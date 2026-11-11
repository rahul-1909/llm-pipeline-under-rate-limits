"""
Stage F: Load Experiments Runner.
Simulates all 5 required load scenarios for Tables 2a and 2b:
1. steady: steady offered rate within capacity
2. burst: sudden burst of concurrent requests
3. sustained_overload: sustained high rate exceeding capacity
4. limit_halved: per-minute limits cut in half (OTPM=500, TPM=4000)
5. daily_exhausted: daily token/request budget exhausted

Features:
- Thread-safe request queue with worker threads and deadline timeout
- Retry policy with exponential backoff and random jitter
- Graceful degradation reply when daily budget is gone
- Direct logging of all requests to results/results.csv
- Tracking of peak queue length and outcome statistics
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import random
import queue
import threading
import concurrent.futures
from typing import Dict, Any, List
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from src.pipeline import RelayPipeline, log_result, RESULTS_CSV
from src.standin_server import StandInModelServer
from src.grader import AnswerGrader

LOAD_QUESTIONS = [
    "What optimizer parameters beta1 and beta2 are used in Attention Is All You Need?",
    "How many layers and attention heads are in the Transformer base model?",
    "What frame rate does the Mimi codec use in Moshi?",
    "What is the theoretical and practical latency of Moshi?",
    "How are chunk boundaries decided in Dynamic Chunking?",
    "What compression ratio in bytes per chunk does Dynamic Chunking achieve?",
    "What dataset was used to fine-tune Moshi for duplex conversation?",
    "What is the label smoothing value used in Transformer training?",
    "What is the maximum path length for self-attention in Table 1 of Attention?",
    "What is the parameter count of the Helium language model?"
]

class LoadScenarioRunner:
    def __init__(self):
        self.pipeline = RelayPipeline()
        self.grader = AnswerGrader()

    def run_scenario(
        self,
        scenario_name: str,
        num_requests: int,
        arrival_pattern: str,  # 'steady', 'burst', 'overload'
        halve_limits: bool = False,
        daily_exhausted: bool = False,
        max_workers: int = 6,
        time_scale: float = 4.0,
        virtual_patience_deadline_s: float = 15.0
    ) -> Dict[str, Any]:
        """
        Executes a load scenario against the stand-in model server.
        Uses time_scale to scale virtual arrival rates and sliding windows.
        Patience deadline is 15s virtual (15 / time_scale real).
        """
        stand_in = StandInModelServer(time_scale=time_scale)
        stand_in.halve_per_minute_limits = halve_limits
        stand_in.daily_budget_exhausted = daily_exhausted
        if daily_exhausted:
            stand_in.daily_budget_limit_requests = 0

        req_queue = queue.Queue()
        peak_queue_len = 0
        queue_lock = threading.Lock()

        scenario_start_mono = time.monotonic()
        results_collector = []
        stop_event = threading.Event()
        real_deadline_s = virtual_patience_deadline_s / time_scale

        def worker():
            nonlocal peak_queue_len
            while not stop_event.is_set() or not req_queue.empty():
                try:
                    item = req_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                with queue_lock:
                    q_len = req_queue.qsize()
                    if q_len > peak_queue_len:
                        peak_queue_len = q_len

                req_id, q_text = item
                try:
                    gen = self.pipeline.process_request(
                        request_id=req_id,
                        question=q_text,
                        scenario=scenario_name,
                        source="stand-in",
                        stand_in=stand_in,
                        retrieval_setting="medium_k4_norerank",
                        max_retries=2,
                        request_deadline_s=real_deadline_s
                    )

                    tokens = []
                    done_event = None
                    for event in gen:
                        if event["type"] == "token":
                            tokens.append(event["content"])
                        elif event["type"] == "done":
                            done_event = event

                    results_collector.append(done_event)
                finally:
                    req_queue.task_done()

        # Start worker threads
        threads = []
        for _ in range(max_workers):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)

        print(f"\n>>> Running Scenario: {scenario_name} (num_requests={num_requests}, pattern={arrival_pattern})", flush=True)

        # Submit requests according to arrival pattern
        for i in range(num_requests):
            req_id = f"load_{scenario_name}_{i+1}"
            q_text = LOAD_QUESTIONS[i % len(LOAD_QUESTIONS)]

            with queue_lock:
                req_queue.put((req_id, q_text))
                q_len = req_queue.qsize()
                if q_len > peak_queue_len:
                    peak_queue_len = q_len

            # Pacing based on pattern (Item 4: offered rates relative to measured 3.6 answers/min capacity)
            # time_scale = 4.0 (1 virtual second = 0.25 real seconds)
            if arrival_pattern == "steady":
                # ~70% of capacity: 2.5 req/min -> 24 virtual seconds = 6.0 real seconds
                time.sleep(24.0 / time_scale)
            elif arrival_pattern == "burst":
                # 20 requests at once
                time.sleep(0.01)
            elif arrival_pattern == "overload":
                # ~3x capacity: 10.8 req/min -> 5.55 virtual seconds = 1.39 real seconds
                time.sleep(5.55 / time_scale)

        # Wait for queue completion
        req_queue.join()
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)

        total_duration_real_s = time.monotonic() - scenario_start_mono
        total_duration_virtual_s = total_duration_real_s * time_scale
        offered_rate_virtual = (num_requests / total_duration_virtual_s) * 60.0

        print(f"Scenario {scenario_name} finished in {total_duration_virtual_s:.1f}s virtual ({total_duration_real_s:.1f}s real) | Offered rate: {offered_rate_virtual:.1f} req/min | Peak queue: {peak_queue_len}", flush=True)
        return {
            "scenario": scenario_name,
            "requests_sent": num_requests,
            "duration_s": round(total_duration_virtual_s, 1),
            "offered_rate": round(offered_rate_virtual, 1),
            "peak_queue_length": peak_queue_len
        }

def run_all_load_scenarios():
    runner = LoadScenarioRunner()

    scenarios = [
        # 1. Steady Load: ~70% capacity (2.5 req/min), 12 requests
        ("steady", 12, "steady", False, False),
        # 2. Sudden Burst: 20 requests at once
        ("burst", 20, "burst", False, False),
        # 3. Sustained Overload: ~3x capacity (10.8 req/min) for 3 minutes (33 requests)
        ("sustained_overload", 33, "overload", False, False),
        # 4. Limit Halved: same offered rate as steady (2.5 req/min), limits halved
        ("limit_halved", 12, "steady", True, False),
        # 5. Daily Budget Exhausted: 10 requests under synthetic daily exhaustion
        ("daily_budget_exhausted", 10, "burst", False, True)
    ]

    summary = []
    for sc_name, count, pattern, halve, daily_ex in scenarios:
        res = runner.run_scenario(
            scenario_name=sc_name,
            num_requests=count,
            arrival_pattern=pattern,
            halve_limits=halve,
            daily_exhausted=daily_ex,
            time_scale=4.0,
            virtual_patience_deadline_s=15.0
        )
        summary.append(res)
        time.sleep(1.0)

    print("\n=== All Load Scenarios Complete ===", flush=True)
    print(pd.DataFrame(summary).to_string(), flush=True)

if __name__ == "__main__":
    run_all_load_scenarios()

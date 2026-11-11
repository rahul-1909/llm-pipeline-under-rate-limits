"""
Run a real steady scenario:
20 requests at about 3 answers/min (~20s spacing between requests)
scenario=steady_real, source=real
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import RelayPipeline, log_result

STEADY_QUESTIONS = [
    "What optimizer settings (beta1, beta2, epsilon) and warmup steps did the Transformer use for training?",
    "For the base Transformer, how many layers, what model dimension, how many attention heads, and what feed-forward size?",
    "In Dynamic Chunking (version 2), what two components decide where chunk boundaries fall?",
    "What is the label smoothing value (epsilon_ls) used during Transformer training, and what effect did the authors observe on perplexity and BLEU score?",
    "In Table 1 of 'Attention Is All You Need', what is the maximum path length between any two positions for a Self-Attention layer compared to a Recurrent layer?",
    "What exact mathematical formula is used for sinusoidal positional encodings at even dimension indices (2i) in Attention Is All You Need?",
    "What conversational dataset was used to fine-tune Moshi to gain its full-duplex spoken dialogue capabilities?",
    "What formula defines the Exponential Moving Average (EMA) applied by the smoothing module in Dynamic Chunking?",
    "What top-1 accuracy does the Transformer reach on ImageNet?",
    "What BLEU score does the Transformer get on WMT 2014 English-to-Czech?",
    "How well does Moshi handle Hindi conversations?",
    "What expansion ratio is used in the bottleneck residual blocks of the ResNet-50 backbone cited in Attention Is All You Need?",
    "What was the total financial training cost in USD or total megawatt-hours of power consumed to train the Moshi 7B foundation model?",
    "In Table 3 of Attention Is All You Need, what English-to-French BLEU score is reported for the Transformer architecture variation with 16 attention heads (row A)?",
    "What BLEU score does Dynamic Chunking reach on the English-to-Japanese translation benchmark?",
    "What optimizer settings (beta1, beta2, epsilon) and warmup steps did the Transformer use for training?",
    "For the base Transformer, how many layers, what model dimension, how many attention heads, and what feed-forward size?",
    "In Dynamic Chunking (version 2), what two components decide where chunk boundaries fall?",
    "What is the label smoothing value (epsilon_ls) used during Transformer training, and what effect did the authors observe on perplexity and BLEU score?",
    "What conversational dataset was used to fine-tune Moshi to gain its full-duplex spoken dialogue capabilities?"
]

def run_steady_real():
    pipeline = RelayPipeline()
    num_requests = len(STEADY_QUESTIONS)
    print(f"=== Starting Real Steady Scenario: {num_requests} requests at ~3/min (20s pacing) ===", flush=True)

    start_mono = time.monotonic()
    for i, q_text in enumerate(STEADY_QUESTIONS):
        req_id = f"steady_real_{i+1}"
        print(f"[{i+1}/{num_requests}] Submitting {req_id}...", flush=True)

        req_start = time.monotonic()
        gen = pipeline.process_request(
            request_id=req_id,
            question=q_text,
            scenario="steady_real",
            source="real",
            retrieval_setting="medium_k4_norerank",
            max_retries=1,
            request_deadline_s=15.0
        )

        tokens = []
        done_event = None
        for ev in gen:
            if ev["type"] == "token":
                tokens.append(ev["content"])
            elif ev["type"] == "done":
                done_event = ev

        elapsed = time.monotonic() - req_start
        status = done_event.get("status", "unknown") if done_event else "unknown"
        ttft = done_event.get("first_token_ms", 0.0) if done_event else 0.0
        tot = done_event.get("total_ms", 0.0) if done_event else 0.0
        print(f"  -> {req_id}: status={status} | TTFT={ttft:.1f}ms | Total={tot:.1f}ms | Elapsed={elapsed:.1f}s", flush=True)

        # Pace at ~3 answers/min: 20s cycle
        if i < num_requests - 1:
            sleep_needed = max(0.5, 20.0 - elapsed)
            print(f"  Pacing: sleeping {sleep_needed:.1f}s to maintain ~3 req/min...", flush=True)
            time.sleep(sleep_needed)

    total_sec = time.monotonic() - start_mono
    offered_rate = (num_requests / total_sec) * 60.0
    print(f"\n=== Real Steady Scenario Complete: {num_requests} requests in {total_sec:.1f}s ({offered_rate:.2f} req/min) ===", flush=True)

if __name__ == "__main__":
    run_steady_real()

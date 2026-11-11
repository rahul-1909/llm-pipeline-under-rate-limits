"""
Test concurrency in-flight limits, exact retry-after behavior,
and thundering-herd collision (3 unjittered vs 3 jittered runs).
Saves results to results/retry_experiments.jsonl.
"""
import os
import time
import json
import random
import concurrent.futures
import httpx
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "results", "retry_experiments.jsonl")
MODEL_NAME = "qwen/qwen3.8-27b"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def log_experiment(event_type: str, data: dict):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event_type, "timestamp": time.time(), **data}) + "\n")

def make_call(client, client_id, max_tokens=15, prompt="Say one word"):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    t0 = time.monotonic()
    resp = client.post(URL, headers=HEADERS, json=payload, timeout=20.0)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    return {
        "client_id": client_id,
        "status_code": resp.status_code,
        "elapsed_ms": round(elapsed_ms, 1),
        "headers": {k: v for k, v in resp.headers.items() if "ratelimit" in k or "retry" in k},
        "body": resp.text if resp.status_code != 200 else resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    }

def trigger_429(client, max_tokens=250):
    for i in range(8):
        r = make_call(client, f"trigger_{i}", max_tokens=max_tokens, prompt="Explain self-attention")
        if r["status_code"] == 429:
            raw_retry = r["headers"].get("retry-after", "5")
            try:
                val = int(raw_retry)
            except ValueError:
                val = 5
            return val, r
        time.sleep(0.05)
    return 5, None

def run_thundering_herd_trials():
    print("=== Running 3 Unjittered vs 3 Jittered Thundering Herd Trials ===")
    with httpx.Client() as client:
        # 1. Three Unjittered Trials
        for trial in range(1, 4):
            print(f"\n[Unjittered Trial {trial}/3] Inducing 429...")
            retry_s, trigger_info = trigger_429(client, max_tokens=250)
            print(f"Triggered 429, retry-after = {retry_s}s. Waiting exactly {retry_s}s (NO jitter)...")
            time.sleep(retry_s)

            # Fire 4 simultaneous calls with zero jitter
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(make_call, client, f"unjittered_t{trial}_c{i}", max_tokens=200, prompt=f"Question {i}") for i in range(4)]
                results = [f.result() for f in futures]

            statuses = [r["status_code"] for r in results]
            print(f"Trial {trial} Unjittered Statuses: {statuses}")
            log_experiment("thundering_herd_unjittered", {
                "trial": trial,
                "waited_seconds": retry_s,
                "statuses": statuses,
                "results": results
            })
            time.sleep(2.0)

        # 2. Three Jittered Trials
        for trial in range(1, 4):
            print(f"\n[Jittered Trial {trial}/3] Inducing 429...")
            retry_s, trigger_info = trigger_429(client, max_tokens=250)
            print(f"Triggered 429, retry-after = {retry_s}s. Applying jitter (retry_after + uniform(0.1, 1.5)s)...")

            def jittered_call(cid):
                jitter = random.uniform(0.1, 1.5)
                time.sleep(retry_s + jitter)
                return make_call(client, cid, max_tokens=200, prompt=f"Jittered Question {cid}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(jittered_call, f"jittered_t{trial}_c{i}") for i in range(4)]
                results = [f.result() for f in futures]

            statuses = [r["status_code"] for r in results]
            print(f"Trial {trial} Jittered Statuses: {statuses}")
            log_experiment("thundering_herd_jittered", {
                "trial": trial,
                "waited_seconds": retry_s,
                "statuses": statuses,
                "results": results
            })
            time.sleep(2.0)

if __name__ == "__main__":
    run_thundering_herd_trials()

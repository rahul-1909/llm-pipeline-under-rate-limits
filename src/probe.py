"""
Stage B: Probe Script.
Sends ~30 real requests to Groq to characterize real service behavior:
- Latency spread (TTFT, total latency)
- Header behavior: x-ratelimit-limit-requests, x-ratelimit-limit-tokens,
  x-ratelimit-remaining-*, x-ratelimit-reset-*, retry-after
- Raw 429 response structure under rapid load
- Token-budget-aware stop condition (caps real usage at ~25,000 tokens)
- Saves all raw interactions to results/probe_raw.jsonl
"""
import os
import sys
import time
import json
import httpx
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

PROBE_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "probe_raw.jsonl")
MODEL_NAME = "qwen/qwen3.8-27b"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment")

# Benchmark questions of varying token sizes
PROBE_PROMPTS = [
    # 1-10: Short queries (~50-100 tokens)
    "What is the learning rate warmup in Attention Is All You Need?",
    "What is the multi-head attention formula?",
    "How many GPUs were used to train the base Transformer?",
    "What does Moshi do?",
    "What is the audio codec used in Moshi?",
    "What is dynamic chunking in sequence modeling?",
    "What dataset was used for English-German translation in Transformer?",
    "What is the vocabulary size in Attention Is All You Need?",
    "What is the frame rate of Moshi's audio tokenizer?",
    "How does dynamic chunking decide chunk boundaries?",

    # 11-20: Medium queries (~200-400 tokens)
    "Describe the difference between the base and big Transformer models in Table 3 of Attention Is All You Need, including layer count, d_model, and training steps.",
    "Explain the dual-stream architecture of Moshi and how it models text and audio simultaneously.",
    "What are the computational complexity advantages of dynamic chunking compared to full quadratic self-attention?",
    "How does Moshi achieve sub-200ms latency for real-time speech conversation?",
    "In Attention Is All You Need, why did the authors choose self-attention over recurrent and convolutional layers?",
    "Summarize the training hyperparameters of Moshi including batch size, optimizer, and weight decay.",
    "Explain the concept of temporal hierarchy in dynamic chunking sequence models.",
    "What regularization techniques were applied during Transformer training in section 5.4?",
    "How does Moshi handle overlapping speech and turn-taking without an explicit turn-taking classifier?",
    "Compare the BLEU scores achieved by Transformer on WMT 2014 English-to-German and English-to-French.",

    # 21-30: Burst queries designed to probe rate limits
    "Probe 21: What is the dimensionality of query, key, and value vectors in Attention?",
    "Probe 22: What is the dimensionality of query, key, and value vectors in Attention?",
    "Probe 23: What is the dimensionality of query, key, and value vectors in Attention?",
    "Probe 24: What is the dimensionality of query, key, and value vectors in Attention?",
    "Probe 25: What is the dimensionality of query, key, and value vectors in Attention?",
    "Probe 26: What is the dimensionality of query, key, and value vectors in Attention?",
    "Probe 27: What is the dimensionality of query, key, and value vectors in Attention?",
    "Probe 28: What is the dimensionality of query, key, and value vectors in Attention?",
    "Probe 29: What is the dimensionality of query, key, and value vectors in Attention?",
    "Probe 30: What is the dimensionality of query, key, and value vectors in Attention?"
]

def run_probe():
    os.makedirs(os.path.dirname(PROBE_OUTPUT_PATH), exist_ok=True)
    # Clear existing probe output
    with open(PROBE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        pass

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    total_tokens_spent = 0
    max_tokens_budget = 25000
    stop_early = False

    print(f"Starting Stage B Probe (~30 requests, model={MODEL_NAME}, token budget={max_tokens_budget})...")

    with httpx.Client(timeout=30.0) as client:
        for idx, prompt in enumerate(PROBE_PROMPTS, 1):
            if stop_early or total_tokens_spent >= max_tokens_budget:
                print(f"Stopping probe early: token budget reached ({total_tokens_spent} tokens).")
                break

            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a concise research assistant. Give a short, factual answer in 2-3 sentences."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 120
            }

            t0 = time.monotonic()
            status_code = None
            resp_headers = {}
            error_body = None
            response_json = {}
            tokens_in = 0
            tokens_out = 0

            try:
                resp = client.post(url, headers=headers, json=payload)
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                status_code = resp.status_code
                resp_headers = {k: v for k, v in resp.headers.items()}

                if resp.status_code == 200:
                    response_json = resp.json()
                    usage = response_json.get("usage", {})
                    tokens_in = usage.get("prompt_tokens", 0)
                    tokens_out = usage.get("completion_tokens", 0)
                    total_tokens_spent += (tokens_in + tokens_out)
                    print(f"[{idx}/30] Status 200 | Latency: {elapsed_ms:.1f}ms | Tokens: {tokens_in} in, {tokens_out} out | Rem Tokens: {resp.headers.get('x-ratelimit-remaining-tokens')}")
                elif resp.status_code == 429:
                    error_body = resp.text
                    print(f"[{idx}/30] Status 429 (RATE LIMITED) | Latency: {elapsed_ms:.1f}ms | Retry-After: {resp.headers.get('retry-after')}")
                    print(f"Raw 429 Body: {error_body[:200]}")
                else:
                    error_body = resp.text
                    print(f"[{idx}/30] Status {resp.status_code} | Latency: {elapsed_ms:.1f}ms | Error: {error_body[:100]}")

                # Check if remaining requests or tokens is dangerously low
                rem_req = resp.headers.get("x-ratelimit-remaining-requests")
                rem_tok = resp.headers.get("x-ratelimit-remaining-tokens")
                if rem_req and int(rem_req) < 20:
                    print(f"Remaining requests dangerously low ({rem_req}). Stopping probe.")
                    stop_early = True

            except Exception as e:
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                status_code = 500
                error_body = str(e)
                print(f"[{idx}/30] Exception: {e}")

            record = {
                "request_num": idx,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt": prompt,
                "status_code": status_code,
                "elapsed_ms": round(elapsed_ms, 1),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "total_tokens_spent_cumulative": total_tokens_spent,
                "rate_limit_headers": {
                    k: v for k, v in resp_headers.items()
                    if k.startswith("x-ratelimit-") or k == "retry-after"
                },
                "all_headers": resp_headers,
                "error_body": error_body
            }

            with open(PROBE_OUTPUT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            # Pacing:
            # For 1-20, pace slightly (0.8s) so we see normal latency spread
            # For 21-30, send without delay to trigger and observe burst rate-limiting
            if idx <= 20:
                time.sleep(0.8)
            else:
                # No sleep during burst probes
                pass

    print(f"\nStage B Probe finished. Total tokens spent: {total_tokens_spent}. Output saved to {PROBE_OUTPUT_PATH}")

if __name__ == "__main__":
    run_probe()

"""
Stage B Extension: Second real probe of 12 full-context pipeline requests.
Paced at 6 seconds apart to stay strictly under the 1,000 OTPM ceiling.
Logged to results/results.csv (scenario=probe_pipeline, source=real)
and results/timings_detail.csv.
"""
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import RelayPipeline

PIPELINE_QUESTIONS = [
    ("pipe-probe-1", "What are the key differences between the base and big models in Table 3 of Attention Is All You Need?"),
    ("pipe-probe-2", "How many layers and attention heads are in the encoder of the Transformer base model?"),
    ("pipe-probe-3", "What is the training dataset and sequence pair count used for English-to-German in Attention?"),
    ("pipe-probe-4", "What hardware and how many GPUs were used to train the Transformer models?"),
    ("pipe-probe-5", "What are the beta1, beta2, and epsilon values used for the Adam optimizer in Attention?"),
    ("pipe-probe-6", "What is Moshi and what is its primary design goal?"),
    ("pipe-probe-7", "What is the audio codec used by Moshi and what is its bitrate or frame rate?"),
    ("pipe-probe-8", "How does Moshi model text and audio tokens simultaneously in its multi-stream architecture?"),
    ("pipe-probe-9", "What is the latency target or overall latency achieved by Moshi in real-time dialogue?"),
    ("pipe-probe-10", "What problem does Dynamic Chunking address in hierarchical sequence modeling?"),
    ("pipe-probe-11", "How are chunk boundaries determined in the Dynamic Chunking paper?"),
    ("pipe-probe-12", "What computational complexity reduction is achieved by Dynamic Chunking over full quadratic attention?")
]

def run_pipeline_probe():
    pipeline = RelayPipeline()
    print("Starting pipeline probe (12 requests, paced at 6s intervals to respect 1,000 OTPM)...")

    for idx, (req_id, q) in enumerate(PIPELINE_QUESTIONS, 1):
        print(f"\n--- [{idx}/12] Request {req_id}: {q[:60]}... ---")
        t0 = time.monotonic()
        gen = pipeline.process_request(
            request_id=req_id,
            question=q,
            scenario="probe_pipeline",
            source="real",
            retrieval_setting="medium_k4_norerank",
            tool_delay_ms=0
        )

        tokens = []
        done_event = None
        for event in gen:
            if event["type"] == "token":
                tokens.append(event["content"])
            elif event["type"] == "done":
                done_event = event

        full_ans = "".join(tokens)
        elapsed = time.monotonic() - t0
        print(f"Status: {done_event.get('status')} | Grade: {done_event.get('grade')} | TTFT: {done_event.get('first_token_ms', 0):.1f}ms | Total: {done_event.get('total_ms', 0):.1f}ms")
        print(f"Tokens in: {done_event.get('tokens_in', 0)}, out: {done_event.get('tokens_out', 0)}")
        print(f"Sample answer: {full_ans[:120]}...")

        # Pace calls by 8 seconds
        if idx < len(PIPELINE_QUESTIONS):
            print("Pacing: sleeping 8.0s...")
            time.sleep(8.0)

    print("\nPipeline probe complete. All calls logged to results/results.csv.")

if __name__ == "__main__":
    run_pipeline_probe()

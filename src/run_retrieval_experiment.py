"""
Stage E: Retrieval Experiment Runner.
Evaluates 20 benchmark questions across 3 settings:
1. small_k3_norerank
2. medium_k4_norerank
3. large_k3_rerank

Runs on real Groq, max_retries=0, dynamically paced using x-ratelimit-remaining-tokens.
Logs every request to results/results.csv and grades with AnswerGrader.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import RelayPipeline, log_result, RESULTS_CSV
from src.grader import AnswerGrader
import pandas as pd

# The 20 benchmark questions combining mine.md (Q1-Q7) and drafted.md (Q8-Q20)
BENCHMARK_QUESTIONS = [
    # --- Questions from questions/mine.md (Q1 - Q7) ---
    {
        "id": "q1",
        "question": "What optimizer settings (beta1, beta2, epsilon) and warmup steps did the Transformer use for training?",
        "type": "answerable",
        "expected_pages": [("1706.03762v7", 7)],
        "key_facts": [["beta1=0.9", "0.9"], ["beta2=0.98", "0.98"], ["1e-9", "10^-9", "10^{-9}", "109"], ["4000", "4,000"]]
    },
    {
        "id": "q2",
        "question": "For the base Transformer, how many layers, what model dimension, how many attention heads, and what feed-forward size?",
        "type": "answerable",
        "expected_pages": [("1706.03762v7", 9), ("1706.03762v7", 3)],
        "key_facts": [["6 layers", "6", "n=6"], ["512", "d_model=512"], ["8 heads", "8", "h=8"], ["2048", "d_ff=2048"]]
    },
    {
        "id": "q3",
        "question": "What frame rate and bitrate does Moshi's Mimi codec use, and what latency does Moshi report?",
        "type": "answerable",
        "expected_pages": [("2410.00037v1", 11), ("2410.00037v1", 24), ("2410.00037v1", 1), ("2410.00037v1", 2)],
        "key_facts": [["12.5 hz", "12.5hz", "12.5"], ["1.1 kbps", "1.1kbps", "1.1"], ["160 ms", "160ms"], ["200 ms", "200ms"]]
    },
    {
        "id": "q4",
        "question": "In Dynamic Chunking (version 2), what two components decide where chunk boundaries fall?",
        "type": "answerable",
        "expected_pages": [("2507.07955v2", 3), ("2507.07955v2", 6)],
        "key_facts": [["routing", "routing module"], ["cosine similarity", "downsampler", "boundary"]]
    },
    {
        "id": "q5",
        "question": "What top-1 accuracy does the Transformer reach on ImageNet?",
        "type": "unanswerable",
        "expected_pages": [],
        "key_facts": []
    },
    {
        "id": "q6",
        "question": "What BLEU score does the Transformer get on WMT 2014 English-to-Czech?",
        "type": "unanswerable",
        "expected_pages": [],
        "key_facts": []
    },
    {
        "id": "q7",
        "question": "How well does Moshi handle Hindi conversations?",
        "type": "unanswerable",
        "expected_pages": [],
        "key_facts": []
    },

    # --- Questions from questions/drafted.md (Q8 - Q20) ---
    {
        "id": "q8",
        "question": "What is the label smoothing value (epsilon_ls) used during Transformer training, and what effect did the authors observe on perplexity and BLEU score?",
        "type": "answerable",
        "expected_pages": [("1706.03762v7", 8)],
        "key_facts": [["0.1", "0.10"], ["perplexity"], ["bleu", "accuracy"]]
    },
    {
        "id": "q9",
        "question": "In Table 1 of 'Attention Is All You Need', what is the maximum path length between any two positions for a Self-Attention layer compared to a Recurrent layer?",
        "type": "answerable",
        "expected_pages": [("1706.03762v7", 6)],
        "key_facts": [["o(1)", "o( 1 )", "constant"], ["o(n)", "o( n )"]]
    },
    {
        "id": "q10",
        "question": "What exact mathematical formula is used for sinusoidal positional encodings at even dimension indices (2i) in Attention Is All You Need?",
        "type": "answerable",
        "expected_pages": [("1706.03762v7", 6)],
        "key_facts": [["sin"], ["10000"], ["d_model", "dmodel", "2i"]]
    },
    {
        "id": "q11",
        "question": "What are the parameter counts of the Helium language model and the Moshi speech-text model in the Moshi paper?",
        "type": "answerable",
        "expected_pages": [("2410.00037v1", 1), ("2410.00037v1", 8)],
        "key_facts": [["7b", "7 billion"]]
    },
    {
        "id": "q12",
        "question": "How does Moshi model parallel audio streams, and whose speech channels do they represent?",
        "type": "answerable",
        "expected_pages": [("2410.00037v1", 1), ("2410.00037v1", 3), ("2410.00037v1", 4)],
        "key_facts": [["user"], ["moshi", "agent"], ["stream", "duplex"]]
    },
    {
        "id": "q13",
        "question": "What conversational dataset was used to fine-tune Moshi to gain its full-duplex spoken dialogue capabilities?",
        "type": "answerable",
        "expected_pages": [("2410.00037v1", 8), ("2410.00037v1", 19)],
        "key_facts": [["fisher"]]
    },
    {
        "id": "q14",
        "question": "What compression ratio (in bytes per chunk) does Dynamic Chunking naturally learn, and what standard tokenization format does it match?",
        "type": "answerable",
        "expected_pages": [("2507.07955v2", 3)],
        "key_facts": [["4.5-5", "4.5 to 5", "4.5"], ["bpe", "byte-pair", "tokenizer"]]
    },
    {
        "id": "q15",
        "question": "What formula defines the Exponential Moving Average (EMA) applied by the smoothing module in Dynamic Chunking?",
        "type": "answerable",
        "expected_pages": [("2507.07955v2", 7)],
        "key_facts": [["p_t", "pt", "p t", "alpha", "α"], ["(5)", "equation 5", "ema", "z_t", "zt", "h_t"]]
    },
    {
        "id": "q16",
        "question": "Compare the attention head count of the Transformer base model with the head count in Moshi's Depth Transformer.",
        "type": "answerable",
        "expected_pages": [("1706.03762v7", 9), ("2410.00037v1", 8)],
        "key_facts": [["8"], ["16"]]
    },
    {
        "id": "q17",
        "question": "What expansion ratio is used in the bottleneck residual blocks of the ResNet-50 backbone cited in Attention Is All You Need?",
        "type": "unanswerable",
        "expected_pages": [],
        "key_facts": []
    },
    {
        "id": "q18",
        "question": "What was the total financial training cost in USD or total megawatt-hours of power consumed to train the Moshi 7B foundation model?",
        "type": "unanswerable",
        "expected_pages": [],
        "key_facts": []
    },
    {
        "id": "q19",
        "question": "In Table 3 of Attention Is All You Need, what English-to-French BLEU score is reported for the Transformer architecture variation with 16 attention heads (row A)?",
        "type": "unanswerable",
        "expected_pages": [],
        "key_facts": []
    },
    {
        "id": "q20",
        "question": "What BLEU score does Dynamic Chunking reach on the English-to-Japanese translation benchmark?",
        "type": "unanswerable",
        "expected_pages": [],
        "key_facts": []
    }
]

SETTINGS = [
    "small_k3_norerank",
    "medium_k4_norerank",
    "large_k3_rerank"
]

def check_and_pace_from_headers(pipeline: RelayPipeline, min_safe_tokens: int = 3500):
    """
    Queries Groq rate limit remaining tokens; sleeps if below min_safe_tokens.
    """
    try:
        raw = pipeline.groq_client.chat.completions.with_raw_response.create(
            model=pipeline.model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5
        )
        rem_tokens = int(raw.headers.get("x-ratelimit-remaining-tokens", 8000))
        rem_requests = int(raw.headers.get("x-ratelimit-remaining-requests", 1000))
        reset_tokens_str = raw.headers.get("x-ratelimit-reset-tokens", "1s")

        # Parse reset string: e.g. '500ms' or '12.5s'
        reset_sec = 2.0
        if "ms" in reset_tokens_str:
            reset_sec = float(reset_tokens_str.replace("ms", "")) / 1000.0
        elif "s" in reset_tokens_str:
            reset_sec = float(reset_tokens_str.replace("s", ""))

        if rem_tokens < min_safe_tokens:
            sleep_duration = max(reset_sec + 1.0, 3.0)
            print(f"Pacing: remaining tokens ({rem_tokens}) below {min_safe_tokens}. Sleeping {sleep_duration:.1f}s until reset...")
            time.sleep(sleep_duration)
        else:
            # Baseline spacing between requests
            time.sleep(1.5)

    except Exception as e:
        time.sleep(3.0)

def run_retrieval_experiments():
    pipeline = RelayPipeline()
    grader = AnswerGrader()

    total_runs = len(SETTINGS) * len(BENCHMARK_QUESTIONS)
    print(f"=== Starting Stage E: Retrieval Experiment (3 settings x 20 questions = {total_runs} runs) ===")

    current_idx = 0
    for setting in SETTINGS:
        print(f"\n=======================================================")
        print(f"   EVALUATING RETRIEVAL SETTING: {setting}")
        print(f"=======================================================\n")

        for q in BENCHMARK_QUESTIONS:
            current_idx += 1
            req_id = f"retrieval_{setting}_{q['id']}"
            print(f"[{current_idx}/{total_runs}] Running {req_id} ({q['type']})...")

            # Execute pipeline
            gen = pipeline.process_request(
                request_id=req_id,
                question=q["question"],
                scenario="retrieval_exp",
                source="real",
                retrieval_setting=setting,
                max_retries=0
            )

            tokens = []
            done_event = None
            for event in gen:
                if event["type"] == "token":
                    tokens.append(event["content"])
                elif event["type"] == "done":
                    done_event = event

            full_answer = "".join(tokens)
            status = done_event.get("status", "ok") if done_event else "error"

            # Grade with deterministic grader
            grade_res = grader.grade(
                question_type=q["type"],
                expected_pages=q["expected_pages"],
                key_facts=q["key_facts"],
                answer_text=full_answer,
                status=status
            )
            final_grade = grade_res["grade"]

            # Update results.csv answer_grade with the deterministic grade
            _update_results_grade(req_id, final_grade)

            ttft = done_event.get("first_token_ms", 0.0) if done_event else 0.0
            tot = done_event.get("total_ms", 0.0) if done_event else 0.0
            print(f"  -> Grade: {final_grade} | TTFT: {ttft:.1f}ms | Total: {tot:.1f}ms | Reason: {grade_res['reason']}")

            # Pace dynamically from headers
            check_and_pace_from_headers(pipeline, min_safe_tokens=3500)

    print("\nStage E Retrieval Experiment finished! All 60 requests graded and logged to results/results.csv.")

def _update_results_grade(request_id: str, grade: str):
    try:
        df = pd.read_csv(RESULTS_CSV)
        mask = df["request_id"] == request_id
        if mask.any():
            df.loc[mask, "answer_grade"] = grade
            df.to_csv(RESULTS_CSV, index=False)
    except Exception as e:
        print(f"Warning: could not update grade in results.csv for {request_id}: {e}")

if __name__ == "__main__":
    run_retrieval_experiments()

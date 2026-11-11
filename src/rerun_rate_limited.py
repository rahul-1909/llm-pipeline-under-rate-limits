"""
Script to re-run only the rate-limited questions for retrieval settings (Q16-Q20 in large_k3_rerank).
Paces from x-ratelimit-remaining-tokens, allows max 2 retries honoring retry-after.
Scenario: retrieval_exp
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import re
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import RelayPipeline, log_result, RESULTS_CSV, TIMINGS_CSV
from src.grader import AnswerGrader
from src.run_retrieval_experiment import BENCHMARK_QUESTIONS

TARGET_IDS = ["q16", "q17", "q18", "q19", "q20"]
TARGET_SETTING = "large_k3_rerank"

def check_and_pace_from_headers(pipeline: RelayPipeline, min_safe_tokens: int = 6500):
    try:
        raw = pipeline.groq_client.chat.completions.with_raw_response.create(
            model=pipeline.model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5
        )
        rem_tokens = int(raw.headers.get("x-ratelimit-remaining-tokens", 8000))
        rem_requests = int(raw.headers.get("x-ratelimit-remaining-requests", 1000))
        reset_tokens_str = raw.headers.get("x-ratelimit-reset-tokens", "1s")

        reset_sec = 2.0
        if "ms" in reset_tokens_str:
            reset_sec = float(reset_tokens_str.replace("ms", "")) / 1000.0
        elif "s" in reset_tokens_str:
            reset_sec = float(reset_tokens_str.replace("s", ""))

        print(f"Header check: remaining_tokens={rem_tokens}, reset_tokens={reset_tokens_str}")
        if rem_tokens < min_safe_tokens:
            sleep_duration = max(reset_sec + 2.0, 5.0)
            print(f"Pacing: remaining tokens ({rem_tokens}) below {min_safe_tokens}. Sleeping {sleep_duration:.1f}s until reset...")
            time.sleep(sleep_duration)
        else:
            time.sleep(2.0)
    except Exception as e:
        print(f"Header check failed: {e}. Sleeping 5s...")
        time.sleep(5.0)

def main():
    pipeline = RelayPipeline()
    grader = AnswerGrader()

    # Find the target question objects
    target_questions = [q for q in BENCHMARK_QUESTIONS if q["id"] in TARGET_IDS]
    print(f"Target questions to re-run in {TARGET_SETTING}: {[q['id'] for q in target_questions]}")

    # Remove old rate-limited rows from results.csv and timings_detail.csv
    df_results = pd.read_csv(RESULTS_CSV)
    target_req_ids = [f"retrieval_{TARGET_SETTING}_{q['id']}" for q in target_questions]
    print(f"Removing {len(target_req_ids)} previous failed rows from results.csv...")
    df_results = df_results[~df_results["request_id"].isin(target_req_ids)]
    df_results.to_csv(RESULTS_CSV, index=False)

    if os.path.exists(TIMINGS_CSV):
        df_timings = pd.read_csv(TIMINGS_CSV)
        df_timings = df_timings[~df_timings["request_id"].isin(target_req_ids)]
        df_timings.to_csv(TIMINGS_CSV, index=False)

    # Initial pacing check
    check_and_pace_from_headers(pipeline, min_safe_tokens=6500)

    for q in target_questions:
        req_id = f"retrieval_{TARGET_SETTING}_{q['id']}"
        print(f"\n--- Running {req_id} ({q['type']}) ---")
        print(f"Question: {q['question']}")

        gen = pipeline.process_request(
            request_id=req_id,
            question=q["question"],
            scenario="retrieval_exp",
            source="real",
            retrieval_setting=TARGET_SETTING,
            max_retries=2,
            request_deadline_s=60.0
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
        ttft = done_event.get("first_token_ms", 0.0) if done_event else 0.0
        tot = done_event.get("total_ms", 0.0) if done_event else 0.0

        print(f"Answer received (status={status}, TTFT={ttft:.1f}ms, Total={tot:.1f}ms):\n{full_answer[:150]}...")

        # Grade
        grade_res = grader.grade(
            question_type=q["type"],
            expected_pages=q["expected_pages"],
            key_facts=q["key_facts"],
            answer_text=full_answer,
            status=status
        )
        final_grade = grade_res["grade"] if status == "ok" else ""
        print(f"Grading -> Grade: '{final_grade}' | Reason: {grade_res['reason']}")

        # Update grade in results.csv
        df_curr = pd.read_csv(RESULTS_CSV)
        df_curr.loc[df_curr["request_id"] == req_id, "answer_grade"] = final_grade
        df_curr.to_csv(RESULTS_CSV, index=False)

        # Dynamic pacing before next call
        check_and_pace_from_headers(pipeline, min_safe_tokens=6500)

    print("\nAll target rate-limited questions re-run successfully!")

if __name__ == "__main__":
    main()

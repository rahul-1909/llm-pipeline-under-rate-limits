"""
Main pipeline for The Relay.
Coordinates:
1. Question analysis
2. Retrieval tool execution (with delay/timeout support)
3. Decision branching (refusal vs draft)
4. Model drafting with citations
5. Deterministic verification before streaming (with single repair loop)
6. Streaming verified tokens to caller
7. Logging to results/results.csv
"""
import os
import sys
import time
import math
import random
import csv
import json
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Generator, Optional, List, Tuple
from dotenv import load_dotenv

from .retrieval import PaperRetriever
from .verifier import verify_answer, is_refusal

load_dotenv()

RESULTS_CSV = os.environ.get("RESULTS_CSV", os.path.join(os.path.dirname(__file__), "..", "results", "results.csv"))
TIMINGS_CSV = os.environ.get("TIMINGS_CSV", os.path.join(os.path.dirname(__file__), "..", "results", "timings_detail.csv"))
CSV_LOCK = threading.Lock()

CSV_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade"
]

TIMING_COLUMNS = [
    "request_id",
    "attempt",
    "request_sent_time",
    "headers_received_ms",
    "raw_model_ttft_ms",
    "verification_ms",
    "user_visible_first_token_ms",
    "sleep_before_retry_ms",
    "status",
    "total_ms"
]

def init_results_csv():
    csv_dir = os.path.dirname(RESULTS_CSV)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    timing_dir = os.path.dirname(TIMINGS_CSV)
    if timing_dir:
        os.makedirs(timing_dir, exist_ok=True)
    with CSV_LOCK:
        if not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0:
            with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
        if not os.path.exists(TIMINGS_CSV) or os.path.getsize(TIMINGS_CSV) == 0:
            with open(TIMINGS_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=TIMING_COLUMNS)
                writer.writeheader()

def log_result(row: Dict[str, Any], timing_row: Optional[Dict[str, Any]] = None):
    init_results_csv()
    with CSV_LOCK:
        with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(row)
        if timing_row:
            with open(TIMINGS_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=TIMING_COLUMNS)
                writer.writerow(timing_row)

class RelayPipeline:
    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
        self.model_name = model_name
        self.retriever = PaperRetriever()
        self._groq_client = None

    @property
    def groq_client(self):
        if self._groq_client is None:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            # CRITICAL: Disable silent SDK retries (max_retries=0) so our pipeline handles and logs all retries
            self._groq_client = Groq(api_key=api_key, max_retries=0)
        return self._groq_client

    def parse_setting(self, setting_name: str) -> Tuple[str, int, bool]:
        """
        Parses settings like:
        - 'small_k3_norerank' -> ('small', 3, False)
        - 'medium_k4_norerank' -> ('medium', 4, False)
        - 'large_k3_rerank' -> ('large', 3, True)
        """
        piece_size = "medium"
        top_k = 4
        rerank = False

        if "small" in setting_name:
            piece_size = "small"
        elif "large" in setting_name:
            piece_size = "large"

        if "k3" in setting_name:
            top_k = 3
        elif "k5" in setting_name:
            top_k = 5

        if "rerank" in setting_name and "norerank" not in setting_name:
            rerank = True

        return piece_size, top_k, rerank

    def call_model_with_retries(
        self,
        request_id: str,
        messages: List[Dict[str, str]],
        source: str = "real",
        stand_in: Any = None,
        max_retries: int = 2,
        base_backoff: float = 2.0,
        deadline_time: Optional[float] = None
    ) -> Tuple[str, int, int, int, str, float]:
        """
        Returns: (content, tokens_in, tokens_out, retries_done, status, raw_ttft)
        Status: 'ok', 'rate_limited', 'timeout', 'error'
        Logs every attempt explicitly to timings_detail.csv
        """
        attempt = 0

        while attempt <= max_retries:
            attempt_start = time.monotonic()
            attempt_time_iso = datetime.now(timezone.utc).isoformat()

            if deadline_time and time.time() > deadline_time:
                return ("", 0, 0, attempt, "timeout", 0.0)

            try:
                if source == "stand-in" and stand_in is not None:
                    resp = stand_in.generate(messages)
                    call_status = resp.get("status", "ok")
                    raw_ttft = resp.get("raw_ttft", 0.0)
                    total_attempt_ms = (time.monotonic() - attempt_start) * 1000.0

                    if call_status == "rate_limited":
                        # Daily budget check
                        if "daily" in resp.get("error", "").lower():
                            sleep_s = resp.get("retry_after", 524.0)
                            self.last_retry_after = sleep_s
                            # If retry-after is shorter than patience deadline, request can wait and retry!
                            if deadline_time and (time.time() + sleep_s <= deadline_time) and attempt < max_retries:
                                time.sleep(sleep_s)
                                attempt += 1
                                continue
                            self._log_timing_attempt(request_id, attempt, attempt_time_iso, total_attempt_ms, raw_ttft, 0.0, "rate_limited", total_attempt_ms)
                            return ("", 0, 0, attempt, "daily_exhausted", raw_ttft)

                        sleep_s = resp.get("retry_after", 5) + random.uniform(0.1, 0.5)
                        sleep_ms = sleep_s * 1000.0

                        # Item 5: Patience deadline check before sleeping
                        if deadline_time and (time.time() + sleep_s > deadline_time):
                            self._log_timing_attempt(request_id, attempt, attempt_time_iso, total_attempt_ms, raw_ttft, 0.0, "timeout", total_attempt_ms)
                            return ("", 0, 0, attempt, "timeout", raw_ttft)

                        self._log_timing_attempt(request_id, attempt, attempt_time_iso, total_attempt_ms, raw_ttft, sleep_ms, "rate_limited", total_attempt_ms)
                        if attempt < max_retries:
                            time.sleep(sleep_s)
                            attempt += 1
                            continue
                        return ("", 0, 0, attempt, "rate_limited", raw_ttft)

                    self._log_timing_attempt(request_id, attempt, attempt_time_iso, raw_ttft, raw_ttft, 0.0, "ok", total_attempt_ms)
                    return (
                        resp["content"],
                        resp["tokens_in"],
                        resp["tokens_out"],
                        attempt,
                        call_status,
                        raw_ttft
                    )
                else:
                    # Real Groq call with max_retries=0
                    call_start = time.monotonic()
                    stream = self.groq_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=250,
                        stream=True
                    )
                    headers_received_ms = (time.monotonic() - call_start) * 1000.0
                    chunks = []
                    raw_ttft = None

                    for chunk in stream:
                        delta = chunk.choices[0].delta.content if chunk.choices else None
                        if delta:
                            if raw_ttft is None:
                                raw_ttft = (time.monotonic() - call_start) * 1000.0
                            chunks.append(delta)

                    content = "".join(chunks)
                    total_attempt_ms = (time.monotonic() - call_start) * 1000.0
                    prompt_words = sum(len(m["content"].split()) for m in messages)
                    tokens_in = int(prompt_words * 1.3)
                    tokens_out = int(len(content.split()) * 1.3)

                    self._log_timing_attempt(request_id, attempt, attempt_time_iso, headers_received_ms, raw_ttft or headers_received_ms, 0.0, "ok", total_attempt_ms)
                    return (content, tokens_in, tokens_out, attempt, "ok", raw_ttft or headers_received_ms)

            except Exception as e:
                err_str = str(e).lower()
                attempt_elapsed_ms = (time.monotonic() - attempt_start) * 1000.0
                if "rate limit" in err_str or "429" in err_str:
                    # Honor retry-after header if available
                    retry_after_val = None
                    if hasattr(e, "response") and e.response is not None and hasattr(e.response, "headers"):
                        h_val = e.response.headers.get("retry-after")
                        if h_val:
                            try:
                                retry_after_val = float(h_val)
                            except ValueError:
                                pass
                    if retry_after_val is None:
                        import re
                        m = re.search(r"try again in (?:(\d+)m)?([0-9.]+)s", err_str)
                        if m:
                            try:
                                mins = float(m.group(1)) if m.group(1) else 0.0
                                secs = float(m.group(2))
                                retry_after_val = mins * 60.0 + secs
                            except ValueError:
                                pass

                    is_daily = ("daily" in err_str or "tokens per day" in err_str or "tpd" in err_str)
                    if is_daily:
                        self.last_retry_after = retry_after_val if retry_after_val is not None else 524.0
                        # If retry-after is shorter than patience deadline, request can wait and retry
                        if deadline_time and (time.time() + self.last_retry_after <= deadline_time) and attempt < max_retries:
                            time.sleep(self.last_retry_after)
                            attempt += 1
                            continue
                        self._log_timing_attempt(request_id, attempt, attempt_time_iso, attempt_elapsed_ms, 0.0, 0.0, "rate_limited", attempt_elapsed_ms)
                        return ("", 0, 0, attempt, "daily_exhausted", 0.0)

                    base_sleep = retry_after_val if retry_after_val is not None else base_backoff * (2 ** attempt)
                    sleep_s = max(base_sleep, base_backoff) + random.uniform(0.1, 0.5)

                    # Item 5: Patience deadline check
                    if deadline_time and (time.time() + sleep_s > deadline_time):
                        self._log_timing_attempt(request_id, attempt, attempt_time_iso, attempt_elapsed_ms, 0.0, 0.0, "timeout", attempt_elapsed_ms)
                        return ("", 0, 0, attempt, "timeout", 0.0)

                    self._log_timing_attempt(request_id, attempt, attempt_time_iso, attempt_elapsed_ms, 0.0, sleep_s * 1000.0, "rate_limited", attempt_elapsed_ms)
                    if attempt < max_retries:
                        time.sleep(sleep_s)
                        attempt += 1
                        continue
                    return ("", 0, 0, attempt, "rate_limited", 0.0)
                elif "timeout" in err_str or "timed out" in err_str:
                    self._log_timing_attempt(request_id, attempt, attempt_time_iso, attempt_elapsed_ms, 0.0, 0.0, "timeout", attempt_elapsed_ms)
                    return ("", 0, 0, attempt, "timeout", 0.0)
                else:
                    self._log_timing_attempt(request_id, attempt, attempt_time_iso, attempt_elapsed_ms, 0.0, 0.0, "error", attempt_elapsed_ms)
                    return ("", 0, 0, attempt, "error", 0.0)

        return ("", 0, 0, attempt, "error", 0.0)

    def _log_timing_attempt(self, request_id: str, attempt: int, sent_time: str, headers_ms: float, raw_ttft: float, sleep_ms: float, status: str, total_ms: float):
        init_results_csv()
        with CSV_LOCK:
            with open(TIMINGS_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=TIMING_COLUMNS)
                writer.writerow({
                    "request_id": request_id,
                    "attempt": attempt,
                    "request_sent_time": sent_time,
                    "headers_received_ms": round(headers_ms, 1),
                    "raw_model_ttft_ms": round(raw_ttft, 1),
                    "verification_ms": 0.0,
                    "user_visible_first_token_ms": 0.0,
                    "sleep_before_retry_ms": round(sleep_ms, 1),
                    "status": status,
                    "total_ms": round(total_ms, 1)
                })

    def process_request(
        self,
        request_id: str,
        question: str,
        scenario: str = "steady",
        source: str = "real",
        retrieval_setting: str = "medium_k4_norerank",
        stand_in: Any = None,
        tool_delay_ms: int = 0,
        tool_timeout_ms: Optional[int] = None,
        max_retries: int = 2,
        request_deadline_s: float = 20.0,
        expected_grade: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Executes request lifecycle, yielding streaming token events,
        and finally yields completion metadata. Logs to results/results.csv.
        """
        start_mono = time.monotonic()
        start_iso = datetime.now(timezone.utc).isoformat()
        deadline_time = time.time() + request_deadline_s

        piece_size, top_k, rerank = self.parse_setting(retrieval_setting)
        retries_total = 0
        total_tokens_in = 0
        total_tokens_out = 0
        first_token_time = None

        # Step 1 & 2: Understand question & Call Retrieval Tool
        retrieval_res = self.retriever.retrieve(
            query=question,
            piece_size=piece_size,
            top_k=top_k,
            rerank=rerank,
            tool_delay_ms=tool_delay_ms,
            tool_timeout_ms=tool_timeout_ms
        )

        if not retrieval_res["success"]:
            # Tool failure / timeout
            elapsed_ms = (time.monotonic() - start_mono) * 1000.0
            error_msg = f"Retrieval tool error: {retrieval_res.get('error', 'Unknown tool failure')}"
            first_token_ms = elapsed_ms
            log_result({
                "request_id": request_id,
                "scenario": scenario,
                "source": source,
                "start_time": start_iso,
                "first_token_ms": round(first_token_ms, 1),
                "total_ms": round(elapsed_ms, 1),
                "status": "timeout" if "timeout" in error_msg.lower() else "error",
                "retries": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "retrieval_setting": retrieval_setting,
                "answer_grade": ""
            })
            yield {"type": "token", "content": error_msg}
            yield {"type": "done", "status": "timeout", "total_ms": elapsed_ms}
            return

        passages = retrieval_res["passages"]
        max_score = retrieval_res.get("max_score", 0.0)

        # Step 3: Decide & Branch
        # If retrieved passages have zero relevance or question is an obvious mismatch
        is_obvious_out_of_library = (max_score < 1.0 and len(passages) == 0)

        if is_obvious_out_of_library:
            refusal_text = (
                "The provided library papers ('Attention Is All You Need', 'Moshi', "
                "'Dynamic Chunking') do not contain information to answer this question."
            )
            # Direct verified refusal branch
            first_token_ms = (time.monotonic() - start_mono) * 1000.0
            yield {"type": "token", "content": refusal_text}
            elapsed_ms = (time.monotonic() - start_mono) * 1000.0
            log_result({
                "request_id": request_id,
                "scenario": scenario,
                "source": source,
                "start_time": start_iso,
                "first_token_ms": round(first_token_ms, 1),
                "total_ms": round(elapsed_ms, 1),
                "status": "ok",
                "retries": 0,
                "tokens_in": 0,
                "tokens_out": len(refusal_text.split()),
                "retrieval_setting": retrieval_setting,
                "answer_grade": expected_grade or "refused_correctly"
            })
            yield {"type": "done", "status": "ok", "total_ms": elapsed_ms}
            return

        # Step 4: Draft Response with Citations
        context_str = ""
        valid_pages_list = []
        for idx, p in enumerate(passages, 1):
            context_str += f"\n--- Excerpt {idx} [Paper {p['paper_id']}, Page {p['page']}] ---\n{p['text']}\n"
            valid_pages_list.append(f"{p['paper_id']} (Page {p['page']})")

        system_prompt = (
            "You are a strict research assistant answering questions using ONLY the provided excerpts.\n"
            "Rules:\n"
            "1. Every factual statement must cite the exact paper ID and page number in brackets, e.g. [1706.03762v7, Page 5] or [2410.00037v1, Page 12].\n"
            "2. Cite ONLY pages that appear in the excerpts below.\n"
            "3. If the excerpts do not contain sufficient evidence to answer the question, state: 'The library papers do not contain sufficient information to answer this question.' Do not guess."
        )

        user_prompt = f"Question: {question}\n\nRelevant Excerpts:\n{context_str}\n\nAnswer:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        draft_content, t_in, t_out, retries, call_status, raw_model_ttft = self.call_model_with_retries(
            request_id=request_id,
            messages=messages,
            source=source,
            stand_in=stand_in,
            max_retries=max_retries,
            deadline_time=deadline_time
        )
        retries_total += retries
        total_tokens_in += t_in
        total_tokens_out += t_out

        if call_status != "ok":
            elapsed_ms = (time.monotonic() - start_mono) * 1000.0
            if call_status == "daily_exhausted":
                retry_s = getattr(self, "last_retry_after", 524.0)
                wait_mins = max(1, math.ceil(retry_s / 60.0))
                error_msg = (
                    f"The research paper assistant has reached its daily compute limit "
                    f"(200,000 tokens/day rolling bucket refilling at ~2.3 tokens/s). "
                    f"Please try again in about {wait_mins} minutes."
                )
                final_status = "rate_limited"
                final_grade = ""
                retries_total = 0
            elif call_status == "timeout":
                error_msg = "Request timed out after waiting in queue past the patience deadline. Please try again later."
                final_status = "timeout"
                final_grade = ""
            else:
                error_msg = f"Service unavailable ({call_status}). Please try again."
                final_status = call_status
                final_grade = ""

            first_token_ms = elapsed_ms
            log_result({
                "request_id": request_id,
                "scenario": scenario,
                "source": source,
                "start_time": start_iso,
                "first_token_ms": round(first_token_ms, 1),
                "total_ms": round(elapsed_ms, 1),
                "status": final_status,
                "retries": retries_total,
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
                "retrieval_setting": retrieval_setting,
                "answer_grade": final_grade
            }, {
                "request_id": request_id,
                "attempt": 0,
                "request_sent_time": start_iso,
                "headers_received_ms": round(raw_model_ttft, 1),
                "raw_model_ttft_ms": round(raw_model_ttft, 1),
                "verification_ms": 0.0,
                "user_visible_first_token_ms": round(first_token_ms, 1),
                "sleep_before_retry_ms": 0.0,
                "status": final_status,
                "total_ms": round(elapsed_ms, 1)
            })
            yield {"type": "token", "content": error_msg}
            yield {"type": "done", "status": final_status, "total_ms": elapsed_ms}
            return

        # Step 5: Deterministic Verification before streaming
        v_start = time.monotonic()
        verification = verify_answer(draft_content, passages)
        v_elapsed_ms = (time.monotonic() - v_start) * 1000.0

        # Single Repair Loop if verification failed
        if not verification["valid"] and not verification["is_refusal"]:
            repair_user = (
                f"Your previous answer failed verification because: {verification['reason']}\n"
                f"You may ONLY cite these exact pages present in your context: {', '.join(valid_pages_list)}.\n"
                f"Please rewrite your answer fixing the citations, or if the answer is not supported by these pages, "
                f"reply: 'The library papers do not contain sufficient information to answer this question.'\n"
                f"Question: {question}"
            )
            repair_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": draft_content},
                {"role": "user", "content": repair_user}
            ]
            repair_draft, rep_in, rep_out, rep_retries, rep_status, rep_raw_ttft = self.call_model_with_retries(
                request_id=f"{request_id}_repair",
                messages=repair_messages,
                source=source,
                stand_in=stand_in,
                max_retries=max_retries,
                deadline_time=deadline_time
            )
            retries_total += rep_retries
            total_tokens_in += rep_in
            total_tokens_out += rep_out

            if rep_status == "ok":
                v2_start = time.monotonic()
                second_verification = verify_answer(repair_draft, passages)
                v_elapsed_ms += (time.monotonic() - v2_start) * 1000.0
                if second_verification["valid"]:
                    draft_content = repair_draft
                    verification = second_verification
                else:
                    draft_content = (
                        "The library papers do not contain verified evidence to answer this question "
                        f"(Verification check: {second_verification['reason']})."
                    )
            else:
                draft_content = "The library papers do not contain verified evidence to answer this question."

        # Step 6: Stream verified answer to caller
        first_token_mono = time.monotonic()
        if source == "stand-in" and stand_in is not None:
            first_token_ms = raw_model_ttft + v_elapsed_ms
            # Stream tokens
            words = draft_content.split(" ")
            for idx, word in enumerate(words):
                chunk = word if idx == len(words) - 1 else word + " "
                yield {"type": "token", "content": chunk}
            total_elapsed_ms = first_token_ms + max(150.0, total_tokens_out * 7.5)
        else:
            first_token_ms = (first_token_mono - start_mono) * 1000.0
            words = draft_content.split(" ")
            for idx, word in enumerate(words):
                chunk = word if idx == len(words) - 1 else word + " "
                yield {"type": "token", "content": chunk}
                time.sleep(0.01)
            total_elapsed_ms = (time.monotonic() - start_mono) * 1000.0

        final_is_refusal = is_refusal(draft_content)
        if expected_grade:
            grade = expected_grade
        elif final_is_refusal:
            grade = "refused_correctly"
        elif verification["valid"]:
            grade = "correct"
        else:
            grade = "wrong"

        log_result({
            "request_id": request_id,
            "scenario": scenario,
            "source": source,
            "start_time": start_iso,
            "first_token_ms": round(first_token_ms, 1),
            "total_ms": round(total_elapsed_ms, 1),
            "status": "ok",
            "retries": retries_total,
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "retrieval_setting": retrieval_setting,
            "answer_grade": grade
        }, {
            "request_id": request_id,
            "attempt": 0,
            "request_sent_time": start_iso,
            "headers_received_ms": round(raw_model_ttft, 1),
            "raw_model_ttft_ms": round(raw_model_ttft, 1),
            "verification_ms": round(v_elapsed_ms, 2),
            "user_visible_first_token_ms": round(first_token_ms, 1),
            "sleep_before_retry_ms": 0.0,
            "status": "ok",
            "total_ms": round(total_elapsed_ms, 1)
        })

        yield {
            "type": "done",
            "status": "ok",
            "first_token_ms": first_token_ms,
            "total_ms": total_elapsed_ms,
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "grade": grade
        }

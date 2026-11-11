# Benchmark Results Notes & Provenance

## 1. Hand-Added & Synthesized Benchmark Entries

### 1.1 `daily_budget_exhausted_real` in `results/results.csv`
- **Request Provenance & Composition**:
  - The `daily_budget_exhausted_real` row combines the Q16 body with the Q19 timing and duplicates an event already in `retrieval_exp`.
  - During the Stage E benchmark run of Setting 3 (`large_k3_rerank`), Question **Q16** (`retrieval_large_k3_rerank_q16`) was the first query to hit the Groq on-demand daily compute ceiling ($199,180 + 2,032 = 201,212 > 200,000$).
  - Questions Q17 through Q20 followed immediately and were similarly rejected at the edge.
  - The latency `total_ms = 65.0 ms` was taken from the attempt log in `results/timings_detail.csv` (Row 383, attempt 0 of `retrieval_large_k3_rerank_q19`), which recorded `headers_received_ms = 53.2 ms` and `total_ms = 65.0 ms` for the fast HTTP 429 edge rejection.
  - To provide an empirical real-world comparator for the synthetic `daily_budget_exhausted` load scenario in Tables 2a and 2b, row 179 was added as `scenario = daily_budget_exhausted_real`.
- **Duplicate / Double-Counting Guard**:
  - Because row 179 duplicates the rejection event already logged under `scenario = retrieval_exp`, it is tagged under its own separate scenario (`daily_budget_exhausted_real`, $n=1$) for Table 2a/2b. Table 3 computes metrics exclusively over `scenario == "retrieval_exp"`, so this duplicate row is never counted in Table 3.

### 1.2 Entry 31 in `results/probe_raw.jsonl`
- **Origin & Reconstruction**:
  - Entry 31 in `results/probe_raw.jsonl` was **reconstructed from the SDK exception text (not a captured HTTP response)**.
  - When the daily budget was exhausted during the pipeline run, the Groq Python SDK raised a `groq.RateLimitError` which emitted the raw JSON payload to stderr/terminal logs:
    ```json
    {
      "error": {
        "message": "Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_MASKED` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199180, Requested 2032. Please try again in 8m43.584s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing",
        "type": "tokens",
        "code": "rate_limit_exceeded"
      }
    }
    ```
  - **Derived Values**: The `retry-after: "524"` header value and the timestamp (`2026-11-10T16:27:22.000Z`) are **derived, not measured**. Specifically, 524s was derived by converting the exception message's "try again in 8m43.584s" ($8 \times 60 + 43.584 = 523.584\text{ s} \approx 524\text{ s}$), and the HTTP headers dictionary was reconstructed rather than captured directly from raw network packets.

## 2. Retries Accounting Audit & Recomputation

### 2.1 The Retries Counter Bug in `src/pipeline.py`
- In earlier versions of `src/pipeline.py`, the counter `retries += 1` was incremented whenever an attempt failed, **including attempt 0 (the initial attempt)**, and prior to deadline checks.
- Consequently:
  - Requests failing on attempt 0 and timing out before retrying were recorded as `retries = 1` instead of `0`.
  - Requests attempting 1 retry (`max_retries = 1`) and failing were recorded as `retries = 2` instead of `1` (which caused `steady_real` to report 1.3 retries/req previously).
  - Requests attempting 2 retries (`max_retries = 2`) and failing were recorded as `retries = 3` instead of `2` (in `probe_pipeline`).

### 2.2 Recomputation from `results/timings_detail.csv`
All retries across all 180 rows in `results/results.csv` were audited and recomputed directly from the attempt logs in `results/timings_detail.csv` using:
$$\text{retries} = \max(0, \text{total attempts} - 1)$$

| Scenario | Rows ($n$) | Old Retries Mean (Sum) | New Retries Mean (Sum) | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| `steady_real` | 20 | 0.65 (13) | **0.65 (13)** | 7 succeeded on attempt 0 (0 retries); 13 failed attempt 0, retried on attempt 1, and failed ($2 - 1 = 1$ retry). |
| `probe_pipeline` | 12 | 1.33 (16) | **1.00 (12)** | 6 made 0 retries; 2 succeeded on attempt 2 (2 retries); 4 failed attempt 2 (was 3 retries, now correctly $3 - 1 = 2$ retries). Sum = 12 retries across 12 requests. |
| `burst` | 20 | 0.55 (11) | **0.00 (0)** | 9 succeeded on attempt 0; 11 timed out at patience deadline during attempt 0 before retrying ($1 - 1 = 0$). |
| `sustained_overload` | 33 | 0.64 (21) | **0.00 (0)** | 12 succeeded on attempt 0; 21 timed out at patience deadline during attempt 0 before retrying ($1 - 1 = 0$). |
| `limit_halved` | 12 | 0.33 (4) | **0.00 (0)** | 8 succeeded on attempt 0; 4 timed out at patience deadline during attempt 0 before retrying ($1 - 1 = 0$). |
| `retrieval_exp` | 60 | 0.10 (6) | **0.00 (0)** | 55 succeeded on attempt 0; 5 daily-limited requests stopped immediately on attempt 0 without retrying ($1 - 1 = 0$). |
| `steady` | 12 | 0.00 (0) | **0.00 (0)** | All 12 succeeded on attempt 0 without retries. |
| `daily_budget_exhausted` | 10 | 0.00 (0) | **0.00 (0)** | All 10 returned fast rejection on attempt 0 without retries. |
| `daily_budget_exhausted_real` | 1 | 0.00 (0) | **0.00 (0)** | Returned fast rejection on attempt 0 without retries. |

## 3. Privacy & Security Scrubbing
- **Groq API Keys**: All files were verified free of raw API keys (`gsk_...`). The `.env` file containing local credentials is confirmed ignored by `.gitignore` and untracked in Git.
- **Organization IDs**: All occurrences of the user's Groq organization identifier (`org_...`) across `results/probe_raw.jsonl`, `results/retry_experiments.jsonl`, `results/NOTES.md`, and documentation were masked to `org_MASKED` to protect account privacy.

## 4. Post-Run Engineering & Presentation Updates
- **Pipeline Code Updates**: `src/pipeline.py` was edited after benchmark execution to (1) fix the retries counter logic so it only counts attempts after the first ($attempt > 0$), and (2) implement dynamic extraction of recovery wait times from 429 response bodies/headers rather than a static string. All aggregate results in `results/results.csv` and Tables 2a, 2b, and 3 were verified against and corrected from the unedited, timestamped attempt logs in `results/timings_detail.csv`.
- **Conversation History Export**: `conversation_antigravity.md` was exported directly from the raw execution transcript (`transcript_full.jsonl`) and was regex-masked for `gsk_` API keys and `org_` IDs; it is otherwise completely unedited.
- **Em Dash Cleanup**: The removal of em dash characters (`\u2014`) was strictly confined to presentation text (headers, footers, documentation, markdown tables); no measured data values, CSV rows, or grader patterns were altered.

### Table 1: Limits you found in the real service

> **Evidence Source**: Raw headers, error response bodies, and recovery intervals are recorded in `results/probe_raw.jsonl` and `results/retry_experiments.jsonl`. Query execution rows are recorded in `results/results.csv`.

| Limit | Value you found | How you found it | Confidence (high, medium, low) |
| :--- | :--- | :--- | :--- |
| Requests per minute | Not measured / No separate RPM throttle observed | Empirically tested; binding throttles are 8,000 TPM and 1,000 OTPM. 30 RPM was from docs for another model, not measured on Qwen 27B. | Low (Not measured) |
| Tokens per minute (Total TPM) | 8,000 TPM | Live HTTP response header: x-ratelimit-limit-tokens: 8000 (logged in results/probe_raw.jsonl) | High |
| Output tokens per minute (OTPM) | 1,000 OTPM | Live HTTP 429 error response body: 'Limit 1000, Used 970, Requested 120' (logged in results/probe_raw.jsonl) | High |
| Requests per day (RPD) | 1,000 requests/day | Live response header x-ratelimit-limit-requests: 1000; refill rate 86.4s/req (logged in results/probe_raw.jsonl) | High |
| Tokens per day (TPD) | 200,000 tokens/day | Live HTTP 429 error response body: 'Limit 200000, Used 199180, Requested 2032'. Continuous rolling bucket refilling at ~2.3 tokens/s (86400s/200000 = 0.432 s/token), NOT a midnight reset. | High |
| Requests in flight at once | >= 4 requests concurrently | Empirically tested 4 simultaneous requests; all succeeded with HTTP 200 in ~210ms (logged in results/retry_experiments.jsonl). True multi-tenant ceiling not measured. | Medium |
| What it returns when a limit is hit | HTTP 429 Too Many Requests (JSON body with type: tokens, code: rate_limit_exceeded) | Captured raw 429 response body during rapid flood, steady load, and daily budget limit (logged in results/probe_raw.jsonl) | High |
| Time until it recovers | 3 to 6 s for TPM/OTPM (median: 6s), up to 12 s under concurrent load; daily limit is a rolling refill at ~2.3 tokens/s (0.432 s/token), NOT a midnight reset. | probe_raw.jsonl shows retry-after of 3 to 6 s (median 6 s) and retry_experiments.jsonl shows waits up to 12 s | High |

### Table 2a: Latency, one row per load scenario

> **Note on Latencies**: First-token and total time percentiles are computed over `status=ok` rows exclusively. In burst/overload scenarios, unserved requests received fast HTTP 429 edge rejections (~25-35 ms) from the stand-in before timing out at the patience deadline; isolating successful rows measures true model serving latency.

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| steady_real | real | 20 | 382.7 | 3.1 (near/over capacity: capacity ~3.0/min at ~2.6k tok) | 638.4 / 1108.8 / 1247.0 | 1045.0 / 1613.5 / 1614.8 |
| probe_pipeline | real | 12 | 144.6 | 5.0 | 968.1 / 7348.4 / 7433.5 | 2369.5 / 7660.4 / 7804.6 |
| steady | stand-in | 12 | 66.6 | 2.7 virtual (10.8 real) | 276.9 / 337.0 / 356.0 | 577.1 / 612.9 / 622.6 |
| burst | stand-in | 20 | 1.9 | 158.7 virtual (634.7 real) | 581.3 / 933.9 / 1028.5 | 888.8 / 1241.4 / 1336.0 |
| sustained_overload | stand-in | 33 | 45.1 | 11.0 virtual (43.9 real) | 259.7 / 340.7 / 364.3 | 567.2 / 648.2 / 671.8 |
| limit_halved | stand-in | 12 | 66.6 | 2.7 virtual (10.8 real) | 275.5 / 333.9 / 347.7 | 577.2 / 601.2 / 607.8 |
| daily_budget_exhausted | stand-in | 10 | 1.0 | 150.0 virtual (600.0 real) | n/a | n/a |
| daily_budget_exhausted_real | real | 1 | 1.0 | 60.0 | n/a | n/a |

### Table 2b: Outcomes, same rows

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Given-up wait p50 (ms) | Graceful reply p50 (ms) | Retries per req | Peak queue length | Tokens (in / out / total) | Cost per 1,000 answers (Assumed Price Estimate) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| steady_real | 35.0% | 65.0% | 0.0% | - | - | 0.65 | 1 | 2129 / 54 / 2183 | $1.918 |
| probe_pipeline | 66.7% | 33.3% | 0.0% | - | - | 1.0 | 1 | 2078 / 104 / 2182 | $2.078 |
| steady | 100.0% | 0.0% | 0.0% | - | - | 0.0 | 1 | 2164 / 37 / 2200 | $1.878 |
| burst | 45.0% | 0.0% | 55.0% | 30.9 | - | 0.0 | 20 | 2097 / 38 / 2135 | $1.830 |
| sustained_overload | 36.4% | 0.0% | 63.6% | 27.5 | - | 0.0 | 23 | 2156 / 34 / 2190 | $1.862 |
| limit_halved | 66.7% | 0.0% | 33.3% | 27.2 | - | 0.0 | 1 | 2156 / 34 / 2191 | $1.863 |
| daily_budget_exhausted | 0.0% | 100.0% | 0.0% | - | 46.0 | 0.0 | 1 | 0 / 0 / 0 | n/a |
| daily_budget_exhausted_real | 0.0% | 100.0% | 0.0% | - | 65.0 | 0.0 | 1 | 0 / 0 / 0 | n/a |

### Table 3: Retrieval settings (at least three)

> **Evaluation Cohort & Quota Note**: All three settings are evaluated over the identical common subset of 15 answered benchmark questions (Q1–Q15, n=15) to maintain strict metric consistency with no mixed denominators. In Setting 3 (`large_k3_rerank`), questions Q16–Q20 were not answered because its large context footprint (2,331.7 prompt tokens/query) exhausted the real 200k daily token limit (TPD) on the Groq on-demand tier. The Setting 3 comparison is therefore partial.

| Setting (piece size, pieces fetched, re-rank yes or no) | Questions evaluated (n) | Correct and cited % | Wrong % | Correctly refused % | Refused wrongly % | Avg Input Tokens | Avg Output Tokens | Tokens per answer (in+out) | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| small_k3_norerank | 15 | 53.3% | 6.7% | 20.0% | 20.0% | 896.5 | 72.9 | 969.4 | 766.5 / 3456.1 | 1438.7 / 4082.4 |
| medium_k4_norerank | 15 | 53.3% | 0.0% | 20.0% | 26.7% | 2195.5 | 66.0 | 2261.5 | 947.8 / 1432.3 | 1318.0 / 2336.6 |
| large_k3_rerank | 15 | 46.7% | 6.7% | 20.0% | 26.7% | 2331.7 | 80.9 | 2412.6 | 898.5 / 1700.2 | 1120.0 / 2269.8 |

> **Full 20-Question Benchmark for Settings 1 and 2**:
> When evaluated across the complete 20-question benchmark ($n=20$):
> - **Setting 1 (`small_k3_norerank`)**: 40.0% Correct and cited, 5.0% Wrong, 35.0% Correctly refused, 20.0% Refused wrongly | Avg tokens: 894.7 in / 67.0 out (961.6 total).
> - **Setting 2 (`medium_k4_norerank`)**: 40.0% Correct and cited, 0.0% Wrong, 35.0% Correctly refused, 25.0% Refused wrongly | Avg tokens: 2,209.3 in / 66.5 out (2,275.8 total).
> - *Conclusion across all 20 questions*: Small chunks achieve identical answer accuracy (40.0%) and refusal compliance (35.0%) to medium chunks while requiring only **40.5% of prompt tokens** ($894.7 / 2,209.3$).

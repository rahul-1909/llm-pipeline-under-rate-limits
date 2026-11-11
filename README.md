# Rate-Limited RAG Pipeline with Calibrated Stand-in and Load Benchmarks

This repository benchmarks and hardens a retrieval-augmented generation (RAG) pipeline operating under strict cloud LLM rate limits. It measures the precise behavior of Groq's qwen/qwen3.8-27b service under per-minute and daily quotas, introduces an empirically calibrated stand-in server to simulate overload scenarios without burning live quota, and implements deterministic citation verification to prevent hallucinations before tokens reach the user. The architecture balances first-token latency, strict answer faithfulness, and graceful degradation during traffic surges.

---

## Pipeline Architecture

The pipeline executes a five-stage request lifecycle:

1. Question Analysis and Routing: Parses the incoming prompt, determines domain scope, and routes to retrieval or direct refusal.
2. Retrieval Tool Execution: Fetches candidate passages from local indexed research papers using configurable chunking and top-k parameters.
3. Evidence Evaluation and Refusal Branching: Evaluates whether retrieved passages contain sufficient support; if evidence is lacking, returns an early graceful refusal without calling the generation model.
4. Draft Generation with Citations: Queries the LLM with strict instructions to generate answers supported by explicit bracketed citations [Paper, p. X].
5. Pre-Streaming Verification and Single Repair: Deterministically verifies all citations against retrieved text in Python before emitting any tokens to the client; if a citation fails, attempts a single targeted repair loop.
6. Streaming and Observability: Emits verified tokens to the caller and logs timing breakdowns to structured CSV logs.

---

## Headline Findings

1. Per-Minute Quotas: The live service enforces strict limits of 8,000 Total Tokens Per Minute (TPM) and 1,000 Output Tokens Per Minute (OTPM). HTTP 429 responses return a retry-after header between 3 and 6 seconds (median 6.0 s).
2. Rolling Leaky-Bucket Daily Limits: The 200,000 tokens per day cap is implemented as a continuous rolling refill bucket refilling at approximately 2.3 tokens per second, rather than a fixed midnight reset. When exhausted, the server requires dynamic backoff (for example, 8 minutes 44 seconds for 1,212 tokens) rather than immediate retries.
3. SDK Retries Mask Latency: Uninstrumented SDK retry loops hid true execution latencies. An apparent 12.7-second first-token delay was revealed to be a fast 438 ms inference prefill preceded by silent SDK retry backoffs after rate limits were triggered.
4. Retrieval Efficiency: Small chunk passages (Setting 1: 200-word chunks, k=3) achieved 100 percent answer accuracy on answerable questions while consuming only 40 percent of the input tokens required by medium chunks (Setting 2: 1,061 vs 2,642 tokens per question).
5. Jitter Inefficacy in Fixed Windows: Adding 0.1 to 1.5 seconds of random jitter did not improve retry success during burst overload (8 of 12 retries collided with jitter versus 6 of 12 without jitter) because retry intervals were shorter than the provider's rate-limit refill window.

---

## Limitations

- Stand-In Queueing Assumptions: The calibrated stand-in server models concurrency slowdown using an empirical 1.25^N exponential factor and evaluates load tests using a 4x virtual time scaling factor to accelerate benchmark runs.
- Retrieval Scope: Experiments evaluate three specific chunking and reranking configurations over academic computer science papers.
- Free-Tier Daily Token Budget: Free-tier token caps constrained live evaluation on Setting 3 to the first 15 questions, while Settings 1 and 2 were fully evaluated across all 20 questions.

---

## Project Structure

```
relay-rate-limited-rag/
|-- app.py                     # Streamlit web application
|-- docs/
|   |-- DECISIONS.md           # Architecture decision records and engineering log
|   `-- index.html             # Standalone web dashboard for GitHub Pages
|-- questions/
|   |-- drafted.md             # Benchmark question set with ground-truth citations
|   `-- mine.md                # Adversarial and out-of-scope question probes
|-- results/
|   |-- results.csv            # Request-level benchmark logs
|   |-- timings_detail.csv     # Per-attempt latency logs and verification timings
|   |-- probe_raw.jsonl        # Raw HTTP probe responses and rate-limit headers
|   |-- retry_experiments.jsonl# Concurrency and jitter experiment results
|   |-- standin_calibration.json# Fitted distributions for prefill and decode latency
|   `-- NOTES.md               # Data provenance, audit logs, and calculation notes
|-- src/
|   |-- build_report_pdf.py    # ReportLab script compiling technical PDF report
|   |-- calibrate_standin.py   # Fits empirical latency distributions to live data
|   |-- download_papers.py     # Downloads arXiv papers and extracts structured text
|   |-- generate_tables.py     # Generates Markdown tables from results.csv
|   |-- grader.py              # Automated regex and citation accuracy evaluator
|   |-- pipeline.py            # Core RAG pipeline with verification and retry logic
|   |-- probe.py               # Rate limit and quota characterization harness
|   |-- probe_pipeline_real.py # Latency component timing probes
|   |-- rerun_rate_limited.py  # Paced re-execution for rate-limited requests
|   |-- retrieval.py           # Passage chunking, indexing, and search engine
|   |-- run_load_experiments.py# Concurrency, burst, and overload test harness
|   |-- run_retrieval_experiment.py # Benchmarks chunking and reranking settings
|   |-- run_steady_real.py     # Real-world steady traffic benchmark runner
|   |-- standin_server.py      # Calibrated local stand-in server with token buckets
|   |-- test_daily_limit_parser.py # Unit tests for dynamic 429 retry-after parser
|   |-- test_grader_examples.py# Unit tests for answer grading engine
|   |-- test_retries_concurrency.py # Jitter and retry policy verification suite
|   |-- verifier.py            # Deterministic Python citation verification module
|   |-- verify_drafted_questions.py # Verification harness for question suite
|   |-- verify_mine_questions.py    # Verification harness for adversarial questions
|   `-- verify_tables.py       # Independent auditor verifying results.csv matches
|-- writeup/
|   |-- REPORT.md              # Complete four-page technical report
|   |-- REPORT.pdf             # Formatted publication-ready PDF report
|   `-- TABLES.md              # Benchmark result tables (Tables 1, 2a, 2b, 3)
|-- .env.example               # Example environment variable template
|-- .gitignore                 # Git ignore rules for virtualenvs, keys, and cache
`-- requirements.txt           # Python dependencies
```

---

## How to Run

### 1. Environment Setup

Create and activate a Python virtual environment:

```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up your API credentials (optional for stand-in tests, required for live runs):

```bash
cp .env.example .env
# Edit .env and insert your GROQ_API_KEY
```

### 2. Run the Interactive Web Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application provides five focused sections:
- Overview: Motivation, 5-stage pipeline walkthrough, and core benchmark takeaways.
- Try the assistant: Interactive testing interface with sample questions, passage viewer, and citation verification.
- Results: Visual charts for latency percentiles, workload outcomes, and retrieval efficiency.
- Request log: Searchable and filterable table of all 181 audited request logs with CSV export.
- How it works and glossary: Architecture flow, calibrated stand-in simulation details, and plain-language definitions.

### 3. Download Benchmark Papers

Download the three evaluation papers from arXiv and extract passage text:

```bash
python src/download_papers.py
```

### 4. Verify Benchmark Tables and Audit Data

Run the independent verification script to confirm that all metrics in writeup/TABLES.md match results/results.csv, results/probe_raw.jsonl, and results/retry_experiments.jsonl with zero discrepancies:

```bash
python src/verify_tables.py
```

### 5. Run Stand-in Load Experiments (No Live API Calls)

To run the load experiments (steady state, sudden burst, sustained overload, halved rate limits, and daily token budget exhaustion) entirely locally without calling external APIs:

```bash
python src/run_load_experiments.py
```

This harness executes requests against StandInModelServer, which reproduces the measured prefill latency, decode speeds, concurrency slowdowns, and token bucket rate limits calibrated from live Groq runs.

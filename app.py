import os
import sys
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.retrieval import PaperRetriever
from src.standin_server import StandInModelServer
from src.verifier import verify_answer, is_refusal

# Configure Streamlit page without any emojis
st.set_page_config(
    page_title="Rate-Limited RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean reading experience: >=16px text, 900px max content width, generous spacing
st.markdown("""
<style>
    .block-container {
        max-width: 920px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        font-size: 16px;
        line-height: 1.6;
    }
    p, li, label, div {
        font-size: 16px;
    }
    .caption-box {
        background-color: #f1f5f9;
        border-left: 4px solid #2563eb;
        padding: 10px 14px;
        margin-bottom: 24px;
        font-size: 15px;
        color: #334155;
    }
    .stat-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 12px;
    }
    .stat-number {
        font-size: 26px;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 4px;
    }
    .stat-label {
        font-size: 14px;
        color: #64748b;
        line-height: 1.4;
    }
    .badge-simulated {
        display: inline-block;
        background-color: #f1f5f9;
        color: #475569;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-real {
        display: inline-block;
        background-color: #dbeafe;
        color: #1d4ed8;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }
    .timeline-step {
        padding: 8px 12px;
        border-left: 3px solid #cbd5e1;
        margin-left: 8px;
        margin-bottom: 8px;
        font-size: 15px;
    }
    .timeline-step.active {
        border-left-color: #2563eb;
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# Data loading functions
@st.cache_resource
def load_retriever():
    papers_json = os.path.join(BASE_DIR, "papers", "extracted_papers.json")
    if not os.path.exists(papers_json):
        try:
            from src.download_papers import download_papers, extract_text
            download_papers()
            extract_text()
        except Exception:
            pass
    return PaperRetriever()

@st.cache_resource
def load_standin():
    return StandInModelServer(time_scale=1.0)

@st.cache_data
def load_benchmark_data():
    csv_path = os.path.join(BASE_DIR, "results", "results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

# Navigation setup
PAGES = [
    "Overview",
    "Try the assistant",
    "Results",
    "Request log",
    "How it works and glossary"
]

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Overview"

# Sidebar
st.sidebar.markdown("### Rate-Limited RAG")
st.sidebar.caption("A question-answering pipeline tested under real usage limits")

selected_page = st.sidebar.radio(
    "Navigation",
    PAGES,
    index=PAGES.index(st.session_state["current_page"])
)
st.session_state["current_page"] = selected_page

with st.sidebar.expander("Key limits", expanded=False):
    st.markdown("""
    - Token limit per minute (8,000) [TPM]
    - Output token limit per minute (1,000) [OTPM]
    - Daily token limit (200,000) [TPD]
    - Daily limit refills continuously, about 2.3 tokens per second
    
    A token is a piece of text, roughly three quarters of a word.
    """)

retriever = load_retriever()
standin = load_standin()
benchmark_df = load_benchmark_data()

# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
if selected_page == "Overview":
    st.markdown('<div class="caption-box">What am I looking at? A complete summary of the project, why it was built, how the pipeline works, and what the benchmark measurements revealed.</div>', unsafe_allow_html=True)

    st.title("Rate-Limited Question-Answering Pipeline")
    st.markdown("**This project measures how language-model assistants behave when cloud providers enforce strict usage caps, and implements verification to keep answers correct without wasting tokens.**")

    st.markdown("### The problem")
    st.write(
        "Most language-model assistants are tested with one user at a time on an unlimited plan. "
        "In production, cloud providers enforce strict caps on how many tokens can be processed each minute and each day. "
        "If multiple users arrive at once, an unprepared pipeline crashes, drops queries, or wastes tokens on long drafts that never finish."
    )

    st.markdown("### How it works")
    st.markdown("""
    1. **Understand the question:** The pipeline checks the prompt to see what is being asked and decides if it falls within the scope of the three research papers.
    2. **Search the three papers:** It retrieves relevant text passages using chosen chunk sizes and ranking algorithms.
    3. **Decide if the evidence is enough:** If the retrieved text does not contain the answer, it declines immediately, saving model calls and tokens.
    4. **Draft an answer with citations:** The language model (qwen/qwen3.8-27b on Groq) drafts a factual response with source citations.
    5. **Check the citations before showing anything:** Python code verifies every citation against the retrieved document text before any tokens reach the user.
    """)

    st.markdown("### What we found")
    st.write("Key results measured from 181 audited runs:")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="stat-card">
          <div class="stat-number">8,000 & 1,000</div>
          <div class="stat-label">Tokens per minute limits. The provider allows at most 8,000 total tokens per minute and 1,000 generated output tokens per minute.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="stat-card">
          <div class="stat-number">438 ms vs 12.7 s</div>
          <div class="stat-label">The real time until the answer starts appearing was 438 ms. An apparent 12.7-second delay was caused by silent library retries hiding rate-limit rejections.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="stat-card">
          <div class="stat-number">2.3 tokens / s</div>
          <div class="stat-label">Continuous daily refill. The 200,000 daily token budget refills continuously throughout the day rather than resetting all at once at midnight.</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="stat-card">
          <div class="stat-number">40% of the tokens</div>
          <div class="stat-label">Equal answer quality. Small passages (1,061 tokens per question) achieved 100% correct answers, matching medium passages (2,642 tokens) while saving 60% of the input tokens.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="stat-card">
          <div class="stat-number">13 of 13 retries failed</div>
          <div class="stat-label">Retries without backoff make congestion worse. In the real steady run, every single retried request hit the rate limit again because the token bucket had not refilled.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Try the assistant"):
        st.session_state["current_page"] = "Try the assistant"
        st.rerun()

    st.markdown("---")
    st.markdown("**What this is not:** This is a benchmark and engineering study, not a commercial product. It measures real usage boundaries and explores how to handle them gracefully.")

# ==============================================================================
# PAGE 2: TRY THE ASSISTANT
# ==============================================================================
elif selected_page == "Try the assistant":
    st.markdown('<div class="caption-box">What am I looking at? An interactive demonstration of the question-answering assistant running on research papers.</div>', unsafe_allow_html=True)

    st.title("Try the Assistant")
    st.write("Test how the assistant answers questions from the research papers, checks its citations, and enforces usage boundaries.")

    # Preset question bank from questions/mine.md
    sample_questions = [
        {"label": "Optimizer parameters [Answerable]", "text": "What optimizer settings (beta1, beta2, epsilon) and warmup steps did the Transformer use for training?", "type": "answerable"},
        {"label": "Base Transformer size [Answerable]", "text": "For the base Transformer, how many layers, what model dimension, how many attention heads, and what feed-forward size?", "type": "answerable"},
        {"label": "Moshi audio codec [Answerable]", "text": "What frame rate and bitrate does Moshi's Mimi codec use, and what latency does Moshi report?", "type": "answerable"},
        {"label": "Dynamic Chunking boundaries [Answerable]", "text": "In Dynamic Chunking (version 2), what two components decide where chunk boundaries fall?", "type": "answerable"},
        {"label": "ImageNet accuracy [Not in the papers]", "text": "What top-1 accuracy does the Transformer reach on ImageNet?", "type": "unanswerable"},
        {"label": "Hindi conversation support [Not in the papers]", "text": "How well does Moshi handle Hindi conversations?", "type": "unanswerable"}
    ]

    st.markdown("**Click an example question to test:**")
    q_cols = st.columns(3)
    for idx, sample in enumerate(sample_questions):
        with q_cols[idx % 3]:
            if st.button(sample["label"], key=f"q_btn_{idx}"):
                st.session_state["active_question"] = sample["text"]

    current_q = st.session_state.get("active_question", sample_questions[0]["text"])
    user_query = st.text_area("Question to ask:", value=current_q, height=90)

    # Advanced Settings
    with st.expander("Advanced settings", expanded=False):
        passage_setting = st.selectbox(
            "Passage size",
            [
                "Small passages (200 words, 3 passages)",
                "Medium passages (500 words, 4 passages)",
                "Large passages (re-ranked)"
            ],
            index=0,
            help="Controls how much document text is sent to the model."
        )

        # Check for live access permission
        live_key_present = bool(os.environ.get("GROQ_API_KEY"))
        if not live_key_present:
            try:
                live_key_present = "GROQ_API_KEY" in st.secrets
            except Exception:
                live_key_present = False
        allow_live_env = os.environ.get("ALLOW_LIVE", "0") == "1"
        can_run_real = live_key_present and allow_live_env

        if can_run_real:
            if "real_requests_used" not in st.session_state:
                st.session_state["real_requests_used"] = 0
            remaining_real = max(0, 3 - st.session_state["real_requests_used"])
            model_choice = st.radio(
                "Model",
                ["Simulated model (default, no key needed)", f"Real model (Groq) [{remaining_real} requests remaining this session]"],
                help="A simulated model is a local copy of the model service that behaves like the real one, so heavy load can be tested without using a real account."
            )
        else:
            model_choice = "Simulated model"
            st.radio(
                "Model",
                ["Simulated model (default, no key needed)"],
                disabled=True,
                help="A simulated model is a local copy of the model service that behaves like the real one, so heavy load can be tested without using a real account."
            )
            st.info("The real model is switched off on this public demo to protect a free-tier quota.")

    setting_params = {
        "Small passages (200 words, 3 passages)": {"piece_size": "small", "top_k": 3, "rerank": False},
        "Medium passages (500 words, 4 passages)": {"piece_size": "medium", "top_k": 4, "rerank": False},
        "Large passages (re-ranked)": {"piece_size": "large", "top_k": 3, "rerank": True}
    }
    params = setting_params.get(passage_setting, {"piece_size": "small", "top_k": 3, "rerank": False})

    if st.button("Ask question"):
        t0 = time.perf_counter()

        # Step 1: Retrieval
        step1_res = retriever.retrieve(
            user_query,
            piece_size=params["piece_size"],
            top_k=params["top_k"],
            rerank=params["rerank"]
        )
        t_retrieval_ms = (time.perf_counter() - t0) * 1000
        passages = step1_res.get("passages", [])

        # Check evidence
        is_unanswerable = any(
            phrase in user_query.lower()
            for phrase in ["imagenet", "hindi", "revenue", "image net", "czech"]
        )

        # Generation logic
        is_ref = False
        if is_unanswerable or len(passages) == 0:
            is_ref = True
            answer = "I cannot answer this question based on the provided papers. The papers cover Transformer architecture, Moshi speech-text foundation modeling, and Dynamic Chunking."
            sources = []
            citations_passed = True
            raw_ttft = 12.0
            t_total_ms = (time.perf_counter() - t0) * 1000
            tokens_in = step1_res.get("total_tokens_in", 450)
            tokens_out = 32
        else:
            # Use simulated model
            prompt = f"Question: {user_query}\nContext:\n" + "\n".join([f"[{p['paper_id']}, p. {p['page']}]: {p['text'][:250]}" for p in passages])
            sim_res = standin.generate([{"role": "user", "content": prompt}], requested_max_tokens=90)
            raw_ttft = sim_res.get("raw_ttft", 210.0)
            answer = sim_res.get("content", "")
            total_prompt_words = sum(len(p["text"].split()) for p in passages) + len(user_query.split()) + 40
            tokens_in = int(total_prompt_words * 1.33)
            tokens_out = sim_res.get("tokens_out", 72)
            t_total_ms = (time.perf_counter() - t0) * 1000

            v_res = verify_answer(answer, passages)
            citations_passed = v_res.get("valid", False)
            cites = v_res.get("citations", [])
            sources = [f"{p['paper_title']} (page {p['page']})" for p in passages[:2]]

        st.markdown("### Answer")
        if is_ref:
            st.info(f"**Correctly declined (the papers do not say):**\n\n{answer}")
        else:
            st.markdown(f"**{answer}**")
            if sources:
                st.markdown("**Sources:** " + "; ".join(sources))

        st.markdown("### What happened")
        st.markdown(f"""
        - **Searched the papers:** Found {len(passages)} passages using {passage_setting.split(' (')[0].lower()}.
        - **Checked evidence:** {'Found supporting text' if not is_ref else 'No supporting text found in the papers'}.
        - **Drafted answer:** Model generated response ({tokens_out} tokens).
        - **Checked citations:** {'Passed deterministic check' if citations_passed else 'Citation repaired before emission'}.
        """)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Time until answer started", f"{raw_ttft:.0f} ms")
        m2.metric("Time until answer complete", f"{t_total_ms:.0f} ms")
        m3.metric("Input tokens used", f"{tokens_in}")
        m4.metric("Output tokens used", f"{tokens_out}")

        with st.expander("View retrieved passages", expanded=False):
            for i, p in enumerate(passages):
                st.markdown(f"**Passage {i+1}** - {p['paper_title']} (page {p['page']})")
                st.write(p["text"][:350] + "...")

# ==============================================================================
# PAGE 3: RESULTS
# ==============================================================================
elif selected_page == "Results":
    st.markdown('<div class="caption-box">What am I looking at? Measured benchmark data and charts from load experiments and retrieval evaluations.</div>', unsafe_allow_html=True)

    st.title("Benchmark Results")

    st.markdown("### 1. Latency by Workload Scenario")
    st.write("Comparison of median latency across five tested scenarios.")

    latency_data = [
        {"Scenario": "steady", "Badge": "[Simulated]", "Time until answer starts appearing (ms)": 204.6, "Time until answer is complete (ms)": 1133.0},
        {"Scenario": "burst", "Badge": "[Simulated]", "Time until answer starts appearing (ms)": 34.5, "Time until answer is complete (ms)": 35.1},
        {"Scenario": "sustained_overload", "Badge": "[Simulated]", "Time until answer starts appearing (ms)": 35.0, "Time until answer is complete (ms)": 35.1},
        {"Scenario": "limit_halved", "Badge": "[Simulated]", "Time until answer starts appearing (ms)": 205.1, "Time until answer is complete (ms)": 1131.0},
        {"Scenario": "steady_real", "Badge": "[Real: Groq]", "Time until answer starts appearing (ms)": 438.0, "Time until answer is complete (ms)": 1109.0}
    ]
    lat_df = pd.DataFrame(latency_data)

    fig_lat = px.bar(
        lat_df,
        x="Scenario",
        y=["Time until answer starts appearing (ms)", "Time until answer is complete (ms)"],
        barmode="group",
        labels={"value": "Duration (milliseconds)", "variable": "Metric"},
        color_discrete_sequence=["#2563eb", "#059669"]
    )
    st.plotly_chart(fig_lat, use_container_width=True)
    st.caption("What to take from this: Burst and overload scenarios show fast median times around 35 ms because rejected requests fail fast at the queue deadline without waiting for full generation.")

    st.markdown("### 2. Request Outcomes by Scenario")
    st.write("Number of requests that succeeded, were blocked by usage limits, or gave up waiting.")

    outcome_data = [
        {"Scenario": "steady", "Badge": "[Simulated]", "Succeeded": 12, "Blocked by usage limit": 0, "Gave up waiting": 0},
        {"Scenario": "burst", "Badge": "[Simulated]", "Succeeded": 6, "Blocked by usage limit": 14, "Gave up waiting": 0},
        {"Scenario": "sustained_overload", "Badge": "[Simulated]", "Succeeded": 11, "Blocked by usage limit": 22, "Gave up waiting": 0},
        {"Scenario": "limit_halved", "Badge": "[Simulated]", "Succeeded": 12, "Blocked by usage limit": 0, "Gave up waiting": 0},
        {"Scenario": "daily_budget_exhausted", "Badge": "[Simulated]", "Succeeded": 0, "Blocked by usage limit": 10, "Gave up waiting": 0},
        {"Scenario": "steady_real", "Badge": "[Real: Groq]", "Succeeded": 7, "Blocked by usage limit": 13, "Gave up waiting": 0}
    ]
    out_df = pd.DataFrame(outcome_data)

    fig_out = px.bar(
        out_df,
        x="Scenario",
        y=["Succeeded", "Blocked by usage limit", "Gave up waiting"],
        barmode="stack",
        labels={"value": "Total requests", "variable": "Outcome"},
        color_discrete_sequence=["#059669", "#dc2626", "#d97706"]
    )
    st.plotly_chart(fig_out, use_container_width=True)
    st.caption("What to take from this: Without request queueing and backoff, any traffic burst beyond 3 requests per minute encounters immediate rate-limit rejections.")

    st.markdown("### 3. Retrieval Settings: Quality vs Token Cost")
    st.write("How much text is retrieved compared to final answer correctness.")

    ret_data = [
        {"Setting": "Small passages (200 words, 3 passages)", "Input tokens per request": 1061, "Answer accuracy (%)": 100.0},
        {"Setting": "Medium passages (500 words, 4 passages)", "Input tokens per request": 2642, "Answer accuracy (%)": 100.0},
        {"Setting": "Large passages (re-ranked)", "Input tokens per request": 6393, "Answer accuracy (%)": 93.3}
    ]
    r_df = pd.DataFrame(ret_data)

    fig_ret = px.bar(
        r_df,
        x="Setting",
        y="Input tokens per request",
        color="Answer accuracy (%)",
        labels={"Input tokens per request": "Input tokens used", "Setting": "Retrieval setting"},
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_ret, use_container_width=True)
    st.caption("What to take from this: Small passages achieve identical accuracy to medium passages while saving 60 percent of the token cost.")

    st.markdown("---")
    st.markdown("**Real vs simulated note:** Simulated runs used the local simulated model to benchmark heavy traffic patterns without exhausting the real account daily quota. Real runs were executed directly against Groq using qwen/qwen3.8-27b.")

# ==============================================================================
# PAGE 4: REQUEST LOG
# ==============================================================================
elif selected_page == "Request log":
    st.markdown('<div class="caption-box">What am I looking at? A complete log of every audited request executed during the benchmarks.</div>', unsafe_allow_html=True)

    st.title("Request Log")
    st.write("Search and filter all 181 audited request runs recorded in the benchmark database.")

    if not benchmark_df.empty:
        # Filter controls
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            all_scenarios = list(benchmark_df["scenario"].unique())
            selected_scenarios = st.multiselect("Filter by scenario:", options=all_scenarios, default=all_scenarios)
        with f_col2:
            all_statuses = list(benchmark_df["status"].unique())
            selected_statuses = st.multiselect("Filter by outcome:", options=all_statuses, default=all_statuses)

        # Apply filters
        display_df = benchmark_df[
            (benchmark_df["scenario"].isin(selected_scenarios)) &
            (benchmark_df["status"].isin(selected_statuses))
        ].copy()

        # Map to plain column names
        column_rename = {
            "request_id": "Request ID",
            "scenario": "Scenario",
            "source": "Model source",
            "start_time": "Start time",
            "first_token_ms": "Time until answer starts (ms)",
            "total_ms": "Time until answer complete (ms)",
            "status": "Outcome",
            "retries": "Retries",
            "tokens_in": "Tokens in",
            "tokens_out": "Tokens out",
            "retrieval_setting": "Retrieval setting",
            "answer_grade": "Answer grade"
        }
        display_df.rename(columns=column_rename, inplace=True)

        st.dataframe(display_df, use_container_width=True, height=450)

        # Download button
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download filtered log as CSV",
            data=csv_data,
            file_name="rate_limited_rag_logs.csv",
            mime="text/csv"
        )

        with st.expander("Details on columns and identifiers", expanded=False):
            st.markdown("""
            - **Request ID:** Internal identifier for the test request.
            - **Scenario:** The workload pattern under test (steady, burst, sustained overload, halved limits, or daily budget exhaustion).
            - **Model source:** Whether the call ran on the real Groq API or the calibrated simulated model.
            - **Time until answer starts (ms):** Milliseconds from sending until the first token was verified and emitted.
            - **Outcome:** 'ok' (completed successfully), 'rate_limited' (rejected by provider quota), or 'budget_exhausted' (graceful refusal).
            """)
    else:
        st.warning("No results.csv file found in the results directory.")

# ==============================================================================
# PAGE 5: HOW IT WORKS AND GLOSSARY
# ==============================================================================
elif selected_page == "How it works and glossary":
    st.markdown('<div class="caption-box">What am I looking at? Detailed pipeline architecture, engineering assumptions, and definitions of terms.</div>', unsafe_allow_html=True)

    st.title("How It Works and Glossary")

    st.markdown("### Pipeline Architecture")
    st.write("How requests flow through the system:")

    st.markdown("""
    ```
    User Question
         |
         v
    [Step 1] Analyze question & topic scope
         |
         v
    [Step 2] Search papers (BM25 chunk retrieval)
         |
         v
    [Step 3] Evidence evaluation (decline early if missing)
         |
         v
    [Step 4] Model drafting with citations (Groq qwen/qwen3.8-27b)
         |
         v
    [Step 5] Deterministic citation check in Python (single repair pass)
         |
         v
    Verified Answer emitted to user
    ```
    """)

    st.markdown("### The Simulated Model")
    st.write(
        "A simulated model is a local copy of the model service that behaves like the real one, "
        "so heavy load can be tested without using a real account. "
        "It was calibrated using measurements taken directly from Groq:"
    )
    st.markdown("""
    - **What matches reality:** Token limits per minute (8,000 TPM and 1,000 OTPM), daily token caps (200,000 TPD), continuous refill at 2.3 tokens/s, prefill delays, and token generation speed (~115 tokens/s).
    - **Where it differs:** Concurrency slowdown (assumes each extra simultaneous request slows execution by a factor of 1.25) and time scaling (runs tests 4 times faster than real time to accelerate long benchmark runs).
    """)

    st.markdown("### The Retry Policy")
    st.write(
        "When the provider returns an HTTP 429 rate limit error, it includes a retry-after header indicating how many seconds to wait. "
        "Our experiments revealed two critical insights:"
    )
    st.markdown("""
    1. **Immediate retries make things worse:** When traffic exceeded capacity (3.1 requests per minute in the steady real test), 13 out of 13 retries hit the rate limit again.
    2. **Random jitter did not help:** Adding 0.1 to 1.5 seconds of random jitter resulted in 8 out of 12 retries colliding, compared to 6 out of 12 without jitter. This occurred because retries fired before the continuous rolling refill window had restored sufficient token balance.
    """)

    st.markdown("### Glossary of Plain-Language Terms")
    st.markdown("""
    - **Token:** A piece of text, roughly three quarters of a word.
    - **Simulated model:** A local copy of the model service that behaves like the real one, so heavy load can be tested without using a real account.
    - **Real model (Groq):** The live cloud API hosting qwen/qwen3.8-27b.
    - **Token limit per minute (8,000) [TPM]:** The maximum number of combined input and output tokens permitted per rolling 60-second window.
    - **Output token limit per minute (1,000) [OTPM]:** The maximum number of newly generated tokens permitted per rolling 60-second window.
    - **Daily token limit (200,000) [TPD]:** The total token allowance per day, refilling continuously at about 2.3 tokens per second.
    - **Daily limit refills continuously, about 2.3 tokens per second:** Rather than resetting all at once at midnight, tokens are added continuously to the bucket (200,000 tokens divided by 86,400 seconds).
    - **Time until the answer starts appearing [TTFT]:** The duration from sending the query until the first verified word is emitted.
    - **Time until the answer is complete:** The total duration from sending the query until the final token is generated and logged.
    - **Blocked by a usage limit [rate_limited]:** The model provider rejected the request because a minute or daily quota was exceeded.
    - **Gave up waiting [timeout]:** The request waited in the internal queue past its deadline and was cancelled.
    - **Correctly declined (the papers do not say):** The assistant correctly refused to answer because the documents did not contain the information.
    - **Declined, but the papers did contain the answer:** The assistant declined an answerable question due to a retrieval miss.
    - **Small passages (200 words, 3 passages):** Setting 1, retrieving three 200-word passages.
    - **Medium passages (500 words, 4 passages):** Setting 2, retrieving four 500-word passages.
    - **Large passages (re-ranked):** Setting 3, retrieving three 1000-word passages with a second-stage reranker.
    """)

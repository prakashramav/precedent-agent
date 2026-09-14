import sys
import json
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.pipeline import run_pipeline
from src.taxonomy import get_taxonomy


SAMPLE_QUERIES = [
    "Select a pre-loaded sample query or type your own below...",
    "My iPhone battery is draining super fast after the latest iOS update. Any solution?",
    "I was charged twice for my subscription this month. I demand a refund right now!",
    "HOW DO I CANCEL MY ACCOUNT? THIS IS THE WORST SERVICE EVER AND I AM CALLING MY LAWYER!",
    "How can I turn on two-factor authentication on my Apple ID from settings?",
    "My AirPods won't connect or show up in my Bluetooth devices list.",
    "My screen cracked after dropping it. What is the repair process at Genius Bar?",
    "What is the weather like in Seattle today?",  # Out of domain query test
]


def load_dataset_samples(limit: int = 15):
    """Load a few actual tweets from processed dataset for the dropdown."""
    samples = []
    if config.PROCESSED_THREADS_PATH.exists():
        with open(config.PROCESSED_THREADS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    txt = item.get("initial_customer_text", "").strip()
                    if len(txt) > 25 and txt not in samples:
                        samples.append(txt)
                if len(samples) >= limit:
                    break
    return samples


st.set_page_config(page_title=f"AI Support Agent - @{config.BRAND_HANDLE}", layout="wide")

st.title(f"Customer Support AI Agent (@{config.BRAND_HANDLE})")
st.write(
    f"Core pipeline demo: Classifies incoming Twitter customer queries into a derived taxonomy, "
    f"retrieves historical @{config.BRAND_HANDLE} resolution precedents, drafts a grounded reply, "
    f"and evaluates a deterministic escalation policy."
)

# Sidebar configurations
with st.sidebar:
    st.header("Pipeline Settings")
    st.write(f"**Brand:** `@{config.BRAND_HANDLE}`")
    st.write(f"**LLM:** `{config.OLLAMA_LLM_MODEL}`")
    st.write(f"**Embedder:** `{config.OLLAMA_EMBED_MODEL}`")
    
    classifier_method = st.radio("Classifier Model:", ["llm", "ml"], index=0, help="Choose between Prompted LLM or Logistic Regression baseline")
    top_k = st.slider("Top-k Precedents:", min_value=1, max_value=5, value=config.TOP_K_RETRIEVAL)
    
    st.divider()
    st.subheader("Audit Log Counter")
    if config.AUDIT_LOG_PATH.exists():
        with open(config.AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
            num_logs = sum(1 for line in f if line.strip())
        st.write(f"Logged decisions: `{num_logs}`")
    else:
        st.write("Logged decisions: `0`")

# 1. Customer Message Input Section
st.subheader("1. Customer Message Input")

dataset_samples = load_dataset_samples(limit=10)
dropdown_options = SAMPLE_QUERIES + dataset_samples
selected_sample = st.selectbox("Pick a sample message from dataset:", dropdown_options, index=0)

default_text = ""
if selected_sample and selected_sample != SAMPLE_QUERIES[0]:
    default_text = selected_sample

customer_message = st.text_area("Customer Message:", value=default_text, height=100, placeholder="Type or paste an incoming customer tweet...")

col1, col2 = st.columns([1, 5])
with col1:
    process_button = st.button("Process Message", type="primary")

if process_button and customer_message.strip():
    with st.spinner("Processing message through pipeline (classification, retrieval, generation, policy)..."):
        output = run_pipeline(
            customer_message=customer_message.strip(),
            classifier_method=classifier_method,
            top_k=top_k,
            compare_baselines=True,
        )

    st.divider()

    # Output 1: Intent Classification + Confidence
    st.subheader("2. Intent Classification")
    ic = output.intent_classification
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(label="Detected Intent", value=ic.intent)
    with col_b:
        st.metric(label="Confidence", value=f"{ic.confidence:.1%}")
    with col_c:
        st.metric(label="Risk Tier", value=ic.risk_tier.upper())

    st.write(f"**Rationale:** {ic.rationale}")
    
    if output.ml_baseline_classification:
        st.write(
            f"*Baseline ML Comparison:* `{output.ml_baseline_classification.intent}` "
            f"({output.ml_baseline_classification.confidence:.1%} confidence)"
        )

    st.divider()

    # Output 2: Top-k Retrieved Historical Precedents
    st.subheader(f"3. Top-{len(output.retrieved_precedents)} Retrieved Historical Precedents")
    st.write("Historical (customer message + brand resolution) pairs retrieved from threads satisfying the resolution proxy:")

    if output.retrieved_precedents:
        table_rows = []
        for p in output.retrieved_precedents:
            table_rows.append({
                "Rank": p["rank"],
                "Similarity Score": f"{p['similarity_score']:.4f}",
                "Past Customer Message": p["customer_text"],
                "Brand Resolution Reply": p["brand_reply"],
                "Thread ID": p["thread_id"],
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    else:
        st.write("No precedents found.")

    st.divider()

    # Output 3: Drafted Reply
    st.subheader("4. Drafted Grounded Reply")
    dr = output.drafted_reply
    st.text_area("Drafted Twitter Reply:", value=dr.reply_text, height=110)
    
    col_d, col_e = st.columns(2)
    with col_d:
        st.write(f"**Cited Precedents:** `{dr.used_precedents}`")
    with col_e:
        st.write(f"**Insufficient Context Flag:** `{dr.insufficient_context}`")
    if dr.grounding_notes:
        st.write(f"**Grounding Notes:** {dr.grounding_notes}")

    st.divider()

    # Output 4: Escalation Decision + Reason String
    st.subheader("5. Escalation Decision & Audit Policy")
    ed = output.escalation_decision
    
    if ed.action == "auto_handle":
        st.success(f"DECISION: {ed.action.upper()}")
    else:
        st.error(f"DECISION: {ed.action.upper()}")

    st.write("**Decision Reasons:**")
    for r in ed.reasons:
        st.write(f"- {r}")

    st.write(f"**Audit ID:** `{ed.audit_id}` (Logged to `{config.AUDIT_LOG_PATH.name}`)")

import sys
import json
import re
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.pipeline import run_pipeline
from src.taxonomy import get_taxonomy, DEFAULT_TAXONOMY
from src.evaluation import (
    evaluate_intent_classification,
    evaluate_escalation_policy,
    evaluate_llm_judge_agreement,
    run_full_evaluation,
)
from src.llm_judge import llm_judge
from src.create_golden_set import GOLDEN_SET_PATH, generate_golden_set


st.set_page_config(
    page_title=f"AI Support Agent - @{config.BRAND_HANDLE}",
    layout="wide",
    initial_sidebar_state="expanded"
)

SAMPLE_QUERIES = [
    "Select a pre-loaded sample query or type your own below...",
    "My iPhone battery is draining super fast after the latest iOS update. Any solution?",
    "I was charged twice for my subscription this month. I demand a refund right now!",
    "HOW DO I CANCEL MY ACCOUNT? THIS IS THE WORST SERVICE EVER AND I AM CALLING MY LAWYER!",
    "How can I turn on two-factor authentication on my Apple ID from settings?",
    "My AirPods won't connect or show up in my Bluetooth devices list.",
    "My screen cracked after dropping it. What is the repair process at Genius Bar?",
    "What is the weather like in Seattle today?",
]


def load_dataset_samples(limit: int = 15):
    """Load a few actual tweets from processed dataset for dropdown."""
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


def load_golden_dataframe() -> pd.DataFrame:
    """Load golden set CSV and ensure text/label columns are object dtype to prevent pandas float64 TypeError."""
    if not GOLDEN_SET_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(GOLDEN_SET_PATH)
    string_cols = [
        "human_true_intent",
        "human_should_escalate",
        "human_notes",
        "human_quality_helpfulness",
        "human_quality_tone",
        "human_quality_grounding",
        "human_quality_conciseness",
        "human_quality_notes",
        "llm_judge_helpfulness",
        "llm_judge_tone",
        "llm_judge_grounding",
        "llm_judge_conciseness",
        "llm_judge_rationale",
        "customer_message",
        "customer_clean_text",
        "initial_brand_reply",
        "candidate_intent",
        "system_predicted_intent",
        "system_drafted_reply",
        "system_escalation_action",
        "system_escalation_reasons",
        "trivial_predicted_intent",
        "trivial_drafted_reply",
        "trivial_escalation_action",
        "simple_predicted_intent",
        "simple_drafted_reply",
        "simple_escalation_action",
        "simple_escalation_reasons",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(object)
    return df


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title(f"🤖 @{config.BRAND_HANDLE} Agent")

app_mode = st.sidebar.radio(
    "Select Mode:",
    [
        "🚀 Live Pipeline Playground",
        "✍️ Golden Set Labeling & Human Judge",
        "📊 Evaluation & Benchmarks Dashboard",
    ],
    index=1,
)

st.sidebar.divider()
st.sidebar.subheader("System Architecture")
st.sidebar.write(f"• **Brand Handle:** `@{config.BRAND_HANDLE}`")
st.sidebar.write(f"• **Generation LLM:** `{config.OLLAMA_LLM_MODEL}`")
st.sidebar.write(f"• **Judge LLM:** `{config.OLLAMA_JUDGE_MODEL}`")
st.sidebar.write(f"• **Embeddings:** `{config.OLLAMA_EMBED_MODEL}` (768-dim)")


# ==============================================================================
# MODE 1: LIVE PIPELINE PLAYGROUND (Task 1 Core)
# ==============================================================================
if app_mode == "🚀 Live Pipeline Playground":
    st.header(f"Customer Support AI Agent (@{config.BRAND_HANDLE})")
    st.write(
        "Interactive Playground: Test single customer messages through the live end-to-end pipeline: "
        "intent classification, RAG resolution retrieval, grounded reply drafting, and escalation policy."
    )

    with st.sidebar:
        st.subheader("Playground Settings")
        classifier_method = st.radio("Classifier Model:", ["llm", "ml"], index=0)
        top_k = st.slider("Top-k Precedents:", min_value=1, max_value=5, value=config.TOP_K_RETRIEVAL)

    st.subheader("1. Customer Message Input")
    dataset_samples = load_dataset_samples(limit=10)
    dropdown_options = SAMPLE_QUERIES + dataset_samples
    selected_sample = st.selectbox("Pick a sample message from dataset:", dropdown_options, index=0)

    default_text = ""
    if selected_sample and selected_sample != SAMPLE_QUERIES[0]:
        default_text = selected_sample

    customer_message = st.text_area(
        "Customer Message:",
        value=default_text,
        height=90,
        placeholder="Type or paste an incoming customer tweet...",
    )

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        process_button = st.button("Process Message", type="primary")

    if process_button and customer_message.strip():
        with st.spinner("Processing message through live pipeline..."):
            output = run_pipeline(
                customer_message=customer_message.strip(),
                classifier_method=classifier_method,
                top_k=top_k,
                compare_baselines=True,
            )

        st.divider()

        # 2. Intent Classification
        st.subheader("2. Intent Classification")
        ic = output.intent_classification
        ca, cb, cc = st.columns(3)
        ca.metric("Detected Intent", ic.intent)
        cb.metric("Confidence", f"{ic.confidence:.1%}")
        cc.metric("Risk Tier", ic.risk_tier.upper())
        st.write(f"**Rationale:** {ic.rationale}")

        if output.ml_baseline_classification:
            st.write(
                f"*ML Baseline:* `{output.ml_baseline_classification.intent}` "
                f"({output.ml_baseline_classification.confidence:.1%} confidence)"
            )

        st.divider()

        # 3. Retrieved Precedents
        st.subheader(f"3. Top-{len(output.retrieved_precedents)} Retrieved Historical Precedents")
        if output.retrieved_precedents:
            rows = []
            for p in output.retrieved_precedents:
                rows.append({
                    "Rank": p["rank"],
                    "Similarity Score": f"{p['similarity_score']:.4f}",
                    "Past Customer Message": p["customer_text"],
                    "Brand Resolution Reply": p["brand_reply"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.divider()

        # 4. Drafted Reply
        st.subheader("4. Drafted Grounded Reply")
        dr = output.drafted_reply
        st.text_area("Drafted Twitter Reply:", value=dr.reply_text, height=100)
        cd, ce = st.columns(2)
        cd.write(f"**Cited Precedents:** `{dr.used_precedents}`")
        ce.write(f"**Insufficient Context Flag:** `{dr.insufficient_context}`")

        st.divider()

        # 5. Escalation Decision
        st.subheader("5. Escalation Decision & Audit Policy")
        ed = output.escalation_decision
        if ed.action == "auto_handle":
            st.success("DECISION: AUTO_HANDLE")
        else:
            st.error("DECISION: ESCALATE (Human Handoff Required)")

        st.write("**Decision Reasons:**")
        for r in ed.reasons:
            st.write(f"- {r}")
        st.caption(f"Audit ID: `{ed.audit_id}` (Logged to `{config.AUDIT_LOG_PATH.name}`)")


# ==============================================================================
# MODE 2: GOLDEN SET LABELING & HUMAN JUDGE (Task 2 Core)
# ==============================================================================
elif app_mode == "✍️ Golden Set Labeling & Human Judge":
    st.header("Golden Set Labeling & Human Quality Judge")
    st.write(
        "Hand-label the stratified 180-example evaluation set for ground-truth intent and escalation safety. "
        "For the flagged 40-example quality subset, hand-score reply quality across the 4 rubric dimensions."
    )

    if not GOLDEN_SET_PATH.exists():
        st.warning(f"Golden evaluation set CSV not found at `{GOLDEN_SET_PATH}`.")
        if st.button("Generate Golden Evaluation Set (180 Examples)", type="primary"):
            with st.spinner("Generating stratified golden set..."):
                generate_golden_set(force=True)
                st.rerun()
        st.stop()

    df = load_golden_dataframe()

    # Calculate labeling progress
    labeled_intent_count = df[df["human_true_intent"].notna() & (df["human_true_intent"].astype(str).str.strip() != "")].shape[0]
    labeled_esc_count = df[df["human_should_escalate"].notna() & (df["human_should_escalate"].astype(str).str.strip() != "")].shape[0]
    quality_subset_df = df[df["is_quality_subset"] == True]
    quality_scored_count = quality_subset_df[
        pd.to_numeric(quality_subset_df["human_quality_helpfulness"], errors="coerce").notna()
    ].shape[0]

    # Metrics Progress Bar
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    pcol1.metric("Total Evaluation Set", f"{len(df)} rows")
    pcol2.metric("Intents Labeled", f"{labeled_intent_count} / {len(df)}", f"{labeled_intent_count/len(df):.1%}")
    pcol3.metric("Escalations Labeled", f"{labeled_esc_count} / {len(df)}", f"{labeled_esc_count/len(df):.1%}")
    pcol4.metric("Quality Subset Scored", f"{quality_scored_count} / {len(quality_subset_df)}", f"{quality_scored_count/len(quality_subset_df):.1%}")

    st.divider()

    # Filters & Navigation
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1.5, 1.5, 2, 3])
    with col_nav1:
        if st.button("⏮️ Previous", use_container_width=True):
            if st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()

    with col_nav2:
        if st.button("⏭️ Next", use_container_width=True):
            if st.session_state.current_index < len(df) - 1:
                st.session_state.current_index += 1
                st.rerun()

    with col_nav3:
        # Find next unlabelled row
        unlabelled_indices = df[
            df["human_true_intent"].isna() | (df["human_true_intent"].astype(str).str.strip() == "")
        ].index.tolist()
        if st.button("⚡ Next Unlabelled", use_container_width=True):
            future_unlabelled = [i for i in unlabelled_indices if i > st.session_state.current_index]
            if future_unlabelled:
                st.session_state.current_index = future_unlabelled[0]
            elif unlabelled_indices:
                st.session_state.current_index = unlabelled_indices[0]
            st.rerun()

    with col_nav4:
        new_idx = st.number_input(
            f"Jump to row (0 to {len(df)-1}):",
            min_value=0,
            max_value=len(df)-1,
            value=st.session_state.current_index,
            step=1
        )
        if new_idx != st.session_state.current_index:
            st.session_state.current_index = new_idx
            st.rerun()

    # Load current row
    curr_idx = st.session_state.current_index
    row = df.iloc[curr_idx]

    # Row Header & Badges
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    bcol1.info(f"**Example #{curr_idx}** (Thread `{row.get('thread_id', 'N/A')}`)")
    
    ttype = row.get("thread_type", "single_turn")
    bcol2.write(f"**Depth:** `{ttype.upper()}` ({row.get('total_turns', 1)} turns)")

    res_cat = row.get("resolution_category", "unknown")
    if row.get("flag_pushed_to_dm", False):
        bcol3.error("⚠️ LIKELY PUSHED TO DM")
    elif res_cat == "resolved_on_platform":
        bcol3.success("✅ RESOLVED ON PLATFORM")
    else:
        bcol3.warning("🔄 UNRESOLVED FOLLOWUP")

    is_qual = bool(row.get("is_quality_subset", False))
    if is_qual:
        bcol4.warning("⭐ Quality Evaluation Subset (1 of 40)")
    else:
        bcol4.write("Standard Evaluation Item")

    # Message View
    st.markdown("#### Incoming Customer Tweet:")
    st.info(f"💬 \"{row.get('customer_message', '')}\"")

    with st.expander("Historical Brand Resolution from Dataset (Context)", expanded=False):
        st.write(f"**@{config.BRAND_HANDLE}:** {row.get('initial_brand_reply', 'No reply')}")

    # Pipeline Outputs
    st.markdown("#### AI Agent Pipeline Predictions (Audit):")
    p_col_a, p_col_b, p_col_c = st.columns(3)
    p_col_a.write(f"• **Predicted Intent:** `{row.get('system_predicted_intent', 'N/A')}`")
    p_col_b.write(f"• **Confidence:** `{float(row.get('system_confidence', 0.0)):.1%}`")
    p_col_c.write(f"• **Risk Tier:** `{str(row.get('system_risk_tier', '')).upper()}`")

    esc_action = str(row.get("system_escalation_action", "")).lower()
    if esc_action == "escalate":
        st.error(f"• **Policy Action:** ESCALATE ({row.get('system_escalation_reasons', '')})")
    else:
        st.success(f"• **Policy Action:** AUTO_HANDLE ({row.get('system_escalation_reasons', 'Passed all safety checks')})")

    st.write(f"• **Drafted Reply:** \"{row.get('system_drafted_reply', '')}\"")

    st.divider()

    # HUMAN LABELING FORM
    st.subheader("Your Ground-Truth Human Labels")
    taxonomy = get_taxonomy()
    intent_options = list(taxonomy.keys())

    # Pre-select current label if exists, else fallback to pipeline predicted intent
    existing_intent = str(row.get("human_true_intent", "")).strip()
    if existing_intent in intent_options:
        intent_index = intent_options.index(existing_intent)
    elif str(row.get("system_predicted_intent", "")).strip() in intent_options:
        intent_index = intent_options.index(str(row.get("system_predicted_intent", "")).strip())
    else:
        intent_index = 0

    with st.form(key=f"label_form_{curr_idx}"):
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            selected_intent = st.selectbox(
                "Correct Intent Category (Ground Truth):",
                options=intent_options,
                index=intent_index,
                help="Select the true primary intent of the customer inquiry"
            )

        with fcol2:
            existing_esc = str(row.get("human_should_escalate", "")).strip().lower()
            esc_default = 0 if existing_esc in ["escalate", "true", "1"] else (1 if existing_esc in ["auto_handle", "false", "0"] else (0 if esc_action == "escalate" else 1))
            
            selected_escalate = st.radio(
                "Human Escalation Decision (Ground Truth):",
                options=["Escalate (Human Agent Required)", "Auto-Handle (Safe to Automate)"],
                index=esc_default,
                help="Would sending an automated reply be appropriate or is human intervention required?"
            )

        human_note = st.text_input(
            "Annotator Notes / Edge Case Rationale:",
            value=str(row.get("human_notes", "") if pd.notna(row.get("human_notes")) else ""),
            placeholder="e.g. Frustrated tone, ambiguous hardware vs battery, billing dispute...",
        )

        # Quality Subset Sub-scores (if applicable)
        quality_scores = {}
        if is_qual:
            st.markdown("---")
            st.markdown("#### ⭐ Reply Quality Rubric (Hand-Score Ground Truth)")
            st.caption("Score the AI Agent's drafted reply above on a strict 1 to 5 integer scale:")

            def parse_int_safe(v, default=4):
                try:
                    return int(float(v)) if pd.notna(v) and str(v).strip() != "" else default
                except Exception:
                    return default

            qcol1, qcol2 = st.columns(2)
            with qcol1:
                q_help = st.slider(
                    "1. Helpfulness & Correctness (1-5):",
                    min_value=1, max_value=5,
                    value=parse_int_safe(row.get("human_quality_helpfulness"), 4),
                    help="1=Misleading/wrong, 3=Generic boilerplate, 5=Specific actionable resolution"
                )
                q_tone = st.slider(
                    "2. Tone Match to Brand (@AppleSupport) (1-5):",
                    min_value=1, max_value=5,
                    value=parse_int_safe(row.get("human_quality_tone"), 4),
                    help="1=Hostile/robotic, 3=Neutral corporate, 5=Warm, empathetic, classic Apple Support voice"
                )
            with qcol2:
                q_ground = st.slider(
                    "3. Grounding / Hallucination Check (1-5):",
                    min_value=1, max_value=5,
                    value=parse_int_safe(row.get("human_quality_grounding"), 4),
                    help="1=Severe hallucination (fake URLs/policies), 3=Generic advice, 5=100% grounded in verified precedents"
                )
                q_concise = st.slider(
                    "4. Conciseness & Twitter Fit (1-5):",
                    min_value=1, max_value=5,
                    value=parse_int_safe(row.get("human_quality_conciseness"), 5),
                    help="1=Exceeds 280 chars or rambling, 3=Slightly wordy, 5=Punchy, crisp fit (<200 chars)"
                )
            
            q_notes = st.text_input(
                "Reply Quality Notes:",
                value=str(row.get("human_quality_notes", "") if pd.notna(row.get("human_quality_notes")) else ""),
                placeholder="Notes on grounding or tone fidelity..."
            )
            quality_scores = {
                "helpfulness": q_help,
                "tone": q_tone,
                "grounding": q_ground,
                "conciseness": q_concise,
                "notes": q_notes,
            }

        # Submission buttons
        sub_col1, sub_col2 = st.columns([1.5, 4])
        with sub_col1:
            submit_label = st.form_submit_button("💾 Save & Next Row", type="primary", use_container_width=True)

    if submit_label:
        # Ensure column is object dtype before assignment to prevent pandas TypeError
        for col in ["human_true_intent", "human_should_escalate", "human_notes", "human_quality_notes"]:
            df[col] = df[col].astype(object)

        df.loc[curr_idx, "human_true_intent"] = str(selected_intent)
        df.loc[curr_idx, "human_should_escalate"] = "escalate" if "Escalate" in selected_escalate else "auto_handle"
        df.loc[curr_idx, "human_notes"] = str(human_note)

        if is_qual:
            for qcol in ["human_quality_helpfulness", "human_quality_tone", "human_quality_grounding", "human_quality_conciseness", "human_quality_notes"]:
                df[qcol] = df[qcol].astype(object)
            df.loc[curr_idx, "human_quality_helpfulness"] = quality_scores["helpfulness"]
            df.loc[curr_idx, "human_quality_tone"] = quality_scores["tone"]
            df.loc[curr_idx, "human_quality_grounding"] = quality_scores["grounding"]
            df.loc[curr_idx, "human_quality_conciseness"] = quality_scores["conciseness"]
            df.loc[curr_idx, "human_quality_notes"] = str(quality_scores["notes"])

        # Save to CSV
        df.to_csv(GOLDEN_SET_PATH, index=False, encoding="utf-8")
        st.success(f"Saved labels for row #{curr_idx}!")

        # Advance to next
        if curr_idx < len(df) - 1:
            st.session_state.current_index = curr_idx + 1
            st.rerun()

    # Section: LLM Judge On-Demand Evaluation
    if is_qual:
        with st.expander("🤖 Run / Inspect Independent LLM Judge (llama3.2:latest)", expanded=False):
            st.write(f"**Judge Model:** `{config.OLLAMA_JUDGE_MODEL}` (Independent model to eliminate self-preference bias)")
            has_llm_scores = pd.notna(row.get("llm_judge_helpfulness")) and str(row.get("llm_judge_helpfulness")).strip() != ""
            
            if has_llm_scores:
                st.write(
                    f"• **Helpfulness:** `{row.get('llm_judge_helpfulness')}/5` | "
                    f"• **Tone:** `{row.get('llm_judge_tone')}/5` | "
                    f"• **Grounding:** `{row.get('llm_judge_grounding')}/5` | "
                    f"• **Conciseness:** `{row.get('llm_judge_conciseness')}/5`"
                )
                st.caption(f"**Rationale:** {row.get('llm_judge_rationale', '')}")

            if st.button("Evaluate with LLM Judge", key=f"eval_judge_{curr_idx}"):
                with st.spinner("Calling independent judge model..."):
                    try:
                        precedents = json.loads(row.get("system_retrieved_precedents", "[]"))
                    except Exception:
                        precedents = []
                    score = llm_judge.evaluate(
                        customer_message=row.get("customer_clean_text", ""),
                        drafted_reply=row.get("system_drafted_reply", ""),
                        retrieved_precedents=precedents,
                        intent_label=selected_intent,
                    )
                    for jcol in ["llm_judge_helpfulness", "llm_judge_tone", "llm_judge_grounding", "llm_judge_conciseness", "llm_judge_rationale"]:
                        df[jcol] = df[jcol].astype(object)
                    df.loc[curr_idx, "llm_judge_helpfulness"] = score.helpfulness
                    df.loc[curr_idx, "llm_judge_tone"] = score.tone
                    df.loc[curr_idx, "llm_judge_grounding"] = score.grounding
                    df.loc[curr_idx, "llm_judge_conciseness"] = score.conciseness
                    df.loc[curr_idx, "llm_judge_rationale"] = str(score.rationale)
                    df.to_csv(GOLDEN_SET_PATH, index=False, encoding="utf-8")
                    st.rerun()


# ==============================================================================
# MODE 3: EVALUATION & BENCHMARKS DASHBOARD
# ==============================================================================
elif app_mode == "📊 Evaluation & Benchmarks Dashboard":
    st.header("Evaluation & Automated Proof Dashboard")
    st.write(
        "Benchmarking the Main Agent against Simple (Keyword + Template) and Trivial Baselines "
        "across Intent Accuracy, Macro-F1, Escalation Precision/Recall (False Negatives), and LLM Judge Agreement."
    )

    if not GOLDEN_SET_PATH.exists():
        st.warning("Golden set CSV not found. Please generate the golden set first.")
        st.stop()

    df = load_golden_dataframe()
    labeled_count = df[df["human_true_intent"].notna() & (df["human_true_intent"].astype(str).str.strip() != "")].shape[0]

    st.write(f"**Evaluation Progress:** `{labeled_count} / {len(df)}` examples labeled by human annotator ({labeled_count/len(df):.1%})")

    if labeled_count == 0:
        st.info("No human labels recorded yet. Switch to **'✍️ Golden Set Labeling & Human Judge'** mode to begin hand-labeling.")
        st.stop()

    # 1. Intent Classification Table
    st.subheader("1. Intent Classification Benchmark")
    sys_intent = evaluate_intent_classification(df, "system_predicted_intent", "human_true_intent")
    sim_intent = evaluate_intent_classification(df, "simple_predicted_intent", "human_true_intent")
    triv_intent = evaluate_intent_classification(df, "trivial_predicted_intent", "human_true_intent")

    intent_summary_rows = [
        {"Model / System": "Main Pipeline (Embedding + ML / LLM)", "Accuracy": f"{sys_intent['accuracy']:.1%}", "Macro-F1": f"{sys_intent['macro_f1']:.4f}", "Macro-Recall": f"{sys_intent['macro_recall']:.4f}"},
        {"Model / System": "Simple Baseline (Keyword Matcher)", "Accuracy": f"{sim_intent['accuracy']:.1%}", "Macro-F1": f"{sim_intent['macro_f1']:.4f}", "Macro-Recall": f"{sim_intent['macro_recall']:.4f}"},
        {"Model / System": "Trivial Baseline (Majority Intent)", "Accuracy": f"{triv_intent['accuracy']:.1%}", "Macro-F1": f"{triv_intent['macro_f1']:.4f}", "Macro-Recall": f"{triv_intent['macro_recall']:.4f}"},
    ]
    st.dataframe(pd.DataFrame(intent_summary_rows), use_container_width=True)

    if sys_intent.get("worst_intents"):
        st.markdown("**Worst-Performing Intents (Main Pipeline):**")
        for wi in sys_intent["worst_intents"]:
            st.write(f"- `{wi['intent']}`: F1 = `{wi['f1']:.4f}` (Support: {wi['support']})")

    st.divider()

    # 2. Escalation Policy Table
    st.subheader("2. Escalation Decision Benchmark (Focus: False Negatives)")
    st.write(
        "False Negatives (bot auto-handles a customer message that should have escalated to a human) "
        "represent the highest financial, security, and brand-damage risk."
    )
    sys_esc = evaluate_escalation_policy(df, "system_escalation_action", "human_should_escalate")
    sim_esc = evaluate_escalation_policy(df, "simple_escalation_action", "human_should_escalate")
    triv_esc = evaluate_escalation_policy(df, "trivial_escalation_action", "human_should_escalate")

    if sys_esc.get("status") != "no_labels":
        esc_rows = []
        for name, res in [("Main Pipeline", sys_esc), ("Simple Baseline", sim_esc), ("Trivial Baseline", triv_esc)]:
            fn_ratio = f"{res['false_negatives']} / {res['true_positives'] + res['false_negatives']}"
            esc_rows.append({
                "Model / System": name,
                "Accuracy": f"{res['accuracy']:.1%}",
                "Precision": f"{res['precision']:.1%}",
                "Recall": f"{res['recall']:.1%}",
                "F1 Score": f"{res['f1']:.4f}",
                "False Negatives (Costly Errors)": fn_ratio,
                "False Negative Rate": f"{res['false_negative_rate']:.1%}",
            })
        st.dataframe(pd.DataFrame(esc_rows), use_container_width=True)

    st.divider()

    # 3. LLM Judge Agreement Table
    st.subheader("3. Reply Quality: LLM-as-a-Judge Agreement (Cohen's Kappa)")
    st.write(
        f"Evaluating agreement between Human ground-truth and Independent Judge (`{config.OLLAMA_JUDGE_MODEL}`) "
        "across the 40 quality examples using Quadratic Weighted Kappa:"
    )
    agreement = evaluate_llm_judge_agreement(df)
    if agreement:
        judge_rows = []
        for dim, res in agreement.items():
            if res.get("status") != "insufficient_data":
                judge_rows.append({
                    "Rubric Dimension": dim.capitalize(),
                    "Quadratic Weighted Kappa": f"{res['quadratic_weighted_kappa']:.4f}",
                    "Unweighted Kappa": f"{res['unweighted_kappa']:.4f}",
                    "Mean Absolute Error (MAE)": f"{res['mae']:.2f}",
                    "Avg Human Score": f"{res['avg_human_score']:.2f} / 5",
                    "Avg LLM Judge Score": f"{res['avg_llm_score']:.2f} / 5",
                    "Evaluated Count": res["valid_count"],
                })
            else:
                judge_rows.append({
                    "Rubric Dimension": dim.capitalize(),
                    "Quadratic Weighted Kappa": "Pending scores",
                    "Unweighted Kappa": "Pending scores",
                    "Mean Absolute Error (MAE)": "N/A",
                    "Avg Human Score": "N/A",
                    "Avg LLM Judge Score": "N/A",
                    "Evaluated Count": res.get("valid_count", 0),
                })
        st.dataframe(pd.DataFrame(judge_rows), use_container_width=True)

    if st.button("Export Full Evaluation JSON", type="secondary"):
        out = run_full_evaluation()
        st.success(f"Full evaluation saved to `{config.LOGS_DIR / 'evaluation_results.json'}`")

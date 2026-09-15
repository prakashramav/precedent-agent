import sys
import os
import re
import json
import random
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.taxonomy import DEFAULT_TAXONOMY
from src.baselines import trivial_baseline, simple_baseline
from src.classifier import ml_classifier
from src.indexer import resolution_index
from src.escalation import evaluate_escalation_policy
from src.ollama_client import ollama_client

GOLDEN_SET_PATH = config.DATA_DIR / "golden_evaluation_set.csv"

DM_REGEX = re.compile(
    r"\b(dm|direct message|message us|reach out in dm|dm us|send a dm|private message)\b|"
    r"twitter\.com/messages|apple\.co/dm",
    re.IGNORECASE
)

INTENT_QUOTAS = {
    "cancellation_churn_complaint": 20,
    "billing_subscription_refund": 22,
    "account_access_security": 23,
    "hardware_physical_damage": 23,
    "connectivity_network_bluetooth": 23,
    "os_update_battery_drain": 23,
    "how_to_feature_inquiry": 23,
    "other_general_inquiry": 23,
}

PRIORITY_ORDER = [
    "cancellation_churn_complaint",
    "billing_subscription_refund",
    "account_access_security",
    "hardware_physical_damage",
    "connectivity_network_bluetooth",
    "os_update_battery_drain",
    "how_to_feature_inquiry",
]


def assign_candidate_intent(text: str) -> str:
    tl = text.lower()
    for ik in PRIORITY_ORDER:
        info = DEFAULT_TAXONOMY.get(ik, {})
        for kw in info.get("sample_keywords", []):
            pattern = rf"\b{re.escape(kw)}\b" if len(kw) > 3 else re.escape(kw)
            if re.search(pattern, tl):
                return ik
    return "other_general_inquiry"


def build_stratified_sample(seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed)

    if not config.PROCESSED_THREADS_PATH.exists():
        raise FileNotFoundError(f"Processed threads not found at {config.PROCESSED_THREADS_PATH}")

    threads = []
    with open(config.PROCESSED_THREADS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                threads.append(json.loads(line))

    print(f"Loaded {len(threads)} reconstructed threads.")

    intent_bins = {ik: [] for ik in INTENT_QUOTAS}

    for t in threads:
        c_text = t.get("initial_customer_text", "").strip()
        b_reply = t.get("brand_final_reply", "").strip()

        if len(c_text) < 20 or len(c_text) > 350 or not b_reply:
            continue

        c_intent = assign_candidate_intent(c_text)
        is_dm = bool(DM_REGEX.search(b_reply))
        ttype = "single_turn" if t.get("total_turns", 1) <= 2 else "multi_turn"

        if is_dm:
            res_cat = "likely_pushed_to_dm"
        elif t.get("went_quiet_after_brand", False):
            res_cat = "resolved_on_platform"
        else:
            res_cat = "unresolved_followup"

        candidate = {
            "thread_id": t.get("thread_id", ""),
            "customer_message": t.get("initial_customer_raw", c_text),
            "customer_clean_text": c_text,
            "total_turns": t.get("total_turns", 1),
            "thread_type": ttype,
            "resolution_category": res_cat,
            "flag_pushed_to_dm": is_dm,
            "initial_brand_reply": b_reply,
            "candidate_intent": c_intent,
        }
        intent_bins[c_intent].append(candidate)

    selected = []
    for ik, quota in INTENT_QUOTAS.items():
        candidates = intent_bins[ik]
        random.shuffle(candidates)

        multi = [c for c in candidates if c["thread_type"] == "multi_turn"]
        single = [c for c in candidates if c["thread_type"] == "single_turn"]

        n_multi = min(len(multi), max(2, int(quota * 0.30)))
        n_single = min(len(single), quota - n_multi)

        chosen = multi[:n_multi] + single[:n_single]
        if len(chosen) < quota:
            remaining = [c for c in candidates if c not in chosen]
            chosen += remaining[: quota - len(chosen)]

        selected.extend(chosen[:quota])

    random.shuffle(selected)
    print(f"Total stratified candidates selected: {len(selected)}")

    # Flag 40 examples (5 per intent) for quality judging subset
    quality_counts = {ik: 0 for ik in INTENT_QUOTAS}
    for item in selected:
        ik = item["candidate_intent"]
        if quality_counts[ik] < 5:
            item["is_quality_subset"] = True
            quality_counts[ik] += 1
        else:
            item["is_quality_subset"] = False

    return selected


def precompute_pipeline_outputs(selected: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Precomputes predictions and precedents:
    - Batch embeds queries with nomic-embed-text.
    - Vectorized NumPy search for historical resolution precedents.
    - Logistic Regression prediction on embeddings.
    - Grounded draft synthesis from top precedents.
    - Deterministic escalation evaluation.
    """
    if not resolution_index.is_built() and resolution_index.vectors is None:
        resolution_index.build_index()
    elif resolution_index.vectors is None:
        resolution_index.load_index()

    ml_classifier.train_or_load()

    total = len(selected)
    clean_texts = [item["customer_clean_text"] for item in selected]

    print(f"Batch embedding all {total} queries with nomic-embed-text...")
    all_query_embeddings = np.array(ollama_client.get_embeddings_batch(clean_texts, batch_size=32))
    
    # Normalize query embeddings
    norms = np.linalg.norm(all_query_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    norm_query_embeddings = all_query_embeddings / norms

    print("Running vectorized retrieval across historical resolution index...")
    sim_matrix = np.dot(norm_query_embeddings, resolution_index.vectors.T)

    print("Running ML intent classification on embeddings...")
    probs = ml_classifier.classifier.predict_proba(all_query_embeddings)
    pred_indices = np.argmax(probs, axis=1)

    taxonomy = DEFAULT_TAXONOMY
    rows = []

    print("Building golden set rows with grounded replies and escalation decisions...")
    for idx, item in enumerate(selected):
        c_text = item["customer_clean_text"]
        raw_msg = item["customer_message"]

        # Intent
        best_idx = pred_indices[idx]
        pred_intent = ml_classifier.classes[best_idx]
        conf = float(probs[idx][best_idx])
        risk_tier = taxonomy.get(pred_intent, {}).get("risk_tier", "medium")

        # Precedents
        sim_scores = sim_matrix[idx]
        top_k_indices = np.argsort(sim_scores)[::-1][:config.TOP_K_RETRIEVAL]
        precedents = []
        for rank, p_idx in enumerate(top_k_indices, 1):
            meta = resolution_index.metadata[p_idx]
            precedents.append({
                "rank": rank,
                "similarity_score": round(float(sim_scores[p_idx]), 4),
                "customer_text": meta.get("customer_text", ""),
                "brand_reply": meta.get("brand_reply", ""),
                "thread_id": meta.get("thread_id", ""),
            })
        top_sim = precedents[0]["similarity_score"] if precedents else 0.0

        # Baselines
        tb = trivial_baseline.predict(c_text)
        sb = simple_baseline.predict(c_text)

        # Grounded Drafted Reply
        top_p = precedents[0] if precedents else {}
        p_reply = top_p.get("brand_reply", "").strip()
        insufficient_context = top_sim < config.SIMILARITY_THRESHOLD

        if p_reply and not insufficient_context:
            drafted_text = f"We'd be glad to help with this. {p_reply}"
        else:
            drafted_text = sb.drafted_reply

        # Escalation Decision
        esc = evaluate_escalation_policy(
            customer_message=raw_msg,
            intent=pred_intent,
            risk_tier=risk_tier,
            confidence=conf,
            top_similarity=top_sim,
            insufficient_context=insufficient_context,
        )

        row = {
            "example_id": idx,
            "thread_id": item["thread_id"],
            "customer_message": raw_msg,
            "customer_clean_text": c_text,
            "total_turns": item["total_turns"],
            "thread_type": item["thread_type"],
            "resolution_category": item["resolution_category"],
            "flag_pushed_to_dm": item["flag_pushed_to_dm"],
            "initial_brand_reply": item["initial_brand_reply"],
            "candidate_intent": item["candidate_intent"],
            "is_quality_subset": item["is_quality_subset"],
            # Baseline predictions
            "trivial_predicted_intent": tb.intent,
            "trivial_drafted_reply": tb.drafted_reply,
            "trivial_escalation_action": tb.action,
            "simple_predicted_intent": sb.intent,
            "simple_drafted_reply": sb.drafted_reply,
            "simple_escalation_action": sb.action,
            "simple_escalation_reasons": "; ".join(sb.reasons),
            # System Pipeline predictions
            "system_predicted_intent": pred_intent,
            "system_confidence": round(conf, 4),
            "system_risk_tier": risk_tier,
            "system_retrieved_precedents": json.dumps(precedents, ensure_ascii=False),
            "system_top_similarity": round(top_sim, 4),
            "system_drafted_reply": drafted_text,
            "system_insufficient_context": insufficient_context,
            "system_escalation_action": esc.action,
            "system_escalation_reasons": "; ".join(esc.reasons),
            # Human labels (empty initially - to be labeled by user)
            "human_true_intent": "",
            "human_should_escalate": "",
            "human_notes": "",
            "human_quality_helpfulness": "",
            "human_quality_tone": "",
            "human_quality_grounding": "",
            "human_quality_conciseness": "",
            "human_quality_notes": "",
            # LLM Judge fields
            "llm_judge_helpfulness": "",
            "llm_judge_tone": "",
            "llm_judge_grounding": "",
            "llm_judge_conciseness": "",
            "llm_judge_rationale": "",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def generate_golden_set(force: bool = False) -> pd.DataFrame:
    GOLDEN_SET_PATH.parent.mkdir(parents=True, exist_ok=True)

    if GOLDEN_SET_PATH.exists() and not force:
        print(f"Golden set already exists at {GOLDEN_SET_PATH}. Loading existing CSV...")
        return pd.read_csv(GOLDEN_SET_PATH)

    print("Generating stratified sample of 180 evaluation candidates...")
    candidates = build_stratified_sample()
    df = precompute_pipeline_outputs(candidates)
    df.to_csv(GOLDEN_SET_PATH, index=False, encoding="utf-8")
    print(f"\nSuccessfully saved {len(df)} golden evaluation examples to {GOLDEN_SET_PATH}")
    return df


if __name__ == "__main__":
    generate_golden_set(force=True)

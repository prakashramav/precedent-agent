import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.taxonomy import DEFAULT_TAXONOMY


def parse_boolean(val: Any) -> Optional[bool]:
    """Robustly parse boolean from human input / CSV strings."""
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip().lower()
    if s in ["true", "1", "yes", "escalate", "t", "y"]:
        return True
    if s in ["false", "0", "no", "auto_handle", "f", "n"]:
        return False
    return None


def evaluate_intent_classification(
    df: pd.DataFrame,
    pred_col: str = "system_predicted_intent",
    true_col: str = "human_true_intent",
) -> Dict[str, Any]:
    """
    Evaluate intent classification accuracy, macro-F1, per-intent metrics,
    and confusion matrix.
    """
    valid = df[df[true_col].notna() & (df[true_col].astype(str).str.strip() != "")].copy()
    if len(valid) == 0:
        return {"status": "no_labels", "count": 0}

    y_true = valid[true_col].astype(str).str.strip()
    y_pred = valid[pred_col].astype(str).str.strip()

    labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    # Per-intent metrics
    per_class_prec, per_class_rec, per_class_f1, class_counts = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_intent = {}
    for i, label in enumerate(labels):
        per_intent[label] = {
            "precision": round(float(per_class_prec[i]), 4),
            "recall": round(float(per_class_rec[i]), 4),
            "f1": round(float(per_class_f1[i]), 4),
            "support": int(class_counts[i]),
        }

    # Find worst intents by F1 (with at least 1 support)
    supported_intents = [(k, v["f1"], v["support"]) for k, v in per_intent.items() if v["support"] > 0]
    worst_intents = sorted(supported_intents, key=lambda x: x[1])

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    return {
        "count": len(valid),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "per_intent": per_intent,
        "worst_intents": [{"intent": k, "f1": f1, "support": s} for k, f1, s in worst_intents[:3]],
        "confusion_matrix": cm_df.to_dict(),
        "labels": labels,
    }


def evaluate_escalation_policy(
    df: pd.DataFrame,
    pred_action_col: str = "system_escalation_action",
    true_col: str = "human_should_escalate",
) -> Dict[str, Any]:
    """
    Evaluate deterministic escalation policy against human ground truth.
    Special focus on False Negatives (system auto-handled but should have escalated).
    """
    valid = df[df[true_col].notna() & (df[true_col].astype(str).str.strip() != "")].copy()
    if len(valid) == 0:
        return {"status": "no_labels", "count": 0}

    valid["y_true"] = valid[true_col].apply(parse_boolean)
    valid = valid[valid["y_true"].notna()].copy()
    if len(valid) == 0:
        return {"status": "no_valid_labels", "count": 0}

    # True = Escalate, False = Auto-handle
    valid["y_pred"] = valid[pred_action_col].astype(str).str.lower().str.strip() == "escalate"

    y_true = valid["y_true"].values
    y_pred = valid["y_pred"].values

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=True, zero_division=0)

    # Breakdown of error types
    # True Positive: should escalate & did escalate
    tp = int(np.sum((y_true == True) & (y_pred == True)))
    # True Negative: safe to auto-handle & did auto-handle
    tn = int(np.sum((y_true == False) & (y_pred == False)))
    # False Positive: safe to auto-handle but escalated (human agent burden)
    fp = int(np.sum((y_true == False) & (y_pred == True)))
    # False Negative: SHOULD escalate but auto-handled (CRITICAL RISK!)
    fn = int(np.sum((y_true == True) & (y_pred == False)))

    total_positives = tp + fn
    fn_rate = round(float(fn / total_positives), 4) if total_positives > 0 else 0.0

    return {
        "count": len(valid),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "false_negative_rate": fn_rate,
    }


def evaluate_llm_judge_agreement(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute agreement (Cohen's Kappa & Quadratic Weighted Kappa)
    between LLM judge scores and human ground truth on the 40 quality examples.
    """
    # Filter for quality subset with both human and LLM scores
    subset = df[df.get("is_quality_subset", False) == True].copy()
    if len(subset) == 0:
        subset = df.copy()

    dimensions = ["helpfulness", "tone", "grounding", "conciseness"]
    agreement_results = {}

    for dim in dimensions:
        h_col = f"human_quality_{dim}"
        l_col = f"llm_judge_{dim}"

        if h_col not in subset.columns or l_col not in subset.columns:
            continue

        valid = subset[
            pd.to_numeric(subset[h_col], errors="coerce").notna() &
            pd.to_numeric(subset[l_col], errors="coerce").notna()
        ].copy()

        if len(valid) < 5:
            agreement_results[dim] = {
                "status": "insufficient_data",
                "valid_count": len(valid),
            }
            continue

        y_human = valid[h_col].astype(int).values
        y_llm = valid[l_col].astype(int).values

        # Unweighted Cohen's Kappa
        try:
            unweighted_kappa = cohen_kappa_score(y_human, y_llm)
        except Exception:
            unweighted_kappa = 0.0

        # Quadratic Weighted Kappa (standard for ordinal 1-5 Likert scales)
        try:
            weighted_kappa = cohen_kappa_score(y_human, y_llm, weights="quadratic")
        except Exception:
            weighted_kappa = 0.0

        # Mean absolute error and average scores
        mae = float(np.mean(np.abs(y_human - y_llm)))
        avg_human = float(np.mean(y_human))
        avg_llm = float(np.mean(y_llm))

        agreement_results[dim] = {
            "valid_count": len(valid),
            "unweighted_kappa": round(float(unweighted_kappa), 4),
            "quadratic_weighted_kappa": round(float(weighted_kappa), 4),
            "mae": round(mae, 4),
            "avg_human_score": round(avg_human, 2),
            "avg_llm_score": round(avg_llm, 2),
        }

    return agreement_results


def run_full_evaluation(csv_path: Path = config.DATA_DIR / "golden_evaluation_set.csv") -> Dict[str, Any]:
    """Run full comparative benchmark across System, Simple Baseline, and Trivial Baseline."""
    if not csv_path.exists():
        print(f"Golden set not found at {csv_path}. Please run src/create_golden_set.py first.")
        return {}

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    labeled_rows = df[df["human_true_intent"].notna() & (df["human_true_intent"].astype(str).str.strip() != "")].shape[0]

    print("=" * 60)
    print(f"EVALUATION SUITE: @{config.BRAND_HANDLE} Customer Support Agent")
    print(f"Dataset: {csv_path.name} | Total Examples: {total_rows} | Labeled: {labeled_rows} ({labeled_rows/total_rows:.1%})")
    print("=" * 60)

    if labeled_rows == 0:
        print("\n[NOTE] No human labels found yet in golden set CSV.")
        print("Please use the Streamlit labeling UI (app.py Mode 2) to complete hand-labeling.")
        return {"status": "unlabeled", "total": total_rows, "labeled": 0}

    # 1. Intent Classification Comparison
    system_intent = evaluate_intent_classification(df, "system_predicted_intent", "human_true_intent")
    simple_intent = evaluate_intent_classification(df, "simple_predicted_intent", "human_true_intent")
    trivial_intent = evaluate_intent_classification(df, "trivial_predicted_intent", "human_true_intent")

    # 2. Escalation Policy Comparison
    system_esc = evaluate_escalation_policy(df, "system_escalation_action", "human_should_escalate")
    simple_esc = evaluate_escalation_policy(df, "simple_escalation_action", "human_should_escalate")
    trivial_esc = evaluate_escalation_policy(df, "trivial_escalation_action", "human_should_escalate")

    # 3. LLM Judge Agreement
    judge_agreement = evaluate_llm_judge_agreement(df)

    results = {
        "metadata": {
            "total_examples": total_rows,
            "labeled_examples": labeled_rows,
            "brand_handle": config.BRAND_HANDLE,
            "judge_model": config.OLLAMA_JUDGE_MODEL,
        },
        "intent_classification": {
            "main_system": system_intent,
            "simple_baseline": simple_intent,
            "trivial_baseline": trivial_intent,
        },
        "escalation_policy": {
            "main_system": system_esc,
            "simple_baseline": simple_esc,
            "trivial_baseline": trivial_esc,
        },
        "llm_judge_agreement": judge_agreement,
    }

    # Print Summary Table
    print("\n1. INTENT CLASSIFICATION BENCHMARK")
    print("-" * 65)
    print(f"{'System / Model':<22} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Macro-Rec':<10}")
    print("-" * 65)
    for name, res in [("Main System", system_intent), ("Simple Baseline", simple_intent), ("Trivial Baseline", trivial_intent)]:
        if res.get("status") != "no_labels":
            print(f"{name:<22} | {res['accuracy']:<10.1%} | {res['macro_f1']:<10.4f} | {res['macro_recall']:<10.4f}")
    print("-" * 65)

    if system_intent.get("worst_intents"):
        print("Worst-performing intents (Main System):")
        for wi in system_intent["worst_intents"]:
            print(f"  - {wi['intent']}: F1={wi['f1']:.4f} (Support={wi['support']})")

    print("\n2. ESCALATION POLICY BENCHMARK (False Negatives = Dangerous Errors)")
    print("-" * 80)
    print(f"{'System / Model':<20} | {'Precision':<10} | {'Recall':<10} | {'F1':<8} | {'False Neg (FN)':<15} | {'FN Rate':<8}")
    print("-" * 80)
    for name, res in [("Main System", system_esc), ("Simple Baseline", simple_esc), ("Trivial Baseline", trivial_esc)]:
        if res.get("status") != "no_labels":
            fn_str = f"{res['false_negatives']} / {res['true_positives'] + res['false_negatives']}"
            print(f"{name:<20} | {res['precision']:<10.1%} | {res['recall']:<10.1%} | {res['f1']:<8.4f} | {fn_str:<15} | {res['false_negative_rate']:<8.1%}")
    print("-" * 80)

    print("\n3. LLM JUDGE AGREEMENT (Cohen's Kappa on 40 Quality Examples)")
    print("-" * 75)
    print(f"{'Dimension':<25} | {'Quadratic Weighted Kappa':<25} | {'MAE':<10} | {'Count':<6}")
    print("-" * 75)
    for dim, res in judge_agreement.items():
        if res.get("status") != "insufficient_data":
            print(f"{dim.capitalize():<25} | {res['quadratic_weighted_kappa']:<25.4f} | {res['mae']:<10.2f} | {res['valid_count']:<6}")
        else:
            print(f"{dim.capitalize():<25} | {'Pending user scores':<25} | {'N/A':<10} | {res.get('valid_count', 0):<6}")
    print("-" * 75)

    # Persist results JSON
    out_path = config.LOGS_DIR / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2, default=str))
    print(f"\nSaved detailed evaluation JSON to {out_path}")

    return results


if __name__ == "__main__":
    run_full_evaluation()

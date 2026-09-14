import sys
import re
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config


class EscalationDecision(BaseModel):
    action: str = Field(..., description="'auto_handle' or 'escalate'")
    reasons: List[str] = Field(default_factory=list, description="List of human-readable trigger reasons")
    intent: str
    risk_tier: str
    confidence: float
    top_similarity: float
    insufficient_context: bool
    triggered_rules: List[str]
    audit_id: str
    timestamp: str


def compute_caps_ratio(text: str) -> float:
    """Calculate ratio of uppercase letters among all alphabetic characters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    caps = [c for c in letters if c.isupper()]
    return len(caps) / len(letters)


def detect_urgent_keywords(text: str) -> List[str]:
    """Detect matching urgency or risk keywords using word boundaries."""
    text_lower = text.lower()
    matched = []
    for kw in config.URGENT_KEYWORDS:
        pattern = rf"\b{re.escape(kw)}\b"
        if re.search(pattern, text_lower):
            matched.append(kw)
    return matched


def evaluate_escalation_policy(
    customer_message: str,
    intent: str,
    risk_tier: str,
    confidence: float,
    top_similarity: float,
    insufficient_context: bool = False,
    audit_log_path: Path = config.AUDIT_LOG_PATH,
) -> EscalationDecision:
    """
    Explicit deterministic escalation policy combining:
      1. Intent risk_tier (high -> escalate)
      2. Classifier confidence (< threshold -> escalate)
      3. Retrieval top-1 similarity (< threshold or insufficient_context -> escalate)
      4. Cheap rule-based urgency / safety keyword check
      5. ALL-CAPS ratio check for frustration/urgency
    """
    reasons = []
    triggered_rules = []

    # Rule 1: High Risk Intent
    if risk_tier.lower() in config.HIGH_RISK_TIERS:
        triggered_rules.append("HIGH_RISK_INTENT")
        reasons.append(f"Intent '{intent}' has risk tier '{risk_tier}', requiring human agent review.")

    # Rule 2: Low Classifier Confidence
    if confidence < config.CONFIDENCE_THRESHOLD:
        triggered_rules.append("LOW_CLASSIFIER_CONFIDENCE")
        reasons.append(
            f"Classifier confidence ({confidence:.2f}) is below acceptable threshold ({config.CONFIDENCE_THRESHOLD:.2f})."
        )

    # Rule 3: Low Retrieval Similarity or Insufficient Context
    if top_similarity < config.SIMILARITY_THRESHOLD:
        triggered_rules.append("LOW_RETRIEVAL_SIMILARITY")
        reasons.append(
            f"Retrieval top-1 similarity ({top_similarity:.2f}) is below grounding threshold ({config.SIMILARITY_THRESHOLD:.2f})."
        )
    if insufficient_context:
        triggered_rules.append("INSUFFICIENT_RETRIEVAL_CONTEXT")
        reasons.append("Model flagged that historical precedents do not adequately cover this customer problem.")

    # Rule 4: Rule-based Urgency / High-Stakes Keywords
    matched_kws = detect_urgent_keywords(customer_message)
    if matched_kws:
        triggered_rules.append("URGENT_KEYWORDS_DETECTED")
        reasons.append(f"Urgent keywords detected in message: {matched_kws}.")

    # Rule 5: ALL CAPS Ratio (Customer Frustration)
    caps_ratio = compute_caps_ratio(customer_message)
    if len(customer_message) >= config.ALL_CAPS_MIN_LENGTH and caps_ratio >= config.ALL_CAPS_RATIO_THRESHOLD:
        triggered_rules.append("EXCESSIVE_ALL_CAPS")
        reasons.append(
            f"High uppercase letter ratio ({caps_ratio:.1%}) indicates heightened customer distress."
        )

    # Final Decision
    if triggered_rules:
        action = "escalate"
    else:
        action = "auto_handle"
        reasons.append("All risk checks passed: low/medium risk tier, high confidence, and strong historical precedent grounding.")

    decision = EscalationDecision(
        action=action,
        reasons=reasons,
        intent=intent,
        risk_tier=risk_tier,
        confidence=round(confidence, 4),
        top_similarity=round(top_similarity, 4),
        insufficient_context=insufficient_context,
        triggered_rules=triggered_rules,
        audit_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Write audit log to JSONL file
    log_escalation_decision(decision, customer_message, audit_log_path)

    return decision


def log_escalation_decision(
    decision: EscalationDecision,
    customer_message: str,
    log_path: Path = config.AUDIT_LOG_PATH,
):
    """Append structured escalation decision to audit log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    audit_entry = decision.model_dump()
    audit_entry["customer_message"] = customer_message

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Test case 1: Normal resolvable issue
    dec1 = evaluate_escalation_policy(
        customer_message="How can I update my payment card in settings?",
        intent="how_to_feature_inquiry",
        risk_tier="low",
        confidence=0.92,
        top_similarity=0.78,
    )
    print("Normal Query Decision:", dec1.action, dec1.reasons)

    # Test case 2: Angry churn threat with ALL CAPS
    dec2 = evaluate_escalation_policy(
        customer_message="YOUR SERVICE IS TERRIBLE I WANT TO CANCEL MY ACCOUNT RIGHT NOW AND CONTACT MY LAWYER",
        intent="cancellation_churn_complaint",
        risk_tier="high",
        confidence=0.88,
        top_similarity=0.62,
    )
    print("\nAngry Query Decision:", dec2.action, dec2.reasons)

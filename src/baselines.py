import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.taxonomy import DEFAULT_TAXONOMY


class BaselineOutput(BaseModel):
    intent: str
    confidence: float
    drafted_reply: str
    action: str  # 'escalate' or 'auto_handle'
    reasons: List[str]


class TrivialBaseline:
    """
    Trivial baseline:
    - Always predicts majority intent ('os_update_battery_drain').
    - Confidence fixed at empirical majority prevalence (~0.32).
    - Always returns one canned reply.
    - Always escalates because confidence (0.32) is below confidence threshold (0.65).
    """

    MAJORITY_INTENT = "os_update_battery_drain"
    MAJORITY_CONFIDENCE = 0.32
    CANNED_REPLY = (
        "Thanks for reaching out to @AppleSupport. Please restart your device and ensure you have "
        "the latest software update installed. You can also visit https://support.apple.com for troubleshooting."
    )

    def predict(self, text: str) -> BaselineOutput:
        # Escalates anything below confidence threshold
        escalate = self.MAJORITY_CONFIDENCE < config.CONFIDENCE_THRESHOLD
        reasons = ["Trivial baseline: confidence below threshold (0.32 < 0.65)"] if escalate else []

        return BaselineOutput(
            intent=self.MAJORITY_INTENT,
            confidence=self.MAJORITY_CONFIDENCE,
            drafted_reply=self.CANNED_REPLY,
            action="escalate" if escalate else "auto_handle",
            reasons=reasons,
        )


class SimpleBaseline:
    """
    Simple baseline:
    - Rule/keyword-based intent classifier (no embeddings, no LLM).
    - Small intent-to-template reply bank.
    - Escalates on keyword absence (low confidence), high-risk intents, or urgent keywords.
    """

    TEMPLATE_BANK = {
        "account_access_security": (
            "We understand how important account security is. Please head over to https://iforgot.apple.com "
            "to safely reset your Apple ID password or follow verification steps."
        ),
        "billing_subscription_refund": (
            "You can review your complete App Store purchase history and request a refund directly at "
            "https://reportaproblem.apple.com. Let us know if you need any further guidance."
        ),
        "cancellation_churn_complaint": (
            "We are very sorry to hear about your experience. Your feedback is important to us; "
            "please share more details regarding your device so we can help resolve your concerns."
        ),
        "os_update_battery_drain": (
            "We're here to help. Check Settings > Battery > Battery Health to inspect your maximum capacity "
            "and see which apps consume the most energy. Let us know what you see."
        ),
        "hardware_physical_damage": (
            "For physical hardware diagnostics and screen repair pricing, visit https://getsupport.apple.com "
            "to find your nearest authorized service provider or schedule a Genius Bar appointment."
        ),
        "connectivity_network_bluetooth": (
            "Let's get you connected. Try going to Settings > General > Reset > Reset Network Settings, "
            "or toggle Bluetooth/Airplane Mode off and back on."
        ),
        "how_to_feature_inquiry": (
            "You can discover step-by-step guides and settings tips for iOS features directly in the Apple Support "
            "app or at https://support.apple.com."
        ),
        "other_general_inquiry": (
            "Thanks for contacting @AppleSupport! How can we help you today? Please let us know your device model "
            "and iOS version."
        ),
    }

    # Priority ordering for keyword intent matching
    INTENT_PRIORITY = [
        "cancellation_churn_complaint",
        "billing_subscription_refund",
        "account_access_security",
        "hardware_physical_damage",
        "connectivity_network_bluetooth",
        "os_update_battery_drain",
        "how_to_feature_inquiry",
    ]

    def __init__(self):
        self.taxonomy = DEFAULT_TAXONOMY

    def predict(self, text: str) -> BaselineOutput:
        text_lower = text.lower()

        # 1. Match intent keywords
        matched_intent = None
        matched_keyword = None

        for intent_key in self.INTENT_PRIORITY:
            info = self.taxonomy.get(intent_key, {})
            keywords = info.get("sample_keywords", [])
            for kw in keywords:
                # Word boundary match where applicable
                pattern = rf"\b{re.escape(kw)}\b" if len(kw) > 3 else re.escape(kw)
                if re.search(pattern, text_lower):
                    matched_intent = intent_key
                    matched_keyword = kw
                    break
            if matched_intent:
                break

        if matched_intent:
            intent = matched_intent
            confidence = 0.80
            keyword_present = True
        else:
            intent = "other_general_inquiry"
            confidence = 0.40
            keyword_present = False

        # 2. Template bank reply
        drafted_reply = self.TEMPLATE_BANK.get(intent, self.TEMPLATE_BANK["other_general_inquiry"])

        # 3. Escalation policy
        reasons = []
        # Rule A: Escalate on keyword absence
        if not keyword_present:
            reasons.append("Simple baseline: keyword absence (unrecognized intent, confidence < 0.65)")

        # Rule B: Escalate on high-risk intents
        risk_tier = self.taxonomy.get(intent, {}).get("risk_tier", "medium")
        if risk_tier == "high":
            reasons.append(f"Simple baseline: high-risk intent detected ({intent})")

        # Rule C: Urgent keywords
        urgent_matches = [kw for kw in config.URGENT_KEYWORDS if kw in text_lower]
        if urgent_matches:
            reasons.append(f"Simple baseline: urgent keywords detected: {', '.join(urgent_matches)}")

        action = "escalate" if len(reasons) > 0 else "auto_handle"

        return BaselineOutput(
            intent=intent,
            confidence=confidence,
            drafted_reply=drafted_reply,
            action=action,
            reasons=reasons,
        )


# Global instances
trivial_baseline = TrivialBaseline()
simple_baseline = SimpleBaseline()


if __name__ == "__main__":
    test_queries = [
        "My battery is draining so fast after iOS update!",
        "I need a refund for double subscription charge, this is fraud!",
        "How do I pair my AirPods?",
        "Where is your headquarters located?",
    ]

    print("=== TRIVIAL BASELINE ===")
    for q in test_queries:
        out = trivial_baseline.predict(q)
        print(f"Q: {q}\n-> Intent: {out.intent} | Action: {out.action} | Reply: {out.drafted_reply[:60]}...\n")

    print("\n=== SIMPLE BASELINE ===")
    for q in test_queries:
        out = simple_baseline.predict(q)
        print(f"Q: {q}\n-> Intent: {out.intent} (conf: {out.confidence}) | Action: {out.action} | Reasons: {out.reasons}\n   Reply: {out.drafted_reply[:70]}...\n")

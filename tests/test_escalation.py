import unittest
from pathlib import Path
import tempfile
from src.escalation import evaluate_escalation_policy, detect_urgent_keywords, compute_caps_ratio


class TestEscalationPolicy(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_log_path = Path(self.temp_dir.name) / "test_audit.jsonl"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_auto_handle_standard_query(self):
        decision = evaluate_escalation_policy(
            customer_message="How do I change the wallpaper on my iPhone?",
            intent="how_to_feature_inquiry",
            risk_tier="low",
            confidence=0.90,
            top_similarity=0.82,
            insufficient_context=False,
            audit_log_path=self.test_log_path,
        )
        self.assertEqual(decision.action, "auto_handle")
        self.assertEqual(len(decision.triggered_rules), 0)

    def test_escalate_high_risk_tier(self):
        decision = evaluate_escalation_policy(
            customer_message="My account was hacked and I cannot login",
            intent="account_access_security",
            risk_tier="high",
            confidence=0.85,
            top_similarity=0.75,
            insufficient_context=False,
            audit_log_path=self.test_log_path,
        )
        self.assertEqual(decision.action, "escalate")
        self.assertIn("HIGH_RISK_INTENT", decision.triggered_rules)

    def test_escalate_low_confidence(self):
        decision = evaluate_escalation_policy(
            customer_message="Random weird query",
            intent="how_to_feature_inquiry",
            risk_tier="low",
            confidence=0.40,  # Below 0.65 threshold
            top_similarity=0.75,
            insufficient_context=False,
            audit_log_path=self.test_log_path,
        )
        self.assertEqual(decision.action, "escalate")
        self.assertIn("LOW_CLASSIFIER_CONFIDENCE", decision.triggered_rules)

    def test_escalate_low_similarity(self):
        decision = evaluate_escalation_policy(
            customer_message="Something completely unseen in history",
            intent="how_to_feature_inquiry",
            risk_tier="low",
            confidence=0.85,
            top_similarity=0.35,  # Below 0.55 threshold
            insufficient_context=False,
            audit_log_path=self.test_log_path,
        )
        self.assertEqual(decision.action, "escalate")
        self.assertIn("LOW_RETRIEVAL_SIMILARITY", decision.triggered_rules)

    def test_escalate_urgent_keywords(self):
        decision = evaluate_escalation_policy(
            customer_message="Give me my refund or I will sue you with my lawyer",
            intent="how_to_feature_inquiry",
            risk_tier="low",
            confidence=0.85,
            top_similarity=0.75,
            insufficient_context=False,
            audit_log_path=self.test_log_path,
        )
        self.assertEqual(decision.action, "escalate")
        self.assertIn("URGENT_KEYWORDS_DETECTED", decision.triggered_rules)

    def test_escalate_all_caps_frustration(self):
        decision = evaluate_escalation_policy(
            customer_message="FIX THIS RIGHT NOW YOU ARE ABSOLUTELY USELESS",
            intent="how_to_feature_inquiry",
            risk_tier="low",
            confidence=0.85,
            top_similarity=0.75,
            insufficient_context=False,
            audit_log_path=self.test_log_path,
        )
        self.assertEqual(decision.action, "escalate")
        self.assertIn("EXCESSIVE_ALL_CAPS", decision.triggered_rules)


if __name__ == "__main__":
    unittest.main()

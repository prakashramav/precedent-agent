import unittest
import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.evaluation import (
    evaluate_intent_classification,
    evaluate_escalation_policy,
    evaluate_llm_judge_agreement,
    parse_boolean,
)


class TestEvaluation(unittest.TestCase):

    def test_parse_boolean(self):
        self.assertTrue(parse_boolean("True"))
        self.assertTrue(parse_boolean("escalate"))
        self.assertTrue(parse_boolean(1))
        self.assertFalse(parse_boolean("False"))
        self.assertFalse(parse_boolean("auto_handle"))
        self.assertFalse(parse_boolean(0))
        self.assertIsNone(parse_boolean(None))
        self.assertIsNone(parse_boolean(""))

    def test_intent_classification_metrics(self):
        data = {
            "pred": ["intent_a", "intent_b", "intent_a", "intent_b"],
            "true": ["intent_a", "intent_b", "intent_b", "intent_b"],
        }
        df = pd.DataFrame(data)
        metrics = evaluate_intent_classification(df, pred_col="pred", true_col="true")
        self.assertEqual(metrics["count"], 4)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertIn("macro_f1", metrics)
        self.assertIn("confusion_matrix", metrics)

    def test_escalation_policy_metrics_false_negatives(self):
        # 4 samples:
        # 1: true=escalate, pred=escalate (TP)
        # 2: true=auto_handle, pred=auto_handle (TN)
        # 3: true=auto_handle, pred=escalate (FP)
        # 4: true=escalate, pred=auto_handle (FN - Dangerous error!)
        data = {
            "system_escalation_action": ["escalate", "auto_handle", "escalate", "auto_handle"],
            "human_should_escalate": ["escalate", "auto_handle", "auto_handle", "escalate"],
        }
        df = pd.DataFrame(data)
        res = evaluate_escalation_policy(df)
        self.assertEqual(res["true_positives"], 1)
        self.assertEqual(res["true_negatives"], 1)
        self.assertEqual(res["false_positives"], 1)
        self.assertEqual(res["false_negatives"], 1)
        self.assertEqual(res["false_negative_rate"], 0.5)  # 1 FN out of 2 true escalations

    def test_llm_judge_agreement_calculation(self):
        # Synthetic ordinal data with high agreement
        data = {
            "is_quality_subset": [True] * 10,
            "human_quality_helpfulness": [4, 5, 4, 3, 5, 2, 4, 5, 3, 4],
            "llm_judge_helpfulness": [4, 5, 4, 3, 4, 2, 4, 5, 3, 4],
            "human_quality_tone": [5, 5, 4, 4, 5, 3, 4, 5, 4, 5],
            "llm_judge_tone": [5, 5, 4, 4, 5, 3, 4, 5, 4, 5],
            "human_quality_grounding": [4, 4, 3, 5, 4, 2, 4, 5, 3, 4],
            "llm_judge_grounding": [4, 4, 3, 5, 4, 2, 4, 5, 3, 4],
            "human_quality_conciseness": [5, 5, 5, 4, 5, 4, 5, 5, 4, 5],
            "llm_judge_conciseness": [5, 5, 5, 4, 5, 4, 5, 5, 4, 5],
        }
        df = pd.DataFrame(data)
        res = evaluate_llm_judge_agreement(df)
        self.assertIn("helpfulness", res)
        self.assertIn("tone", res)
        self.assertGreater(res["helpfulness"]["quadratic_weighted_kappa"], 0.8)


if __name__ == "__main__":
    unittest.main()

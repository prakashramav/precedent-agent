import unittest
import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.create_golden_set import GOLDEN_SET_PATH


class TestGoldenSet(unittest.TestCase):

    def test_golden_set_file_and_columns(self):
        self.assertTrue(GOLDEN_SET_PATH.exists(), f"Golden set file missing at {GOLDEN_SET_PATH}")
        df = pd.read_csv(GOLDEN_SET_PATH)

        # 1. Total row count within range (150-250)
        self.assertEqual(len(df), 180)

        # 2. Key required columns present
        required_cols = [
            "example_id",
            "thread_id",
            "customer_message",
            "thread_type",
            "resolution_category",
            "flag_pushed_to_dm",
            "is_quality_subset",
            "system_predicted_intent",
            "system_confidence",
            "system_escalation_action",
            "human_true_intent",
            "human_should_escalate",
            "human_quality_helpfulness",
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Column {col} missing in golden set")

        # 3. Exactly 40 quality subset examples
        quality_count = df["is_quality_subset"].sum()
        self.assertEqual(quality_count, 40)

        # 4. Stratification across single and multi-turn
        thread_types = set(df["thread_type"].unique())
        self.assertIn("single_turn", thread_types)
        self.assertIn("multi_turn", thread_types)

        # 5. Stratification across DM push flag
        dm_flags = set(df["flag_pushed_to_dm"].unique())
        self.assertIn(True, dm_flags)
        self.assertIn(False, dm_flags)


if __name__ == "__main__":
    unittest.main()

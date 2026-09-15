import unittest
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.baselines import trivial_baseline, simple_baseline, TrivialBaseline, SimpleBaseline


class TestBaselines(unittest.TestCase):

    def setUp(self):
        self.trivial = TrivialBaseline()
        self.simple = SimpleBaseline()

    def test_trivial_baseline_always_majority_and_escalates(self):
        msg = "My screen is cracked and I need a repair."
        res = self.trivial.predict(msg)
        self.assertEqual(res.intent, "os_update_battery_drain")
        self.assertLess(res.confidence, 0.65)
        self.assertEqual(res.action, "escalate")
        self.assertIn("support.apple.com", res.drafted_reply)

    def test_simple_baseline_intent_keyword_match(self):
        # Battery keyword
        res_battery = self.simple.predict("My iPhone battery is draining so fast after iOS update!")
        self.assertEqual(res_battery.intent, "os_update_battery_drain")
        self.assertEqual(res_battery.confidence, 0.80)

        # Refund keyword (High risk -> escalate)
        res_refund = self.simple.predict("I want a refund for this double subscription charge!")
        self.assertEqual(res_refund.intent, "billing_subscription_refund")
        self.assertEqual(res_refund.action, "escalate")

        # AirPods Bluetooth keyword
        res_bt = self.simple.predict("My airpods are disconnecting from bluetooth.")
        self.assertEqual(res_bt.intent, "connectivity_network_bluetooth")

    def test_simple_baseline_keyword_absence_escalation(self):
        # Out of domain query with no known keywords
        res = self.simple.predict("What is the weather in Seattle?")
        self.assertEqual(res.intent, "other_general_inquiry")
        self.assertEqual(res.confidence, 0.40)
        self.assertEqual(res.action, "escalate")
        self.assertTrue(any("keyword absence" in r for r in res.reasons))


if __name__ == "__main__":
    unittest.main()

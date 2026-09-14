import unittest
import pandas as pd
from src.data_loader import clean_tweet_text, reconstruct_threads, extract_resolution_pairs


class TestDataLoader(unittest.TestCase):

    def test_clean_tweet_text(self):
        sample = "@AppleSupport my phone is broken! Check https://example.com/pic.jpg @user_123"
        res = clean_tweet_text(sample)
        self.assertEqual(res["clean_text"], "my phone is broken! Check")
        self.assertIn("@AppleSupport", res["mentions"])
        self.assertIn("@user_123", res["mentions"])
        self.assertIn("https://example.com/pic.jpg", res["urls"])
        self.assertEqual(res["raw_text"], sample)

    def test_reconstruct_threads_and_outcomes(self):
        data = [
            # Thread 1: Customer asks -> Brand replies -> Thread quiet (Resolution proxy = True)
            {
                "tweet_id": "101",
                "author_id": "115712",
                "inbound": True,
                "text": "@AppleSupport how to update?",
                "response_tweet_id": "102",
                "in_response_to_tweet_id": None,
                "created_at": "2017-10-31"
            },
            {
                "tweet_id": "102",
                "author_id": "AppleSupport",
                "inbound": False,
                "text": "@115712 Go to settings -> general -> update.",
                "response_tweet_id": None,
                "in_response_to_tweet_id": "101",
                "created_at": "2017-10-31"
            },
            # Thread 2: Customer asks -> Brand replies -> Customer complains further (Resolution proxy = False)
            {
                "tweet_id": "201",
                "author_id": "99999",
                "inbound": True,
                "text": "@AppleSupport still broken",
                "response_tweet_id": "202",
                "in_response_to_tweet_id": None,
                "created_at": "2017-10-31"
            },
            {
                "tweet_id": "202",
                "author_id": "AppleSupport",
                "inbound": False,
                "text": "@99999 Please restart your device.",
                "response_tweet_id": "203",
                "in_response_to_tweet_id": "201",
                "created_at": "2017-10-31"
            },
            {
                "tweet_id": "203",
                "author_id": "99999",
                "inbound": True,
                "text": "@AppleSupport that did not work at all!",
                "response_tweet_id": None,
                "in_response_to_tweet_id": "202",
                "created_at": "2017-10-31"
            }
        ]
        df = pd.DataFrame(data)
        threads = reconstruct_threads(df, brand_handle="AppleSupport")
        
        self.assertEqual(len(threads), 2)
        
        t1 = next(t for t in threads if t["thread_id"] == "101")
        self.assertTrue(t1["brand_replied"])
        self.assertFalse(t1["further_customer_complaint"])
        self.assertTrue(t1["went_quiet_after_brand"])
        self.assertTrue(t1["resolution_proxy"])

        t2 = next(t for t in threads if t["thread_id"] == "201")
        self.assertTrue(t2["brand_replied"])
        self.assertTrue(t2["further_customer_complaint"])
        self.assertFalse(t2["went_quiet_after_brand"])
        self.assertFalse(t2["resolution_proxy"])

        # Extract resolution pairs
        pairs = extract_resolution_pairs(threads)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["thread_id"], "101")


if __name__ == "__main__":
    unittest.main()

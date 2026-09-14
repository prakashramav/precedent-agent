import unittest
import numpy as np
from src.indexer import HistoricalResolutionIndex


class TestIndexer(unittest.TestCase):

    def test_cosine_similarity_ranking(self):
        index = HistoricalResolutionIndex()
        # Mock 3 normalized vectors
        # Vector 0: [1, 0, 0]
        # Vector 1: [0.7071, 0.7071, 0]
        # Vector 2: [0, 1, 0]
        index.vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.7071, 0.7071, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        index.metadata = [
            {"thread_id": "1", "customer_text": "query A", "brand_reply": "reply A"},
            {"thread_id": "2", "customer_text": "query B", "brand_reply": "reply B"},
            {"thread_id": "3", "customer_text": "query C", "brand_reply": "reply C"},
        ]

        # Query aligned with vector 0
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        scores = np.dot(index.vectors, q)
        top_indices = np.argsort(scores)[::-1]

        self.assertEqual(top_indices[0], 0)
        self.assertAlmostEqual(scores[0], 1.0, places=3)
        self.assertAlmostEqual(scores[1], 0.7071, places=3)
        self.assertAlmostEqual(scores[2], 0.0, places=3)


if __name__ == "__main__":
    unittest.main()

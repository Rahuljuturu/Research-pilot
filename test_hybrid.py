import unittest
from hybrid import BM25, reciprocal_rank_fusion

class RetrievalUnitTests(unittest.TestCase):
    def test_keyword_ranking(self):
        bm = BM25(["transformer uses attention heads", "apples and fruit orchard", "self attention transformer"])
        scores = bm.scores("transformer attention")
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(scores[1], 0)

    def test_fusion_rewards_agreement(self):
        ranked, scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
        self.assertEqual(ranked[0], "b")
        self.assertGreater(scores["b"], scores["c"])

if __name__ == "__main__":
    unittest.main()

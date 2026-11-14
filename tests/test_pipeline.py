"""Basic smoke tests for the key data-contract behaviors."""
import unittest
import pandas as pd
from src.training.models import PopularityRecommender, ItemItemCFRecommender

class PipelineSmokeTests(unittest.TestCase):
    def setUp(self):
        self.users = pd.DataFrame({"user_id": ["u1", "u2"]})
        self.items = pd.DataFrame({"item_id": ["i1", "i2", "i3"], "category": ["A", "A", "B"]})
        self.train = pd.DataFrame({
            "interaction_id": ["e1", "e2", "e3", "e4"],
            "user_id": ["u1", "u1", "u2", "u2"],
            "item_id": ["i1", "i2", "i1", "i3"],
            "rating": [5, 4, 4, 3],
        })

    def test_models_return_unseen_items(self):
        for model in [PopularityRecommender().fit(self.train, self.items), ItemItemCFRecommender().fit(self.train, self.items)]:
            recs = model.recommend("u1", {"i1", "i2"}, k=2)
            self.assertTrue(set(recs).isdisjoint({"i1", "i2"}))
            self.assertLessEqual(len(recs), 2)

if __name__ == "__main__":
    unittest.main()

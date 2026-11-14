"""Popularity baseline and item-item collaborative filtering recommender."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.config import MODELS


class PopularityRecommender:
    def fit(self, train: pd.DataFrame, items: pd.DataFrame):
        stats = train.groupby("item_id").agg(count=("interaction_id", "count"), avg_rating=("rating", "mean"))
        # Bayesian-smoothed ranking prevents low-volume items from dominating by chance.
        global_mean = float(train["rating"].mean())
        m = 5
        stats["score"] = (stats["count"] / (stats["count"] + m)) * stats["avg_rating"] + (m / (stats["count"] + m)) * global_mean
        self.ranking = stats["score"].sort_values(ascending=False).index.tolist()
        self.items = items["item_id"].astype(str).tolist()
        return self

    def recommend(self, user_id: str, seen: set[str], k: int = 5):
        recs = [i for i in self.ranking if i not in seen]
        if len(recs) < k:
            recs += [i for i in self.items if i not in seen and i not in recs]
        return recs[:k]


class ItemItemCFRecommender:
    def fit(self, train: pd.DataFrame, items: pd.DataFrame):
        matrix = train.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="mean", fill_value=0)
        self.user_ids = matrix.index.astype(str).tolist()
        self.item_ids = matrix.columns.astype(str).tolist()
        # Treat 4-5 star interactions as positive implicit feedback; lower ratings are 0.
        self.matrix = np.where(matrix.to_numpy(dtype=float) >= 4.0, matrix.to_numpy(dtype=float) - 3.0, 0.0)
        self.similarity = cosine_similarity(self.matrix.T)
        np.fill_diagonal(self.similarity, 0.0)
        self.item_lookup = {item: idx for idx, item in enumerate(self.item_ids)}
        self.user_lookup = {user: idx for idx, user in enumerate(self.user_ids)}
        self.all_items = items["item_id"].astype(str).tolist()
        self.popular_fallback = (train.groupby("item_id").size().sort_values(ascending=False).index.astype(str).tolist())
        return self

    def recommend(self, user_id: str, seen: set[str], k: int = 5):
        scores = np.zeros(len(self.item_ids), dtype=float)
        uidx = self.user_lookup.get(str(user_id))
        if uidx is not None:
            user_vector = self.matrix[uidx]
            scores = user_vector @ self.similarity
        candidates = [(self.item_ids[i], scores[i]) for i in range(len(self.item_ids)) if self.item_ids[i] not in seen]
        candidates.sort(key=lambda x: (-x[1], x[0]))
        recs = [item for item, score in candidates if score > 0][:k]
        if len(recs) < k:
            fallback = [i for i in self.popular_fallback if i not in seen and i not in recs]
            fallback += [i for i in self.all_items if i not in seen and i not in recs and i not in fallback]
            recs.extend(fallback[:k-len(recs)])
        return recs[:k]


def save_model(model, filename: str):
    path = MODELS / filename
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path

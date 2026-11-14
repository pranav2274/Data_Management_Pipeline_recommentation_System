#!/usr/bin/env python3
"""Simple offline prediction entry point using the trained feature/model artifacts."""
import argparse
import pickle
import pandas as pd
from src.config import MODELS, PROCESSED, FEATURES


def recommend(user_id: str, model_name: str = "cf", k: int = 5):
    model_file = "item_item_cf_model.pkl" if model_name == "cf" else "popularity_model.pkl"
    with open(MODELS / model_file, "rb") as f:
        model = pickle.load(f)
    train = pd.read_csv(PROCESSED / "train_interactions.csv")
    item_features = pd.read_csv(FEATURES / "item_features.csv")
    known_users = set(train["user_id"].astype(str))
    if user_id not in known_users:
        seen = set()
        print(f"User {user_id} has no training history; returning catalog/popularity fallback.")
    else:
        seen = set(train.loc[train["user_id"].astype(str) == user_id, "item_id"].astype(str))
    recs = model.recommend(user_id, seen, k=k)
    result = item_features[item_features["item_id"].astype(str).isin(recs)].copy()
    result["rank"] = result["item_id"].astype(str).map({item: i + 1 for i, item in enumerate(recs)})
    return result.sort_values("rank")[["rank", "item_id", "title", "category", "price", "item_popularity", "item_avg_rating"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate top-K recommendations for a user")
    parser.add_argument("--user", required=True, help="User ID such as U0001")
    parser.add_argument("--model", choices=["cf", "popularity"], default="cf")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    print(recommend(args.user, args.model, args.k).to_string(index=False))

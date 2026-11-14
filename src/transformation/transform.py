"""Per-user chronological holdout and reusable analytical aggregates."""
import pandas as pd
from src.config import PROCESSED


def transform_data(users: pd.DataFrame, items: pd.DataFrame, interactions: pd.DataFrame, train_fraction: float = 0.8):
    ordered = interactions.sort_values(["user_id", "timestamp", "interaction_id"]).copy()
    ordered["user_order"] = ordered.groupby("user_id").cumcount()
    ordered["user_count"] = ordered.groupby("user_id")["interaction_id"].transform("count")
    ordered["train_cut"] = (ordered["user_count"] * train_fraction).apply(lambda x: max(1, int(x)))
    train = ordered[ordered["user_order"] < ordered["train_cut"]].drop(columns=["user_order", "user_count", "train_cut"])
    test = ordered[ordered["user_order"] >= ordered["train_cut"]].drop(columns=["user_order", "user_count", "train_cut"])

    # Summaries are computed from training data only to prevent test leakage.
    user_activity = train.groupby("user_id").agg(
        user_activity_count=("interaction_id", "count"), user_avg_rating=("rating", "mean"),
        last_interaction=("timestamp", "max"),
    ).reset_index()
    item_popularity = train.groupby("item_id").agg(
        item_popularity=("interaction_id", "count"), item_avg_rating=("rating", "mean"),
    ).reset_index()
    interaction_matrix = train.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="mean", fill_value=0)

    train.to_csv(PROCESSED / "train_interactions.csv", index=False)
    test.to_csv(PROCESSED / "test_interactions.csv", index=False)
    user_activity.to_csv(PROCESSED / "user_activity_summary.csv", index=False)
    item_popularity.to_csv(PROCESSED / "item_popularity.csv", index=False)
    interaction_matrix.to_csv(PROCESSED / "user_item_matrix.csv")
    return train, test, user_activity, item_popularity, interaction_matrix

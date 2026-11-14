"""Training-time feature layer; the same tables are joined at recommendation time."""
import pandas as pd
from src.config import FEATURES


def build_features(users, items, train_interactions, user_activity, item_popularity):
    reference_time = train_interactions["timestamp"].max()
    preferred = (train_interactions.merge(items[["item_id", "category"]], on="item_id")
                 .groupby(["user_id", "category"]).size().rename("n").reset_index())
    preferred = preferred.sort_values(["user_id", "n", "category"], ascending=[True, False, True]).drop_duplicates("user_id")
    preferred = preferred.rename(columns={"category": "preferred_category"})[["user_id", "preferred_category"]]

    user_features = users.merge(user_activity, on="user_id", how="left").merge(preferred, on="user_id", how="left")
    user_features["user_activity_count"] = user_features["user_activity_count"].fillna(0).astype(int)
    user_features["user_avg_rating"] = user_features["user_avg_rating"].fillna(train_interactions["rating"].mean())
    user_features["days_since_last_interaction"] = (
        reference_time - pd.to_datetime(user_features["last_interaction"], utc=True)
    ).dt.total_seconds().div(86400).fillna(9999).clip(lower=0)
    user_features["preferred_category"] = user_features["preferred_category"].fillna("Unknown")

    item_features = items.merge(item_popularity, on="item_id", how="left")
    item_features["item_popularity"] = item_features["item_popularity"].fillna(0).astype(int)
    item_features["item_avg_rating"] = item_features["item_avg_rating"].fillna(train_interactions["rating"].mean())

    interaction_features = train_interactions.merge(
        user_features[["user_id", "user_activity_count", "user_avg_rating", "preferred_category", "days_since_last_interaction"]],
        on="user_id", how="left"
    ).merge(item_features[["item_id", "category", "item_popularity", "item_avg_rating"]], on="item_id", how="left")

    user_features.to_csv(FEATURES / "user_features.csv", index=False)
    item_features.to_csv(FEATURES / "item_features.csv", index=False)
    interaction_features.to_csv(FEATURES / "interaction_features.csv", index=False)
    return user_features, item_features, interaction_features

"""Standardization, safe dimension imputation, and clean interaction persistence."""
import pandas as pd
from src.config import PROCESSED


def prepare_data(users: pd.DataFrame, items: pd.DataFrame, interactions: pd.DataFrame):
    users = users.copy()
    items = items.copy()
    interactions = interactions.copy()

    users["user_id"] = users["user_id"].astype(str).str.strip()
    users["age"] = pd.to_numeric(users["age"], errors="coerce").fillna(users["age"].median()).astype(int)
    users["country"] = users["country"].fillna("Unknown").astype(str).str.upper()
    users["signup_date"] = pd.to_datetime(users["signup_date"], errors="coerce", utc=True)

    items["item_id"] = items["item_id"].astype(str).str.strip()
    items["title"] = items["title"].fillna("Untitled item").astype(str).str.strip()
    items["category"] = items["category"].fillna("Uncategorized").astype(str).str.strip().str.title()
    items["price"] = pd.to_numeric(items["price"], errors="coerce").fillna(items["price"].median())

    interactions["user_id"] = interactions["user_id"].astype(str).str.strip()
    interactions["item_id"] = interactions["item_id"].astype(str).str.strip()
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    interactions["event_type"] = interactions["event_type"].fillna("view").astype(str).str.lower()
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], utc=True, errors="coerce")
    interactions = interactions.dropna(subset=["user_id", "item_id", "rating", "timestamp"])

    users.to_csv(PROCESSED / "users_clean.csv", index=False)
    items.to_csv(PROCESSED / "items_clean.csv", index=False)
    interactions.to_csv(PROCESSED / "interactions_clean.csv", index=False)
    return users, items, interactions

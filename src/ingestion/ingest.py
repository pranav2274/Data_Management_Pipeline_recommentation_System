"""CSV ingestion and deterministic synthetic data generation."""
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import RAW, RANDOM_SEED, SYNTHETIC_USERS, SYNTHETIC_ITEMS, INTERACTIONS_PER_USER

CATEGORIES = ["Electronics", "Books", "Home", "Sports", "Fashion", "Beauty"]


def generate_synthetic_data(force: bool = False) -> None:
    """Create reproducible users/items/interactions CSVs, including known bad rows."""
    files = [RAW / "users.csv", RAW / "items.csv", RAW / "interactions.csv"]
    if all(p.exists() for p in files) and not force:
        return

    rng = np.random.default_rng(RANDOM_SEED)
    users = pd.DataFrame({
        "user_id": [f"U{i:04d}" for i in range(1, SYNTHETIC_USERS + 1)],
        "age": rng.integers(18, 66, size=SYNTHETIC_USERS),
        "country": rng.choice(["IN", "US", "GB", "CA", "DE"], size=SYNTHETIC_USERS),
        "signup_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 700, size=SYNTHETIC_USERS), unit="D"),
    })

    items = pd.DataFrame({
        "item_id": [f"I{i:04d}" for i in range(1, SYNTHETIC_ITEMS + 1)],
        "title": [f"Catalog item {i:03d}" for i in range(1, SYNTHETIC_ITEMS + 1)],
        "category": rng.choice(CATEGORIES, size=SYNTHETIC_ITEMS),
        "price": np.round(rng.uniform(5, 500, size=SYNTHETIC_ITEMS), 2),
    })

    rows = []
    interaction_counter = 1
    # Each user has a latent preferred category, creating a learnable preference signal.
    user_preferred_category = rng.integers(0, len(CATEGORIES), size=SYNTHETIC_USERS)
    item_categories = items["category"].tolist()
    for u_idx, user in users.iterrows():
        chosen = rng.choice(SYNTHETIC_ITEMS, size=INTERACTIONS_PER_USER, replace=False)
        for item_idx in chosen:
            cat_idx = CATEGORIES.index(item_categories[item_idx])
            is_preferred = cat_idx == user_preferred_category[u_idx]
            positive_probability = 0.90 if is_preferred else 0.08
            if rng.random() < positive_probability:
                rating = int(rng.choice([4, 5], p=[0.40, 0.60]))
            else:
                rating = int(rng.choice([1, 2, 3], p=[0.15, 0.35, 0.50]))
            event_type = "purchase" if rating >= 4 and rng.random() < 0.45 else ("click" if rng.random() < 0.7 else "view")
            # Spread interactions over time, while retaining per-user chronology.
            days = int(rng.integers(0, 540))
            ts = pd.Timestamp("2025-01-01") + pd.Timedelta(days=days, minutes=int(rng.integers(0, 1440)))
            rows.append({
                "interaction_id": f"E{interaction_counter:07d}",
                "user_id": user["user_id"],
                "item_id": items.iloc[item_idx]["item_id"],
                "rating": rating,
                "event_type": event_type,
                "timestamp": ts.isoformat(),
            })
            interaction_counter += 1

    interactions = pd.DataFrame(rows)
    # Introduce four deterministic quality defects for the validation demonstration.
    bad = interactions.iloc[[0, 1, 2, 3]].copy()
    bad.loc[bad.index[0], "rating"] = 9                         # invalid rating
    bad.loc[bad.index[1], "user_id"] = "U9999"                  # orphan user key
    bad.loc[bad.index[2], "timestamp"] = "not-a-timestamp"      # invalid timestamp
    bad.loc[bad.index[3], "item_id"] = "I9999"                  # orphan item key
    interactions = pd.concat([interactions, bad], ignore_index=True)
    # Add an exact duplicate interaction id/row to test duplicate handling.
    interactions = pd.concat([interactions, interactions.iloc[[4]]], ignore_index=True)

    users.to_csv(RAW / "users.csv", index=False)
    items.to_csv(RAW / "items.csv", index=False)
    interactions.to_csv(RAW / "interactions.csv", index=False)


def ingest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three raw CSV entities."""
    missing = [name for name in ["users.csv", "items.csv", "interactions.csv"] if not (RAW / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw input(s): {missing}. Run generate_synthetic_data() first.")
    users = pd.read_csv(RAW / "users.csv")
    items = pd.read_csv(RAW / "items.csv")
    interactions = pd.read_csv(RAW / "interactions.csv")
    return users, items, interactions

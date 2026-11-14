"""Rule-based data quality checks with row-level quarantine."""
import json
import pandas as pd
from src.config import ERRORS, REPORTS


def validate_data(users: pd.DataFrame, items: pd.DataFrame, interactions: pd.DataFrame):
    report = {"checks": {}, "summary": {}}
    errors = []

    def check(name, passed, detail):
        report["checks"][name] = {"passed": bool(passed), "detail": detail}

    check("users_missing_values", not users.isna().any().any(), f"{int(users.isna().sum().sum())} missing cell(s)")
    check("items_missing_values", not items.isna().any().any(), f"{int(items.isna().sum().sum())} missing cell(s)")
    check("interactions_missing_values", not interactions.isna().any().any(), f"{int(interactions.isna().sum().sum())} missing cell(s)")
    check("users_duplicate_keys", not users["user_id"].duplicated().any(), f"{int(users['user_id'].duplicated().sum())} duplicate user_id(s)")
    check("items_duplicate_keys", not items["item_id"].duplicated().any(), f"{int(items['item_id'].duplicated().sum())} duplicate item_id(s)")
    check("interaction_duplicate_ids", not interactions["interaction_id"].duplicated().any(), f"{int(interactions['interaction_id'].duplicated().sum())} duplicate interaction_id(s)")

    numeric_age = pd.to_numeric(users["age"], errors="coerce")
    check("user_age_numeric", numeric_age.notna().all(), f"{int(numeric_age.isna().sum())} non-numeric age(s)")
    numeric_price = pd.to_numeric(items["price"], errors="coerce")
    check("item_price_numeric", numeric_price.notna().all(), f"{int(numeric_price.isna().sum())} non-numeric price(s)")

    parsed_ts = pd.to_datetime(interactions["timestamp"], errors="coerce", utc=True)
    check("timestamps_valid", parsed_ts.notna().all(), f"{int(parsed_ts.isna().sum())} invalid timestamp(s)")
    ratings = pd.to_numeric(interactions["rating"], errors="coerce")
    valid_rating = ratings.between(1, 5) & (ratings % 1 == 0)
    check("interaction_rating_numeric", ratings.notna().all(), f"{int(ratings.isna().sum())} non-numeric rating(s)")
    check("ratings_between_1_and_5", valid_rating.all(), f"{int((~valid_rating).sum())} invalid rating(s)")

    user_ids, item_ids = set(users["user_id"].astype(str)), set(items["item_id"].astype(str))
    orphan_users = ~interactions["user_id"].astype(str).isin(user_ids)
    orphan_items = ~interactions["item_id"].astype(str).isin(item_ids)
    check("referential_integrity_users", not orphan_users.any(), f"{int(orphan_users.sum())} interaction(s) reference unknown users")
    check("referential_integrity_items", not orphan_items.any(), f"{int(orphan_items.sum())} interaction(s) reference unknown items")

    # Row-level quarantine. Duplicate IDs: keep the first occurrence, quarantine subsequent rows.
    seen = set()
    for idx, row in interactions.iterrows():
        reasons = []
        iid = str(row.get("interaction_id", ""))
        if iid in seen:
            reasons.append("duplicate_interaction_id")
        seen.add(iid)
        r = pd.to_numeric(pd.Series([row.get("rating")]), errors="coerce").iloc[0]
        if pd.isna(r) or r < 1 or r > 5 or float(r).is_integer() is False:
            reasons.append("invalid_rating")
        if pd.isna(pd.to_datetime(row.get("timestamp"), errors="coerce", utc=True)):
            reasons.append("invalid_timestamp")
        if str(row.get("user_id")) not in user_ids:
            reasons.append("unknown_user_id")
        if str(row.get("item_id")) not in item_ids:
            reasons.append("unknown_item_id")
        if reasons:
            rec = row.to_dict()
            rec["error_reasons"] = ";".join(reasons)
            errors.append(rec)

    error_df = pd.DataFrame(errors)
    clean = interactions.drop_duplicates(subset=["interaction_id"], keep="first").copy()
    clean["rating"] = pd.to_numeric(clean["rating"], errors="coerce")
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce", utc=True)
    valid_mask = (
        clean["rating"].between(1, 5)
        & clean["timestamp"].notna()
        & clean["user_id"].astype(str).isin(user_ids)
        & clean["item_id"].astype(str).isin(item_ids)
    )
    clean = clean.loc[valid_mask].copy()
    clean["rating"] = clean["rating"].astype(float)

    error_df.to_csv(ERRORS / "invalid_interactions.csv", index=False)
    report["summary"] = {
        "users_rows": int(len(users)), "items_rows": int(len(items)),
        "interaction_rows_received": int(len(interactions)),
        "valid_interactions": int(len(clean)), "quarantined_interactions": int(len(error_df)),
        "raw_checks_passed": bool(all(v["passed"] for v in report["checks"].values())),
        "quarantine_completed": True,
        "clean_interactions_passed": bool(clean["rating"].between(1, 5).all() and clean["timestamp"].notna().all() and clean["user_id"].astype(str).isin(user_ids).all() and clean["item_id"].astype(str).isin(item_ids).all()),
    }
    with open(REPORTS / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return clean, error_df, report

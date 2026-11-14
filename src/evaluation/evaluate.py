"""Top-K ranking evaluation: precision, recall, NDCG, coverage and diversity."""
import math
import pandas as pd
from src.config import REPORTS, TOP_K, MIN_RELEVANT_RATING


def _dcg(relevant_flags):
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevant_flags))


def evaluate_model(model, model_name: str, train: pd.DataFrame, test: pd.DataFrame, items: pd.DataFrame, k: int = TOP_K):
    seen_by_user = train.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(str))).to_dict()
    truth = test[test["rating"] >= MIN_RELEVANT_RATING].groupby("user_id")["item_id"].apply(lambda s: set(s.astype(str))).to_dict()
    category = items.set_index("item_id")["category"].astype(str).to_dict()
    user_list = sorted(set(test["user_id"].astype(str)))
    precisions, recalls, ndcgs, diversities = [], [], [], []
    all_recommended = set()

    for user in user_list:
        actual = truth.get(user, set())
        recs = model.recommend(user, seen_by_user.get(user, set()), k=k)
        all_recommended.update(recs)
        hits = [1 if item in actual else 0 for item in recs]
        precisions.append(sum(hits) / k)
        recalls.append(sum(hits) / len(actual) if actual else 0.0)
        ideal = _dcg([1] * min(len(actual), k))
        ndcgs.append(_dcg(hits) / ideal if ideal else 0.0)
        cats = [category.get(item, "Unknown") for item in recs]
        pairs = len(cats) * (len(cats) - 1) / 2
        same_category_pairs = sum(1 for i in range(len(cats)) for j in range(i + 1, len(cats)) if cats[i] == cats[j])
        diversities.append(1 - same_category_pairs / pairs if pairs else 0.0)

    return {
        "model": model_name,
        f"precision@{k}": float(sum(precisions) / len(precisions)) if precisions else 0.0,
        f"recall@{k}": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0,
        "coverage": float(len(all_recommended) / len(items)) if len(items) else 0.0,
        "diversity": float(sum(diversities) / len(diversities)) if diversities else 0.0,
        "evaluated_users": len(user_list),
        "note": "Ranking-only recommender; explicit rating prediction (RMSE/MAE) is not performed.",
    }


def evaluate_and_save(popularity, collaborative, train, test, items):
    results = [
        evaluate_model(popularity, "Popularity baseline", train, test, items),
        evaluate_model(collaborative, "Item-item collaborative filtering", train, test, items),
    ]
    df = pd.DataFrame(results)
    df.to_csv(REPORTS / "evaluation_results.csv", index=False)
    return df

"""Dependency-aware Python workflow. Schedule with cron/task scheduler if needed."""
import logging
import json
from datetime import datetime, timezone
from src.config import REPORTS, RANDOM_SEED, TRAIN_FRACTION, TOP_K
from src.ingestion.ingest import generate_synthetic_data, ingest
from src.validation.validate import validate_data
from src.preparation.prepare import prepare_data
from src.transformation.transform import transform_data
from src.features.build_features import build_features
from src.training.models import PopularityRecommender, ItemItemCFRecommender, save_model
from src.training.tracking import log_experiment
from src.evaluation.evaluate import evaluate_and_save


def run_pipeline(force_data: bool = False):
    log_path = REPORTS / "pipeline.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()], force=True)
    logger = logging.getLogger("recommendation_pipeline")
    run_summary = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "tasks": []}

    def task(name, fn, *args, **kwargs):
        logger.info("START %s", name)
        result = fn(*args, **kwargs)
        logger.info("SUCCESS %s", name)
        run_summary["tasks"].append({"task": name, "status": "success"})
        return result

    try:
        task("generate_or_check_synthetic_data", generate_synthetic_data, force=force_data)
        users, items, raw_interactions = task("ingestion", ingest)
        clean_interactions, error_rows, validation_report = task("validation_and_quarantine", validate_data, users, items, raw_interactions)
        users, items, clean_interactions = task("preparation", prepare_data, users, items, clean_interactions)
        train, test, user_activity, item_popularity, matrix = task(
            "time_split_and_derived_datasets", transform_data, users, items, clean_interactions, TRAIN_FRACTION
        )
        user_features, item_features, interaction_features = task(
            "feature_management", build_features, users, items, train, user_activity, item_popularity
        )

        popularity = task("train_popularity_baseline", PopularityRecommender().fit, train, items)
        collab = task("train_item_item_collaborative_filtering", ItemItemCFRecommender().fit, train, items)
        popularity_path = save_model(popularity, "popularity_model.pkl")
        collab_path = save_model(collab, "item_item_cf_model.pkl")

        results = task("evaluation", evaluate_and_save, popularity, collab, train, test, items)
        for row in results.to_dict(orient="records"):
            name = row.pop("model")
            log_experiment(name, {"top_k": TOP_K, "train_fraction": TRAIN_FRACTION, "random_seed": RANDOM_SEED}, row,
                           [popularity_path if "Popularity" in name else collab_path, REPORTS / "evaluation_results.csv"])

        run_summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_summary["status"] = "success"
        run_summary["counts"] = {
            "users": int(len(users)), "items": int(len(items)), "raw_interactions": int(len(raw_interactions)),
            "valid_interactions": int(len(clean_interactions)), "quarantined_interactions": int(len(error_rows)),
            "train_interactions": int(len(train)), "test_interactions": int(len(test)),
            "interaction_feature_rows": int(len(interaction_features)),
        }
        (REPORTS / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("Results: %s", REPORTS / "evaluation_results.csv")
        return run_summary
    except Exception as exc:
        run_summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_summary["status"] = "failed"
        run_summary["error"] = repr(exc)
        (REPORTS / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        logger.exception("PIPELINE FAILED")
        raise

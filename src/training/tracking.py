"""Small file-based experiment tracker (parameters, metrics, artifacts, run IDs)."""
import json
import uuid
from datetime import datetime, timezone
from src.config import EXPERIMENTS


def log_experiment(model_name, params, metrics, artifacts):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    payload = {
        "run_id": run_id,
        "model_name": model_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": params,
        "metrics": metrics,
        "artifacts": [str(p) for p in artifacts],
    }
    path = EXPERIMENTS / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (EXPERIMENTS / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    return path

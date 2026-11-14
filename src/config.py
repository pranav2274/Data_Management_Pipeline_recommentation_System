from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
FEATURES = DATA / "features"
ERRORS = DATA / "errors"
MODELS = ROOT / "models"
EXPERIMENTS = ROOT / "experiments"
REPORTS = ROOT / "reports"

RANDOM_SEED = 42
TRAIN_FRACTION = 0.80
TOP_K = 5
MIN_RELEVANT_RATING = 4.0
SYNTHETIC_USERS = 300
SYNTHETIC_ITEMS = 150
INTERACTIONS_PER_USER = 35

for path in [RAW, PROCESSED, FEATURES, ERRORS, MODELS, EXPERIMENTS, REPORTS]:
    path.mkdir(parents=True, exist_ok=True)

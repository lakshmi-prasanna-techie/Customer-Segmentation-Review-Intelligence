from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = None  # Set to "data/your_dataset.csv" when using real data
RANDOM_STATE = 42

BASE_NUMERIC_COLS = [
    "recency_days",
    "order_frequency",
    "avg_order_value",
    "review_count",
    "avg_star_rating",
]

POSITIVE_WORDS = {
    "great", "amazing", "love", "fast", "excellent", "perfect", "smooth",
    "helpful", "best", "reliable", "awesome", "worth", "recommend", "happy",
}

NEGATIVE_WORDS = {
    "broken", "late", "slow", "terrible", "bad", "refund", "damaged", "poor",
    "worst", "delay", "defective", "disappointed", "useless", "issue", "cancel",
}

COMPLAINT_KEYWORDS = {
    "refund", "broken", "damaged", "late", "delay", "defective", "cancel", "support",
}

K_MIN = 2
K_MAX = 7
KMEANS_N_INIT = 25
SVD_COMPONENTS = 3
TFIDF_MAX_FEATURES = 300

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
}

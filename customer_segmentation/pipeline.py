from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

CSV_PATH = None
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "recency_days",
    "order_frequency",
    "avg_order_value",
    "review_count",
    "avg_star_rating",
    "sentiment_score",
    "complaint_mention_rate",
]

ZERO_FILL_COLS = ["review_count", "complaint_mention_rate", "order_frequency"]
MEDIAN_FILL_COLS = ["recency_days", "avg_order_value", "avg_star_rating", "sentiment_score"]


def load_or_generate_data(csv_path: str | None) -> pd.DataFrame:
    if csv_path and Path(csv_path).exists():
        print(f"Loading dataset from: {csv_path}")
        return pd.read_csv(csv_path)

    print("Using synthetic customer + review intelligence dataset...")
    rng = np.random.default_rng(42)
    n_samples = 1500

    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n_samples)],
        "recency_days": rng.integers(1, 180, size=n_samples),
        "order_frequency": rng.integers(1, 35, size=n_samples),
        "avg_order_value": rng.gamma(shape=3.0, scale=35.0, size=n_samples),
        "review_count": rng.integers(0, 20, size=n_samples),
        "avg_star_rating": np.clip(rng.normal(4.1, 0.8, size=n_samples), 1.0, 5.0),
        "sentiment_score": np.clip(rng.normal(0.25, 0.45, size=n_samples), -1.0, 1.0),
        "complaint_mention_rate": np.clip(rng.beta(1.5, 8.0, size=n_samples), 0.0, 1.0),
    })

    loyal_idx = rng.choice(n_samples, size=450, replace=False)
    df.loc[loyal_idx, "recency_days"] = rng.integers(1, 30, size=450)
    df.loc[loyal_idx, "order_frequency"] = rng.integers(12, 35, size=450)
    df.loc[loyal_idx, "avg_order_value"] *= 1.4
    df.loc[loyal_idx, "avg_star_rating"] = np.clip(rng.normal(4.7, 0.3, size=450), 1.0, 5.0)
    df.loc[loyal_idx, "sentiment_score"] = np.clip(rng.normal(0.7, 0.2, size=450), -1.0, 1.0)
    df.loc[loyal_idx, "complaint_mention_rate"] = np.clip(rng.beta(1.0, 15.0, size=450), 0.0, 1.0)

    remaining_idx = np.setdiff1d(np.arange(n_samples), loyal_idx)
    critics_idx = rng.choice(remaining_idx, size=350, replace=False)
    df.loc[critics_idx, "recency_days"] = rng.integers(60, 180, size=350)
    df.loc[critics_idx, "avg_star_rating"] = np.clip(rng.normal(2.2, 0.6, size=350), 1.0, 5.0)
    df.loc[critics_idx, "sentiment_score"] = np.clip(rng.normal(-0.5, 0.25, size=350), -1.0, 1.0)
    df.loc[critics_idx, "complaint_mention_rate"] = np.clip(rng.beta(5.0, 2.5, size=350), 0.0, 1.0)

    return df


def preprocess_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    clean_df = df.copy()
    for col in feature_cols:
        if col not in clean_df.columns:
            raise ValueError(f"Missing required column: '{col}'")
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    for col in ZERO_FILL_COLS:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].fillna(0.0)

    for col in MEDIAN_FILL_COLS:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    return clean_df


def fit_kmeans_with_silhouette(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 7):
    scores = {}
    print("\n--- Silhouette Score Evaluation ---")
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        print(f"k = {k} -> silhouette_score = {score:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\nBest k by silhouette score: k = {best_k} (score = {scores[best_k]:.4f})")

    best_kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    segments = best_kmeans.fit_predict(X_scaled)
    return best_kmeans, segments, scores


def plot_pca_segments(X_scaled: np.ndarray, segments: np.ndarray, out_path: Path):
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(X_scaled)
    total_var = pca.explained_variance_ratio_.sum()

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=segments,
        cmap="tab10",
        alpha=0.75,
        s=28,
        edgecolors="none",
    )
    plt.title(f"2D PCA View of Customer Segments (Explained Variance: {total_var:.1%})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Segment ID")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved 2D PCA plot to: {out_path}")
    return pca


def train_xgboost_classifier(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- XGBoost Held-Out Evaluation ---")
    print(classification_report(y_test, y_pred, digits=4))

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    print("--- XGBoost Feature Importances ---")
    print(importances.round(4))
    return model


def main():
    raw_df = load_or_generate_data(CSV_PATH)
    clean_df = preprocess_features(raw_df, FEATURE_COLS)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_df[FEATURE_COLS])

    kmeans_model, segments, _ = fit_kmeans_with_silhouette(X_scaled, k_min=2, k_max=7)
    clean_df["segment"] = segments

    print("\n--- Segment Profile Summary (Mean Values) ---")
    print(clean_df.groupby("segment")[FEATURE_COLS].mean().round(2))

    pca_model = plot_pca_segments(
        X_scaled, segments, ARTIFACTS_DIR / "pca_segments_2d.png"
    )

    xgb_model = train_xgboost_classifier(clean_df[FEATURE_COLS], clean_df["segment"])

    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    joblib.dump(kmeans_model, ARTIFACTS_DIR / "kmeans_model.joblib")
    joblib.dump(pca_model, ARTIFACTS_DIR / "pca_model.joblib")
    joblib.dump(xgb_model, ARTIFACTS_DIR / "xgb_segment_model.joblib")
    clean_df.to_csv(ARTIFACTS_DIR / "segmented_customers.csv", index=False)
    print(f"\nSaved models and segmented data to '{ARTIFACTS_DIR}/'")

    loaded_xgb = joblib.load(ARTIFACTS_DIR / "xgb_segment_model.joblib")
    sample_new_customers = pd.DataFrame([
        {
            "recency_days": 10,
            "order_frequency": 22,
            "avg_order_value": 115.0,
            "review_count": 8,
            "avg_star_rating": 4.8,
            "sentiment_score": 0.75,
            "complaint_mention_rate": 0.02,
        },
        {
            "recency_days": 120,
            "order_frequency": 3,
            "avg_order_value": 42.0,
            "review_count": 4,
            "avg_star_rating": 2.0,
            "sentiment_score": -0.6,
            "complaint_mention_rate": 0.65,
        },
    ])

    preds = loaded_xgb.predict(sample_new_customers[FEATURE_COLS])
    probs = loaded_xgb.predict_proba(sample_new_customers[FEATURE_COLS])

    print("\n--- Scoring New Customers ---")
    for idx, (pred, prob) in enumerate(zip(preds, probs)):
        print(f"Customer {idx + 1} -> Predicted Segment: {pred} | Class Probs: {np.round(prob, 3)}")


if __name__ == "__main__":
    main()
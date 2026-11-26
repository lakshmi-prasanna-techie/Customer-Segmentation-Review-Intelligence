import json
import joblib
from config import ARTIFACTS_DIR, CSV_PATH
from data_loader import load_or_generate_data
from nlp_features import build_feature_matrix
from clustering import assign_business_personas, evaluate_and_fit_kmeans, scale_features
from viz import plot_feature_importance, plot_pca_with_centroids, plot_validation_curve
from classifier import train_and_evaluate_xgboost


def main():
    raw_df = load_or_generate_data(CSV_PATH)
    feat_df, feature_cols, tfidf, svd = build_feature_matrix(raw_df, fit=True)

    scaler, X_scaled = scale_features(feat_df, feature_cols)
    kmeans, segments, val_df = evaluate_and_fit_kmeans(X_scaled)
    feat_df["segment"] = segments

    persona_map = assign_business_personas(feat_df)
    feat_df["persona"] = feat_df["segment"].map(lambda s: persona_map[s]["persona"])
    feat_df["recommended_action"] = feat_df["segment"].map(lambda s: persona_map[s]["recommended_action"])

    print("\n[main] Persona Summary Table")
    summary_cols = [
        "recency_days", "order_frequency", "avg_order_value",
        "avg_star_rating", "nlp_sentiment_score", "nlp_complaint_rate",
    ]
    print(feat_df.groupby(["segment", "persona"])[summary_cols].mean().round(2))

    plot_validation_curve(val_df, int(kmeans.n_clusters))
    pca = plot_pca_with_centroids(X_scaled, segments, kmeans, persona_map)

    xgb_model, cv_f1, test_f1, importances = train_and_evaluate_xgboost(feat_df[feature_cols], feat_df["segment"])
    plot_feature_importance(importances)

    bundle = {
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "xgb_model": xgb_model,
        "tfidf": tfidf,
        "svd": svd,
        "feature_cols": feature_cols,
        "persona_map": persona_map,
    }
    joblib.dump(bundle, ARTIFACTS_DIR / "segmentation_bundle.joblib")
    feat_df.to_csv(ARTIFACTS_DIR / "segmented_customers.csv", index=False)

    best_sil = float(val_df.loc[val_df["k"] == kmeans.n_clusters, "silhouette"].iloc[0])
    best_db = float(val_df.loc[val_df["k"] == kmeans.n_clusters, "davies_bouldin"].iloc[0])

    metrics = {
        "n_customers": int(len(feat_df)),
        "n_features": int(len(feature_cols)),
        "selected_k": int(kmeans.n_clusters),
        "silhouette_score": round(best_sil, 4),
        "davies_bouldin_index": round(best_db, 4),
        "xgb_5fold_cv_macro_f1": round(cv_f1, 4),
        "xgb_test_macro_f1": round(test_f1, 4),
        "personas": persona_map,
    }
    (ARTIFACTS_DIR / "pipeline_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[main] Done! All artifacts saved to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()

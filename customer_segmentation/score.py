from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from config import ARTIFACTS_DIR
from nlp_features import build_feature_matrix


def score_customers(new_df: pd.DataFrame, bundle_path: Path = ARTIFACTS_DIR / "segmentation_bundle.joblib") -> pd.DataFrame:
    bundle = joblib.load(bundle_path)
    tfidf = bundle["tfidf"]
    svd = bundle["svd"]
    xgb_model = bundle["xgb_model"]
    feature_cols = bundle["feature_cols"]
    persona_map = bundle["persona_map"]

    feat_df, _, _, _ = build_feature_matrix(new_df, tfidf=tfidf, svd=svd, fit=False)
    preds = xgb_model.predict(feat_df[feature_cols])
    probs = xgb_model.predict_proba(feat_df[feature_cols])

    out = new_df.copy().reset_index(drop=True)
    out["predicted_segment"] = preds
    out["predicted_persona"] = [persona_map[int(s)]["persona"] for s in preds]
    out["recommended_action"] = [persona_map[int(s)]["recommended_action"] for s in preds]
    out["confidence"] = np.round(probs.max(axis=1), 4)
    return out


if __name__ == "__main__":
    sample_incoming = pd.DataFrame([
        {
            "customer_id": "NEW_001",
            "recency_days": 8,
            "order_frequency": 24,
            "avg_order_value": 110.0,
            "review_count": 9,
            "avg_star_rating": 4.9,
            "review_text": "Amazing quality and fast delivery, love it and recommend to everyone",
        },
        {
            "customer_id": "NEW_002",
            "recency_days": 135,
            "order_frequency": 3,
            "avg_order_value": 45.0,
            "review_count": 3,
            "avg_star_rating": 1.8,
            "review_text": "Broken on arrival and late delivery, want a refund immediately",
        },
    ])
    print(score_customers(sample_incoming)[["customer_id", "predicted_segment", "predicted_persona", "confidence"]])

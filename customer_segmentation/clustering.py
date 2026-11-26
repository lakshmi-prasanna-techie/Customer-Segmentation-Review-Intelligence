import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from config import KMEANS_N_INIT, K_MAX, K_MIN, RANDOM_STATE


def scale_features(df: pd.DataFrame, feature_cols: list[str]):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    return scaler, X_scaled


def evaluate_and_fit_kmeans(X_scaled: np.ndarray, k_min: int = K_MIN, k_max: int = K_MAX):
    rows = []
    print("\n[clustering] Evaluating K-Means across candidate cluster counts...")
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        rows.append({"k": k, "silhouette": sil, "davies_bouldin": db})
        print(f"  k={k} -> Silhouette={sil:.4f} | Davies-Bouldin={db:.4f}")

    val_df = pd.DataFrame(rows)
    candidates = val_df[val_df["k"] >= 3] if (val_df["k"] >= 3).any() else val_df
    best_row = candidates.loc[candidates["silhouette"].idxmax()]
    best_k = int(best_row["k"])

    print(
        f"[clustering] Selected k={best_k} "
        f"(Silhouette={best_row['silhouette']:.4f}, Davies-Bouldin={best_row['davies_bouldin']:.4f})"
    )

    final_km = KMeans(n_clusters=best_k, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
    segments = final_km.fit_predict(X_scaled)
    return final_km, segments, val_df


def assign_business_personas(df: pd.DataFrame) -> dict[int, dict[str, str]]:
    summary = df.groupby("segment")[
        ["recency_days", "order_frequency", "avg_star_rating", "nlp_sentiment_score"]
    ].mean()

    persona_map: dict[int, dict[str, str]] = {}
    for seg_id, row in summary.iterrows():
        if row["avg_star_rating"] >= 4.4 and row["order_frequency"] >= 14:
            name = "Champions & Brand Advocates"
            action = "Enroll in VIP loyalty tier, request referrals, offer early product access"
        elif row["avg_star_rating"] < 3.0 or row["nlp_sentiment_score"] < -0.1:
            name = "At-Risk Dissatisfied Reviewers"
            action = "Trigger priority support outreach, service recovery voucher, QA review"
        elif row["recency_days"] > 110:
            name = "Dormant / Lapsed Buyers"
            action = "Launch targeted win-back email/offer with personalized picks"
        else:
            name = "Core Steady Customers"
            action = "Promote bundles and cross-sell complementary categories"

        persona_map[int(seg_id)] = {"persona": name, "recommended_action": action}
    return persona_map

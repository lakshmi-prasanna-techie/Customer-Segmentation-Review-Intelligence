from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from config import ARTIFACTS_DIR, RANDOM_STATE


def plot_validation_curve(val_df: pd.DataFrame, best_k: int, out_path: Path = ARTIFACTS_DIR / "cluster_validation.png"):
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(val_df["k"], val_df["silhouette"], marker="o", color="#1f77b4", label="Silhouette Score")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Silhouette Score", color="#1f77b4")
    ax1.axvline(best_k, color="gray", linestyle="--", alpha=0.65, label=f"Selected k={best_k}")

    ax2 = ax1.twinx()
    ax2.plot(val_df["k"], val_df["davies_bouldin"], marker="s", color="#d62728", label="Davies-Bouldin")
    ax2.set_ylabel("Davies-Bouldin Index", color="#d62728")

    plt.title("K-Means Cluster Validation Curve")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pca_with_centroids(
    X_scaled: np.ndarray,
    segments: np.ndarray,
    kmeans: KMeans,
    persona_map: dict[int, dict[str, str]],
    out_path: Path = ARTIFACTS_DIR / "pca_segments_2d.png",
):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    centroids_2d = pca.transform(kmeans.cluster_centers_)
    total_var = pca.explained_variance_ratio_.sum()

    plt.figure(figsize=(10, 6.5))
    for seg_id in sorted(np.unique(segments)):
        mask = segments == seg_id
        label = f"Seg {seg_id}: {persona_map[seg_id]['persona']} (n={mask.sum()})"
        plt.scatter(coords[mask, 0], coords[mask, 1], alpha=0.65, s=26, label=label)

    plt.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c="black",
        marker="X",
        s=180,
        edgecolors="white",
        linewidths=1.5,
        label="Cluster Centroids",
        zorder=5,
    )

    plt.title(f"Customer Segments in 2D PCA Space ({total_var:.1%} Variance Explained)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.legend(loc="best", frameon=True, fontsize=8.5)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return pca


def plot_feature_importance(
    importances: pd.Series,
    out_path: Path = ARTIFACTS_DIR / "xgb_feature_importance.png",
):
    plt.figure(figsize=(8.5, 5))
    importances.sort_values().plot(kind="barh", color="#2b5c8f")
    plt.title("XGBoost Feature Importance for Segment Classification")
    plt.xlabel("Gain Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

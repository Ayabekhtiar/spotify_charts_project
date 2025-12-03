"""
Clustering utilities for Spotify track features, built on top of PCA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

from .pca import PCAResult, _infer_feature_columns, compute_pca


@dataclass
class ClusterResult:
    labels: np.ndarray
    kmeans: KMeans
    pca_result: PCAResult


def compute_kmeans_clusters(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    n_clusters: int = 5,
    n_components: int = 2,
    standardize: bool = True,
    sample: Optional[int] = 5000,
) -> ClusterResult:
    """
    Run k-means clustering on PCA-transformed feature space.
    """
    cols = _infer_feature_columns(df, feature_cols)
    clean_df = df.dropna(subset=cols)

    if sample is not None and len(clean_df) > sample:
        clean_df = clean_df.sample(sample, random_state=42)

    pca_res = compute_pca(
        clean_df,
        feature_cols=cols,
        n_components=n_components,
        standardize=standardize,
    )

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(pca_res.components)

    return ClusterResult(labels=labels, kmeans=kmeans, pca_result=pca_res)


def plot_clustered_pca(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    n_clusters: int = 5,
    n_components: int = 2,
    standardize: bool = True,
    sample: Optional[int] = 2000,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Visualise k-means clusters in PCA space.
    """
    cols = _infer_feature_columns(df, feature_cols)
    clean_df = df.dropna(subset=cols)

    if sample is not None and len(clean_df) > sample:
        clean_df = clean_df.sample(sample, random_state=42)

    cluster_res = compute_kmeans_clusters(
        clean_df,
        feature_cols=cols,
        n_clusters=n_clusters,
        n_components=n_components,
        standardize=standardize,
        sample=None,  # already sampled above
    )

    pcs = cluster_res.pca_result.components
    clean_df = clean_df.iloc[: pcs.shape[0]].copy()
    clean_df["PC1"] = pcs[:, 0]
    if pcs.shape[1] > 1:
        clean_df["PC2"] = pcs[:, 1]
    clean_df["cluster"] = cluster_res.labels.astype(str)

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=clean_df,
        x="PC1",
        y="PC2" if "PC2" in clean_df.columns else None,
        hue="cluster",
        palette="tab10",
        alpha=0.7,
        s=40,
        ax=ax,
    )

    ax.set_title(f"k-means Clusters (k={n_clusters}) in PCA Space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    sns.despine()
    fig.tight_layout()
    return fig, ax, cluster_res



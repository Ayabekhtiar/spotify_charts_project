"""
PCA utilities for visualising Spotify track features.

The main entrypoints are:
- compute_pca: returns transformed PCA components and the fitted objects
- plot_pca_scatter: quick 2D scatter plot of PCA1 vs PCA2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


NUMERIC_DEFAULT_FEATURES: List[str] = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


@dataclass
class PCAResult:
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler
    pca: PCA


def _infer_feature_columns(df: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None) -> List[str]:
    """
    Infer a sensible set of numeric feature columns if none are provided.
    Falls back to all numeric columns if our preferred list is not present.
    """
    if feature_cols:
        return [c for c in feature_cols if c in df.columns]

    preferred = [c for c in NUMERIC_DEFAULT_FEATURES if c in df.columns]
    if preferred:
        return preferred

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return numeric_cols


def compute_pca(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    n_components: int = 2,
    standardize: bool = True,
) -> PCAResult:
    """
    Compute PCA on selected numeric columns of the dataframe.
    """
    cols = _infer_feature_columns(df, feature_cols)
    if not cols:
        raise ValueError("No numeric feature columns found for PCA.")

    X = df[cols].dropna().to_numpy()

    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = StandardScaler(with_mean=False, with_std=False)
        X_scaled = scaler.fit_transform(X)  # effectively a no-op

    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X_scaled)

    return PCAResult(
        components=components,
        explained_variance_ratio=pca.explained_variance_ratio_,
        feature_names=cols,
        scaler=scaler,
        pca=pca,
    )


def plot_pca_scatter(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    hue: Optional[str] = None,
    n_components: int = 2,
    standardize: bool = True,
    sample: Optional[int] = 2000,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Compute PCA and plot a fancy 2D scatter plot (PC1 vs PC2).
    """
    # Drop rows with missing features (and missing hue if specified)
    cols = _infer_feature_columns(df, feature_cols)
    plot_df = df.dropna(subset=cols if hue is None else list(cols) + [hue])

    if sample is not None and len(plot_df) > sample:
        plot_df = plot_df.sample(sample, random_state=42)

    pca_res = compute_pca(plot_df, feature_cols=cols, n_components=n_components, standardize=standardize)

    pcs = pca_res.components
    plot_df = plot_df.iloc[: pcs.shape[0]].copy()
    plot_df["PC1"] = pcs[:, 0]
    if pcs.shape[1] > 1:
        plot_df["PC2"] = pcs[:, 1]

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2" if "PC2" in plot_df.columns else None,
        hue=hue,
        palette="viridis",
        alpha=0.7,
        s=40,
        ax=ax,
    )

    ax.set_title("PCA of Spotify Track Features")
    ax.set_xlabel(f"PC1 ({pca_res.explained_variance_ratio[0] * 100:.1f}% var)")
    if len(pca_res.explained_variance_ratio) > 1:
        ax.set_ylabel(f"PC2 ({pca_res.explained_variance_ratio[1] * 100:.1f}% var)")

    sns.despine()
    fig.tight_layout()
    return fig, ax, pca_res



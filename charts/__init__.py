"""
Reusable chart utilities for the Spotify charts project.

This package exposes high-level functions for:
- PCA visualisation of tracks
- Clustering visualisation (e.g. k-means on PCA space)
"""

from .pca import compute_pca, plot_pca_scatter
from .clustering import compute_kmeans_clusters, plot_clustered_pca

__all__ = [
    "compute_pca",
    "plot_pca_scatter",
    "compute_kmeans_clusters",
    "plot_clustered_pca",
]


"""Data visualization package for Spotify charts analysis."""

from .data_loading import load_data, get_id_list
from .track_analysis import get_track_appearances, calculate_track_lifecycle
from .plotting import (
    plot_track_rankings,
    plot_time_series,
    plot_weekly_trends,
    plot_track_lifecycle,
    plot_audio_feature_distributions,
    plot_streams_heatmap,
    plot_streams_vs_features,
    plot_correlation_heatmap,
    plot_roc_curves,
    plot_best_confusion_matrix,
    prepare_longevity_data,
    plot_longevity_distribution,
    prepare_longevity_groups,
    plot_longevity_feature_comparison,
    plot_longevity_correlations,
    plot_longevity_scatter_plots,
    plot_longevity_clustering,
    plot_longevity_curves,
    plot_longevity_ml_results,
)
from .analysis import get_stream_correlations, print_summary_insights

__all__ = [
    # Data loading
    'load_data',
    'get_id_list',
    # Track analysis
    'get_track_appearances',
    'calculate_track_lifecycle',
    # Plotting
    'plot_track_rankings',
    'plot_time_series',
    'plot_weekly_trends',
    'plot_track_lifecycle',
    'plot_audio_feature_distributions',
    'plot_streams_heatmap',
    'plot_streams_vs_features',
    'plot_correlation_heatmap',
    'plot_roc_curves',
    'plot_best_confusion_matrix',
    # Longevity plotting
    'prepare_longevity_data',
    'plot_longevity_distribution',
    'prepare_longevity_groups',
    'plot_longevity_feature_comparison',
    'plot_longevity_correlations',
    'plot_longevity_scatter_plots',
    'plot_longevity_clustering',
    'plot_longevity_curves',
    'plot_longevity_ml_results',
    # Analysis
    'get_stream_correlations',
    'print_summary_insights'
]


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
)
from .analysis import get_stream_correlations, print_summary_insights
from .ml_features import engineer_ml_features, split_train_test
from .ml_models import (
    train_stream_prediction_models,
    plot_prediction_results,
    prepare_classification_data,
    train_classification_models,
    plot_classification_results,
    compare_models,
    plot_feature_importance,
)

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
    # Analysis
    'get_stream_correlations',
    'print_summary_insights',
    # ML Features
    'engineer_ml_features',
    'split_train_test',
    # ML Models
    'train_stream_prediction_models',
    'plot_prediction_results',
    'prepare_classification_data',
    'train_classification_models',
    'plot_classification_results',
    'compare_models',
    'plot_feature_importance',
]


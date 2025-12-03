"""Analysis functions for Spotify charts data."""

import pandas as pd


def get_stream_correlations(songs, audio_features):
    """Calculate and return stream correlations with audio features."""
    correlation_features = audio_features + ['streams']
    correlation_data = songs[correlation_features].corr()
    
    # Show top correlations with streams
    stream_correlations = correlation_data['streams'].drop('streams').sort_values(ascending=False)
    return stream_correlations


def print_summary_insights(songs, track_appearances, models_dict):
    """Print key findings and insights from the analysis."""
    regression_results = models_dict.get('regression', {})
    classification_results = models_dict.get('classification', {})
    xgboost_available = models_dict.get('xgboost_available', False)
    
    print("=" * 60)
    print("KEY FINDINGS AND INSIGHTS")
    print("=" * 60)
    
    print("\n1. DATA OVERVIEW:")
    print(f"   - Total unique tracks: {songs['track_id'].nunique():,}")
    print(f"   - Total track-week observations: {len(songs):,}")
    print(f"   - Date range: {songs['week_date'].min().date()} to {songs['week_date'].max().date()}")
    print(f"   - Average streams per track-week: {songs['streams'].mean():,.0f}")
    print(f"   - Median streams per track-week: {songs['streams'].median():,.0f}")
    
    print("\n2. TRACK PERFORMANCE:")
    top_track = track_appearances.iloc[0]
    print(f"   - Most appearing track: '{top_track['track_name']}' by {top_track['artist_names']}")
    print(f"     Appeared in {int(top_track['appearance_count'])} weeks")
    print(f"     Average streams: {top_track['avg_streams']:,.0f}")
    
    


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
    
    print("\n3. STREAM PREDICTION MODEL PERFORMANCE:")
    if regression_results:
        best_reg_model = "XGBoost" if xgboost_available and regression_results.get('xgb_r2', 0) > regression_results.get('rf_r2', 0) else "Random Forest"
        best_reg_r2 = regression_results.get('xgb_r2', regression_results.get('rf_r2', 0))
        print(f"   - Best model: {best_reg_model} (R² = {best_reg_r2:.4f})")
        print(f"   - Historical features (lag streams, rolling averages) are highly important")
        print(f"   - Audio features contribute to prediction but less than temporal patterns")
    
    print("\n4. APPEARANCE CLASSIFICATION MODEL PERFORMANCE:")
    if classification_results:
        best_clf_model = "XGBoost" if xgboost_available and classification_results.get('xgb_clf_auc', 0) > classification_results.get('rf_clf_auc', 0) else "Random Forest"
        best_clf_auc = classification_results.get('xgb_clf_auc', classification_results.get('rf_clf_auc', 0))
        print(f"   - Best model: {best_clf_model} (ROC-AUC = {best_clf_auc:.4f})")
        print(f"   - Track history and temporal features are key predictors")
        print(f"   - Model can reasonably predict if a track will appear in a given week")
    
    print("\n5. RECOMMENDATIONS:")
    print("   - For stream prediction: Use ensemble methods (Random Forest/XGBoost)")
    print("   - Focus on historical performance metrics (previous streams, trends)")
    print("   - Consider track lifecycle stage (weeks since first appearance)")
    print("   - Audio features provide context but temporal patterns dominate")
    print("   - For appearance prediction: Track history and age are most important")
    print("   - Consider seasonality and release timing in future models")
    
    print("\n6. FUTURE IMPROVEMENTS:")
    print("   - Add external features (holidays, events, marketing campaigns)")
    print("   - Implement time series models (LSTM, Prophet) for better temporal patterns")
    print("   - Include artist-level features and cross-track relationships")
    print("   - Experiment with deep learning for complex feature interactions")
    print("   - Consider multi-step ahead forecasting")


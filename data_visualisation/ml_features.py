"""Machine learning feature engineering functions."""

import pandas as pd
import numpy as np


def engineer_ml_features(songs):
    """Create all ML features and return cleaned dataset with available features list."""
    # Create a copy for feature engineering
    ml_data = songs.copy().sort_values(['track_id', 'week_date'])
    
    # Time features
    ml_data['year'] = ml_data['week_date'].dt.year
    ml_data['month'] = ml_data['week_date'].dt.month
    ml_data['week_of_year'] = ml_data['week_date'].dt.isocalendar().week
    ml_data['day_of_year'] = ml_data['week_date'].dt.dayofyear
    
    # Convert release_date to datetime if not already
    ml_data['release_date'] = pd.to_datetime(ml_data['release_date'], errors='coerce')
    
    # Days since release
    ml_data['days_since_release'] = (ml_data['week_date'] - ml_data['release_date']).dt.days
    ml_data['weeks_since_release'] = ml_data['days_since_release'] / 7
    
    # Historical features: lag features for streams
    ml_data = ml_data.sort_values(['track_id', 'week_date'])
    ml_data['prev_week_streams'] = ml_data.groupby('track_id')['streams'].shift(1)
    ml_data['prev_2week_streams'] = ml_data.groupby('track_id')['streams'].shift(2)
    ml_data['prev_3week_streams'] = ml_data.groupby('track_id')['streams'].shift(3)
    
    # Rolling averages
    ml_data['rolling_avg_3w'] = ml_data.groupby('track_id')['streams'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    ml_data['rolling_avg_5w'] = ml_data.groupby('track_id')['streams'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    # Track-level aggregations (using historical data only)
    ml_data['track_max_streams'] = ml_data.groupby('track_id')['streams'].transform(
        lambda x: x.shift(1).expanding().max()
    )
    ml_data['track_mean_streams'] = ml_data.groupby('track_id')['streams'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    # Appearance history features
    ml_data['weeks_since_first_appearance'] = (
        ml_data.groupby('track_id')['week_date'].transform(lambda x: (x - x.min()).dt.days / 7)
    )
    ml_data['total_appearances'] = ml_data.groupby('track_id').cumcount() + 1
    
    # Track age features
    track_first_appearance = ml_data.groupby('track_id')['week_date'].min().reset_index()
    track_first_appearance.columns = ['track_id', 'first_appearance_date']
    ml_data = ml_data.merge(track_first_appearance, on='track_id', how='left')
    ml_data['track_age_weeks'] = ((ml_data['week_date'] - ml_data['first_appearance_date']).dt.days / 7).round()
    
    # Fill NaN values for lag features (first appearances)
    ml_data['prev_week_streams'] = ml_data['prev_week_streams'].fillna(0)
    ml_data['prev_2week_streams'] = ml_data['prev_2week_streams'].fillna(0)
    ml_data['prev_3week_streams'] = ml_data['prev_3week_streams'].fillna(0)
    ml_data['rolling_avg_3w'] = ml_data['rolling_avg_3w'].fillna(0)
    ml_data['rolling_avg_5w'] = ml_data['rolling_avg_5w'].fillna(0)
    ml_data['track_max_streams'] = ml_data['track_max_streams'].fillna(0)
    ml_data['track_mean_streams'] = ml_data['track_mean_streams'].fillna(0)
    ml_data['days_since_release'] = ml_data['days_since_release'].fillna(ml_data['days_since_release'].median())
    ml_data['weeks_since_release'] = ml_data['weeks_since_release'].fillna(ml_data['weeks_since_release'].median())
    
    # Select features for ML
    audio_feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 
                          'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                          'key', 'mode', 'time_signature']
    
    metadata_cols = ['duration_ms', 'explicit']
    
    time_feature_cols = ['year', 'month', 'week_of_year', 'day_of_year', 
                         'days_since_release', 'weeks_since_release', 'track_age_weeks']
    
    historical_feature_cols = ['prev_week_streams', 'prev_2week_streams', 'prev_3week_streams',
                               'rolling_avg_3w', 'rolling_avg_5w', 'track_max_streams', 
                               'track_mean_streams', 'weeks_since_first_appearance', 'total_appearances']
    
    all_feature_cols = audio_feature_cols + metadata_cols + time_feature_cols + historical_feature_cols
    
    # Filter to only include columns that exist
    available_features = [col for col in all_feature_cols if col in ml_data.columns]
    
    # Remove rows with missing values in key features
    ml_data_clean = ml_data[available_features + ['streams', 'track_id', 'week_date']].dropna(
        subset=available_features + ['streams']
    )
    
    print(f"Original data shape: {songs.shape}")
    print(f"ML data shape after feature engineering: {ml_data_clean.shape}")
    print(f"Number of features: {len(available_features)}")
    print(f"\nAvailable features: {available_features}")
    
    return ml_data_clean, available_features


def split_train_test(ml_data, test_size=0.2, time_based=True):
    """Split data for train/test. If time_based=True, split by time periods."""
    if time_based:
        # Time-based train/test split: use specified percentage of weeks for training
        sorted_weeks = sorted(ml_data['week_date'].unique())
        split_idx = int(len(sorted_weeks) * (1 - test_size))
        train_weeks = sorted_weeks[:split_idx]
        test_weeks = sorted_weeks[split_idx:]
        
        train_data = ml_data[ml_data['week_date'].isin(train_weeks)]
        test_data = ml_data[ml_data['week_date'].isin(test_weeks)]
        
        print(f"Training set: {len(train_data)} samples from {len(train_weeks)} weeks")
        print(f"Test set: {len(test_data)} samples from {len(test_weeks)} weeks")
        print(f"Train date range: {train_weeks[0]} to {train_weeks[-1]}")
        print(f"Test date range: {test_weeks[0]} to {test_weeks[-1]}")
        
        return train_data, test_data, train_weeks, test_weeks
    else:
        # Random split (not recommended for time series)
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(ml_data, test_size=test_size, random_state=42)
        return train_data, test_data, None, None


def prepare_longevity_features(songs, track_appearances):
    """
    Prepare features for longevity prediction using only non-temporal characteristics.
    
    Args:
        songs: DataFrame with track-week level data
        track_appearances: DataFrame with track-level appearance statistics (from get_track_appearances)
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target vector with longevity categories (0=Short, 1=Long),
           where Short corresponds to ≤8 weeks on chart and Long to ≥9 weeks
        feature_names: List of feature names
        longevity_df: DataFrame with track-level data and longevity categories
    """
    # Merge track_appearances to get weeks_on_chart (appearance_count)
    longevity_df = track_appearances[['track_id', 'appearance_count']].copy()
    longevity_df = longevity_df.rename(columns={'appearance_count': 'weeks_on_chart'})
    
    # Create longevity categories
    # Short: ≤8 weeks, Long: ≥9 weeks
    def categorize_longevity(weeks):
        if weeks <= 8:
            return 'Short'
        else:
            return 'Long'
    
    longevity_df['longevity_category'] = longevity_df['weeks_on_chart'].apply(categorize_longevity)
    
    # Get track-level features (one row per track)
    # Use first appearance or mean if multiple values exist
    track_features = songs.groupby('track_id').agg({
        # Audio features - use mean if multiple values
        'danceability': 'mean',
        'energy': 'mean',
        'acousticness': 'mean',
        'instrumentalness': 'mean',
        'valence': 'mean',
        'speechiness': 'mean',
        'tempo': 'mean',
        # Metadata - use first or mode
        'duration_ms': 'first',
        'explicit': 'first',
        # Artist info - use first
        'id_artists': 'first',
    }).reset_index()
    
    # Merge with longevity categories
    longevity_df = longevity_df.merge(track_features, on='track_id', how='inner')
    
    # Extract artist count from id_artists
    def count_artists(id_artists):
        """Count number of artists from id_artists column."""
        if pd.isna(id_artists):
            return 1  # Default to 1 if missing
        try:
            # Handle if it's a string representation of a list
            if isinstance(id_artists, str):
                # Try to evaluate as Python literal
                import ast
                try:
                    parsed = ast.literal_eval(id_artists)
                    if isinstance(parsed, (list, tuple)):
                        return len(parsed)
                    else:
                        return 1
                except:
                    # If parsing fails, treat as single artist
                    return 1
            elif isinstance(id_artists, (list, tuple)):
                return len(id_artists)
            else:
                return 1
        except:
            return 1
    
    longevity_df['num_artists'] = longevity_df['id_artists'].apply(count_artists)
    
    # Select features for ML (non-temporal only)
    audio_features = ['danceability', 'energy', 'acousticness', 'instrumentalness', 
                     'valence', 'speechiness', 'tempo']
    metadata_features = ['duration_ms', 'explicit']
    artist_features = ['num_artists']
    
    all_feature_cols = audio_features + metadata_features + artist_features
    
    # Filter to only include columns that exist
    available_features = [col for col in all_feature_cols if col in longevity_df.columns]
    
    # Prepare feature matrix X
    X = longevity_df[available_features].copy()
    
    # Handle missing values - drop rows with any missing values
    missing_mask = X.isna().any(axis=1)
    if missing_mask.sum() > 0:
        print(f"Dropping {missing_mask.sum()} rows with missing values")
        X = X.dropna()
        longevity_df = longevity_df.loc[X.index]
    
    # Prepare target vector y (convert categories to numeric)
    category_map = {'Short': 0, 'Long': 1}
    y = longevity_df['longevity_category'].map(category_map).values
    
    # Remove rows where y is NaN (shouldn't happen, but safety check)
    valid_mask = ~pd.isna(y)
    X = X.loc[valid_mask]
    y = y[valid_mask]
    longevity_df = longevity_df.loc[valid_mask]
    
    print(f"\nLongevity Prediction Data Preparation:")
    print(f"Total tracks: {len(X)}")
    print(f"Features: {available_features}")
    print(f"\nLongevity category distribution:")
    print(longevity_df['longevity_category'].value_counts().sort_index())
    print(f"\nFeature matrix shape: {X.shape}")
    
    return X.values, y, available_features, longevity_df


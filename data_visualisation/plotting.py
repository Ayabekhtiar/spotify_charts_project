"""Plotting functions for visualizing Spotify charts data."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


def plot_track_rankings(track_appearances, top_n=20):
    """Plot top N most appearing tracks."""
    top_tracks = track_appearances.head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(len(top_tracks)), top_tracks['appearance_count'], color='steelblue')
    ax.set_yticks(range(len(top_tracks)))
    ax.set_yticklabels([f"{row['track_name']} - {row['artist_names']}" 
                        for _, row in top_tracks.iterrows()], fontsize=9)
    ax.set_xlabel('Number of Weeks Appeared', fontsize=12)
    ax.set_title(f'Top {top_n} Most Frequently Appearing Tracks', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_tracks.iterrows()):
        ax.text(row['appearance_count'] + 0.5, i, f"{int(row['appearance_count'])}", 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    return top_tracks


def plot_time_series(songs, track_appearances, top_n=10, by='total_streams'):
    """Plot time series of streams for top tracks."""
    top_tracks = track_appearances.sort_values(by, ascending=False).head(top_n)['track_id'].tolist()
    top_tracks_data = songs[songs['track_id'].isin(top_tracks)].copy()
    top_tracks_data = top_tracks_data.sort_values(['track_id', 'week_date'])
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for track_id in top_tracks:
        track_data = top_tracks_data[top_tracks_data['track_id'] == track_id]
        track_name = track_data['track_name'].iloc[0]
        artist = track_data['artist_names'].iloc[0]
        ax.plot(track_data['week_date'], track_data['streams'], 
                label=f"{track_name} - {artist}", linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Week Date', fontsize=12)
    ax.set_ylabel('Streams', fontsize=12)
    ax.set_title(f'Streams Over Time for Top {top_n} Tracks (by {by.replace("_", " ").title()})', 
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_weekly_trends(songs):
    """Plot aggregate weekly trends."""
    weekly_stats = songs.groupby('week_date').agg({
        'streams': ['sum', 'mean', 'count']
    }).reset_index()
    weekly_stats.columns = ['week_date', 'total_streams', 'avg_streams', 'num_tracks']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    axes[0].plot(weekly_stats['week_date'], weekly_stats['total_streams'], 
                 linewidth=2, color='steelblue', marker='o', markersize=4)
    axes[0].set_xlabel('Week Date', fontsize=12)
    axes[0].set_ylabel('Total Streams', fontsize=12)
    axes[0].set_title('Total Streams Per Week', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].plot(weekly_stats['week_date'], weekly_stats['avg_streams'], 
                linewidth=2, color='coral', marker='o', markersize=4)
    axes[1].set_xlabel('Week Date', fontsize=12)
    axes[1].set_ylabel('Average Streams', fontsize=12)
    axes[1].set_title('Average Streams Per Track Per Week', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    return weekly_stats


def plot_track_lifecycle(track_lifecycle):
    """Visualize track lifecycle patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Distribution of weeks active
    axes[0, 0].hist(track_lifecycle['weeks_active'], bins=50, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Weeks Active', fontsize=11)
    axes[0, 0].set_ylabel('Number of Tracks', fontsize=11)
    axes[0, 0].set_title('Distribution of Track Lifespan (Weeks Active)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Weeks to peak
    axes[0, 1].hist(track_lifecycle['weeks_to_peak'], bins=50, color='coral', edgecolor='black')
    axes[0, 1].set_xlabel('Weeks to Peak', fontsize=11)
    axes[0, 1].set_ylabel('Number of Tracks', fontsize=11)
    axes[0, 1].set_title('Distribution of Weeks to Peak Streams', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Peak streams vs weeks active
    axes[1, 0].scatter(track_lifecycle['weeks_active'], track_lifecycle['peak_streams'], 
                      alpha=0.3, s=20, color='green')
    axes[1, 0].set_xlabel('Weeks Active', fontsize=11)
    axes[1, 0].set_ylabel('Peak Streams', fontsize=11)
    axes[1, 0].set_title('Peak Streams vs Track Lifespan', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Average streams vs weeks active
    axes[1, 1].scatter(track_lifecycle['weeks_active'], track_lifecycle['avg_streams'], 
                       alpha=0.3, s=20, color='purple')
    axes[1, 1].set_xlabel('Weeks Active', fontsize=11)
    axes[1, 1].set_ylabel('Average Streams', fontsize=11)
    axes[1, 1].set_title('Average Streams vs Track Lifespan', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.suptitle('Track Lifecycle Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\nTrack lifecycle statistics:")
    print(f"Average weeks active: {track_lifecycle['weeks_active'].mean():.1f}")
    print(f"Median weeks active: {track_lifecycle['weeks_active'].median():.1f}")
    print(f"Average weeks to peak: {track_lifecycle['weeks_to_peak'].mean():.1f}")
    print(f"Median weeks to peak: {track_lifecycle['weeks_to_peak'].median():.1f}")


def plot_audio_feature_distributions(songs, numeric_cols):
    """Plot basic distribution plots for key audio features."""
    available = [c for c in numeric_cols if c in songs.columns]
    
    songs[available].hist(figsize=(14, 10), bins=30)
    plt.suptitle("Distributions of core audio features", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_streams_heatmap(songs, track_appearances, top_n=20):
    """Plot heatmap of streams by track and week for top N tracks."""
    top_tracks = track_appearances.head(top_n)['track_id'].tolist()
    heatmap_data = songs[songs['track_id'].isin(top_tracks)].copy()
    
    # Create pivot table for heatmap
    pivot_data = heatmap_data.pivot_table(
        values='streams',
        index='track_id',
        columns='week_date',
        aggfunc='sum',
        fill_value=0
    )
    
    # Get track names for better labels
    track_labels = heatmap_data.groupby('track_id').agg({
        'track_name': 'first',
        'artist_names': 'first'
    }).reset_index()
    track_labels['label'] = track_labels['track_name'] + ' - ' + track_labels['artist_names']
    track_labels = track_labels.set_index('track_id')['label']
    
    # Reindex pivot to match labels
    pivot_data = pivot_data.reindex(track_labels.index)
    pivot_data.index = [track_labels[idx][:50] + '...' if len(track_labels[idx]) > 50 else track_labels[idx] 
                        for idx in pivot_data.index]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(pivot_data, cmap='YlOrRd', cbar_kws={'label': 'Streams'}, 
                ax=ax, linewidths=0.1, linecolor='gray')
    ax.set_title('Streams Heatmap: Top 20 Tracks by Appearance Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Week Date', fontsize=12)
    ax.set_ylabel('Track', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_streams_vs_features(songs, audio_features):
    """Plot scatter plots of streams vs audio features."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, feature in enumerate(audio_features):
        if feature in songs.columns:
            ax = axes[idx]
            # Sample data for faster plotting if dataset is large
            plot_data = songs.sample(min(50000, len(songs)), random_state=42) if len(songs) > 50000 else songs
            ax.scatter(plot_data[feature], plot_data['streams'], alpha=0.3, s=10)
            ax.set_xlabel(feature, fontsize=11)
            ax.set_ylabel('Streams', fontsize=11)
            ax.set_title(f'Streams vs {feature.capitalize()}', fontsize=12)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Stream Distribution by Audio Features', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(songs, audio_features):
    """Plot correlation heatmap between audio features and streams."""
    correlation_features = audio_features + ['streams']
    correlation_data = songs[correlation_features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Heatmap: Audio Features and Streams', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_roc_curves(y_test, y_pred_proba_dict, model_names):
    """Plot ROC curves for classification models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, y_pred_proba in y_pred_proba_dict.items():
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: Appearance Classification Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_best_confusion_matrix(y_test, y_pred, model_name):
    """Plot confusion matrix for the best classification model."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(f"Confusion Matrix: {model_name} Appearance Classifier", fontweight='bold')
    plt.tight_layout()
    plt.show()


#
# Longevity Analysis Visualization Functions
#

def prepare_longevity_data(songs, track_appearances):
    """
    Prepare longevity analysis data by merging track appearances with audio features.
    
    Returns:
        DataFrame with track-level data including weeks_on_chart and audio features
    """
    longevity_data = track_appearances[['track_id', 'track_name', 'artist_names', 'appearance_count']].copy()
    
    # Add audio features
    track_features = songs.groupby('track_id').first()[['danceability', 'energy', 'acousticness', 
                                                         'instrumentalness', 'valence', 'speechiness', 
                                                         'tempo', 'duration_ms']].reset_index()
    longevity_data = longevity_data.merge(track_features, on='track_id', how='left')
    
    # Rename for clarity
    longevity_data = longevity_data.rename(columns={'appearance_count': 'weeks_on_chart'})
    
    return longevity_data


def plot_longevity_distribution(longevity_data):
    """Plot distribution of weeks on chart (histogram and KDE)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(longevity_data['weeks_on_chart'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Weeks on Chart', fontsize=12)
    axes[0].set_ylabel('Number of Tracks', fontsize=12)
    axes[0].set_title('Distribution of Weeks on Chart (Histogram)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # KDE
    sns.kdeplot(data=longevity_data, x='weeks_on_chart', ax=axes[1], fill=True, color='coral')
    axes[1].set_xlabel('Weeks on Chart', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Distribution of Weeks on Chart (KDE)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    skewness = longevity_data['weeks_on_chart'].skew()
    print(f"\nDistribution is {'right-skewed' if skewness > 0 else 'left-skewed'} (Skewness: {skewness:.2f})")
    return skewness


def prepare_longevity_groups(longevity_data, short_threshold=8, long_threshold=15):
    """
    Prepare groups for comparing short-lived vs long-lasting hits.
    
    Returns:
        tuple: (short_lived_df, long_lasting_df, comparison_df)
    """
    short_lived = longevity_data[longevity_data['weeks_on_chart'] <= short_threshold].copy()
    long_lasting = longevity_data[longevity_data['weeks_on_chart'] >= long_threshold].copy()
    
    short_lived['group'] = f'Short-lived (≤{short_threshold} weeks)'
    long_lasting['group'] = f'Long-lasting (≥{long_threshold} weeks)'
    
    comparison_data = pd.concat([short_lived, long_lasting], ignore_index=True)
    
    return short_lived, long_lasting, comparison_data


def plot_longevity_feature_comparison(comparison_data, short_lived, long_lasting, 
                                     features=None):
    """Plot boxplots comparing audio features between short-lived and long-lasting hits."""
    if features is None:
        features = ['danceability', 'energy', 'acousticness', 'instrumentalness', 
                   'valence', 'speechiness', 'tempo', 'duration_ms']
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        if feature in comparison_data.columns:
            sns.boxplot(data=comparison_data, x='group', y=feature, ax=axes[idx], 
                       palette=['lightcoral', 'lightblue'])
            axes[idx].set_title(f'{feature.capitalize()}', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for feature in features:
        if feature in short_lived.columns and feature in long_lasting.columns:
            short_mean = short_lived[feature].mean()
            long_mean = long_lasting[feature].mean()
            diff = long_mean - short_mean
            print(f"{feature:20s}: Short={short_mean:7.3f}, Long={long_mean:7.3f}, Diff={diff:7.3f}")


def plot_longevity_correlations(longevity_data, features=None):
    """
    Plot correlation heatmap between audio features and weeks on chart.
    
    Returns:
        Series with correlations sorted by absolute value
    """
    if features is None:
        features = ['danceability', 'energy', 'acousticness', 'instrumentalness', 
                   'valence', 'speechiness', 'tempo', 'duration_ms']
    
    corr_features = ['weeks_on_chart'] + features
    df_corr = longevity_data[corr_features].corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(df_corr, dtype=bool))  # Mask upper triangle
    sns.heatmap(df_corr, mask=mask, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap: Audio Features vs Weeks on Chart', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Show correlations with weeks_on_chart
    longevity_correlations = df_corr['weeks_on_chart'].drop('weeks_on_chart').sort_values(ascending=False)
    print("\nCorrelations with Weeks on Chart:")
    print(longevity_correlations)
    
    return longevity_correlations


def plot_longevity_scatter_plots(longevity_data, top_features, n_features=3):
    """Plot scatter plots with regression lines for top correlated features."""
    top_features = top_features.abs().nlargest(n_features).index.tolist()
    
    fig, axes = plt.subplots(1, n_features, figsize=(18, 5))
    
    for idx, feature in enumerate(top_features):
        # Scatter plot
        axes[idx].scatter(longevity_data[feature], longevity_data['weeks_on_chart'], 
                         alpha=0.5, s=20, color='steelblue')
        
        # Linear regression line
        z = np.polyfit(longevity_data[feature].dropna(), 
                      longevity_data.loc[longevity_data[feature].notna(), 'weeks_on_chart'], 1)
        p = np.poly1d(z)
        axes[idx].plot(longevity_data[feature].sort_values(), 
                      p(longevity_data[feature].sort_values()), 
                      "r--", alpha=0.8, linewidth=2, label='Linear fit')
        
        # LOWESS smoothing (if scipy available)
        try:
            sorted_idx = longevity_data[feature].sort_values().index
            sorted_feature = longevity_data.loc[sorted_idx, feature]
            sorted_weeks = longevity_data.loc[sorted_idx, 'weeks_on_chart']
            
            # Simple moving average as LOWESS approximation
            window = max(10, len(sorted_feature) // 20)
            smoothed = sorted_weeks.rolling(window=window, center=True).mean()
            axes[idx].plot(sorted_feature, smoothed, 
                          "g-", alpha=0.6, linewidth=2, label='Smoothed trend')
        except:
            pass
        
        axes[idx].set_xlabel(feature.capitalize(), fontsize=11)
        axes[idx].set_ylabel('Weeks on Chart', fontsize=11)
        axes[idx].set_title(f'{feature.capitalize()} vs Weeks on Chart', fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop {n_features} features analyzed: {', '.join(top_features)}")


def plot_longevity_clustering(longevity_data, cluster_features=None, n_clusters=4):
    """
    Perform K-means clustering on audio features and visualize with PCA.
    
    Returns:
        DataFrame with cluster assignments and PCA coordinates
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    if cluster_features is None:
        cluster_features = ['danceability', 'energy', 'acousticness', 'instrumentalness', 
                           'valence', 'speechiness', 'tempo']
    
    # Get data with no missing values
    cluster_data = longevity_data[cluster_features + ['weeks_on_chart', 'track_id']].dropna()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_data[cluster_features])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    cluster_data = cluster_data.copy()
    cluster_data['cluster'] = cluster_labels
    
    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    cluster_data['pca_1'] = X_pca[:, 0]
    cluster_data['pca_2'] = X_pca[:, 1]
    
    # Plot clusters
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA scatter plot
    scatter = axes[0].scatter(cluster_data['pca_1'], cluster_data['pca_2'], 
                             c=cluster_data['cluster'], cmap='viridis', 
                             s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
    axes[0].set_title('PCA 2D Cluster Plot', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Cluster')
    
    # Average weeks on chart by cluster
    cluster_weeks = cluster_data.groupby('cluster')['weeks_on_chart'].mean().sort_values(ascending=False)
    axes[1].bar(range(len(cluster_weeks)), cluster_weeks.values, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Cluster', fontsize=11)
    axes[1].set_ylabel('Average Weeks on Chart', fontsize=11)
    axes[1].set_title('Average Weeks on Chart by Cluster', fontsize=13, fontweight='bold')
    axes[1].set_xticks(range(len(cluster_weeks)))
    axes[1].set_xticklabels([f'Cluster {i}' for i in cluster_weeks.index])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print cluster characteristics
    print("\nCluster Characteristics:")
    for cluster_id in sorted(cluster_data['cluster'].unique()):
        cluster_subset = cluster_data[cluster_data['cluster'] == cluster_id]
        avg_weeks = cluster_subset['weeks_on_chart'].mean()
        print(f"\nCluster {cluster_id}:")
        print(f"  Average weeks on chart: {avg_weeks:.1f}")
        print(f"  Number of tracks: {len(cluster_subset)}")
        print(f"  Feature means:")
        for feat in cluster_features:
            print(f"    {feat}: {cluster_subset[feat].mean():.3f}")
    
    return cluster_data


def plot_longevity_curves(songs, track_features, feature='danceability', n_quantiles=3):
    """
    Plot longevity curves showing average chart rank over time by feature quantiles.
    
    Args:
        songs: DataFrame with track-week level data
        track_features: DataFrame with track-level features
        feature: Feature name to split by quantiles
        n_quantiles: Number of quantiles to create
    """
    # Calculate rank for each track in each week
    songs_with_rank = songs.copy()
    songs_with_rank['rank'] = songs_with_rank.groupby('week_date')['streams'].rank(ascending=False)
    
    # Merge with track features
    songs_with_rank = songs_with_rank.merge(
        track_features[['track_id', feature]], 
        on='track_id', 
        how='left',
        suffixes=('', '_track')
    )
    
    # Use track-level feature value
    songs_with_rank[f'{feature}_track'] = songs_with_rank.groupby('track_id')[f'{feature}_track'].transform('first')
    
    # Split by feature quantiles
    songs_with_rank[f'{feature}_q'] = pd.qcut(
        songs_with_rank[f'{feature}_track'], 
        q=n_quantiles, 
        labels=['Low', 'Medium', 'High']
    )
    
    # Calculate weeks since first appearance for each track
    track_first_week = songs_with_rank.groupby('track_id')['week_date'].min().reset_index()
    track_first_week.columns = ['track_id', 'first_week']
    songs_with_rank = songs_with_rank.merge(track_first_week, on='track_id', how='left')
    songs_with_rank['weeks_since_first'] = (
        (songs_with_rank['week_date'] - songs_with_rank['first_week']).dt.days / 7
    ).round().astype(int)
    
    # Calculate average rank by weeks since first appearance and quantile
    longevity_curves = songs_with_rank.groupby([f'{feature}_q', 'weeks_since_first'])['rank'].mean().reset_index()
    
    # Plot longevity curves
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for q in ['Low', 'Medium', 'High']:
        q_data = longevity_curves[longevity_curves[f'{feature}_q'] == q]
        ax.plot(q_data['weeks_since_first'], q_data['rank'], 
                marker='o', label=f'{q} {feature.capitalize()}', linewidth=2, markersize=4)
    
    ax.set_xlabel('Weeks Since First Appearance', fontsize=11)
    ax.set_ylabel('Average Chart Rank', fontsize=11)
    ax.set_title(f'Longevity Curves: Average Rank Over Time by {feature.capitalize()}', 
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Lower rank = better position
    
    plt.tight_layout()
    plt.show()
    
    return longevity_curves


def plot_longevity_ml_results(longevity_results, y_test, feature_names):
    """
    Plot confusion matrices and feature importance for longevity ML models.
    
    Returns:
        DataFrame with feature importance
    """
    xgboost_available = longevity_results['xgboost_available']
    fig, axes = plt.subplots(1, 3 if xgboost_available else 2, figsize=(18, 5))
    if not xgboost_available:
        axes = [axes[0], axes[1]]
    
    class_names = longevity_results['class_names']
    predictions = longevity_results['predictions']
    results = longevity_results['results']
    
    # Logistic Regression
    if 'lr' in predictions and predictions['lr'] is not None:
        cm_lr = confusion_matrix(y_test, predictions['lr'])
        lr_results = results.get('lr', {})
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                   xticklabels=class_names, yticklabels=class_names, cbar=False)
        axes[0].set_xlabel('Predicted', fontsize=11)
        axes[0].set_ylabel('Actual', fontsize=11)
        acc = lr_results.get('acc', 0)
        f1 = lr_results.get('f1', 0)
        axes[0].set_title(f'Logistic Regression\n(Acc={acc:.3f}, F1={f1:.3f})', fontsize=12)
    
    # Random Forest
    if 'rf' in predictions and predictions['rf'] is not None:
        cm_rf = confusion_matrix(y_test, predictions['rf'])
        rf_results = results.get('rf', {})
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                   xticklabels=class_names, yticklabels=class_names, cbar=False)
        axes[1].set_xlabel('Predicted', fontsize=11)
        axes[1].set_ylabel('Actual', fontsize=11)
        acc = rf_results.get('acc', 0)
        f1 = rf_results.get('f1', 0)
        axes[1].set_title(f'Random Forest\n(Acc={acc:.3f}, F1={f1:.3f})', fontsize=12)
    
    # XGBoost
    if xgboost_available and 'xgb' in predictions and predictions['xgb'] is not None:
        cm_xgb = confusion_matrix(y_test, predictions['xgb'])
        xgb_results = results.get('xgb', {})
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Purples', ax=axes[2],
                   xticklabels=class_names, yticklabels=class_names, cbar=False)
        axes[2].set_xlabel('Predicted', fontsize=11)
        axes[2].set_ylabel('Actual', fontsize=11)
        acc = xgb_results.get('acc', 0)
        f1 = xgb_results.get('f1', 0)
        axes[2].set_title(f'XGBoost\n(Acc={acc:.3f}, F1={f1:.3f})', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Plot feature importance (using Random Forest)
    rf_model = longevity_results['models']['rf']
    feature_importance = None
    if rf_model is not None:
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(feature_importance)), feature_importance['importance'], color='steelblue')
        ax.set_yticks(range(len(feature_importance)))
        ax.set_yticklabels(feature_importance['feature'], fontsize=10)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Feature Importance for Longevity Prediction (Random Forest)', 
                    fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
        
        print("\nTop Features for Longevity Prediction:")
        print(feature_importance.to_string(index=False))
    
    return feature_importance


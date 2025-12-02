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


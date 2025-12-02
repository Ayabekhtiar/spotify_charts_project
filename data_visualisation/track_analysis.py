"""Track analysis functions for calculating statistics and lifecycle metrics."""

import pandas as pd


def get_track_appearances(songs):
    """Calculate track appearance statistics."""
    track_appearances = songs.groupby('track_id').agg({
        'track_name': 'first',
        'artist_names': 'first',
        'week_date': 'count',
        'streams': ['sum', 'mean', 'max']
    }).reset_index()
    
    track_appearances.columns = ['track_id', 'track_name', 'artist_names', 
                                  'appearance_count', 'total_streams', 'avg_streams', 'max_streams']
    track_appearances = track_appearances.sort_values('appearance_count', ascending=False)
    return track_appearances


def calculate_track_lifecycle(songs):
    """Calculate track lifecycle metrics."""
    # Basic lifecycle stats
    track_lifecycle = songs.groupby('track_id').agg({
        'week_date': ['min', 'max'],
        'streams': ['max', 'mean'],
        'track_name': 'first',
        'artist_names': 'first'
    }).reset_index()
    
    track_lifecycle.columns = ['track_id', 'first_appearance', 'last_appearance', 
                               'max_streams', 'avg_streams', 'track_name', 'artist_names']
    
    # Calculate track age
    track_lifecycle['weeks_active'] = (
        (track_lifecycle['last_appearance'] - track_lifecycle['first_appearance']).dt.days / 7
    ).round().astype(int)
    
    # Find peak week for each track
    peak_weeks = songs.loc[songs.groupby('track_id')['streams'].idxmax()][['track_id', 'week_date', 'streams']]
    peak_weeks = peak_weeks.rename(columns={'week_date': 'peak_week', 'streams': 'peak_streams_value'})
    
    # Merge peak week info (avoiding column name conflicts)
    track_lifecycle = track_lifecycle.merge(peak_weeks, on='track_id', how='left')
    
    # Calculate weeks to peak
    track_lifecycle['weeks_to_peak'] = (
        (track_lifecycle['peak_week'] - track_lifecycle['first_appearance']).dt.days / 7
    ).round().astype(int)
    
    # Use peak_streams_value as peak_streams
    track_lifecycle['peak_streams'] = track_lifecycle['peak_streams_value']
    
    return track_lifecycle


"""Data loading functions for Spotify charts visualization."""

import pandas as pd


def load_data(data_path="data/gold/"):
    """Load and preprocess the songs dataset."""
    songs = pd.read_parquet(data_path + "songs_with_features.parquet")
    songs['week_date'] = pd.to_datetime(songs['week_date'])
    return songs


def get_id_list(songs):
    """Group by artist_names and track_name and list track ids.
    
    Returns:
        tuple: (number of songs without track_id, number of songs with multiple track_ids)
    """
    # group by artist_names and track_name and list track ids
    id_list = songs.groupby(["artist_names", "track_name"])["track_id"].apply(set).reset_index()

    id_list["length"] = id_list["track_id"].apply(len)
    
    return id_list[id_list["length"] == 0].size, id_list[id_list["length"] > 1].size


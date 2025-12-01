import pandas as pd

import os

def merge_data():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    tracks_path = os.path.join(base_dir, "data", "raw", "tracks.csv")
    songs_path = os.path.join(base_dir, "data", "processed", "combined_songs.csv")
    output_path = os.path.join(base_dir, "data", "processed", "songs_with_features.csv")

    if not os.path.exists(tracks_path) or not os.path.exists(songs_path):
        print("Input files not found.")
        return

    # Load your files
    tracks = pd.read_csv(tracks_path)
    songs = pd.read_csv(songs_path)

    # Rename if necessary so both files have the same ID column name
    songs.rename(columns={"track_id": "id"}, inplace=True)

    # Merge on the track ID
    merged = songs.merge(tracks, on="id", how="left")

    # Save
    merged.to_csv(output_path, index=False)
    print(f"✓ Merged data saved to: {output_path}")

if __name__ == "__main__":
    merge_data()

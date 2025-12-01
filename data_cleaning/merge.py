import pandas as pd

import os

def merge_data(tracks_path, songs_path, output_path):
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

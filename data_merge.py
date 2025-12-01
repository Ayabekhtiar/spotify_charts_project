import pandas as pd

# Load your files
tracks = pd.read_csv("tracks.csv")
songs = pd.read_csv("combined_songs.csv")

# Rename if necessary so both files have the same ID column name
songs.rename(columns={"track_id": "id"}, inplace=True)

# Merge on the track ID
merged = songs.merge(tracks, on="id", how="left")

# Save
merged.to_csv("songs_with_features.csv", index=False)

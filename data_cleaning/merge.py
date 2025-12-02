import pandas as pd

import os

def merge_data(tracks_path, songs_path, output_path):
    if not os.path.exists(tracks_path) or not os.path.exists(songs_path):
        print("Input files not found.")
        return

    # Load your files
    tracks = pd.read_csv(tracks_path)
    songs = pd.read_csv(songs_path)
    
    #Ensure there will not be any duplicates in tracks "id" dataframe
    tracks = tracks.drop_duplicates(subset=["id"])
    
    # remove from tracks the columns that are already in songs except ""
    for c in songs.columns:
        if c in tracks.columns and c not in ["id", "track_id"]:
            tracks.drop(columns=[c], inplace=True)
    
    # Rename if necessary so both files have the same ID column name
    tracks.rename(columns={"id": "track_id"}, inplace=True)
    
    


    # Merge on the track ID
    merged = songs.merge(tracks, on="track_id", how="left")

    # Save
    merged.to_csv(output_path, index=False)
    print(f"✓ Merged data saved to: {output_path}")

if __name__ == "__main__":
    merge_data()

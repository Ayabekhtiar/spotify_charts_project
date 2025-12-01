import pandas as pd
import os

data_folder = "data"

# Load and combine all CSVs
dfs = [pd.read_csv(os.path.join(data_folder, f))
       for f in os.listdir(data_folder) if f.endswith(".csv")]

combined_df = pd.concat(dfs, ignore_index=True)

# Remove duplicates (choose proper subset if needed)
unique_df = combined_df.drop_duplicates()
cols_to_drop = ["rank", "peak_rank", "previous_rank", "weeks_on_chart", "streams"]
unique_df = unique_df.drop(columns=cols_to_drop)
unique_df["uri"] = unique_df["uri"].str.replace("spotify:track:", "", regex=False)
unique_df = unique_df.rename(columns={"uri": "track_id"})


# Save to CSV
output_path = "combined_songs.csv"
unique_df.to_csv(output_path, index=False)

print(f"✓ File saved as: {output_path}")




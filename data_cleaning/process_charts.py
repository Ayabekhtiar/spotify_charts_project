import pandas as pd
import os

def process_all_charts():
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

    # Load and combine all CSVs
    dfs = [pd.read_csv(os.path.join(data_folder, f))
           for f in os.listdir(data_folder) if f.endswith(".csv")]

    if not dfs:
        print("No CSV files found in data/raw")
        return

    combined_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (choose proper subset if needed)
    unique_df = combined_df.drop_duplicates()
    cols_to_drop = ["rank", "peak_rank", "previous_rank", "weeks_on_chart", "streams"]
    unique_df = unique_df.drop(columns=cols_to_drop)
    unique_df["uri"] = unique_df["uri"].str.replace("spotify:track:", "", regex=False)
    unique_df = unique_df.rename(columns={"uri": "track_id"})


    # Save to CSV
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "combined_songs.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    unique_df.to_csv(output_path, index=False)

    print(f"✓ File saved as: {output_path}")

if __name__ == "__main__":
    process_all_charts()




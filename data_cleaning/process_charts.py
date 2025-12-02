import pandas as pd
import os

def process_all_charts(data_folder, output_path):
    
    dfs_names = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    dfs = [pd.read_csv(os.path.join(data_folder, f)) for f in dfs_names]
    
    # In each csv add a column "week_date" with the name of the csv as value for all its rows (eg. "2016-12-29" for "regional-global-weekly-2016-12-29.csv")
    for df_name, df in zip(dfs_names, dfs):
        df["week_date"] = df_name.replace("regional-global-weekly-", "").replace(".csv", "")

    if not dfs:
        print("No CSV files found in", data_folder)
        return

    combined_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (choose proper subset if needed)
    unique_df = combined_df.drop_duplicates()
    cols_to_drop = ["rank", "peak_rank", "previous_rank", "weeks_on_chart", "streams"]
    unique_df = unique_df.drop(columns=cols_to_drop)
    unique_df["uri"] = unique_df["uri"].str.replace("spotify:track:", "", regex=False)
    unique_df = unique_df.rename(columns={"uri": "track_id"})


    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    unique_df.to_csv(output_path, index=False)

    print(f"✓ File saved as: {output_path}")

if __name__ == "__main__":
    process_all_charts()




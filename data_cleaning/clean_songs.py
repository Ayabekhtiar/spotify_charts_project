import os
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm


#
# Verification helpers for weekly charts
#

def list_weekly_chart_files(directory: str) -> List[str]:
    """List all CSV filenames in the given directory."""
    return [f for f in os.listdir(directory) if f.endswith(".csv")]


def extract_dates_from_filenames(files: List[str]) -> List[pd.Timestamp]:
    """Extract dates from weekly chart CSV filenames."""
    dates: List[pd.Timestamp] = []
    for f in files:
        try:
            date_str = f.replace("regional-global-weekly-", "").replace(".csv", "")
            dates.append(pd.to_datetime(date_str))
        except Exception:
            # Skip invalid file names
            continue
    return sorted(dates)


def get_expected_weekly_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Create expected weekly date range from start to end (Thursday frequency)."""
    return pd.date_range(start=start, end=end, freq="W-THU")


def summarize_weekly_date_gaps(dates: List[pd.Timestamp]) -> None:
    """Print summary of missing and extra dates versus expectation."""
    if not dates:
        print("No data files found.")
        return

    start, end = dates[0], dates[-1]
    expected = get_expected_weekly_dates(start, end)
    missing = expected.difference(dates)
    extra = set(dates) - set(expected)

    print("First date:", start.date())
    print("Last date:", end.date())
    print("Total files:", len(dates))
    print("Expected weeks:", len(expected))

    print("\nMissing weeks:")
    for m in missing:
        print(m.date())

    print("\nUnexpected extra dates:")
    for e in sorted(extra):
        print(e.date())


#
# Song de-duplication / ID correction helpers
#

def create_song_dict(
    songs: pd.DataFrame,
    column_to_datetime: str = "release_date",
    *,
    id_col1: str = "artist_names",
    id_col2: str = "track_name",
    saved_col1: str = "track_id",
    saved_col2: str = "source",
) -> Dict[Tuple[Any, Any], List[Any]]:
    """
    Build a dictionary keyed by (artist_names, track_name) that stores:
        [track_id, source, parsed_release_date]
    preferring the *earliest* release_date for each key.
    """
    song_dict: Dict[Tuple[Any, Any], List[Any]] = {}

    songs = songs.copy()
    songs[column_to_datetime] = pd.to_datetime(songs[column_to_datetime], errors="coerce")

    for _, row in tqdm(songs.iterrows(), total=len(songs), desc="Processing rows"):
        key = (row[id_col1], row[id_col2])
        value = [row[saved_col1], row[saved_col2], row[column_to_datetime]]

        if key in song_dict:
            curr_dt = row[column_to_datetime]
            saved_dt = song_dict[key][2]
            try:
                if pd.notnull(curr_dt) and pd.notnull(saved_dt):
                    if curr_dt < saved_dt:
                        song_dict[key] = value
                elif pd.notnull(curr_dt) and pd.isnull(saved_dt):
                    song_dict[key] = value
            except Exception:
                # If datetimes are unexpected types, just keep existing
                pass
        else:
            song_dict[key] = value

    return song_dict


def update_song_rows_with_dict(songs: pd.DataFrame, song_dict: Dict[Tuple[Any, Any], List[Any]]) -> pd.DataFrame:
    """
    For each row in songs, if (artist_names, track_name) exists in song_dict,
    potentially update track_id, source and release_date when dates differ.
    """
    songs_updated = songs.copy()
    updated_count = 0

    for idx, row in tqdm(songs_updated.iterrows(), total=len(songs_updated), desc="Updating songs"):
        key = (row["artist_names"], row["track_name"])
        dict_vals = song_dict.get(key)
        if dict_vals is not None:
            dict_track_id, dict_source, dict_release_date = dict_vals
            row_release_date = row["release_date"]

            dict_release_date_ts = pd.to_datetime(dict_release_date, errors="coerce")
            row_release_date_ts = pd.to_datetime(row_release_date, errors="coerce")

            if pd.isnull(dict_release_date_ts) and pd.isnull(row_release_date_ts):
                continue

            if not pd.isnull(dict_release_date_ts) and not pd.isnull(row_release_date_ts):
                if dict_release_date_ts != row_release_date_ts:
                    songs_updated.at[idx, "track_id"] = dict_track_id
                    songs_updated.at[idx, "source"] = dict_source
                    songs_updated.at[idx, "release_date"] = dict_release_date
                    updated_count += 1
            else:
                songs_updated.at[idx, "track_id"] = dict_track_id
                songs_updated.at[idx, "source"] = dict_source
                songs_updated.at[idx, "release_date"] = dict_release_date
                updated_count += 1

    print("Number of songs updated: {:_}/{:_}".format(updated_count, songs_updated.shape[0]))
    return songs_updated


#
# Filling missing values using within-dataset proxies
#

def build_first_non_nan_dict_casted(songs: pd.DataFrame, columns_to_fill: List[str]) -> Dict[Any, Dict[str, Any]]:
    """
    Build a dict:
        {track_id: {col: first non-NaN value in that group for col}}
    Ensures fill values are compatible with column dtypes to avoid setting errors.
    """
    proxy_info: Dict[Any, Dict[str, Any]] = {}
    col_dtypes = songs.dtypes.to_dict()

    for track_id, group in songs.groupby("track_id"):
        info: Dict[str, Any] = {}
        for col in columns_to_fill:
            non_nan_vals = group[col].dropna()
            value = non_nan_vals.iloc[0] if not non_nan_vals.empty else None

            dtype = col_dtypes.get(col, None)
            if pd.api.types.is_float_dtype(dtype) or pd.api.types.is_integer_dtype(dtype):
                if value is None:
                    value = np.nan
                else:
                    try:
                        value = dtype.type(value)
                    except Exception:
                        value = np.nan
            info[col] = value
        proxy_info[track_id] = info

    return proxy_info


def fill_with_proxy_dict_compat(songs: pd.DataFrame, columns_to_fill: List[str]) -> pd.DataFrame:
    """
    For each row, fill NaNs in columns_to_fill with the value from proxy_info[track_id][col],
    ensuring dtype compatibility (avoid pandas FutureWarning).
    Prints the number of rows (not values) that had at least one NaN in columns_to_fill filled.
    """
    proxy_info = build_first_non_nan_dict_casted(songs, columns_to_fill)
    songs_filled = songs.copy()
    dtypes = songs.dtypes.to_dict()

    rows_filled_mask = pd.Series([False] * len(songs_filled), index=songs_filled.index)

    for col in columns_to_fill:
        na_mask = songs_filled[col].isna()
        valid_track_mask = songs_filled["track_id"].notna() & songs_filled["track_id"].isin(proxy_info)
        update_mask = na_mask & valid_track_mask

        rows_filled_mask = rows_filled_mask | update_mask

        fill_values = songs_filled.loc[update_mask, "track_id"].map(lambda tid: proxy_info[tid][col])

        dtype = dtypes.get(col, None)
        if pd.api.types.is_float_dtype(dtype):
            fill_values = fill_values.astype("float64")
        elif pd.api.types.is_integer_dtype(dtype):
            fill_values = fill_values.astype("Int64")

        songs_filled.loc[update_mask, col] = fill_values

    n_rows_filled = rows_filled_mask.sum()
    print(f"Number of rows filled: {n_rows_filled:_}")
    return songs_filled


#
# Filling missing values from external enrichment dataframes
#

def fill_missing_from_dfs(
    songs: pd.DataFrame,
    columns_to_fill: List[str],
    col_id: str,
    *dfs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tries to fill missing values in `songs` DataFrame, for columns in `columns_to_fill`,
    based on lookup in multiple provided DataFrames using `col_id` as the match key.
    """
    songs = songs.copy()

    initial_missing = songs[columns_to_fill].isna().sum().sum()
    print(f"Total missing values *before* processing DF: {initial_missing:_}")

    for df in dfs:
        available_cols = [c for c in columns_to_fill if c in df.columns and c in songs.columns]
        print(f"Available columns: {available_cols}")
        if not available_cols or col_id not in df.columns:
            continue

        df_non_null = df.dropna(subset=[col_id])
        df_unique = df_non_null.drop_duplicates(subset=[col_id], keep="first")

        for col in available_cols:
            src_dtype = df_unique[col].dtype
            tgt_dtype = songs[col].dtype
            if pd.api.types.is_numeric_dtype(tgt_dtype) and not pd.api.types.is_numeric_dtype(src_dtype):
                songs[col] = songs[col].astype("object")

        df_lookup = df_unique.set_index(col_id)[available_cols].to_dict(orient="index")
        print(f"Size of lookup dictionary : {len(df_lookup)}")

        n_rows = len(songs)
        for i in tqdm(range(n_rows), desc="Enriching songs", leave=True):
            song_id = songs.iloc[i][col_id]
            if song_id in df_lookup:
                for col in available_cols:
                    if pd.isna(songs.iloc[i][col]):
                        val = df_lookup[song_id].get(col, None)
                        if pd.notna(val):
                            songs.iat[i, songs.columns.get_loc(col)] = val

    final_missing = songs[columns_to_fill].isna().sum().sum()
    print(f"Total missing values *after* processing DF: {final_missing:_}")

    return songs


#
# Parquet utility functions
#

def prepare_df_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a dataframe for saving to parquet by converting problematic columns.
    Converts release_date to string to avoid ArrowTypeError with mixed types.
    
    Args:
        df: DataFrame to prepare
        
    Returns:
        DataFrame with columns converted for parquet compatibility
    """
    df_parquet = df.copy()
    if 'release_date' in df_parquet.columns:
        # Convert to datetime first, then to string to ensure consistency
        df_parquet['release_date'] = pd.to_datetime(df_parquet['release_date'], errors='coerce').astype(str)
    return df_parquet



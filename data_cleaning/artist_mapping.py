import itertools
import math
from typing import Set, Dict, Any

import pandas as pd


def get_unique_ids_from_column(df: pd.DataFrame, column: str) -> Set[Any]:
    """Aggregate all ids in a specified df[column] into a set."""
    all_ids = set()
    for ids in df[column]:
        if isinstance(ids, str):
            try:
                parsed_ids = eval(ids)
    
                if isinstance(parsed_ids, (list, set, tuple)):
                    all_ids.update(parsed_ids)
                else:
                    all_ids.add(parsed_ids)
            except Exception:
                all_ids.add(ids)
    return all_ids


def get_all_combinations(all_names_artists: Set[Any], digits: str = '0123456789') -> list:
    """
    Generate all combinations of digits needed to create unique IDs for artists.
    
    Args:
        all_names_artists: Set of artist names that need IDs
        digits: String of digits to use for combinations
        
    Returns:
        List of tuples representing digit combinations
    """
    n = 1
    while math.perm(len(digits), n) < len(all_names_artists):
        n += 1
        if n > 10:
            raise ValueError("Too many combinations")
    
    combs = list(itertools.permutations(digits, n))
    
    return combs


def get_artist_to_id(all_names_artists: Set[Any]) -> Dict[Any, tuple]:
    """
    Create a mapping from artist names to generated IDs.
    
    Args:
        all_names_artists: Set of artist names
        
    Returns:
        Dictionary mapping artist names to ID tuples
    """
    combs = get_all_combinations(all_names_artists)
    return dict(zip(all_names_artists, combs[:len(all_names_artists)]))


def update_id_artists_with_mapping(songs_gold: pd.DataFrame) -> pd.DataFrame:
    """
    Updates the 'id_artists' column in songs_gold by mapping artist names (in 'artists' column)
    to their ids using the artist_to_id_dict. The result will be a list of ids for each row.
    
    Args:
        songs_gold: DataFrame with 'artists' column containing artist names
        
    Returns:
        DataFrame with updated 'id_artists' column
    """
    all_names_artists = get_unique_ids_from_column(songs_gold, 'artists')
    artist_to_id_dict = get_artist_to_id(all_names_artists)
    
    def map_artists_to_ids(artists_entry):
        # handle if artists_entry is a string representation of a list or just a string
        if isinstance(artists_entry, str):
            try:
                parsed = eval(artists_entry)
                if isinstance(parsed, (list, tuple)):
                    return [artist_to_id_dict.get(a, None) for a in parsed]
                else:
                    return [artist_to_id_dict.get(parsed, None)]
            except Exception:
                # fallback: single name as string
                return [artist_to_id_dict.get(artists_entry, None)]
        elif isinstance(artists_entry, (list, tuple)):
            return [artist_to_id_dict.get(a, None) for a in artists_entry]
        else:
            return [artist_to_id_dict.get(artists_entry, None)]
    
    songs_gold = songs_gold.copy()
    songs_gold['id_artists'] = songs_gold['artists'].apply(map_artists_to_ids)
    return songs_gold


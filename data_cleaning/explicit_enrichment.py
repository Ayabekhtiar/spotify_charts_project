import os
import json
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from google.api_core.exceptions import ResourceExhausted
from google import genai


def _get_gemini_client():
    """Get or create Gemini client instance."""
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise EnvironmentError("Please set the GOOGLE_API_KEY environment variable.")
    return genai.Client(api_key=GOOGLE_API_KEY)


def gemini_check_if_explicit(artist_names: str, track_name: str, retry: int = 3) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """
    Queries Gemini API if the song is explicit (18+ or with insults and so on) and expects json {'thinking': ..., 'value': (0|1)}.
    
    Args:
        artist_names: Name(s) of the artist(s)
        track_name: Name of the track
        retry: Number of retry attempts
        
    Returns:
        Tuple of (value, result_obj) where value is 0 or 1 (or None if failed), and result_obj is the full response
    """
    client = _get_gemini_client()
    prompt = (
        f"Given the song by these artists: {artist_names} and with title: {track_name}, "
        "does it contain explicit (age=18+, due to insult, bad words, violence or sexual content) material/lyrics? "
        "Answer in this json format: {'thinking':'<brief reason>', 'value':0 or 1}. "
        "Only 'value':1 means explicit, 0 means clean."
    )
    for attempt in range(retry):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            try:
                answer = response.text.strip()
                if answer.startswith('```'):
                    answer = answer.strip('`')
                # Try to load as JSON
                result = json.loads(answer.replace("'", '"'))
                value = result.get('value', None)
                if value in (0, 1):
                    return value, result
            except Exception:
                pass
        except ResourceExhausted:
            time.sleep(3)
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(1 + attempt)
    return None, None


def enrich_explicit_via_gemini(
    df: pd.DataFrame,
    explicit_col: str = "explicit",
    artist_col: str = "artist_names",
    track_name_col: str = "track_name",
    track_id_col: str = "track_id",
    previous_dict_path: Optional[str] = None,
    save_dict_path: Optional[str] = None,
    save_every: int = 50
) -> Dict[str, Any]:
    """
    Enrich explicit column using Gemini API with caching.
    
    Args:
        df: DataFrame to enrich
        explicit_col: Name of the explicit column
        artist_col: Name of the artist column
        track_name_col: Name of the track name column
        track_id_col: Name of the track ID column
        previous_dict_path: Path to load previous cache from
        save_dict_path: Path to save cache to
        save_every: Save progress every N updates
        
    Returns:
        Dictionary mapping track_id to explicit value
    """
    # Load cache if present
    if previous_dict_path and os.path.exists(previous_dict_path):
        with open(previous_dict_path, "r") as f:
            explicit_map = json.load(f)
    else:
        explicit_map = {}

    answered_ids = set(explicit_map.keys())
    value_by_id = dict(explicit_map)  # for in-loop dynamic dp-style cache

    total = df.shape[0]
    updated_count = 0

    for idx, row in tqdm(df.iterrows(), total=total, desc="Enriching explicit via Gemini"):
        track_id = str(row[track_id_col])

        # Skip if explicit is NOT nan (i.e., is filled already)
        val = row[explicit_col]
        if pd.notna(val) and not (isinstance(val, float) and np.isnan(val)):
            # Already has answer, skip
            continue

        # Check if already in dict (either from loaded or from live answers this run)
        if track_id in value_by_id:
            continue

        # Otherwise, need to call Gemini
        artist_names_val = row[artist_col]
        track_name_val = row[track_name_col]
        #print(f"Calling Gemini for {track_id}")
        value, result_obj = gemini_check_if_explicit(artist_names_val, track_name_val)
        value_by_id[track_id] = value  # dynamic programming: remember answer right away

        updated_count += 1
        # Only update explicit_map when it's actually not None (to not overwrite previous ones with None)
        explicit_map[track_id] = value

        # Save progress every so many updates
        if save_dict_path and updated_count % save_every == 0:
            with open(save_dict_path, "w") as f:
                json.dump(explicit_map, f, indent=2)

    # Save again when all is done
    if save_dict_path:
        with open(save_dict_path, "w") as f:
            json.dump(explicit_map, f, indent=2)
    return explicit_map


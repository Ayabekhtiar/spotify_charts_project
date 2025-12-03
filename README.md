# Spotify Charts Project

This project scrapes, cleans, and analyzes Spotify Charts data using a [medallion architecture](https://www.databricks.com/glossary/medallion-architecture) (Bronze → Silver → Gold) for data processing 

## Project Structure

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── data_cleaning.ipynb          # Main notebook for data cleaning
├── data_visualisation.ipynb      # Notebook for visualization and analysis
├── data/
│   ├── bronze/                   # Raw data layer (CSV format)
│   │   ├── data/                 # Weekly chart CSV files
│   │   ├── tracks.csv            # Track features from Kaggle, initially from Spotify Web API
│   │   └── ...                   # Additional enrichment data
│   ├── silver/                   # Cleaned and merged data (Parquet format)
│   │   ├── combined_songs.parquet
│   │   ├── songs_with_features.parquet
│   │   └── explicit_cache.json   # Cache for explicit content detection
│   └── gold/                     # Final processed data (Parquet format)
│       └── songs_with_features.parquet
├── data_scraping/                # Web scraping for data collection
│   └── scraper.py               # Selenium scraper for Spotify Charts
├── data_cleaning/                # Data cleaning 
│   ├── __init__.py
│   ├── process_charts.py        # Combines weekly charts into one file
│   ├── merge.py                 # Merges charts with track features
│   ├── clean_songs.py           # Data cleaning and missing value handling
│   ├── artist_mapping.py        # Artist ID mapping and normalization
│   └── explicit_enrichment.py   # Explicit column completion via Gemini API
├── data_visualisation/           # Data visualization and analysis
│   ├── __init__.py
│   ├── data_loading.py          # Load gold data for analysis
│   ├── track_analysis.py        # Track appearance and lifecycle analysis
│   ├── plotting.py              # Visualization functions
│   ├── ml_features.py           # Random forest features
│   ├── ml_models.py             # Random forest construction functions
│   ├── analysis.py              # Statistical analysis and insights
└
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
2. **Data preparation**:
   - Open every `.zip` files in the `data/bronze` directory.
   - Weekly chart files should be in `data/bronze/data/`
   - 
3. **(Optional) Configure API keys** (if using explicit content enrichment):
   - Set up Google Gemini API key for explicit content detection
   - See `data_cleaning/explicit_enrichment.py` for configuration

## Data Architecture

The project follows a **medallion architecture** with three data layers:

- **Bronze Layer** (`data/bronze/`): Raw, unprocessed data
  - Weekly chart CSV files from scraping
  - Track features CSV from Spotify API
  - Format: CSV

- **Silver Layer** (`data/silver/`): Cleaned and merged data
  - Combined weekly charts
  - Merged with track features
  - Format: Parquet (for better performance and compression)

- **Gold Layer** (`data/gold/`): Final processed data ready for analysis
  - Fully cleaned and enriched data
  - Missing values handled
  - Format: Parquet

## Usage

### 1. (Optional) Scraping Data

Run the scraper to download weekly charts:
```bash
python data_scraping/scraper.py
```

This will download weekly chart CSV files to `data/bronze/data/`.

### 2. (Pre-ran) Data Cleaning Pipeline

Open `data_cleaning.ipynb` and run the cells to:

1. **Process weekly charts** (`process_charts`):
   - Combines all weekly chart CSV files
   - Removes duplicates and unnecessary columns
   - Saves to `data/silver/combined_songs.parquet`

2. **Merge with track features** (`merge_data`):
   - Merges chart data with track features from `tracks.csv`
   - Saves to `data/silver/songs_with_features.parquet`

3. **Data cleaning**:
   - Handles missing values using proxy dictionaries
   - Normalizes artist IDs
   - Enriches explicit content information (optional, uses Gemini API)

4. **Final processing**:
   - Creates gold layer data
   - Saves to `data/gold/songs_with_features.parquet`

### 3. Visualization and Analysis

Open `data_visualisation.ipynb` to:

1. **Load data**:
   ```python
   from data_visualisation import load_data
   songs = load_data()
   ```

2. **Track analysis**:
   - Track appearances and rankings
   - Track lifecycle analysis
   - Time series trends

3. **Visualizations**:
   - Track rankings and trends
   - Audio feature distributions
   - Correlation heatmaps
   - Streams vs. features analysis


## Module Overview

### `data_cleaning/`

- **`process_charts.py`**: Combines weekly chart files into a single dataset
- **`merge.py`**: Merges chart data with track audio features
- **`clean_songs.py`**: Handles missing values, data cleaning, and Parquet conversion utilities
- **`artist_mapping.py`**: Maps artist names to unique IDs for normalization
- **`explicit_enrichment.py`**: Uses Gemini API to detect explicit content in tracks

### `data_visualisation/`

- **`data_loading.py`**: Loads and preprocesses gold layer data
- **`track_analysis.py`**: Analyzes track appearances and calculates lifecycle metrics
- **`plotting.py`**: Creates various visualizations (rankings, trends, distributions, etc.)
- **`analysis.py`**: Statistical analysis and correlation calculations

## Data Formats

- **Bronze layer**: CSV files (raw data)
- **Silver/Gold layers**: Parquet files (for better performance, compression, and type preservation)
- The code includes migration helpers to automatically convert existing CSV files to Parquet when needed

## Features

- **Medallion Architecture**: Bronze → Silver → Gold data pipeline
- **Data Cleaning**: Comprehensive missing value handling and data normalization
- **Artist Mapping**: Automatic artist ID generation and normalization
- **Explicit Content Detection**: Integration with Google Gemini API
- **Visualization**: Comprehensive plotting and analysis tools
- **Type Safety**: Parquet format preserves data types and improves performance

## Notes

- A Spotify account is needed for the web scraping
- The project automatically handles CSV to Parquet migration for existing data files
- All silver and gold layer files use Parquet format for optimal performance
- Bronze layer files remain in CSV format as they are raw input data
- Explicit content enrichment requires a Google Gemini API key

# Spotify Charts Project

This project scrapes, cleans, and analyzes Spotify Charts data.

## Project Structure

```
.
├── .gitignore
├── README.md
├── data_cleaning.ipynb       # Notebook to run data processing and verification
├── data
│   ├── raw                   # Raw data (scraped CSVs, tracks.csv)
│   └── processed             # Processed data (combined_songs.csv, songs_with_features.csv)
├── data_cleaning             # Module for data processing
│   ├── process_charts.py     # Combines weekly charts into one file
│   └── merge.py              # Merges charts with track features
└── data_scraping             # Module for web scraping
    └── scraper.py            # Selenium scraper for Spotify Charts
```

## Setup

1.  Ensure you have the required dependencies installed (pandas, selenium, etc.).
2.  Place `tracks.csv` in `data/raw` if it's not already there.

## Usage

### Scraping Data
Run the scraper to download weekly charts:
```bash
python data_scraping/scraper.py
```

### Cleaning and Merging Data
Open `data_cleaning.ipynb` and run the cells to:
1.  Process the downloaded charts.
2.  Merge with track features.
3.  Verify data integrity.
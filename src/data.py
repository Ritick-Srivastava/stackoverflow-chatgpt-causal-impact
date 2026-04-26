import time
import pandas as pd
from pytrends.request import TrendReq

from src.config import (
    ALL_KEYWORDS, START_DATE, END_DATE,
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
)


def _pull_single(keyword, start_date, end_date, geo='', sleep_seconds=5):
    """Pull one keyword in its own request so it gets full 0–100 resolution."""
    pytrends = TrendReq(hl='en-US', tz=360)
    timeframe = f"{start_date} {end_date}"
    print(f"  Pulling '{keyword}' ... ", end="", flush=True)
    time.sleep(sleep_seconds)
    pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()
    if df.empty:
        raise ValueError(
            f"pytrends returned empty data for '{keyword}'.\n"
            "Google may be rate-limiting — wait 60 s and retry."
        )
    df = df.drop(columns=["isPartial"], errors="ignore")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    inferred = pd.infer_freq(df.index)
    if inferred is None or any(c in str(inferred) for c in ("W", "D")):
        df = df.resample("MS").mean().round(1)
    print(f"done ({len(df)} rows)")
    return df[[keyword]]


def pull_trends(keywords=None, start_date=None, end_date=None,
                geo='', sleep_seconds=5):
    """
    Pull each keyword in its own Google Trends request so every series gets
    full 0–100 resolution instead of being compressed by a dominant keyword.
    Returns a combined monthly DataFrame indexed by date.
    """
    if keywords is None:
        keywords = ALL_KEYWORDS
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = END_DATE

    print(f"Pulling {len(keywords)} keywords separately (one request each):")
    frames = [_pull_single(kw, start_date, end_date, geo, sleep_seconds)
              for kw in keywords]
    df = pd.concat(frames, axis=1)
    return df


def load_or_pull(force_pull=False):
    """
    Load raw trends from cache if available, otherwise pull fresh.
    Set force_pull=True to bypass cache and re-hit the API.
    """
    raw_path = RAW_DATA_DIR / "trends_raw.csv"

    if not force_pull and raw_path.exists():
        print(f"Loading cached raw data from {raw_path}")
        return pd.read_csv(raw_path, index_col="date", parse_dates=True)

    df = pull_trends()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path)
    print(f"Saved raw data → {raw_path}")
    return df


def process_trends(df):
    """
    Clean raw trends DataFrame:
      - Restrict to configured date range
      - Drop fully-NaN rows
      - Interpolate isolated missing months (max 2 consecutive gaps)
    """
    df = df.copy()
    df = df.loc[START_DATE:END_DATE]
    df = df.dropna(how="all")
    df = df.interpolate(method="time", limit=2)
    return df


def save_processed(df):
    """Save processed DataFrame to data/processed/trends_processed.csv."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / "trends_processed.csv"
    df.to_csv(path)
    print(f"Saved processed data → {path}")
    return path


def load_processed():
    """Load processed trends data. Raises FileNotFoundError if missing."""
    path = PROCESSED_DATA_DIR / "trends_processed.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}.\n"
            "Run notebook 01_data_collection.ipynb first."
        )
    return pd.read_csv(path, index_col="date", parse_dates=True)

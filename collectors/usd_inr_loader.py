import pandas as pd
import requests
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
FRED_API_KEY = "621b94f9b74ff588a2467751ed44b533"


def fetch_usd_inr_from_fred(start_date: str = '2010-01-01', end_date: str = '2025-05-06') -> pd.DataFrame:
    """
    Fetches monthly USD to INR exchange rate from FRED and returns a cleaned DataFrame.
    """
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY is not set in environment variables.")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "DEXINUS",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["observations"]

    df = pd.DataFrame(data)[["date", "value"]]
    df.columns = ["date", "usd_inr"]
    df["date"] = pd.to_datetime(df["date"])
    df["usd_inr"] = pd.to_numeric(df["usd_inr"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    
    return df


def save_usd_inr_csv(save_path: Path = Path("data/raw/usd_inr.csv")):
    df = fetch_usd_inr_from_fred()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved USD/INR data to {save_path}")

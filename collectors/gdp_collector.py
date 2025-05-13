# collectors/gdp_collector.py

import os
import pandas as pd
from pandas_datareader import wb

def fetch_india_gdp(output_path="data/raw/gdp.csv"):
    """
    Fetches India's quarterly GDP from the World Bank API and saves it as CSV.
    """
    # World Bank indicator for GDP (current US$)
    indicator = 'NY.GDP.MKTP.CD'
    country = 'IN'  # India
    print("Fetching GDP data from World Bank...")

    # Download GDP data from World Bank
    df = wb.download(indicator=indicator, country=country, start=2005, end=2024)
    df = df.reset_index()
    df.rename(columns={'year': 'date', 'NY.GDP.MKTP.CD': 'gdp_usd'}, inplace=True)

    # Sort and save
    df = df.sort_values('date')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"GDP data saved to {output_path}")

    return df

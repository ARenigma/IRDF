# collectors/repo_rate_collector.py

import pandas as pd
import os
from playwright.sync_api import sync_playwright

def fetch_repo_rate_playwright(output_path="data/raw/repo_rate.csv"):
    url = "https://www.moneycontrol.com/economic-indicators/india-Interest-Rate-4870699"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)

        # Wait for table to load
        page.wait_for_selector("table")

        # Extract table HTML
        html = page.content()
        browser.close()

    # Parse all tables and find the one with repo rates
    tables = pd.read_html(html)
    repo_df = None
    for table in tables:
        if {'Date', 'RBI Interest Rate Decision (%)'}.issubset(set(table.columns)):
            repo_df = table
            break

    if repo_df is None:
        raise ValueError("Could not find the repo rate table.")

    # Clean up
    repo_df.columns = ['date', 'repo_rate']
    repo_df['date'] = pd.to_datetime(repo_df['date'], dayfirst=True, errors='coerce')
    repo_df['repo_rate'] = pd.to_numeric(repo_df['repo_rate'], errors='coerce')
    repo_df = repo_df.dropna()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    repo_df.to_csv(output_path, index=False)
    print(f"✅ Repo rate data saved to {output_path}")

    return repo_df

# Direct run
if __name__ == "__main__":
    fetch_repo_rate_playwright()

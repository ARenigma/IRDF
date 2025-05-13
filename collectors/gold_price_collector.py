# collectors/gold_price_collector.py

import os
import pandas as pd
from datetime import datetime

def fetch_gold_price(file_path="data/raw/gold_price.xlsx", output_path="data/raw/gold_price_processed.csv"):
    """
    Loads and processes gold price data from Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to the gold price Excel file
    output_path : str
        Path to save the processed CSV file
        
    Returns:
    --------
    DataFrame containing the gold price data with various derived metrics
    """
    print(f"Loading gold price data from {file_path}...")
    
    try:
        # Load gold price data
        gold_df = pd.read_excel(file_path)
        
        # Rename columns based on the actual Excel structure
        if 'Date' in gold_df.columns and 'Price' in gold_df.columns:
            gold_df.rename(columns={'Date': 'date', 'Price': 'gold_price'}, inplace=True)
        elif len(gold_df.columns) >= 2:
            # If column names are different, rename the first two columns
            gold_df = gold_df.iloc[:, 0:2]
            gold_df.columns = ['date', 'gold_price']
            print("Warning: Assuming first column is date and second column is gold price")
        else:
            raise ValueError("Unexpected gold price Excel format")
        
        # Ensure date column is datetime
        gold_df['date'] = pd.to_datetime(gold_df['date'])
        
        # Sort by date
        gold_df = gold_df.sort_values('date')
        
        # Calculate month-over-month percentage change
        gold_df['gold_price_mom'] = gold_df['gold_price'].pct_change() * 100
        
        # Calculate year-over-year percentage change
        gold_df['gold_price_yoy'] = gold_df['gold_price'].pct_change(12) * 100
        
        # Calculate volatility (rolling standard deviation)
        gold_df['gold_volatility_1m'] = gold_df['gold_price'].rolling(window=30).std()
        gold_df['gold_volatility_3m'] = gold_df['gold_price'].rolling(window=90).std()
        
        # Calculate rolling averages
        gold_df['gold_price_30d_avg'] = gold_df['gold_price'].rolling(window=30).mean()
        gold_df['gold_price_90d_avg'] = gold_df['gold_price'].rolling(window=90).mean()
        gold_df['gold_price_180d_avg'] = gold_df['gold_price'].rolling(window=180).mean()
        
        # Calculate trend indicator (difference between short-term and long-term average)
        gold_df['gold_trend'] = gold_df['gold_price_30d_avg'] - gold_df['gold_price_180d_avg']
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gold_df.to_csv(output_path, index=False)
        print(f"Processed gold price data saved to {output_path}")
        
        return gold_df
        
    except Exception as e:
        print(f"Error processing gold price data: {e}")
        return None 
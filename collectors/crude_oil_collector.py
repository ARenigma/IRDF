# collectors/crude_oil_collector.py

import os
import pandas as pd
from datetime import datetime

def fetch_crude_oil_price(file_path="data/raw/crude_oil_price.xlsx", output_path="data/raw/crude_oil_price_processed.csv"):
    """
    Loads and processes crude oil price data from Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to the crude oil price Excel file
    output_path : str
        Path to save the processed CSV file
        
    Returns:
    --------
    DataFrame containing the crude oil price data with various derived metrics
    """
    print(f"Loading crude oil price data from {file_path}...")
    
    try:
        # Load crude oil price data
        oil_df = pd.read_excel(file_path)
        
        # Rename columns based on the actual Excel structure
        if 'Date' in oil_df.columns and 'Price' in oil_df.columns:
            oil_df.rename(columns={'Date': 'date', 'Price': 'oil_price'}, inplace=True)
        elif len(oil_df.columns) >= 2:
            # If column names are different, rename the first two columns
            oil_df = oil_df.iloc[:, 0:2]
            oil_df.columns = ['date', 'oil_price']
            print("Warning: Assuming first column is date and second column is oil price")
        else:
            raise ValueError("Unexpected crude oil price Excel format")
        
        # Ensure date column is datetime
        oil_df['date'] = pd.to_datetime(oil_df['date'])
        
        # Sort by date
        oil_df = oil_df.sort_values('date')
        
        # Calculate month-over-month percentage change
        oil_df['oil_price_mom'] = oil_df['oil_price'].pct_change() * 100
        
        # Calculate year-over-year percentage change
        oil_df['oil_price_yoy'] = oil_df['oil_price'].pct_change(12) * 100
        
        # Calculate volatility (rolling standard deviation)
        oil_df['oil_volatility_1m'] = oil_df['oil_price'].rolling(window=30).std()
        oil_df['oil_volatility_3m'] = oil_df['oil_price'].rolling(window=90).std()
        
        # Calculate rolling averages
        oil_df['oil_price_30d_avg'] = oil_df['oil_price'].rolling(window=30).mean()
        oil_df['oil_price_90d_avg'] = oil_df['oil_price'].rolling(window=90).mean()
        oil_df['oil_price_180d_avg'] = oil_df['oil_price'].rolling(window=180).mean()
        
        # Calculate price momentum
        oil_df['oil_momentum'] = oil_df['oil_price_30d_avg'] / oil_df['oil_price_90d_avg'] - 1
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        oil_df.to_csv(output_path, index=False)
        print(f"Processed crude oil price data saved to {output_path}")
        
        return oil_df
        
    except Exception as e:
        print(f"Error processing crude oil price data: {e}")
        return None 
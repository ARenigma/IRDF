# collectors/cpi_collector.py

import os
import pandas as pd
from datetime import datetime

def fetch_india_cpi(input_path="data/raw/cpi_index.csv", output_path="data/raw/cpi.csv"):
    """
    Loads and processes India's Consumer Price Index data from local CSV file.
    
    Parameters:
    -----------
    input_path : str
        Path to the CPI index CSV file
    output_path : str
        Path to save the processed CSV file
        
    Returns:
    --------
    DataFrame containing the CPI data with date and cpi_index columns
    """
    print(f"Loading CPI data from {input_path}...")
    
    try:
        # Load CPI data from CSV
        cpi_df = pd.read_csv(input_path)
        
        # Rename columns based on the actual CSV structure
        if 'Date' in cpi_df.columns and 'CPI' in cpi_df.columns:
            cpi_df.rename(columns={'Date': 'date', 'CPI': 'cpi_index'}, inplace=True)
        elif len(cpi_df.columns) >= 2:
            # If column names are different, rename the first two columns
            cpi_df = cpi_df.iloc[:, 0:2]
            cpi_df.columns = ['date', 'cpi_index']
            print("Warning: Assuming first column is date and second column is CPI")
        else:
            raise ValueError("Unexpected CPI CSV format")
        
        # Ensure date column is datetime
        cpi_df['date'] = pd.to_datetime(cpi_df['date'])
        
        # Sort by date
        cpi_df = cpi_df.sort_values('date')
        
        # Calculate monthly inflation (month-over-month)
        cpi_df['inflation_mom'] = cpi_df['cpi_index'].pct_change() * 100
        
        # Calculate annual inflation (year-over-year)
        cpi_df['inflation_yoy'] = cpi_df['cpi_index'].pct_change(12) * 100
        
        # Calculate 3-month and 6-month rolling averages
        cpi_df['cpi_3m_avg'] = cpi_df['cpi_index'].rolling(window=3).mean()
        cpi_df['cpi_6m_avg'] = cpi_df['cpi_index'].rolling(window=6).mean()
        
        # Save to processed file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cpi_df.to_csv(output_path, index=False)
        print(f"Processed CPI data saved to {output_path}")
        
        return cpi_df
        
    except Exception as e:
        print(f"Error processing CPI data: {e}")
        return None

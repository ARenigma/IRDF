# collectors/iip_collector.py

import os
import pandas as pd
import numpy as np
from datetime import datetime

def fetch_iip_data(durable_path="data/raw/iip_consumer_durable.xlsx", 
                  nondurable_path="data/raw/iip_consumer_nondurable.xlsx",
                  output_path="data/raw/iip_combined.csv"):
    """
    Loads and processes India's Index of Industrial Production (IIP) data for consumer
    durables and non-durables from Excel files.
    
    Parameters:
    -----------
    durable_path : str
        Path to the IIP consumer durables Excel file
    nondurable_path : str
        Path to the IIP consumer non-durables Excel file
    output_path : str
        Path to save the combined processed CSV file
        
    Returns:
    --------
    DataFrame containing the combined IIP data with date and various IIP metrics
    """
    print(f"Loading IIP data from {durable_path} and {nondurable_path}...")
    
    try:
        # Load IIP consumer durables data
        durable_df = pd.read_excel(durable_path)
        
        # Rename columns based on the actual Excel structure
        if 'Date' in durable_df.columns and 'Index' in durable_df.columns:
            durable_df.rename(columns={'Date': 'date', 'Index': 'iip_durable'}, inplace=True)
        elif len(durable_df.columns) >= 2:
            # If column names are different, rename the first two columns
            durable_df = durable_df.iloc[:, 0:2]
            durable_df.columns = ['date', 'iip_durable']
            print("Warning: Assuming first column is date and second column is IIP durable index")
        else:
            raise ValueError("Unexpected IIP durable Excel format")
        
        # Ensure date column is datetime
        durable_df['date'] = pd.to_datetime(durable_df['date'])
        
        # Ensure IIP values are numeric
        durable_df['iip_durable'] = pd.to_numeric(durable_df['iip_durable'], errors='coerce')
        
        # Sort by date
        durable_df = durable_df.sort_values('date')
        
        # Load IIP consumer non-durables data
        nondurable_df = pd.read_excel(nondurable_path)
        
        # Rename columns based on the actual Excel structure
        if 'Date' in nondurable_df.columns and 'Index' in nondurable_df.columns:
            nondurable_df.rename(columns={'Date': 'date', 'Index': 'iip_nondurable'}, inplace=True)
        elif len(nondurable_df.columns) >= 2:
            # If column names are different, rename the first two columns
            nondurable_df = nondurable_df.iloc[:, 0:2]
            nondurable_df.columns = ['date', 'iip_nondurable']
            print("Warning: Assuming first column is date and second column is IIP non-durable index")
        else:
            raise ValueError("Unexpected IIP non-durable Excel format")
        
        # Ensure date column is datetime
        nondurable_df['date'] = pd.to_datetime(nondurable_df['date'])
        
        # Ensure IIP values are numeric
        nondurable_df['iip_nondurable'] = pd.to_numeric(nondurable_df['iip_nondurable'], errors='coerce')
        
        # Sort by date
        nondurable_df = nondurable_df.sort_values('date')
        
        # Merge the two datasets
        iip_df = pd.merge(durable_df, nondurable_df, on='date', how='outer')
        
        # Sort by date
        iip_df = iip_df.sort_values('date')
        
        # Calculate combined IIP index (simple average)
        iip_df['iip_combined'] = (iip_df['iip_durable'] + iip_df['iip_nondurable']) / 2
        
        # Calculate year-over-year growth rates
        iip_df['iip_durable_yoy'] = iip_df['iip_durable'].pct_change(12) * 100
        iip_df['iip_nondurable_yoy'] = iip_df['iip_nondurable'].pct_change(12) * 100
        iip_df['iip_combined_yoy'] = iip_df['iip_combined'].pct_change(12) * 100
        
        # Calculate month-over-month growth rates
        iip_df['iip_durable_mom'] = iip_df['iip_durable'].pct_change() * 100
        iip_df['iip_nondurable_mom'] = iip_df['iip_nondurable'].pct_change() * 100
        iip_df['iip_combined_mom'] = iip_df['iip_combined'].pct_change() * 100
        
        # Calculate 3-month and 6-month rolling averages
        for col in ['iip_durable', 'iip_nondurable', 'iip_combined']:
            iip_df[f'{col}_3m_avg'] = iip_df[col].rolling(window=3).mean()
            iip_df[f'{col}_6m_avg'] = iip_df[col].rolling(window=6).mean()
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        iip_df.to_csv(output_path, index=False)
        print(f"Processed IIP data saved to {output_path}")
        
        return iip_df
        
    except Exception as e:
        print(f"Error processing IIP data: {e}")
        return None 
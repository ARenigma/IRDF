# collectors/wpi_collector.py

import os
import pandas as pd
from datetime import datetime

def fetch_wpi_data(file_path="data/raw/wpi_index.xlsx", output_path="data/raw/wpi_processed.csv"):
    """
    Loads and processes India's Wholesale Price Index (WPI) data from Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to the WPI Excel file
    output_path : str
        Path to save the processed CSV file
        
    Returns:
    --------
    DataFrame containing the WPI data with date, wpi_index, and wpi_inflation columns
    """
    print(f"Loading WPI data from {file_path}...")
    
    try:
        # Load WPI data
        wpi_df = pd.read_excel(file_path)
        
        # Rename columns based on the actual Excel structure
        if 'Date' in wpi_df.columns and 'WPI Index' in wpi_df.columns:
            wpi_df.rename(columns={'Date': 'date', 'WPI Index': 'wpi_index'}, inplace=True)
        elif len(wpi_df.columns) >= 2:
            # If column names are different, rename the first two columns
            wpi_df = wpi_df.iloc[:, 0:2]
            wpi_df.columns = ['date', 'wpi_index']
            print("Warning: Assuming first column is date and second column is WPI index")
        else:
            raise ValueError("Unexpected WPI Excel format")
        
        # Ensure date column is datetime
        wpi_df['date'] = pd.to_datetime(wpi_df['date'])
        
        # Sort by date
        wpi_df = wpi_df.sort_values('date')
        
        # Calculate WPI inflation (year-over-year percentage change)
        wpi_df['wpi_inflation'] = wpi_df['wpi_index'].pct_change(12) * 100
        
        # Calculate month-over-month changes
        wpi_df['wpi_mom'] = wpi_df['wpi_index'].pct_change() * 100
        
        # Calculate 3-month and 6-month rolling averages
        wpi_df['wpi_3m_avg'] = wpi_df['wpi_index'].rolling(window=3).mean()
        wpi_df['wpi_6m_avg'] = wpi_df['wpi_index'].rolling(window=6).mean()
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        wpi_df.to_csv(output_path, index=False)
        print(f"Processed WPI data saved to {output_path}")
        
        return wpi_df
        
    except Exception as e:
        print(f"Error processing WPI data: {e}")
        return None 
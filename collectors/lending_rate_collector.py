# collectors/lending_rate_collector.py

import os
import pandas as pd
from datetime import datetime

def fetch_lending_rate(file_path="data/raw/lending_rate.xlsx", output_path="data/raw/lending_rate_processed.csv"):
    """
    Loads and processes India's commercial bank lending rate data from Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to the lending rate Excel file
    output_path : str
        Path to save the processed CSV file
        
    Returns:
    --------
    DataFrame containing the lending rate data with date and lending_rate columns
    """
    print(f"Loading lending rate data from {file_path}...")
    
    try:
        # Load lending rate data
        lending_df = pd.read_excel(file_path)
        
        # Rename columns based on the actual Excel structure
        if 'Date' in lending_df.columns and 'Lending Rate' in lending_df.columns:
            lending_df.rename(columns={'Date': 'date', 'Lending Rate': 'lending_rate'}, inplace=True)
        elif len(lending_df.columns) >= 2:
            # If column names are different, rename the first two columns
            lending_df = lending_df.iloc[:, 0:2]
            lending_df.columns = ['date', 'lending_rate']
            print("Warning: Assuming first column is date and second column is lending rate")
        else:
            raise ValueError("Unexpected lending rate Excel format")
        
        # Ensure date column is datetime
        lending_df['date'] = pd.to_datetime(lending_df['date'])
        
        # Ensure lending_rate is numeric
        lending_df['lending_rate'] = pd.to_numeric(lending_df['lending_rate'], errors='coerce')
        
        # Sort by date
        lending_df = lending_df.sort_values('date')
        
        # Calculate spread over repo rate (if available)
        try:
            repo_df = pd.read_csv("data/raw/repo_rate_processed.csv")
            repo_df['date'] = pd.to_datetime(repo_df['date'])
            
            # Ensure interest_rate is numeric
            repo_df['interest_rate'] = pd.to_numeric(repo_df['interest_rate'], errors='coerce')
            
            repo_df = repo_df.sort_values('date')
            
            # Merge with repo rate data
            merged_df = pd.merge_asof(lending_df, repo_df[['date', 'interest_rate']], 
                                     on='date', direction='nearest')
            
            # Calculate spread (difference between lending rate and repo rate)
            merged_df['rate_spread'] = merged_df['lending_rate'] - merged_df['interest_rate']
            
            # Keep only the columns we need from the merged dataframe
            lending_df = merged_df[['date', 'lending_rate', 'interest_rate', 'rate_spread']]
        except Exception as e:
            print(f"Could not calculate spread over repo rate: {e}")
        
        # Calculate month-over-month changes
        lending_df['lending_rate_mom'] = lending_df['lending_rate'].pct_change() * 100
        
        # Calculate 3-month and 6-month rolling averages
        lending_df['lending_rate_3m_avg'] = lending_df['lending_rate'].rolling(window=3).mean()
        lending_df['lending_rate_6m_avg'] = lending_df['lending_rate'].rolling(window=6).mean()
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lending_df.to_csv(output_path, index=False)
        print(f"Processed lending rate data saved to {output_path}")
        
        return lending_df
        
    except Exception as e:
        print(f"Error processing lending rate data: {e}")
        return None 
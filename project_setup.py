"""
India Retail Demand Forecaster
--------------------------------
A macroeconomic signal engine that predicts retail demand in India based on economic indicators.
This project demonstrates skills required for the Data Scientist I position.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# Import collectors
from collectors.gdp_collector import fetch_india_gdp
from collectors.cpi_collector import fetch_india_cpi
from collectors.iip_collector import fetch_iip_data
from collectors.wpi_collector import fetch_wpi_data
from collectors.lending_rate_collector import fetch_lending_rate
from collectors.usd_inr_loader import save_usd_inr_csv
from collectors.gold_price_collector import fetch_gold_price
from collectors.crude_oil_collector import fetch_crude_oil_price

save_usd_inr_csv()


# Set up directories
os.makedirs('data', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('collectors', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Project configuration
CONFIG = {
    'data_sources': {
        'macro_indicators': {
            'gdp': 'https://www.mospi.gov.in/data', # Example URL, you'll need to find actual sources
            'inflation': 'https://www.rbi.org.in/Scripts/Statistics.aspx',
            'interest_rates': 'https://www.rbi.org.in/Scripts/Statistics.aspx',
            'lending_rates': 'https://www.rbi.org.in/Scripts/Statistics.aspx',
            'unemployment': 'https://www.mospi.gov.in/data',
            'consumer_confidence': 'https://www.rbi.org.in/Scripts/Statistics.aspx',
            'wpi': 'https://eaindustry.nic.in/wpi_press_release.asp',
            'iip': 'https://www.mospi.gov.in/data'
        },
        'retail_data': {
            'retail_sales': 'https://www.mospi.gov.in/data',
        }
    },
    'timeframe': {
        'start_date': '2015-01-01',
        'end_date': '2023-12-31',
        'forecast_horizon': 6  # months
    },
    'model_params': {
        'test_size': 0.2,
        'random_state': 42
    }
}

def load_repo_rate(file_path="data/raw/repo_rate.xlsx"):
    """
    Load and process RBI repo rate data from Excel file.
    Returns a DataFrame with dates and interest rates.
    """
    try:
        print(f"Loading repo rate data from {file_path}...")
        # Read the Excel file
        repo_df = pd.read_excel(file_path)
        
        # Rename columns (assuming the Excel has date and rate columns)
        # Adjust column names based on the actual Excel structure
        if 'Date' in repo_df.columns and 'Repo Rate' in repo_df.columns:
            repo_df.rename(columns={'Date': 'date', 'Repo Rate': 'interest_rate'}, inplace=True)
        elif len(repo_df.columns) >= 2:
            # If column names are different, rename the first two columns
            repo_df = repo_df.iloc[:, 0:2]
            repo_df.columns = ['date', 'interest_rate']
            print("Warning: Assuming first column is date and second column is repo rate")
        else:
            raise ValueError("Unexpected Excel format")
        
        # Ensure date column is datetime
        repo_df['date'] = pd.to_datetime(repo_df['date'])
        
        # Ensure interest_rate is numeric
        repo_df['interest_rate'] = pd.to_numeric(repo_df['interest_rate'], errors='coerce')
        
        # Sort by date
        repo_df = repo_df.sort_values('date')
        
        return repo_df
    
    except Exception as e:
        print(f"Error loading repo rate data: {e}")
        return None

def load_unemployment_rate(file_path="data/raw/unemployment_rate.xlsx"):
    """
    Load and process unemployment rate data from Excel file.
    Returns a DataFrame with dates and unemployment rates.
    """
    try:
        print(f"Loading unemployment rate data from {file_path}...")
        # Read the Excel file
        unemp_df = pd.read_excel(file_path)
        
        # Rename columns based on the actual Excel structure
        if 'Date' in unemp_df.columns and 'Unemployment Rate' in unemp_df.columns:
            unemp_df.rename(columns={'Date': 'date', 'Unemployment Rate': 'unemployment'}, inplace=True)
        elif len(unemp_df.columns) >= 2:
            # If column names are different, rename the first two columns
            unemp_df = unemp_df.iloc[:, 0:2]
            unemp_df.columns = ['date', 'unemployment']
            print("Warning: Assuming first column is date and second column is unemployment rate")
        else:
            raise ValueError("Unexpected Excel format")
        
        # Ensure date column is datetime
        unemp_df['date'] = pd.to_datetime(unemp_df['date'])
        
        # Ensure unemployment rate is numeric
        unemp_df['unemployment'] = pd.to_numeric(unemp_df['unemployment'], errors='coerce')
        
        # Sort by date
        unemp_df = unemp_df.sort_values('date')
        
        return unemp_df
    
    except Exception as e:
        print(f"Error loading unemployment rate data: {e}")
        return None

def fetch_data():
    """
    Function to fetch data from APIs or data sources.
    Uses real data collectors where available, and generates synthetic data for other indicators.
    """
    print("Fetching macroeconomic and retail data...")
    
    date_range = pd.date_range(
        start=CONFIG['timeframe']['start_date'],
        end=CONFIG['timeframe']['end_date'],
        freq='M'
    )
    
    # Fetch real GDP data using the collector
    gdp_df = fetch_india_gdp("data/raw/gdp.csv")
    
    # Convert GDP data to monthly frequency (if needed) and align with our date range
    # Note: World Bank GDP data is typically annual, so we'll need to convert it
    if 'date' in gdp_df.columns and 'gdp_usd' in gdp_df.columns:
        gdp_df['date'] = pd.to_datetime(gdp_df['date'])
        gdp_df = gdp_df.set_index('date')
        
        # If annual data, resample to monthly frequency with interpolation
        if len(gdp_df) < len(date_range):
            gdp_df = gdp_df.resample('M').interpolate(method='cubic')
    
        # Calculate GDP growth rate (year-over-year percentage change)
        gdp_df['gdp_growth'] = gdp_df['gdp_usd'].pct_change(12) * 100
        
        # Align with our date range
        gdp_df = gdp_df.reindex(date_range)
        gdp_df['gdp_growth'] = gdp_df['gdp_growth'].interpolate(method='cubic')
    else:
        # Fallback to synthetic data if the format is unexpected
        print("Warning: GDP data format unexpected, using synthetic data")
        # Generate synthetic GDP growth data
        np.random.seed(42)
    gdp_quarterly = np.random.normal(loc=7.5, scale=1.5, size=len(date_range)//3 + 1)
    gdp_monthly = np.interp(
        np.arange(len(date_range)), 
        np.arange(0, len(date_range), 3), 
        gdp_quarterly
    )[:len(date_range)]
    
    # Add seasonality and trend to GDP
    trend = np.linspace(0, 2, len(date_range))
    seasonality = 0.5 * np.sin(np.arange(len(date_range)) * (2*np.pi/12))
    gdp_monthly = gdp_monthly + trend + seasonality
    
    gdp_df = pd.DataFrame(index=date_range)
    gdp_df['gdp_growth'] = gdp_monthly
    
    # Fetch real CPI data and calculate inflation
    try:
        cpi_df = fetch_india_cpi("data/raw/cpi_index.csv", "data/raw/cpi.csv")
        
        # Process CPI data
        if 'date' in cpi_df.columns and 'cpi_index' in cpi_df.columns:
            cpi_df['date'] = pd.to_datetime(cpi_df['date'])
            cpi_df = cpi_df.set_index('date')
            
            # Calculate inflation as year-over-year percentage change in CPI
            cpi_df['inflation'] = cpi_df['cpi_index'].pct_change(12) * 100
            
            # Align with our date range
            cpi_df = cpi_df.reindex(date_range)
            cpi_df['inflation'] = cpi_df['inflation'].interpolate(method='cubic')
            
            # Ensure we don't have NaN values
            if cpi_df['inflation'].isna().any():
                print("Warning: Missing inflation values, filling with interpolation")
                cpi_df['inflation'] = cpi_df['inflation'].interpolate()
            
            inflation = cpi_df['inflation']
        else:
            raise ValueError("Unexpected CPI data format")
    except Exception as e:
        print(f"Warning: Failed to process CPI data, using synthetic data. Error: {e}")
        # Generate synthetic inflation data as fallback
        np.random.seed(42)
    inflation_base = np.random.normal(loc=5.0, scale=1.0, size=len(date_range))
    inflation = np.cumsum(inflation_base - 5.0) / 10 + 5.0
    inflation = inflation + 0.8 * np.sin(np.arange(len(date_range)) * (2*np.pi/12))
    
    # Load WPI data from Excel file
    wpi_df = fetch_wpi_data("data/raw/wpi_index.xlsx", "data/raw/wpi_processed.csv")
    
    if wpi_df is not None and len(wpi_df) > 0:
        # Convert to time series
        wpi_df['date'] = pd.to_datetime(wpi_df['date'])
        wpi_df = wpi_df.set_index('date')
        
        # Check if we need to resample
        if len(wpi_df) < len(date_range):
            # Resample to monthly frequency
            wpi_monthly = wpi_df.resample('M').interpolate(method='cubic')
        else:
            wpi_monthly = wpi_df
            
        # Align with our date range
        wpi_monthly = wpi_monthly.reindex(date_range)
        
        # Fill any missing values with interpolation
        for col in wpi_monthly.columns:
            if wpi_monthly[col].isna().any():
                print(f"Interpolating missing {col} values")
                wpi_monthly[col] = wpi_monthly[col].interpolate(method='linear')
        
        # Extract the columns we'll use for modeling
        wpi_index = wpi_monthly['wpi_index']
        wpi_inflation = wpi_monthly['wpi_inflation']
        wpi_mom = wpi_monthly['wpi_mom']
        wpi_3m_avg = wpi_monthly['wpi_3m_avg']
        wpi_6m_avg = wpi_monthly['wpi_6m_avg']
    else:
        print("Warning: Using synthetic WPI data")
        # Generate synthetic WPI data as fallback
        np.random.seed(42)
        # Base index
        wpi_index_base = np.random.normal(loc=0, scale=0.5, size=len(date_range))
        
        # Create the WPI index with trend and seasonality
        trend = np.linspace(100, 140, len(date_range))  # General upward trend
        seasonality = 2 * np.sin(np.arange(len(date_range)) * (2*np.pi/12))
        wpi_index = trend + np.cumsum(wpi_index_base) + seasonality
        
        # Calculate WPI inflation (year-over-year change)
        wpi_inflation = np.zeros(len(date_range))
        for i in range(12, len(date_range)):
            wpi_inflation[i] = (wpi_index[i] / wpi_index[i-12] - 1) * 100
        
        # Calculate month-over-month changes
        wpi_mom = np.zeros(len(date_range))
        for i in range(1, len(date_range)):
            wpi_mom[i] = (wpi_index[i] / wpi_index[i-1] - 1) * 100
        
        # Calculate rolling averages
        wpi_3m_avg = np.zeros(len(date_range))
        wpi_6m_avg = np.zeros(len(date_range))
        
        for i in range(len(date_range)):
            if i >= 2:
                wpi_3m_avg[i] = np.mean(wpi_index[max(0, i-2):i+1])
            else:
                wpi_3m_avg[i] = wpi_index[i]
                
            if i >= 5:
                wpi_6m_avg[i] = np.mean(wpi_index[max(0, i-5):i+1])
            else:
                wpi_6m_avg[i] = wpi_index[i]
    
    # Load lending rate data from Excel file
    lending_df = fetch_lending_rate("data/raw/lending_rate.xlsx", "data/raw/lending_rate_processed.csv")
    
    if lending_df is not None and len(lending_df) > 0:
        # Convert to time series
        lending_df['date'] = pd.to_datetime(lending_df['date'])
        lending_df = lending_df.set_index('date')
        
        # Check if we need to resample (lending rates might be published at irregular intervals)
        if len(lending_df) < len(date_range):
            # For lending rates, it's better to forward fill and then interpolate
            # as these rates typically remain constant until changed
            lending_daily = lending_df.resample('D').ffill()
            lending_monthly = lending_daily.resample('M').last()
        else:
            lending_monthly = lending_df
            
        # Align with our date range
        lending_monthly = lending_monthly.reindex(date_range)
        
        # Fill any missing values with interpolation
        for col in lending_monthly.columns:
            if lending_monthly[col].isna().any():
                print(f"Interpolating missing {col} values")
                lending_monthly[col] = lending_monthly[col].interpolate(method='linear')
        
        # Extract the columns we'll use for modeling
        lending_rate = lending_monthly['lending_rate']
        
        # Extract additional metrics if available
        if 'rate_spread' in lending_monthly.columns:
            rate_spread = lending_monthly['rate_spread']
        else:
            rate_spread = None
            
        if 'lending_rate_mom' in lending_monthly.columns:
            lending_rate_mom = lending_monthly['lending_rate_mom']
        else:
            lending_rate_mom = None
    else:
        print("Warning: Using synthetic lending rate data")
        # Generate synthetic lending rate data as fallback
        np.random.seed(42)
        # Base values with small random changes
        lending_base = np.random.normal(loc=0, scale=0.05, size=len(date_range))
        
        # Create lending rate with trend and persistence
        # Usually lending rates are a few percentage points above the repo/policy rate
        lending_rate = np.cumsum(lending_base) / 10 + 9.0  # Starting around 9%
        
        # Ensure rates don't go too low
        lending_rate = np.maximum(lending_rate, 7.0)
        
        # No rate spread available with synthetic data
        rate_spread = None
        lending_rate_mom = None
    
    # Load repo rate data from Excel file
    repo_df = load_repo_rate("data/raw/repo_rate.xlsx")
    
    if repo_df is not None and len(repo_df) > 0:
        # Convert to time series and resample if necessary (repo rates might be published at irregular intervals)
        repo_df = repo_df.set_index('date')
        
        # Forward fill for dates where no rate change was announced
        # This creates a daily series where rates are carried forward until the next change
        repo_daily = repo_df.resample('D').ffill()
        
        # Now resample to monthly frequency (taking the last rate of each month)
        repo_monthly = repo_daily.resample('M').last()
        
        # Align with our date range
        repo_monthly = repo_monthly.reindex(date_range)
        
        # Fill any missing values with interpolation
        if repo_monthly['interest_rate'].isna().any():
            print("Interpolating missing repo rate values")
            repo_monthly['interest_rate'] = repo_monthly['interest_rate'].interpolate(method='linear')
        
        interest_rates = repo_monthly['interest_rate']
        
        # Create a processed repo rate CSV for other collectors to use
        repo_df.reset_index().to_csv("data/raw/repo_rate_processed.csv", index=False)
    else:
        print("Warning: Using synthetic interest rate data")
        # Generate synthetic interest rate data as fallback
        np.random.seed(42)
    interest_base = np.random.normal(loc=0, scale=0.1, size=len(date_range))
    interest_rates = np.cumsum(interest_base) / 5 + 6.0
    
    # Calculate rate spread if we have both lending and repo rates but no spread yet
    if rate_spread is None and lending_rate is not None and interest_rates is not None:
        rate_spread = lending_rate - interest_rates
    
    # Load unemployment rate data from Excel file
    unemp_df = load_unemployment_rate("data/raw/unemployment_rate.xlsx")
    
    if unemp_df is not None and len(unemp_df) > 0:
        # Convert to time series and resample to monthly frequency
        unemp_df = unemp_df.set_index('date')
        
        # Check if we need to resample (data might be quarterly or monthly)
        if len(unemp_df) < len(date_range):
            # Resample to monthly frequency
            # For unemployment, interpolation is appropriate as it changes gradually
            unemp_monthly = unemp_df.resample('M').interpolate(method='cubic')
        else:
            unemp_monthly = unemp_df
        
        # Align with our date range
        unemp_monthly = unemp_monthly.reindex(date_range)
        
        # Fill any missing values with interpolation
        if unemp_monthly['unemployment'].isna().any():
            print("Interpolating missing unemployment rate values")
            unemp_monthly['unemployment'] = unemp_monthly['unemployment'].interpolate(method='linear')
        
        unemployment = unemp_monthly['unemployment']
    else:
        print("Warning: Using synthetic unemployment rate data")
        # Generate synthetic unemployment data as fallback
        np.random.seed(42)
    unemployment_base = np.random.normal(loc=0, scale=0.1, size=len(date_range))
    unemployment = np.cumsum(unemployment_base) / 10 + 7.5
    unemployment = unemployment + 0.3 * np.sin(np.arange(len(date_range)) * (2*np.pi/12))
    
    # Load IIP data from Excel files
    iip_df = fetch_iip_data(
        durable_path="data/raw/iip_consumer_durable.xlsx",
        nondurable_path="data/raw/iip_consumer_nondurable.xlsx",
        output_path="data/raw/iip_combined.csv"
    )
    
    if iip_df is not None and len(iip_df) > 0:
        # Convert to time series and align with our date range
        iip_df['date'] = pd.to_datetime(iip_df['date'])
        iip_df = iip_df.set_index('date')
        
        # Check if we need to resample (data might be available at different frequency)
        if len(iip_df) < len(date_range):
            # Resample to monthly frequency
            iip_monthly = iip_df.resample('M').interpolate(method='cubic')
        else:
            iip_monthly = iip_df
            
        # Align with our date range
        iip_monthly = iip_monthly.reindex(date_range)
        
        # Fill any missing values with interpolation
        for col in iip_monthly.columns:
            if iip_monthly[col].isna().any():
                print(f"Interpolating missing {col} values")
                iip_monthly[col] = iip_monthly[col].interpolate(method='linear')
        
        # Extract the columns we'll use for modeling
        iip_durable = iip_monthly['iip_durable']
        iip_nondurable = iip_monthly['iip_nondurable']
        iip_combined = iip_monthly['iip_combined']
        
        # Extract YoY growth rates
        iip_durable_yoy = iip_monthly['iip_durable_yoy']
        iip_nondurable_yoy = iip_monthly['iip_nondurable_yoy']
        iip_combined_yoy = iip_monthly['iip_combined_yoy']
    else:
        print("Warning: Using synthetic IIP data")
        # Generate synthetic IIP data as fallback
        np.random.seed(42)
        # Base values
        iip_durable_base = np.random.normal(loc=0, scale=1.0, size=len(date_range))
        iip_nondurable_base = np.random.normal(loc=0, scale=0.7, size=len(date_range))
        
        # Add trend and seasonality
        trend = np.linspace(0, 20, len(date_range))
        seasonality_durable = 15 * np.sin(np.arange(len(date_range)) * (2*np.pi/12) + np.pi/4)
        seasonality_nondurable = 8 * np.sin(np.arange(len(date_range)) * (2*np.pi/12))
        
        # Create the indices
        iip_durable = 100 + np.cumsum(iip_durable_base) + trend + seasonality_durable
        iip_nondurable = 100 + np.cumsum(iip_nondurable_base) + trend + seasonality_nondurable
        iip_combined = (iip_durable + iip_nondurable) / 2
        
        # Calculate YoY growth rates (simple percentage changes)
        iip_durable_yoy = np.zeros(len(date_range))
        iip_nondurable_yoy = np.zeros(len(date_range))
        iip_combined_yoy = np.zeros(len(date_range))
        
        for i in range(12, len(date_range)):
            iip_durable_yoy[i] = (iip_durable[i] / iip_durable[i-12] - 1) * 100
            iip_nondurable_yoy[i] = (iip_nondurable[i] / iip_nondurable[i-12] - 1) * 100
            iip_combined_yoy[i] = (iip_combined[i] / iip_combined[i-12] - 1) * 100
    
    # Load gold price data from Excel file
    gold_df = fetch_gold_price("data/raw/gold_price.xlsx", "data/raw/gold_price_processed.csv")
    
    if gold_df is not None and len(gold_df) > 0:
        # Convert to time series
        gold_df['date'] = pd.to_datetime(gold_df['date'])
        gold_df = gold_df.set_index('date')
        
        # Check if we need to resample
        if len(gold_df) < len(date_range):
            # Resample to monthly frequency
            gold_monthly = gold_df.resample('M').last()
        else:
            gold_monthly = gold_df
            
        # Align with our date range
        gold_monthly = gold_monthly.reindex(date_range)
        
        # Fill any missing values with interpolation
        for col in gold_monthly.columns:
            if gold_monthly[col].isna().any():
                print(f"Interpolating missing {col} values")
                gold_monthly[col] = gold_monthly[col].interpolate(method='linear')
        
        # Extract the columns we'll use for modeling
        gold_price = gold_monthly['gold_price']
        gold_price_yoy = gold_monthly['gold_price_yoy']
        gold_price_mom = gold_monthly['gold_price_mom']
        gold_trend = gold_monthly['gold_trend']
    else:
        print("Warning: Using synthetic gold price data")
        # Generate synthetic gold price data as fallback
        np.random.seed(42)
        # Base gold price with trend
        gold_base = np.random.normal(loc=0, scale=50, size=len(date_range))
        trend = np.linspace(1000, 2000, len(date_range))  # Upward trend for gold over time
        gold_price = trend + np.cumsum(gold_base) / 10
        
        # Calculate YoY and MoM changes
        gold_price_yoy = np.zeros(len(date_range))
        gold_price_mom = np.zeros(len(date_range))
        
        for i in range(12, len(date_range)):
            gold_price_yoy[i] = (gold_price[i] / gold_price[i-12] - 1) * 100
            
        for i in range(1, len(date_range)):
            gold_price_mom[i] = (gold_price[i] / gold_price[i-1] - 1) * 100
            
        # Simple gold trend indicator (difference from 6-month moving average)
        gold_ma = np.zeros(len(date_range))
        for i in range(len(date_range)):
            if i >= 6:
                gold_ma[i] = np.mean(gold_price[i-6:i])
            else:
                gold_ma[i] = gold_price[i]
                
        gold_trend = gold_price - gold_ma
    
    # Load crude oil price data from Excel file
    oil_df = fetch_crude_oil_price("data/raw/crude_oil_price.xlsx", "data/raw/crude_oil_price_processed.csv")
    
    if oil_df is not None and len(oil_df) > 0:
        # Convert to time series
        oil_df['date'] = pd.to_datetime(oil_df['date'])
        oil_df = oil_df.set_index('date')
        
        # Check if we need to resample
        if len(oil_df) < len(date_range):
            # Resample to monthly frequency
            oil_monthly = oil_df.resample('M').last()
        else:
            oil_monthly = oil_df
            
        # Align with our date range
        oil_monthly = oil_monthly.reindex(date_range)
        
        # Fill any missing values with interpolation
        for col in oil_monthly.columns:
            if oil_monthly[col].isna().any():
                print(f"Interpolating missing {col} values")
                oil_monthly[col] = oil_monthly[col].interpolate(method='linear')
        
        # Extract the columns we'll use for modeling
        oil_price = oil_monthly['oil_price']
        oil_price_yoy = oil_monthly['oil_price_yoy']
        oil_price_mom = oil_monthly['oil_price_mom']
        oil_momentum = oil_monthly['oil_momentum']
    else:
        print("Warning: Using synthetic crude oil price data")
        # Generate synthetic oil price data as fallback
        np.random.seed(43)  # Different seed than gold
        # Base oil price with trend and cyclicality
        oil_base = np.random.normal(loc=0, scale=3, size=len(date_range))
        # Oil prices tend to be more volatile than gold
        oil_price = 60 + np.cumsum(oil_base)
        # Add some cyclicality that's common in oil markets
        oil_price = oil_price + 15 * np.sin(np.arange(len(date_range)) * (2*np.pi/24))
        
        # Calculate YoY and MoM changes
        oil_price_yoy = np.zeros(len(date_range))
        oil_price_mom = np.zeros(len(date_range))
        
        for i in range(12, len(date_range)):
            oil_price_yoy[i] = (oil_price[i] / oil_price[i-12] - 1) * 100
            
        for i in range(1, len(date_range)):
            oil_price_mom[i] = (oil_price[i] / oil_price[i-1] - 1) * 100
            
        # Simple momentum indicator
        oil_ma_short = np.zeros(len(date_range))
        oil_ma_long = np.zeros(len(date_range))
        
        for i in range(len(date_range)):
            if i >= 3:
                oil_ma_short[i] = np.mean(oil_price[i-3:i])
            else:
                oil_ma_short[i] = oil_price[i]
                
            if i >= 9:
                oil_ma_long[i] = np.mean(oil_price[i-9:i])
            else:
                oil_ma_long[i] = oil_price[i]
                
        oil_momentum = oil_ma_short / oil_ma_long - 1
    
    # Create macroeconomic dataframe
    macro_df = pd.DataFrame(index=date_range)
    macro_df['gdp_growth'] = gdp_df['gdp_growth']
    macro_df['inflation'] = inflation
    macro_df['interest_rate'] = interest_rates
    macro_df['lending_rate'] = lending_rate
    if rate_spread is not None:
        macro_df['rate_spread'] = rate_spread
    macro_df['unemployment'] = unemployment
    
    # Add IIP data
    macro_df['iip_durable'] = iip_durable
    macro_df['iip_nondurable'] = iip_nondurable
    macro_df['iip_combined'] = iip_combined
    macro_df['iip_durable_yoy'] = iip_durable_yoy
    macro_df['iip_nondurable_yoy'] = iip_nondurable_yoy
    macro_df['iip_combined_yoy'] = iip_combined_yoy
    
    # Add WPI data
    macro_df['wpi_index'] = wpi_index
    macro_df['wpi_inflation'] = wpi_inflation
    macro_df['wpi_mom'] = wpi_mom
    macro_df['wpi_3m_avg'] = wpi_3m_avg
    macro_df['wpi_6m_avg'] = wpi_6m_avg
    
    # Add Gold price data
    macro_df['gold_price'] = gold_price
    macro_df['gold_price_yoy'] = gold_price_yoy
    macro_df['gold_price_mom'] = gold_price_mom
    macro_df['gold_trend'] = gold_trend
    
    # Add Crude Oil price data
    macro_df['oil_price'] = oil_price
    macro_df['oil_price_yoy'] = oil_price_yoy
    macro_df['oil_price_mom'] = oil_price_mom
    macro_df['oil_momentum'] = oil_momentum
    
    macro_df.reset_index(inplace=True)
    macro_df.rename(columns={'index': 'date'}, inplace=True)
    
    # Generate retail sales data influenced by macroeconomic factors
    # Base retail sales
    retail_base = 1000 + np.cumsum(np.random.normal(loc=5, scale=10, size=len(date_range)))
    
    # Add influence from macro factors
    gdp_monthly = macro_df['gdp_growth'].values
    retail_sales = (
        retail_base +
        # Positive impact from GDP and IIP
        20 * (gdp_monthly - np.mean(gdp_monthly)) +
        10 * (iip_combined_yoy - np.mean(iip_combined_yoy[~np.isnan(iip_combined_yoy)])) +
        # WPI impact (can be positive if it indicates economic growth, negative if it's cost pressure)
        8 * (wpi_mom - np.mean(wpi_mom[~np.isnan(wpi_mom)])) -
        # Negative impact from inflation, interest rates, unemployment, and lending rates
        15 * (inflation - np.mean(inflation)) -
        30 * (interest_rates - np.mean(interest_rates)) -
        20 * (lending_rate - np.mean(lending_rate)) -
        25 * (unemployment - np.mean(unemployment)) -
        # Impact from gold and oil prices
        # Higher oil prices typically have a negative impact on retail sales (cost pressure)
        12 * (oil_price_mom - np.mean(oil_price_mom[~np.isnan(oil_price_mom)])) -
        # Gold can be both positive (wealth effect when prices rise) and negative (inflation hedge)
        # Here we model a small positive impact (reflecting India's cultural affinity for gold)
        5 * (gold_price_mom - np.mean(gold_price_mom[~np.isnan(gold_price_mom)]))
    )
    
    # Add strong seasonality to retail sales (holiday season effects)
    retail_seasonality = 200 * np.sin(np.arange(len(date_range)) * (2*np.pi/12) + np.pi/6)
    retail_sales = retail_sales + retail_seasonality
    
    # Ensure no negative values
    retail_sales = np.maximum(retail_sales, 100)
    
    # Create retail dataframe
    retail_df = pd.DataFrame({
        'date': date_range,
        'retail_sales': retail_sales
    })
    
    # Save to CSV
    macro_df.to_csv('data/macro_indicators.csv', index=False)
    retail_df.to_csv('data/retail_sales.csv', index=False)
    
    print("Data fetching complete. Files saved to data directory.")
    return macro_df, retail_df

def explore_data(macro_df, retail_df):
    """
    Perform exploratory data analysis on the datasets.
    """
    print("Exploring data...")
    
    # Merge datasets on date
    full_data = pd.merge(macro_df, retail_df, on='date')
    
    # Set date as index
    full_data.set_index('date', inplace=True)
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(full_data.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(full_data.isnull().sum())
    
    # Calculate correlations
    print("\nCorrelation Matrix:")
    correlation = full_data.corr()
    print(correlation['retail_sales'].sort_values(ascending=False))
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Economic Indicators and Retail Sales')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png')
    
    # Visualize time series
    plt.figure(figsize=(15, 10))
    
    # Plot macroeconomic indicators
    plt.subplot(3, 1, 1)
    plt.plot(full_data.index, full_data['gdp_growth'], label='GDP Growth (%)')
    plt.plot(full_data.index, full_data['inflation'], label='Inflation (%)')
    plt.plot(full_data.index, full_data['interest_rate'], label='Interest Rate (%)')
    plt.legend()
    plt.title('Macroeconomic Indicators Over Time')
    
    plt.subplot(3, 1, 2)
    plt.plot(full_data.index, full_data['unemployment'], label='Unemployment (%)')
    plt.legend()
    plt.title('Unemployment Rate Over Time')
    
    plt.subplot(3, 1, 3)
    plt.plot(full_data.index, full_data['retail_sales'], color='red', label='Retail Sales')
    plt.legend()
    plt.title('Retail Sales Over Time')
    
    plt.tight_layout()
    plt.savefig('visualizations/time_series_overview.png')
    
    return full_data

def feature_engineering(full_data):
    """
    Create additional features from the raw data.
    """
    print("Engineering features...")
    
    # Make a copy of the data
    df = full_data.copy()
    
    # Create lag features (previous months' values)
    for col in ['gdp_growth', 'inflation', 'interest_rate', 'unemployment']:
        for lag in [1, 3, 6, 12]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Create rolling mean features
    for col in ['gdp_growth', 'inflation', 'interest_rate', 'unemployment']:
        for window in [3, 6, 12]:
            df[f'{col}_rolling_{window}'] = df[col].rolling(window=window).mean()
    
    # Create rolling standard deviation (volatility)
    for col in ['gdp_growth', 'inflation', 'interest_rate']:
        for window in [3, 6, 12]:
            df[f'{col}_volatility_{window}'] = df[col].rolling(window=window).std()
    
    # Create rate of change features
    for col in ['gdp_growth', 'inflation', 'interest_rate', 'unemployment']:
        df[f'{col}_pct_change'] = df[col].pct_change() * 100
        df[f'{col}_diff'] = df[col].diff()
    
    # Create seasonal features
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    
    # Create cyclical encoding of month
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # Create interaction features
    df['gdp_inflation_ratio'] = df['gdp_growth'] / df['inflation']
    df['consumer_interest_diff'] = df['interest_rate'] * 10
    
    # Create a "misery index" (unemployment + inflation)
    df['misery_index'] = df['unemployment'] + df['inflation']
    
    # Drop rows with NaN values (created by lagging and rolling operations)
    df.dropna(inplace=True)
    
    print(f"Features created. Dataset shape: {df.shape}")
    
    # Save engineered features
    df.to_csv('data/engineered_features.csv')
    
    return df

def main():
    """
    Main function to execute the pipeline.
    """
    print("Starting India Retail Demand Forecaster project...")
    
    # Step 1: Fetch Data
    macro_df, retail_df = fetch_data()
    
    # Step 2: Explore Data
    full_data = explore_data(macro_df, retail_df)
    
    # Step 3: Feature Engineering
    features_df = feature_engineering(full_data)
    
    print("Initial setup complete!")
    
if __name__ == "__main__":
    main()
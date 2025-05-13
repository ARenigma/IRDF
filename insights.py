"""
India Retail Demand Forecaster - Insights Generator
--------------------------------------------------
This module analyzes relationships between key economic indicators and retail demand,
with a special focus on the impact of gold prices and crude oil prices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def load_data_sources():
    """
    Loads individual data sources without relying on the full pipeline.
    
    Returns:
    --------
    dict
        Dictionary of DataFrames with each data source
    """
    print("Loading individual data sources...")
    
    data_sources = {}
    
    # Define paths for key data sources
    file_paths = {
        'retail': 'data/retail_sales.csv',
        'gold': 'data/raw/gold_price_processed.csv',
        'oil': 'data/raw/crude_oil_price_processed.csv',
        'macro': 'data/macro_indicators.csv',
    }
    
    # Load each data source
    for source, path in file_paths.items():
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                # Convert date column to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    data_sources[source] = df
                    print(f"Loaded {source} data: {df.shape[0]} rows, {df.shape[1]} columns")
            else:
                print(f"Warning: {path} not found, skipping {source}")
        except Exception as e:
            print(f"Error loading {source} data: {e}")
    
    return data_sources

def merge_needed_signals(data_sources):
    """
    Merges only the required signals for our insights, with simpler error handling.
    
    Parameters:
    -----------
    data_sources : dict
        Dictionary of DataFrames from load_data_sources
        
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with signals needed for analysis
    """
    print("Merging selected signals...")
    
    # Start with retail data if available
    merged_df = None
    if 'retail' in data_sources:
        merged_df = data_sources['retail'].copy()
        print(f"Starting with retail data ({merged_df.shape[0]} rows)")
    else:
        print("Warning: Retail data not found, insights will be limited")
        # Create an empty dataframe with a datetime index
        merged_df = pd.DataFrame(index=pd.date_range(start='2015-01-01', end='2023-12-31', freq='M'))
    
    # Add gold price data if available
    if 'gold' in data_sources:
        gold_df = data_sources['gold']
        if merged_df.index.name == gold_df.index.name:
            # Merge on index
            merged_df = merged_df.join(gold_df, how='outer')
        else:
            # If indices don't match, reset indices and merge on date
            merged_df = merged_df.reset_index()
            gold_df = gold_df.reset_index()
            merged_df = pd.merge(merged_df, gold_df, on='date', how='outer')
            merged_df.set_index('date', inplace=True)
        print(f"Added gold price data")
    
    # Add crude oil data if available
    if 'oil' in data_sources:
        oil_df = data_sources['oil']
        if merged_df.index.name == oil_df.index.name:
            # Merge on index
            merged_df = merged_df.join(oil_df, how='outer')
        else:
            # If indices don't match, reset indices and merge on date
            merged_df = merged_df.reset_index()
            oil_df = oil_df.reset_index()
            merged_df = pd.merge(merged_df, oil_df, on='date', how='outer')
            merged_df.set_index('date', inplace=True)
        print(f"Added crude oil price data")
    
    # Add macro data if available
    if 'macro' in data_sources:
        macro_df = data_sources['macro']
        if merged_df.index.name == macro_df.index.name:
            # Merge on index
            merged_df = merged_df.join(macro_df, how='outer')
        else:
            # If indices don't match, reset indices and merge on date
            merged_df = merged_df.reset_index()
            macro_df = macro_df.reset_index()
            merged_df = pd.merge(merged_df, macro_df, on='date', how='outer')
            merged_df.set_index('date', inplace=True)
        print(f"Added macro indicator data")
    
    # Sort by date and handle missing values with simple interpolation
    merged_df = merged_df.sort_index()
    for col in merged_df.columns:
        if merged_df[col].dtype.kind in 'fc' and merged_df[col].isna().any():
            merged_df[col] = merged_df[col].interpolate(method='linear')
    
    print(f"Final merged dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    
    # Save merged dataset
    os.makedirs('data/processed', exist_ok=True)
    merged_df.to_csv('data/processed/insights_data.csv')
    
    return merged_df

def analyze_gold_price_impact(df):
    """
    Analyzes the impact of gold prices on retail demand in India.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged DataFrame containing gold price and retail sales data
    """
    print("\nAnalyzing gold price impact on retail demand...")
    
    # Check if necessary columns exist
    gold_cols = [col for col in df.columns if 'gold' in col.lower()]
    if not gold_cols:
        print("Warning: No gold price columns found in dataset")
        return
    
    if 'retail_sales' not in df.columns:
        print("Warning: Retail sales column not found in dataset")
        return
    
    # Create directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Correlation analysis
    corr_data = pd.DataFrame()
    corr_data['retail_sales'] = df['retail_sales']
    for col in gold_cols:
        corr_data[col] = df[col]
    
    correlation = corr_data.corr()
    
    # Print correlations with retail sales
    print("\nGold Price Correlations with Retail Sales:")
    gold_retail_corr = correlation['retail_sales'].drop('retail_sales').sort_values(ascending=False)
    for col, corr in gold_retail_corr.items():
        print(f"{col}: {corr:.4f}")
    
    # Visualize correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Between Gold Prices and Retail Sales')
    plt.tight_layout()
    plt.savefig('visualizations/gold_retail_correlation.png')
    plt.close()
    
    # 2. Time series comparison
    plt.figure(figsize=(12, 6))
    
    # Select the most relevant gold price column
    main_gold_col = [col for col in gold_cols if 'price' in col and not ('yoy' in col or 'mom' in col)]
    if main_gold_col:
        main_gold_col = main_gold_col[0]
    else:
        main_gold_col = gold_cols[0]
    
    # Normalize to the same scale for visualization
    gold_normalized = (df[main_gold_col] - df[main_gold_col].min()) / (df[main_gold_col].max() - df[main_gold_col].min())
    retail_normalized = (df['retail_sales'] - df['retail_sales'].min()) / (df['retail_sales'].max() - df['retail_sales'].min())
    
    # Plot
    plt.plot(gold_normalized.index, gold_normalized, 'g-', label=f'{main_gold_col} (normalized)')
    plt.plot(retail_normalized.index, retail_normalized, 'b-', label='Retail Sales (normalized)')
    
    plt.title('Gold Price vs. Retail Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/gold_retail_time_series.png')
    plt.close()
    
    # 3. Lag analysis
    lag_correlation = []
    max_lag = 12  # Up to 12 months lag
    
    for lag in range(max_lag + 1):
        if lag == 0:
            lag_corr = df[main_gold_col].corr(df['retail_sales'])
        else:
            lag_corr = df[main_gold_col].shift(lag).corr(df['retail_sales'])
        lag_correlation.append((lag, lag_corr))
    
    lags, correlations = zip(*lag_correlation)
    
    plt.figure(figsize=(10, 6))
    plt.bar(lags, correlations)
    plt.title('Correlation Between Gold Price and Retail Sales at Different Lags')
    plt.xlabel('Lag (months)')
    plt.ylabel('Correlation')
    plt.xticks(lags)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/gold_retail_lag_correlation.png')
    plt.close()
    
    # Find the lag with the strongest correlation
    max_corr_lag = max(lag_correlation, key=lambda x: abs(x[1]))
    print(f"\nStrongest correlation between gold price and retail sales at lag {max_corr_lag[0]} months: {max_corr_lag[1]:.4f}")
    
    # 4. Seasonal analysis
    monthly_gold = df[main_gold_col].groupby(df.index.month).mean()
    monthly_retail = df['retail_sales'].groupby(df.index.month).mean()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Gold price on left y-axis
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Average Gold Price', color='g')
    ax1.plot(monthly_gold.index, monthly_gold.values, 'g-', marker='o')
    ax1.tick_params(axis='y', labelcolor='g')
    
    # Retail sales on right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Retail Sales', color='b')
    ax2.plot(monthly_retail.index, monthly_retail.values, 'b-', marker='s')
    ax2.tick_params(axis='y', labelcolor='b')
    
    plt.title('Seasonal Patterns: Gold Price vs. Retail Sales')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    fig.tight_layout()
    plt.savefig('visualizations/gold_retail_seasonality.png')
    plt.close()
    
    print("\nGold price impact analysis complete. Visualizations saved.")

def analyze_oil_price_impact(df):
    """
    Analyzes the impact of crude oil prices on retail demand in India.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged DataFrame containing crude oil price and retail sales data
    """
    print("\nAnalyzing crude oil price impact on retail demand...")
    
    # Check if necessary columns exist
    oil_cols = [col for col in df.columns if 'oil' in col.lower()]
    if not oil_cols:
        print("Warning: No crude oil price columns found in dataset")
        return
    
    if 'retail_sales' not in df.columns:
        print("Warning: Retail sales column not found in dataset")
        return
    
    # Create directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Correlation analysis
    corr_data = pd.DataFrame()
    corr_data['retail_sales'] = df['retail_sales']
    for col in oil_cols:
        corr_data[col] = df[col]
    
    correlation = corr_data.corr()
    
    # Print correlations with retail sales
    print("\nCrude Oil Price Correlations with Retail Sales:")
    oil_retail_corr = correlation['retail_sales'].drop('retail_sales').sort_values(ascending=False)
    for col, corr in oil_retail_corr.items():
        print(f"{col}: {corr:.4f}")
    
    # Visualize correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Between Crude Oil Prices and Retail Sales')
    plt.tight_layout()
    plt.savefig('visualizations/oil_retail_correlation.png')
    plt.close()
    
    # 2. Time series comparison
    plt.figure(figsize=(12, 6))
    
    # Select the most relevant oil price column
    main_oil_col = [col for col in oil_cols if 'price' in col and not ('yoy' in col or 'mom' in col)]
    if main_oil_col:
        main_oil_col = main_oil_col[0]
    else:
        main_oil_col = oil_cols[0]
    
    # Normalize to the same scale for visualization
    oil_normalized = (df[main_oil_col] - df[main_oil_col].min()) / (df[main_oil_col].max() - df[main_oil_col].min())
    retail_normalized = (df['retail_sales'] - df['retail_sales'].min()) / (df['retail_sales'].max() - df['retail_sales'].min())
    
    # Plot
    plt.plot(oil_normalized.index, oil_normalized, 'r-', label=f'{main_oil_col} (normalized)')
    plt.plot(retail_normalized.index, retail_normalized, 'b-', label='Retail Sales (normalized)')
    
    plt.title('Crude Oil Price vs. Retail Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/oil_retail_time_series.png')
    plt.close()
    
    # 3. Lag analysis
    lag_correlation = []
    max_lag = 12  # Up to 12 months lag
    
    for lag in range(max_lag + 1):
        if lag == 0:
            lag_corr = df[main_oil_col].corr(df['retail_sales'])
        else:
            lag_corr = df[main_oil_col].shift(lag).corr(df['retail_sales'])
        lag_correlation.append((lag, lag_corr))
    
    lags, correlations = zip(*lag_correlation)
    
    plt.figure(figsize=(10, 6))
    plt.bar(lags, correlations)
    plt.title('Correlation Between Oil Price and Retail Sales at Different Lags')
    plt.xlabel('Lag (months)')
    plt.ylabel('Correlation')
    plt.xticks(lags)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/oil_retail_lag_correlation.png')
    plt.close()
    
    # Find the lag with the strongest correlation
    max_corr_lag = max(lag_correlation, key=lambda x: abs(x[1]))
    print(f"\nStrongest correlation between oil price and retail sales at lag {max_corr_lag[0]} months: {max_corr_lag[1]:.4f}")
    
    print("\nCrude oil price impact analysis complete. Visualizations saved.")

def analyze_combined_commodities_impact(df):
    """
    Analyzes the combined impact of gold and oil prices on retail demand.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged DataFrame containing gold price, oil price, and retail sales data
    """
    print("\nAnalyzing combined commodity price impact on retail demand...")
    
    # Check if necessary columns exist
    gold_cols = [col for col in df.columns if 'gold' in col.lower() and 'price' in col.lower() and not ('yoy' in col or 'mom' in col)]
    oil_cols = [col for col in df.columns if 'oil' in col.lower() and 'price' in col.lower() and not ('yoy' in col or 'mom' in col)]
    
    if not gold_cols or not oil_cols:
        print("Warning: Missing gold or oil price columns")
        return
    
    if 'retail_sales' not in df.columns:
        print("Warning: Retail sales column not found")
        return
    
    # Select main columns
    main_gold_col = gold_cols[0]
    main_oil_col = oil_cols[0]
    
    # First, let's make sure we're only working with data points that have values for all three variables
    analysis_df = df.copy()
    analysis_df = analysis_df.dropna(subset=[main_gold_col, main_oil_col, 'retail_sales'])
    
    if len(analysis_df) < 10:
        print(f"Warning: Not enough complete data points for analysis (only {len(analysis_df)} rows). Skipping regression analysis.")
        return
    
    print(f"Using {len(analysis_df)} complete data points for analysis")
    
    # 1. Create a scatter plot with retail sales as dependent variable
    plt.figure(figsize=(10, 8))
    
    # Create a colormap based on retail sales
    sc = plt.scatter(
        analysis_df[main_gold_col], 
        analysis_df[main_oil_col], 
        c=analysis_df['retail_sales'], 
        cmap='viridis', 
        alpha=0.7,
        s=50
    )
    
    plt.colorbar(sc, label='Retail Sales')
    plt.title('Gold Price vs. Oil Price Colored by Retail Sales')
    plt.xlabel('Gold Price')
    plt.ylabel('Crude Oil Price')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/gold_oil_retail_scatter.png')
    plt.close()
    
    # 2. Create a 3D scatter plot
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(
            analysis_df[main_gold_col],
            analysis_df[main_oil_col],
            analysis_df['retail_sales'],
            c=analysis_df.index.month,
            cmap='hsv',
            s=50,
            alpha=0.7
        )
        
        ax.set_xlabel('Gold Price')
        ax.set_ylabel('Crude Oil Price')
        ax.set_zlabel('Retail Sales')
        ax.set_title('3D Relationship: Gold, Oil, and Retail Sales')
        
        plt.tight_layout()
        plt.savefig('visualizations/gold_oil_retail_3d.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create 3D plot: {e}")
    
    # 3. Create a simple regression analysis report
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data - we're using the already filtered analysis_df
    X = analysis_df[[main_gold_col, main_oil_col]].copy()
    y = analysis_df['retail_sales'].copy()
    
    # Double-check for any remaining NaNs just to be safe
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    # Verify we still have enough data
    if len(X) < 10:
        print(f"Warning: After removing NaNs, not enough data points left (only {len(X)} rows). Skipping regression.")
        return
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit regression model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Get coefficients and intercept
    gold_coef = model.coef_[0]
    oil_coef = model.coef_[1]
    intercept = model.intercept_
    
    # Calculate R-squared
    r_squared = model.score(X_scaled, y)
    
    # Display regression results
    print("\nRegression Analysis Results:")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Coefficients:")
    print(f"  - Gold Price: {gold_coef:.4f}")
    print(f"  - Oil Price: {oil_coef:.4f}")
    print(f"Intercept: {intercept:.4f}")
    
    # Calculate relative importance
    abs_coeffs = np.abs([gold_coef, oil_coef])
    rel_importance = abs_coeffs / np.sum(abs_coeffs) * 100
    
    print(f"\nRelative Importance in Explaining Retail Sales:")
    print(f"  - Gold Price: {rel_importance[0]:.2f}%")
    print(f"  - Oil Price: {rel_importance[1]:.2f}%")
    
    # Visualize coefficient importance
    plt.figure(figsize=(8, 6))
    plt.bar(['Gold Price', 'Oil Price'], abs_coeffs)
    plt.title('Absolute Impact on Retail Sales')
    plt.ylabel('Coefficient Magnitude')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/gold_oil_coefficient_impact.png')
    plt.close()
    
    print("\nCombined commodity impact analysis complete. Visualizations saved.")

def main():
    """
    Main function to run the insights analysis.
    """
    print("Starting insights generation...")
    
    # Load data sources
    data_sources = load_data_sources()
    
    # Merge needed signals
    df = merge_needed_signals(data_sources)
    
    # Analyze gold price impact
    analyze_gold_price_impact(df)
    
    # Analyze oil price impact
    analyze_oil_price_impact(df)
    
    # Analyze combined impact
    analyze_combined_commodities_impact(df)
    
    print("\nInsights generation complete!")

if __name__ == "__main__":
    main()
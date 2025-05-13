"""
India Retail Demand Forecaster - Data Analysis and Cleaning
----------------------------------------------------------
This module provides functions for analyzing data quality issues and implementing
advanced cleaning strategies for the retail demand forecasting project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dateutil.parser import parse
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

def analyze_missingness(df: pd.DataFrame, 
                       output_path: str = 'visualizations/missing_values_heatmap.png') -> pd.DataFrame:
    """
    Analyzes missing values in the dataset and creates a visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to analyze
    output_path : str
        Path to save the missing values heatmap
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics of missing values
    """
    print("Analyzing missing values...")
    
    # Calculate missing values statistics
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_stats = pd.DataFrame({
        'Missing Values': missing_values,
        'Missing Percent': missing_percent.round(2)
    }).sort_values('Missing Percent', ascending=False)
    
    # Print summary
    print("\nMissing Values Summary:")
    print(missing_stats[missing_stats['Missing Values'] > 0])
    
    # Create missing values heatmap
    if not missing_stats.empty and missing_stats['Missing Values'].max() > 0:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        
        # Create missing values by column bar chart
        missing_cols = missing_stats[missing_stats['Missing Values'] > 0]
        if not missing_cols.empty:
            plt.figure(figsize=(12, 6))
            missing_cols['Missing Percent'].plot(kind='bar')
            plt.title('Percentage of Missing Values by Column')
            plt.ylabel('Percent Missing')
            plt.xlabel('Column')
            plt.tight_layout()
            plt.savefig('visualizations/missing_values_by_column.png')
            plt.close()
    
    return missing_stats

def analyze_data_quality(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Performs a comprehensive data quality analysis including outliers,
    inconsistencies, and potential errors.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to analyze
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of data quality issue summaries
    """
    print("Analyzing data quality issues...")
    
    # Initialize results dictionary
    results = {}
    
    # 1. Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    # 2. Look for outliers using Z-scores
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    z_scores = pd.DataFrame(index=df.index)
    
    for col in numeric_cols:
        z_scores[col] = np.abs((df[col] - df[col].mean()) / df[col].std())
    
    # Flag outliers with Z-score > 3
    outliers = (z_scores > 3).sum()
    outlier_summary = pd.DataFrame({
        'Outliers Count': outliers,
        'Outliers Percent': (outliers / len(df) * 100).round(2)
    }).sort_values('Outliers Count', ascending=False)
    
    print("\nOutliers Summary (Z-score > 3):")
    print(outlier_summary[outlier_summary['Outliers Count'] > 0])
    results['outliers'] = outlier_summary
    
    # 3. Check for suspicious data patterns or inconsistencies
    # For time series data, check for large jumps
    jumps = {}
    for col in numeric_cols:
        if col in df.columns:  # Ensure column exists
            pct_change = df[col].pct_change().abs()
            large_jumps = (pct_change > 0.5).sum()  # 50% change threshold
            jumps[col] = large_jumps
    
    jumps_summary = pd.DataFrame({
        'Large Jumps Count': jumps,
        'Large Jumps Percent': {k: (v / len(df) * 100).round(2) for k, v in jumps.items()}
    }).sort_values('Large Jumps Count', ascending=False)
    
    print("\nLarge Value Jumps Summary (>50% change):")
    print(jumps_summary[jumps_summary['Large Jumps Count'] > 0])
    results['jumps'] = jumps_summary
    
    # 4. Check for data inconsistencies using basic statistics
    stats = df.describe().T
    stats['cv'] = stats['std'] / stats['mean']  # Coefficient of variation
    
    # Flag columns with unusually high coefficient of variation
    high_cv = stats[stats['cv'] > 1].sort_values('cv', ascending=False)
    print("\nColumns with high variability (CV > 1):")
    print(high_cv)
    results['high_variability'] = high_cv
    
    # 5. Create visualizations
    if not df.empty:
        # Time series plots for numeric columns
        if isinstance(df.index, pd.DatetimeIndex):
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(numeric_cols[:min(len(numeric_cols), 9)]):
                if col in df.columns:
                    plt.subplot(3, 3, i+1)
                    plt.plot(df.index, df[col])
                    plt.title(col)
                    plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('visualizations/data_quality_time_series.png')
            plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('visualizations/data_quality_correlation.png')
        plt.close()
    
    return results

def time_alignment_analysis(dfs: Dict[str, pd.DataFrame], 
                           date_column: str = 'date') -> Dict[str, pd.DataFrame]:
    """
    Analyzes data sources for time alignment issues and gaps.
    
    Parameters:
    -----------
    dfs : Dict[str, pd.DataFrame]
        Dictionary of dataframes to analyze, keyed by data source name
    date_column : str
        Name of the date column in each dataframe
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Time alignment analysis results
    """
    print("Analyzing time alignment across data sources...")
    
    # Initialize results
    results = {}
    
    # Extract date ranges for each data source
    date_ranges = {}
    frequencies = {}
    
    for source, df in dfs.items():
        if date_column in df.columns:
            df_dates = pd.to_datetime(df[date_column])
            start_date = df_dates.min()
            end_date = df_dates.max()
            date_ranges[source] = (start_date, end_date)
            
            # Estimate data frequency
            if len(df_dates) > 1:
                date_diffs = df_dates.sort_values().diff().dropna()
                median_diff = date_diffs.median()
                frequencies[source] = median_diff
                
                # Check for gaps in time series
                large_gaps = date_diffs[date_diffs > median_diff * 2]
                if not large_gaps.empty:
                    print(f"\nFound {len(large_gaps)} large gaps in {source}:")
                    # Simplified approach to just report the gaps without trying to find adjacent dates
                    for i, (date, gap) in enumerate(large_gaps.items()):
                        print(f"  - Gap of {gap.days} days at {date.strftime('%Y-%m-%d')}")
    
    # Create summary of date ranges
    if date_ranges:
        date_range_df = pd.DataFrame.from_dict(
            {source: {'Start Date': dr[0], 'End Date': dr[1]} for source, dr in date_ranges.items()},
            orient='index'
        )
        date_range_df['Duration (days)'] = (date_range_df['End Date'] - date_range_df['Start Date']).dt.days
        
        if frequencies:
            # Safely handle various TimeDelta formats
            freq_series = {}
            for source, freq in frequencies.items():
                if pd.notna(freq) and hasattr(freq, 'days'):
                    freq_series[source] = f"{freq.days} days"
                elif pd.notna(freq):
                    # In case it's a different time format
                    freq_series[source] = f"{freq}"
                else:
                    freq_series[source] = "unknown"
                    
            date_range_df['Estimated Frequency'] = pd.Series(freq_series)
        
        print("\nDate Range Summary by Data Source:")
        print(date_range_df)
        results['date_ranges'] = date_range_df
        
        # Visualize time coverage
        plt.figure(figsize=(12, 6))
        for i, (source, (start, end)) in enumerate(date_ranges.items()):
            plt.plot([start, end], [i, i], 'o-', linewidth=2, label=source)
        plt.yticks(range(len(date_ranges)), date_ranges.keys())
        plt.xlabel('Date')
        plt.title('Time Coverage by Data Source')
        plt.grid(True, axis='x')
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/time_coverage.png')
        plt.close()
    
    return results

def advanced_data_cleaning(df: pd.DataFrame, 
                          methods: Dict[str, Dict] = None) -> pd.DataFrame:
    """
    Applies advanced data cleaning techniques to the dataset including
    handling missing values, outliers, and inconsistencies.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to clean
    methods : Dict[str, Dict]
        Dictionary specifying cleaning methods for each column
        Example: {'column_name': {'missing': 'interpolate', 'outliers': 'clip'}}
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    print("Applying advanced data cleaning...")
    
    # Create a copy to avoid modifying the original dataframe
    cleaned_df = df.copy()
    
    # Default methods if none specified
    if methods is None:
        # Default methods for different column types
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower()]
        
        methods = {}
        # For numeric columns
        for col in numeric_cols:
            methods[col] = {'missing': 'interpolate', 'outliers': 'winsorize'}
        
        # For date columns
        for col in date_cols:
            methods[col] = {'missing': 'none'}  # Don't try to fill date columns
    
    # Apply cleaning methods column by column
    for col, col_methods in methods.items():
        if col not in cleaned_df.columns:
            print(f"Warning: Column {col} not found in dataframe, skipping")
            continue
        
        # Handle missing values
        if 'missing' in col_methods and cleaned_df[col].isna().any():
            method = col_methods['missing']
            missing_count = cleaned_df[col].isna().sum()
            
            if method == 'interpolate':
                cleaned_df[col] = cleaned_df[col].interpolate(method='linear')
                print(f"Interpolated {missing_count} missing values in {col}")
            elif method == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                print(f"Filled {missing_count} missing values with mean in {col}")
            elif method == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                print(f"Filled {missing_count} missing values with median in {col}")
            elif method == 'mode':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
                print(f"Filled {missing_count} missing values with mode in {col}")
            elif method == 'forward':
                cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
                print(f"Forward-filled {missing_count} missing values in {col}")
            elif method == 'backward':
                cleaned_df[col] = cleaned_df[col].fillna(method='bfill')
                print(f"Backward-filled {missing_count} missing values in {col}")
            elif method == 'remove':
                cleaned_df = cleaned_df.dropna(subset=[col])
                print(f"Removed {missing_count} rows with missing values in {col}")
            elif method != 'none':
                print(f"Warning: Unknown missing value method {method} for column {col}")
        
        # Handle outliers for numeric columns
        if 'outliers' in col_methods and cleaned_df[col].dtype in ['float64', 'int64']:
            method = col_methods['outliers']
            
            # Calculate z-scores
            z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
            outliers = (z_scores > 3).sum()
            
            if outliers > 0:
                if method == 'winsorize':
                    # Cap at 3 standard deviations
                    upper_limit = cleaned_df[col].mean() + 3 * cleaned_df[col].std()
                    lower_limit = cleaned_df[col].mean() - 3 * cleaned_df[col].std()
                    cleaned_df[col] = cleaned_df[col].clip(lower=lower_limit, upper=upper_limit)
                    print(f"Winsorized {outliers} outliers in {col}")
                elif method == 'remove':
                    cleaned_df = cleaned_df[z_scores <= 3]
                    print(f"Removed {outliers} rows with outliers in {col}")
                elif method == 'median':
                    median = cleaned_df[col].median()
                    cleaned_df.loc[z_scores > 3, col] = median
                    print(f"Replaced {outliers} outliers with median in {col}")
                elif method != 'none':
                    print(f"Warning: Unknown outlier method {method} for column {col}")
    
    # Check for any remaining missing values
    remaining_missing = cleaned_df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} missing values remain after cleaning")
    
    return cleaned_df

def build_complete_dataset(data_paths: Dict[str, str], 
                          output_path: str = 'data/processed/complete_dataset.csv',
                          apply_cleaning: bool = True) -> pd.DataFrame:
    """
    Loads all data sources, performs quality analysis, applies advanced cleaning,
    and creates a complete dataset for modeling.
    
    Parameters:
    -----------
    data_paths : Dict[str, str]
        Dictionary mapping data source names to file paths
    output_path : str
        Path to save the complete dataset
    apply_cleaning : bool
        Whether to apply advanced cleaning techniques
        
    Returns:
    --------
    pd.DataFrame
        Complete cleaned dataset ready for modeling
    """
    print("Building complete dataset from all sources...")
    
    # Load all data sources
    data_sources = {}
    for source, path in data_paths.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                data_sources[source] = df
                print(f"Loaded {source} data: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"Error loading {source} data from {path}: {e}")
        else:
            print(f"Warning: File {path} not found, skipping {source} data")
    
    if not data_sources:
        raise ValueError("No valid data sources could be loaded")
    
    # Analyze time alignment across sources
    time_analysis = time_alignment_analysis(data_sources)
    
    # Merge all data sources on date
    merged_df = None
    for source, df in data_sources.items():
        if 'date' not in df.columns:
            print(f"Warning: No 'date' column found in {source}, skipping")
            continue
            
        if merged_df is None:
            merged_df = df.copy()
        else:
            merged_df = pd.merge(merged_df, df, on='date', how='outer')
    
    if merged_df is None:
        raise ValueError("Failed to create merged dataset")
    
    # Set date as index for analysis and cleaning
    merged_df.set_index('date', inplace=True)
    
    # Perform data quality analysis
    analyze_missingness(merged_df)
    quality_analysis = analyze_data_quality(merged_df)
    
    # Apply advanced cleaning if requested
    if apply_cleaning:
        cleaned_df = advanced_data_cleaning(merged_df)
    else:
        cleaned_df = merged_df
    
    # Reset index for saving
    cleaned_df.reset_index(inplace=True)
    
    # Save complete dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"Complete dataset saved to {output_path}")
    
    return cleaned_df

def main():
    """
    Main function to execute the data analysis and cleaning pipeline.
    """
    print("Starting data analysis and cleaning process...")
    
    # Define data paths
    data_paths = {
        'retail': 'data/retail_sales.csv',
        'macro': 'data/macro_indicators.csv',
        'gold': 'data/raw/gold_price_processed.csv',
        'oil': 'data/raw/crude_oil_price_processed.csv',
        'iip': 'data/raw/iip_combined.csv',
        'cpi': 'data/raw/cpi.csv',
        'lending': 'data/raw/lending_rate_processed.csv',
        'wpi': 'data/raw/wpi_processed.csv'
    }
    
    # Build complete dataset
    complete_df = build_complete_dataset(data_paths)
    
    print("Data analysis and cleaning complete!")
    
if __name__ == "__main__":
    main() 
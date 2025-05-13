"""
India Retail Demand Forecaster - Data Cleanup Utilities
------------------------------------------------------
This module provides utilities for cleaning and preprocessing raw data files
before they are used in the main pipeline.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union, Tuple
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def scan_data_quality(directory: str = 'data/raw', report_file: str = 'data/data_quality_report.csv') -> pd.DataFrame:
    """
    Scans all CSV files in a directory and provides a data quality report.
    
    Parameters:
    -----------
    directory : str
        Directory containing CSV files to scan
    report_file : str
        Path to save the data quality report
        
    Returns:
    --------
    pd.DataFrame
        Data quality report
    """
    print(f"Scanning data quality in {directory}...")
    
    # Get all CSV files in the directory
    csv_files = glob.glob(f"{directory}/*.csv")
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return pd.DataFrame()
    
    # Initialize results list
    results = []
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"Analyzing {file_name}...")
        
        try:
            # Read the file
            df = pd.read_csv(file_path)
            
            # Get basic stats
            row_count = len(df)
            col_count = len(df.columns)
            
            # Check for date column
            date_col = None
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        pd.to_datetime(df[col])
                        date_col = col
                        break
                    except:
                        continue
            
            # Calculate missing values stats
            missing_stats = df.isna().sum()
            missing_cols = (missing_stats > 0).sum()
            total_missing = missing_stats.sum()
            missing_pct = (total_missing / (row_count * col_count)) * 100
            
            # Calculate outlier stats (using Z-score > 3)
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            outlier_counts = {}
            total_outliers = 0
            
            for col in numeric_cols:
                if df[col].isna().all():
                    continue
                    
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = (z_scores > 3).sum()
                outlier_counts[col] = outliers
                total_outliers += outliers
            
            outlier_pct = (total_outliers / (row_count * len(numeric_cols))) * 100 if len(numeric_cols) > 0 else 0
            
            # Add to results
            results.append({
                'file_name': file_name,
                'row_count': row_count,
                'column_count': col_count,
                'date_column': date_col if date_col else 'Not found',
                'missing_columns': missing_cols,
                'missing_values': total_missing,
                'missing_pct': round(missing_pct, 2),
                'outlier_count': total_outliers,
                'outlier_pct': round(outlier_pct, 2),
                'needs_cleaning': (missing_pct > 0 or outlier_pct > 1)
            })
            
        except Exception as e:
            print(f"Error analyzing {file_name}: {e}")
            results.append({
                'file_name': file_name,
                'row_count': 0,
                'column_count': 0,
                'date_column': 'Error',
                'missing_columns': 0,
                'missing_values': 0,
                'missing_pct': 0,
                'outlier_count': 0,
                'outlier_pct': 0,
                'needs_cleaning': True,
                'error': str(e)
            })
    
    # Create report DataFrame
    report_df = pd.DataFrame(results)
    
    # Save report
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    report_df.to_csv(report_file, index=False)
    
    print(f"Data quality report saved to {report_file}")
    
    # Print summary
    files_needing_cleanup = report_df[report_df['needs_cleaning']]['file_name'].tolist()
    if files_needing_cleanup:
        print(f"\nFiles requiring cleanup ({len(files_needing_cleanup)}):")
        for file in files_needing_cleanup:
            print(f"- {file}")
    else:
        print("\nAll files appear to be clean (no missing values or significant outliers).")
    
    return report_df

def clean_dataset(file_path: str, output_dir: str = 'data/processed', 
                 methods: Dict[str, str] = None, visualize: bool = True) -> pd.DataFrame:
    """
    Cleans a single dataset with configurable methods.
    
    Parameters:
    -----------
    file_path : str
        Path to the file to clean
    output_dir : str
        Directory to save the cleaned file
    methods : Dict[str, str]
        Dictionary of cleaning methods to apply to specific columns.
        Keys are column names, values are methods:
        - 'drop': Drop rows with NaN in this column
        - 'mean': Replace NaNs with mean
        - 'median': Replace NaNs with median
        - 'mode': Replace NaNs with mode
        - 'forward': Forward fill
        - 'backward': Backward fill
        - 'interpolate': Linear interpolation
        - 'knn': KNN imputation
        - 'zero': Replace NaNs with 0
        If None, automated method selection is used
    visualize : bool
        Whether to generate visualization of data before and after cleaning
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataset
    """
    file_name = os.path.basename(file_path)
    print(f"Cleaning {file_name}...")
    
    try:
        # Read the file
        df = pd.read_csv(file_path)
        original_df = df.copy()  # Save original for comparison
        
        # Identify date column and convert to datetime
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_col = col
                    break
                except:
                    continue
        
        if date_col:
            print(f"Identified date column: {date_col}")
            # Sort by date
            df = df.sort_values(date_col)
        
        # Check for columns with all NaN
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            print(f"Dropping columns with all NaN values: {all_nan_cols}")
            df = df.drop(columns=all_nan_cols)
        
        # Identify columns with missing values
        missing_cols = df.columns[df.isna().any()].tolist()
        print(f"Columns with missing values: {missing_cols}")
        
        # Apply cleaning methods
        if methods is None:
            methods = {}
            # Automatically determine appropriate methods for each column
            for col in missing_cols:
                if df[col].dtype == 'object':
                    # For categorical data, use mode
                    methods[col] = 'mode'
                elif date_col and col != date_col:
                    # For numeric time series data, use interpolation
                    methods[col] = 'interpolate'
                else:
                    # Default to median
                    methods[col] = 'median'
        
        # Apply methods to each column
        for col in missing_cols:
            method = methods.get(col, 'interpolate')
            missing_count = df[col].isna().sum()
            
            if method == 'drop':
                df = df.dropna(subset=[col])
                print(f"  - {col}: Dropped {missing_count} rows with NaN values")
                
            elif method == 'mean':
                df[col] = df[col].fillna(df[col].mean())
                print(f"  - {col}: Filled {missing_count} NaNs with mean")
                
            elif method == 'median':
                df[col] = df[col].fillna(df[col].median())
                print(f"  - {col}: Filled {missing_count} NaNs with median")
                
            elif method == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else np.nan)
                print(f"  - {col}: Filled {missing_count} NaNs with mode")
                
            elif method == 'forward':
                df[col] = df[col].fillna(method='ffill')
                print(f"  - {col}: Filled {missing_count} NaNs with forward fill")
                
            elif method == 'backward':
                df[col] = df[col].fillna(method='bfill')
                print(f"  - {col}: Filled {missing_count} NaNs with backward fill")
                
            elif method == 'interpolate':
                df[col] = df[col].interpolate(method='linear')
                print(f"  - {col}: Filled {missing_count} NaNs with linear interpolation")
                
            elif method == 'zero':
                df[col] = df[col].fillna(0)
                print(f"  - {col}: Filled {missing_count} NaNs with zeros")
                
            elif method == 'knn':
                try:
                    from sklearn.impute import KNNImputer
                    
                    # Apply KNN imputation just to this column
                    imputer = KNNImputer(n_neighbors=5)
                    df[col] = imputer.fit_transform(df[[col]])
                    print(f"  - {col}: Filled {missing_count} NaNs with KNN imputation")
                except ImportError:
                    print("KNN imputation requires scikit-learn. Falling back to interpolation.")
                    df[col] = df[col].interpolate(method='linear')
                    print(f"  - {col}: Filled {missing_count} NaNs with linear interpolation")
            else:
                print(f"  - {col}: Unknown method '{method}', leaving NaNs as is")
                
        # Check if we still have NaNs
        remaining_nans = df.isna().sum().sum()
        if remaining_nans > 0:
            print(f"Warning: {remaining_nans} NaN values remain after cleaning.")
            print("Applying final cleaning pass with forward and backward fill...")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Handle outliers (using Z-score > 3)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if col == date_col:
                continue
                
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = (z_scores > 3)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                print(f"  - {col}: Found {outlier_count} outliers")
                
                # Instead of dropping, we'll cap outliers at 3 std devs from mean
                cap_high = df[col].mean() + 3 * df[col].std()
                cap_low = df[col].mean() - 3 * df[col].std()
                
                # Apply capping
                df.loc[df[col] > cap_high, col] = cap_high
                df.loc[df[col] < cap_low, col] = cap_low
                print(f"      Capped outliers to within 3 std devs of the mean")
        
        # Save clean file
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file_name.replace('.csv', '_cleaned.csv'))
        df.to_csv(output_file, index=False)
        print(f"Cleaned file saved to {output_file}")
        
        # Visualize before and after if requested
        if visualize and numeric_cols.size > 0:
            try:
                os.makedirs('visualizations/data_cleaning', exist_ok=True)
                
                # Pick up to 4 columns to visualize
                cols_to_viz = numeric_cols[:min(4, len(numeric_cols))]
                
                # Create before/after comparison plots
                plt.figure(figsize=(15, 10))
                for i, col in enumerate(cols_to_viz):
                    plt.subplot(len(cols_to_viz), 2, i*2+1)
                    if date_col:
                        plt.plot(original_df[date_col], original_df[col], 'r-', alpha=0.7)
                    else:
                        plt.plot(original_df[col], 'r-', alpha=0.7)
                    plt.title(f"Before: {col}")
                    plt.grid(True)
                    
                    plt.subplot(len(cols_to_viz), 2, i*2+2)
                    if date_col:
                        plt.plot(df[date_col], df[col], 'g-')
                    else:
                        plt.plot(df[col], 'g-')
                    plt.title(f"After: {col}")
                    plt.grid(True)
                
                plt.tight_layout()
                viz_file = os.path.join('visualizations/data_cleaning', file_name.replace('.csv', '_cleaning.png'))
                plt.savefig(viz_file)
                plt.close()
                print(f"Visualization saved to {viz_file}")
            except Exception as e:
                print(f"Error creating visualization: {e}")
        
        return df
        
    except Exception as e:
        print(f"Error cleaning {file_name}: {e}")
        return None

def batch_clean_datasets(report_df: Optional[pd.DataFrame] = None, 
                         data_dir: str = 'data/raw',
                         output_dir: str = 'data/processed') -> Dict[str, pd.DataFrame]:
    """
    Cleans multiple datasets based on the data quality report.
    
    Parameters:
    -----------
    report_df : pd.DataFrame
        Data quality report from scan_data_quality
    data_dir : str
        Directory containing raw data files
    output_dir : str
        Directory to save cleaned files
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of cleaned datasets
    """
    # Generate report if not provided
    if report_df is None:
        report_df = scan_data_quality(data_dir)
    
    # Get files needing cleanup
    files_to_clean = report_df[report_df['needs_cleaning']]['file_name'].tolist()
    
    if not files_to_clean:
        print("No files require cleaning")
        return {}
    
    cleaned_dfs = {}
    
    for file_name in files_to_clean:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            cleaned_df = clean_dataset(file_path, output_dir)
            if cleaned_df is not None:
                cleaned_dfs[file_name] = cleaned_df
    
    return cleaned_dfs

def verify_date_consistency(data_dir: str = 'data/raw', fix: bool = True) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Verifies that all datasets have consistent date formats and ranges.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data files
    fix : bool
        Whether to fix inconsistencies
        
    Returns:
    --------
    Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]
        Dictionary mapping file names to (start_date, end_date) tuples
    """
    print(f"Verifying date consistency in {data_dir}...")
    
    # Get all CSV files
    csv_files = glob.glob(f"{data_dir}/*.csv")
    
    date_ranges = {}
    date_formats = {}
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        
        try:
            # Read the file
            df = pd.read_csv(file_path)
            
            # Find date column
            date_col = None
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        dates = pd.to_datetime(df[col])
                        date_col = col
                        break
                    except:
                        continue
            
            if date_col:
                # Check the original format
                date_samples = df[date_col].dropna().head(3).tolist()
                date_format = None
                
                # Try to detect format
                if date_samples:
                    for sample in date_samples:
                        if isinstance(sample, str):
                            if '-' in sample:
                                if sample.count('-') == 2:
                                    date_format = 'YYYY-MM-DD'
                            elif '/' in sample:
                                if sample.count('/') == 2:
                                    date_format = 'MM/DD/YYYY'
                            elif '.' in sample:
                                if sample.count('.') == 2:
                                    date_format = 'DD.MM.YYYY'
                
                # Convert to datetime
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    
                    # Get min and max dates
                    min_date = df[date_col].min()
                    max_date = df[date_col].max()
                    
                    date_ranges[file_name] = (min_date, max_date)
                    date_formats[file_name] = date_format
                    
                    print(f"{file_name}: {date_col} range {min_date.date()} to {max_date.date()}, format: {date_format}")
                except Exception as e:
                    print(f"Error converting dates in {file_name}: {e}")
            else:
                print(f"{file_name}: No date column found")
                
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    # Find the overall date range
    if date_ranges:
        all_min_dates = [dates[0] for dates in date_ranges.values()]
        all_max_dates = [dates[1] for dates in date_ranges.values()]
        
        overall_min_date = min(all_min_dates)
        overall_max_date = max(all_max_dates)
        
        print(f"\nOverall date range: {overall_min_date.date()} to {overall_max_date.date()}")
        
        # Report files with different date ranges
        for file_name, (min_date, max_date) in date_ranges.items():
            min_diff = (min_date - overall_min_date).days
            max_diff = (overall_max_date - max_date).days
            
            if min_diff > 30 or max_diff > 30:
                print(f"Warning: {file_name} has a significantly different date range")
                print(f"  Starts {min_diff} days after the earliest dataset")
                print(f"  Ends {max_diff} days before the latest dataset")
                
                if fix:
                    print(f"  Fixing will require data extension for {file_name}")
    
    return date_ranges

def main():
    """
    Main function to run the data cleaning utilities.
    """
    print("Running data cleanup utilities...")
    
    # Scan data quality
    report_df = scan_data_quality()
    
    # Verify date consistency
    date_ranges = verify_date_consistency(fix=False)
    
    # Clean datasets
    cleaned_dfs = batch_clean_datasets(report_df)
    
    print("\nData cleanup complete!")
    
    return report_df, date_ranges, cleaned_dfs

if __name__ == "__main__":
    main() 
"""
India Retail Demand Forecaster - Feature Engineering Module
----------------------------------------------------------
This module provides functions for loading, merging, and engineering features
from economic indicators for retail demand forecasting.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def merge_all_signals(data_paths: Dict[str, str], 
                     date_column: str = 'date',
                     interpolate: bool = True,
                     interpolation_method: str = 'linear') -> pd.DataFrame:
    """
    Loads all cleaned CSVs and merges them on the date column.
    
    Parameters:
    -----------
    data_paths : Dict[str, str]
        Dictionary mapping signal names to file paths
    date_column : str
        Name of the date column in each CSV
    interpolate : bool
        Whether to interpolate missing values after merging
    interpolation_method : str
        Method to use for interpolation ('linear', 'cubic', etc.)
        
    Returns:
    --------
    pd.DataFrame
        Merged dataframe with all signals
    """
    print("Merging economic signals...")
    
    # Initialize with the first dataset
    if not data_paths:
        raise ValueError("No data paths provided")
    
    # Load all dataframes
    dfs = {}
    for signal_name, file_path in data_paths.items():
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping")
            continue
            
        try:
            df = pd.read_csv(file_path)
            # Ensure date column is datetime
            df[date_column] = pd.to_datetime(df[date_column])
            dfs[signal_name] = df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not dfs:
        raise ValueError("No valid dataframes could be loaded")
    
    # Start with the first dataframe
    signal_name = list(dfs.keys())[0]
    merged_df = dfs[signal_name].copy()
    print(f"Starting with {signal_name} ({merged_df.shape[0]} rows)")
    
    # Merge all other dataframes
    for signal_name, df in list(dfs.items())[1:]:
        print(f"Merging {signal_name} ({df.shape[0]} rows)")
        merged_df = pd.merge(merged_df, df, on=date_column, how='outer')
    
    # Sort by date
    merged_df = merged_df.sort_values(date_column)
    
    # Set date as index
    merged_df.set_index(date_column, inplace=True)
    
    # Interpolate missing values if requested
    if interpolate:
        print("Interpolating missing values...")
        for col in merged_df.columns:
            if merged_df[col].isna().any():
                missing_pct = merged_df[col].isna().mean() * 100
                print(f"  - {col}: {missing_pct:.1f}% missing values")
                merged_df[col] = merged_df[col].interpolate(method=interpolation_method)
    
    print(f"Final merged dataframe: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    return merged_df

def generate_lag_features(df: pd.DataFrame, 
                         columns: List[str], 
                         lags: Union[int, List[int]] = 3) -> pd.DataFrame:
    """
    Adds lagged versions of each specified column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with time series data
    columns : List[str]
        List of columns to create lags for
    lags : Union[int, List[int]]
        Number of lags to generate or specific list of lag periods
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional lag features
    """
    print("Generating lag features...")
    result_df = df.copy()
    
    # Convert single lag to list
    if isinstance(lags, int):
        lags = list(range(1, lags + 1))
    
    # Create lag features
    for col in columns:
        if col not in result_df.columns:
            print(f"Warning: Column '{col}' not found in dataframe, skipping")
            continue
            
        for lag in lags:
            lag_col_name = f"{col}_lag_{lag}"
            result_df[lag_col_name] = result_df[col].shift(lag)
            print(f"Created {lag_col_name}")
    
    return result_df

def generate_rolling_features(df: pd.DataFrame, 
                             columns: List[str], 
                             windows: List[int] = [3, 6, 12],
                             functions: Dict[str, callable] = {'mean': np.mean, 'std': np.std}) -> pd.DataFrame:
    """
    Adds rolling window calculations for key variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with time series data
    columns : List[str]
        List of columns to create rolling features for
    windows : List[int]
        List of window sizes in periods
    functions : Dict[str, callable]
        Dictionary mapping function names to function objects
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional rolling features
    """
    print("Generating rolling features...")
    result_df = df.copy()
    
    # Create rolling features
    for col in columns:
        if col not in result_df.columns:
            print(f"Warning: Column '{col}' not found in dataframe, skipping")
            continue
            
        for window in windows:
            for func_name, func in functions.items():
                feature_name = f"{col}_{func_name}_{window}"
                result_df[feature_name] = result_df[col].rolling(window=window).apply(func)
                print(f"Created {feature_name}")
    
    return result_df

def generate_date_features(df: pd.DataFrame, 
                          include_holidays: bool = True,
                          include_diwali: bool = True,
                          include_budget: bool = True) -> pd.DataFrame:
    """
    Adds calendar-based features from the date index.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index
    include_holidays : bool
        Whether to include national holiday dummy variables
    include_diwali : bool
        Whether to include Diwali festival features
    include_budget : bool
        Whether to include budget announcement features
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional date-based features
    """
    print("Generating date features...")
    result_df = df.copy()
    
    # Make sure we have a datetime index
    if not isinstance(result_df.index, pd.DatetimeIndex):
        if 'date' in result_df.columns:
            result_df['date'] = pd.to_datetime(result_df['date'])
            result_df.set_index('date', inplace=True)
        else:
            raise ValueError("Dataframe must have a datetime index or a 'date' column")
    
    # Basic date components
    result_df['year'] = result_df.index.year
    result_df['quarter'] = result_df.index.quarter
    result_df['month'] = result_df.index.month
    result_df['day_of_month'] = result_df.index.day
    result_df['day_of_week'] = result_df.index.dayofweek
    
    # Cyclical encoding of month and quarter
    result_df['month_sin'] = np.sin(2 * np.pi * result_df.index.month / 12)
    result_df['month_cos'] = np.cos(2 * np.pi * result_df.index.month / 12)
    result_df['quarter_sin'] = np.sin(2 * np.pi * result_df.index.quarter / 4)
    result_df['quarter_cos'] = np.cos(2 * np.pi * result_df.index.quarter / 4)
    
    # Festival seasons (simplified approach)
    if include_diwali:
        # Approximate Diwali months (typically falls in Oct-Nov, but exact dates vary by year)
        result_df['is_diwali_season'] = result_df.index.month.isin([10, 11]).astype(int)
        
        # Diwali preparation period (1-2 months before)
        result_df['is_pre_diwali'] = result_df.index.month.isin([8, 9]).astype(int)
        
        print("Added Diwali season features (approximated by month)")
    
    # Budget announcements (typically in February in India)
    if include_budget:
        result_df['is_budget_month'] = (result_df.index.month == 2).astype(int)
        
        # Pre and post budget periods
        result_df['is_pre_budget'] = (result_df.index.month == 1).astype(int)
        result_df['is_post_budget'] = (result_df.index.month == 3).astype(int)
        
        print("Added budget announcement features")
    
    # Major Indian holidays (simplified approach)
    if include_holidays:
        # Major holiday seasons:
        # - Diwali/Deepavali (Oct-Nov)
        # - Holi (Feb-Mar)
        # - Durga Puja/Navratri (Sep-Oct)
        # - Christmas/New Year (Dec)
        holiday_months = [3, 9, 10, 11, 12]  # Approximate holiday months
        result_df['is_holiday_season'] = result_df.index.month.isin(holiday_months).astype(int)
        
        print("Added holiday season features")
    
    return result_df

def prepare_features_and_target(df: pd.DataFrame,
                               target_column: str = 'retail_sales',
                               normalize: bool = True,
                               normalization_method: str = 'standard',
                               test_size: float = 0.2,
                               fill_na_strategy: str = 'knn') -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Optional[Union[StandardScaler, MinMaxScaler]]]:
    """
    Prepares features and target for modeling, including normalization and train-test split.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features and target
    target_column : str
        Name of the target column
    normalize : bool
        Whether to normalize features
    normalization_method : str
        Method to use for normalization ('standard' or 'minmax')
    test_size : float
        Proportion of data to use for testing
    fill_na_strategy : str
        Strategy for handling missing values:
        - 'drop': Drop rows with any missing values
        - 'mean': Fill with column mean
        - 'median': Fill with column median
        - 'forward': Forward fill (use previous valid value)
        - 'backward': Backward fill (use next valid value)
        - 'interpolate': Linear interpolation between valid values
        - 'knn': K-Nearest Neighbors imputation
        - 'iterative': Iterative imputation (uses relationships between features)
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Optional[Union[StandardScaler, MinMaxScaler]]]
        (X_train, y_train, X_test, y_test, scaler)
    """
    print("Preparing features and target...")
    result_df = df.copy()
    
    # Check if target column exists
    if target_column not in result_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    # Check for missing values
    missing_count = result_df.isna().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values across all columns")
        
        # Report columns with missing values
        cols_with_missing = result_df.columns[result_df.isna().any()].tolist()
        print(f"Columns with missing values: {cols_with_missing}")
        
        # Handle missing values based on selected strategy
        if fill_na_strategy == 'drop':
            initial_rows = len(result_df)
            result_df = result_df.dropna()
            print(f"Dropped rows with missing values. Rows reduced from {initial_rows} to {len(result_df)}")
            
        elif fill_na_strategy == 'mean':
            # Simple mean imputation
            result_df = result_df.fillna(result_df.mean())
            print("Filled missing values with column means")
            
        elif fill_na_strategy == 'median':
            # Simple median imputation
            result_df = result_df.fillna(result_df.median())
            print("Filled missing values with column medians")
            
        elif fill_na_strategy == 'forward':
            result_df = result_df.fillna(method='ffill')
            # Double check for any remaining NaNs at the beginning
            if result_df.isna().any().any():
                # Fill any remaining NaNs with backward fill
                result_df = result_df.fillna(method='bfill')
            print("Filled missing values using forward fill (with backward fill for initial values)")
            
        elif fill_na_strategy == 'backward':
            result_df = result_df.fillna(method='bfill')
            # Double check for any remaining NaNs at the end
            if result_df.isna().any().any():
                # Fill any remaining NaNs with forward fill
                result_df = result_df.fillna(method='ffill')
            print("Filled missing values using backward fill (with forward fill for ending values)")
            
        elif fill_na_strategy == 'interpolate':
            # Linear interpolation for each column
            for col in result_df.columns:
                if result_df[col].isna().any():
                    result_df[col] = result_df[col].interpolate(method='linear')
            
            # Check for any remaining NaNs at the edges
            if result_df.isna().any().any():
                # Fill edge values with forward and backward fill
                result_df = result_df.fillna(method='ffill').fillna(method='bfill')
            
            print("Filled missing values using linear interpolation (with fill methods for edges)")
            
        elif fill_na_strategy == 'knn':
            try:
                from sklearn.impute import KNNImputer
                
                # Store index and column information
                original_index = result_df.index
                original_columns = result_df.columns
                
                # KNN imputation
                imputer = KNNImputer(n_neighbors=5)
                result_values = imputer.fit_transform(result_df)
                
                # Recreate DataFrame with original index and columns
                result_df = pd.DataFrame(result_values, index=original_index, columns=original_columns)
                print("Filled missing values using KNN imputation with 5 neighbors")
                
            except ImportError:
                print("KNN imputation requires scikit-learn. Falling back to interpolation.")
                # Fallback to interpolation
                for col in result_df.columns:
                    if result_df[col].isna().any():
                        result_df[col] = result_df[col].interpolate(method='linear')
                
                # Check for any remaining NaNs at the edges
                if result_df.isna().any().any():
                    # Fill edge values with forward and backward fill
                    result_df = result_df.fillna(method='ffill').fillna(method='bfill')
        
        elif fill_na_strategy == 'iterative':
            try:
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                
                # Store index and column information
                original_index = result_df.index
                original_columns = result_df.columns
                
                # Iterative imputation
                imputer = IterativeImputer(max_iter=10, random_state=42)
                result_values = imputer.fit_transform(result_df)
                
                # Recreate DataFrame with original index and columns
                result_df = pd.DataFrame(result_values, index=original_index, columns=original_columns)
                print("Filled missing values using iterative imputation")
                
            except ImportError:
                print("Iterative imputation requires scikit-learn. Falling back to interpolation.")
                # Fallback to interpolation
                for col in result_df.columns:
                    if result_df[col].isna().any():
                        result_df[col] = result_df[col].interpolate(method='linear')
                
                # Check for any remaining NaNs at the edges
                if result_df.isna().any().any():
                    # Fill edge values with forward and backward fill
                    result_df = result_df.fillna(method='ffill').fillna(method='bfill')
        
        else:
            raise ValueError(f"Unknown fill_na_strategy: {fill_na_strategy}")
        
        # Verify all missing values have been handled
        remaining_missing = result_df.isna().sum().sum()
        if remaining_missing > 0:
            print(f"Warning: {remaining_missing} missing values still remain. Applying mean imputation for remaining NaNs.")
            result_df = result_df.fillna(result_df.mean())
    
    # Split features and target
    X = result_df.drop(columns=[target_column])
    y = result_df[target_column]
    
    # For time series, we'll use a chronological split
    split_idx = int(len(result_df) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Split data: train={len(X_train)} rows, test={len(X_test)} rows")
    
    # Normalize features if requested
    scaler = None
    if normalize:
        if normalization_method == 'standard':
            scaler = StandardScaler()
            print("Using StandardScaler for normalization")
        elif normalization_method == 'minmax':
            scaler = MinMaxScaler()
            print("Using MinMaxScaler for normalization")
        else:
            raise ValueError(f"Unknown normalization_method: {normalization_method}")
        
        # Fit on training data only
        X_train_values = scaler.fit_transform(X_train)
        X_test_values = scaler.transform(X_test)
        
        # Convert back to dataframes
        X_train = pd.DataFrame(X_train_values, index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_values, index=X_test.index, columns=X_test.columns)
    
    # Final check for NaN values
    if X_train.isna().any().any() or X_test.isna().any().any() or y_train.isna().any() or y_test.isna().any():
        print("Warning: NaN values still present after preprocessing")
    else:
        print("All NaN values successfully handled")
    
    return X_train, y_train, X_test, y_test, scaler 
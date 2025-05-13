"""
India Retail Demand Forecaster - Data Preparation
-------------------------------------------------
This module provides comprehensive data preprocessing to handle data quality
issues and prepare clean, properly formatted data for modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from sklearn.impute import KNNImputer, SimpleImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_raw_data(data_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load raw data from various sources.
    
    Parameters:
    -----------
    data_paths : Dict[str, str]
        Dictionary mapping data type to file path
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of loaded dataframes
    """
    print("Loading raw data sources...")
    
    data_sources = {}
    
    # Load each data source
    for source_name, file_path in data_paths.items():
        try:
            # Check if file exists
            if not Path(file_path).exists():
                print(f"Warning: File {file_path} for source {source_name} not found")
                continue
                
            # Load data with appropriate parser based on file extension
            file_ext = file_path.split('.')[-1].lower()
            
            if file_ext == 'csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['xls', 'xlsx']:
                df = pd.read_excel(file_path)
            else:
                print(f"Warning: Unsupported file format for {file_path}")
                continue
                
            # Check if data was loaded successfully
            if df.empty:
                print(f"Warning: Empty dataframe loaded from {file_path}")
                continue
                
            # Normalize column names
            df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
            
            # Basic sanity check
            print(f"Loaded {source_name}: {df.shape[0]} rows, {df.shape[1]} columns")
            
            data_sources[source_name] = df
            
        except Exception as e:
            print(f"Error loading {source_name} from {file_path}: {e}")
    
    print(f"Successfully loaded {len(data_sources)} data sources")
    return data_sources

def standardize_date_format(dfs: Dict[str, pd.DataFrame], date_column: str = 'date') -> Dict[str, pd.DataFrame]:
    """
    Standardize date formats across all dataframes.
    
    Parameters:
    -----------
    dfs : Dict[str, pd.DataFrame]
        Dictionary of dataframes
    date_column : str
        Name of the date column
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of dataframes with standardized dates
    """
    print("Standardizing date formats...")
    
    standardized_dfs = {}
    
    for source_name, df in dfs.items():
        try:
            # Check if date column exists
            if date_column not in df.columns:
                print(f"Warning: No date column '{date_column}' in {source_name}")
                standardized_dfs[source_name] = df
                continue
            
            # Convert to datetime
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            
            # Check for NULL dates after conversion
            null_dates = df[date_column].isna().sum()
            if null_dates > 0:
                print(f"Warning: {null_dates} null dates in {source_name} after conversion")
                # Drop rows with null dates
                df = df.dropna(subset=[date_column])
                
            # Set the date as index
            df = df.set_index(date_column).sort_index()
            
            # Add to standardized dataframes
            standardized_dfs[source_name] = df
            
            print(f"Standardized dates for {source_name}: {df.index.min()} to {df.index.max()}")
            
        except Exception as e:
            print(f"Error standardizing dates for {source_name}: {e}")
            standardized_dfs[source_name] = df
    
    return standardized_dfs

def detect_and_handle_outliers(dfs: Dict[str, pd.DataFrame], methods: List[str] = ['zscore', 'iqr']) -> Dict[str, pd.DataFrame]:
    """
    Detect and handle outliers in numeric columns.
    
    Parameters:
    -----------
    dfs : Dict[str, pd.DataFrame]
        Dictionary of dataframes
    methods : List[str]
        Outlier detection methods to use
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of dataframes with outliers handled
    """
    print("Detecting and handling outliers...")
    
    processed_dfs = {}
    
    for source_name, df in dfs.items():
        try:
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            
            if len(numeric_cols) == 0:
                print(f"Warning: No numeric columns in {source_name}")
                processed_dfs[source_name] = df
                continue
            
            # Create a copy of the dataframe
            df_processed = df.copy()
            
            # Track outliers
            total_outliers = 0
            
            # Process each numeric column
            for col in numeric_cols:
                outliers_mask = np.zeros(len(df), dtype=bool)
                
                # Z-score method
                if 'zscore' in methods:
                    z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
                    z_outliers = np.where(z_scores > 3)[0]
                    outliers_mask = outliers_mask | (np.abs(z_scores) > 3)
                
                # IQR method
                if 'iqr' in methods:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    iqr_outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
                    outliers_mask = outliers_mask | iqr_outliers
                
                # Count outliers
                col_outliers = np.sum(outliers_mask)
                if col_outliers > 0:
                    print(f"  {source_name}.{col}: {col_outliers} outliers detected ({col_outliers/len(df)*100:.1f}%)")
                    total_outliers += col_outliers
                    
                    # Handle outliers - cap at percentiles
                    lower_bound = df[col].quantile(0.01)
                    upper_bound = df[col].quantile(0.99)
                    df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Add to processed dataframes
            processed_dfs[source_name] = df_processed
            
            print(f"Processed {source_name}: {total_outliers} total outliers handled")
            
        except Exception as e:
            print(f"Error handling outliers for {source_name}: {e}")
            processed_dfs[source_name] = df
    
    return processed_dfs

def impute_missing_values(dfs: Dict[str, pd.DataFrame], method: str = 'knn') -> Dict[str, pd.DataFrame]:
    """
    Impute missing values in dataframes.
    
    Parameters:
    -----------
    dfs : Dict[str, pd.DataFrame]
        Dictionary of dataframes
    method : str
        Imputation method ('mean', 'median', 'knn', or 'time')
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of dataframes with missing values imputed
    """
    print("Imputing missing values...")
    
    imputed_dfs = {}
    
    for source_name, df in dfs.items():
        try:
            # Check for missing values
            missing = df.isna().sum().sum()
            if missing == 0:
                print(f"No missing values in {source_name}")
                imputed_dfs[source_name] = df
                continue
            
            print(f"Found {missing} missing values in {source_name}")
            
            # Create a copy of the dataframe
            df_imputed = df.copy()
            
            # Select imputation method
            if method == 'mean':
                # Simple mean imputation
                imputer = SimpleImputer(strategy='mean')
                df_imputed.loc[:, :] = imputer.fit_transform(df)
                
            elif method == 'median':
                # Median imputation
                imputer = SimpleImputer(strategy='median')
                df_imputed.loc[:, :] = imputer.fit_transform(df)
                
            elif method == 'knn':
                # KNN imputation
                try:
                    imputer = KNNImputer(n_neighbors=5)
                    df_imputed.loc[:, :] = imputer.fit_transform(df)
                except Exception as e:
                    print(f"KNN imputation failed for {source_name}: {e}")
                    print("Falling back to time-based interpolation")
                    method = 'time'
            
            if method == 'time':
                # For time series, use interpolation
                df_imputed = df.interpolate(method='time').bfill().ffill()
            
            # Verify all missing values are handled
            remaining = df_imputed.isna().sum().sum()
            if remaining > 0:
                print(f"Warning: {remaining} missing values remain in {source_name} after imputation")
                # Final fallback - fill with column means
                df_imputed = df_imputed.fillna(df_imputed.mean())
            
            # Add to imputed dataframes
            imputed_dfs[source_name] = df_imputed
            
            print(f"Imputed {source_name} using {method} method")
            
        except Exception as e:
            print(f"Error imputing values for {source_name}: {e}")
            imputed_dfs[source_name] = df
    
    return imputed_dfs

def transform_variables(dfs: Dict[str, pd.DataFrame], transformations: Dict[str, str] = None) -> Dict[str, pd.DataFrame]:
    """
    Apply transformations to variables to improve linearity and reduce heteroskedasticity.
    
    Parameters:
    -----------
    dfs : Dict[str, pd.DataFrame]
        Dictionary of dataframes
    transformations : Dict[str, str]
        Dictionary mapping column names to transformation methods
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of dataframes with transformations applied
    """
    print("Applying variable transformations...")
    
    # Default transformations
    if transformations is None:
        transformations = {
            'retail_sales': 'log',  # Log transform to reduce heteroskedasticity
            'gdp': 'log',
            'gold_price': 'log',
            'oil_price': 'log'
        }
    
    transformed_dfs = {}
    
    for source_name, df in dfs.items():
        try:
            # Create a copy of the dataframe
            df_transformed = df.copy()
            
            # Apply transformations
            for col, transform in transformations.items():
                # Check if column exists
                if col not in df.columns:
                    continue
                
                # Check for negative or zero values that can't be log-transformed
                if transform == 'log' and (df[col] <= 0).any():
                    print(f"Warning: Can't apply log transform to {col} in {source_name} due to non-positive values")
                    continue
                
                # Apply transformation
                if transform == 'log':
                    df_transformed[f'log_{col}'] = np.log(df[col])
                    print(f"Applied log transform to {col} in {source_name}")
                    
                elif transform == 'sqrt':
                    if (df[col] < 0).any():
                        print(f"Warning: Can't apply sqrt transform to {col} in {source_name} due to negative values")
                        continue
                    df_transformed[f'sqrt_{col}'] = np.sqrt(df[col])
                    print(f"Applied sqrt transform to {col} in {source_name}")
                    
                elif transform == 'boxcox':
                    if (df[col] <= 0).any():
                        print(f"Warning: Can't apply Box-Cox transform to {col} in {source_name} due to non-positive values")
                        continue
                    transformed_data, lambda_value = stats.boxcox(df[col])
                    df_transformed[f'boxcox_{col}'] = transformed_data
                    print(f"Applied Box-Cox transform to {col} in {source_name} (lambda={lambda_value:.4f})")
            
            # Add to transformed dataframes
            transformed_dfs[source_name] = df_transformed
            
        except Exception as e:
            print(f"Error transforming variables for {source_name}: {e}")
            transformed_dfs[source_name] = df
    
    return transformed_dfs

def align_time_series(dfs: Dict[str, pd.DataFrame], frequency: str = 'MS') -> Dict[str, pd.DataFrame]:
    """
    Align time series to a common frequency.
    
    Parameters:
    -----------
    dfs : Dict[str, pd.DataFrame]
        Dictionary of dataframes
    frequency : str
        Pandas frequency string (MS = month start)
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of dataframes with aligned time series
    """
    print(f"Aligning time series to {frequency} frequency...")
    
    aligned_dfs = {}
    
    # First, find the common date range
    start_dates = []
    end_dates = []
    
    for source_name, df in dfs.items():
        # Make sure the dataframe has a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"Warning: {source_name} does not have a DatetimeIndex")
            aligned_dfs[source_name] = df
            continue
        
        start_dates.append(df.index.min())
        end_dates.append(df.index.max())
    
    if not start_dates or not end_dates:
        print("Warning: No valid date ranges found")
        return dfs
    
    # Find the common date range
    common_start = max(start_dates)
    common_end = min(end_dates)
    
    print(f"Common date range: {common_start} to {common_end}")
    
    # Create a common date range with the specified frequency
    common_dates = pd.date_range(start=common_start, end=common_end, freq=frequency)
    
    # Align each dataframe to the common dates
    for source_name, df in dfs.items():
        try:
            # Skip if not a DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                aligned_dfs[source_name] = df
                continue
            
            # Reindex to the common dates
            df_aligned = df.asfreq(frequency)
            
            # Reindex to the common date range
            df_aligned = df_aligned.reindex(common_dates)
            
            # Interpolate missing values
            df_aligned = df_aligned.interpolate(method='time').bfill().ffill()
            
            # Add to aligned dataframes
            aligned_dfs[source_name] = df_aligned
            
            print(f"Aligned {source_name}: {len(df_aligned)} observations at {frequency} frequency")
            
        except Exception as e:
            print(f"Error aligning time series for {source_name}: {e}")
            aligned_dfs[source_name] = df
    
    return aligned_dfs

def merge_data_sources(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple data sources into a single dataframe.
    
    Parameters:
    -----------
    dfs : Dict[str, pd.DataFrame]
        Dictionary of dataframes
        
    Returns:
    --------
    pd.DataFrame
        Merged dataframe
    """
    print("Merging data sources...")
    
    # Start with the first dataframe (retail sales)
    if 'retail' not in dfs:
        print("Error: Retail sales data not found")
        return pd.DataFrame()
    
    # Create a new dataframe with a DatetimeIndex from all our aligned data
    # This ensures we have a proper datetime index for the merged dataframe
    first_df = next(iter(dfs.values()))
    if not isinstance(first_df.index, pd.DatetimeIndex):
        print("Error: First dataframe doesn't have a DatetimeIndex. Cannot merge properly.")
        return first_df
        
    # Get the common index
    common_index = first_df.index
    
    # Create an empty dataframe with the common index
    merged_df = pd.DataFrame(index=common_index)
    
    # Add data from each source, keeping the DatetimeIndex
    for source_name, df in dfs.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"Warning: {source_name} doesn't have a DatetimeIndex, skipping")
            continue
            
        # Add source suffix to avoid column name conflicts (except for retail)
        if source_name == 'retail':
            # For retail, keep original column names
            for col in df.columns:
                merged_df[col] = df[col]
            print(f"Added {source_name}: {len(df.columns)} columns")
        else:
            # For other sources, add source suffix
            for col in df.columns:
                col_name = f"{col}_{source_name}"
                merged_df[col_name] = df[col]
            print(f"Added {source_name}: {len(df.columns)} columns")

    # Check for columns with all missing values
    null_cols = merged_df.columns[merged_df.isna().all()]
    if len(null_cols) > 0:
        print(f"Warning: {len(null_cols)} columns have all missing values, dropping")
        merged_df = merged_df.drop(columns=null_cols)

    # Final check for any remaining missing values
    missing = merged_df.isna().sum().sum()
    if missing > 0:
        print(f"Warning: {missing} missing values remain after merging")
        # Interpolate remaining missing values
        merged_df = merged_df.interpolate(method='time').bfill().ffill()

    print(f"Final merged dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")

    return merged_df

def prepare_data(data_paths: Dict[str, str], output_path: str = 'data/processed/features_dataset.csv') -> pd.DataFrame:
    """
    Run the complete data preparation pipeline.
    
    Parameters:
    -----------
    data_paths : Dict[str, str]
        Dictionary mapping data type to file path
    output_path : str
        Path to save the processed dataset
        
    Returns:
    --------
    pd.DataFrame
        Processed dataset ready for modeling
    """
    print("\n" + "="*80)
    print("INDIA RETAIL DEMAND FORECASTER - DATA PREPARATION PIPELINE")
    print("="*80 + "\n")
    
    # 1. Load raw data
    raw_data = load_raw_data(data_paths)
    
    # 2. Standardize date formats
    dated_data = standardize_date_format(raw_data)
    
    # 3. Handle outliers
    cleaned_data = detect_and_handle_outliers(dated_data)
    
    # 4. Impute missing values
    imputed_data = impute_missing_values(cleaned_data, method='knn')
    
    # 5. Align time series
    aligned_data = align_time_series(imputed_data)
    
    # 6. Apply variable transformations
    transformed_data = transform_variables(aligned_data)
    
    # 7. Merge data sources
    final_df = merge_data_sources(transformed_data)
    
    # 8. Save processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path)
    print(f"Saved processed dataset to {output_path}")
    
    # 9. Create summary visualizations
    create_data_summary_visualizations(final_df)
    
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80 + "\n")
    
    return final_df

def create_data_summary_visualizations(df: pd.DataFrame) -> None:
    """
    Create summary visualizations for the processed dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed dataset
    """
    print("Creating data summary visualizations...")
    
    # Create visualizations directory
    os.makedirs('visualizations/data_quality', exist_ok=True)
    
    # Make sure the dataframe has a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex) and 'date' in df.columns:
        df = df.set_index('date')
        print("Set 'date' column as index")
    
    # Check if we have a DatetimeIndex now
    has_date_index = isinstance(df.index, pd.DatetimeIndex)
    
    # 1. Time series plots for key variables
    key_vars = ['retail_sales']
    if 'log_retail_sales' in df.columns:
        key_vars.append('log_retail_sales')
        
    # Add other key variables if they exist
    potential_vars = ['gdp', 'inflation', 'interest_rate', 'gold_price', 'oil_price',
                     'log_gdp', 'log_gold_price', 'log_oil_price']
    
    for var in potential_vars:
        if var in df.columns:
            key_vars.append(var)
    
    # Limit to 9 variables for the plot
    if len(key_vars) > 9:
        key_vars = key_vars[:9]
    
    # Plot time series
    fig, axes = plt.subplots(len(key_vars), 1, figsize=(12, 3*len(key_vars)))
    if len(key_vars) == 1:
        axes = [axes]
    
    for i, var in enumerate(key_vars):
        if var not in df.columns:
            print(f"Warning: Variable {var} not found in dataframe")
            continue
            
        # Plot the time series
        if has_date_index:
            # If we have a date index, use it for the x-axis
            axes[i].plot(df.index, df[var])
            # Set x-ticks every 12 points (approximately 1 year)
            tick_indices = list(range(0, len(df), 12))
            tick_labels = [df.index[j].strftime('%Y-%m') if j < len(df) else '' for j in tick_indices]
            axes[i].set_xticks([df.index[j] for j in tick_indices if j < len(df)])
            axes[i].set_xticklabels(tick_labels, rotation=45)
        else:
            # If we don't have a date index, use simple numeric indices
            axes[i].plot(df[var])
            if len(df) > 20:
                # Only show every 12th tick if we have lots of data
                axes[i].set_xticks(list(range(0, len(df), 12)))
                
        axes[i].set_title(var)
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/data_quality/key_variables_time_series.png')
    plt.close()
    
    # 2. Correlation matrix
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('visualizations/data_quality/correlation_matrix.png')
    plt.close()
    
    # 3. Scatter plots between retail sales and key predictors
    potential_predictors = ['gdp', 'inflation', 'interest_rate', 'gold_price', 'oil_price',
                          'log_gdp', 'log_gold_price', 'log_oil_price']
    
    predictors = []
    for pred in potential_predictors:
        if pred in df.columns:
            predictors.append(pred)
    
    # Limit to 5 predictors
    if len(predictors) > 5:
        predictors = predictors[:5]
    
    if predictors and ('retail_sales' in df.columns or 'log_retail_sales' in df.columns):
        target = 'log_retail_sales' if 'log_retail_sales' in df.columns else 'retail_sales'
        
        fig, axes = plt.subplots(len(predictors), 1, figsize=(10, 4*len(predictors)))
        if len(predictors) == 1:
            axes = [axes]
        
        for i, pred in enumerate(predictors):
            axes[i].scatter(df[pred], df[target], alpha=0.7)
            axes[i].set_title(f'{target} vs {pred}')
            axes[i].set_xlabel(pred)
            axes[i].set_ylabel(target)
            axes[i].grid(True)
            
            try:
                # Add a linear regression line
                z = np.polyfit(df[pred].values, df[target].values, 1)
                p = np.poly1d(z)
                axes[i].plot(df[pred].values, p(df[pred].values), "r--", alpha=0.7)
            except Exception as e:
                print(f"Warning: Could not add regression line for {pred}: {e}")
        
        plt.tight_layout()
        plt.savefig('visualizations/data_quality/scatter_plots.png')
        plt.close()
    
    # 4. Distribution plots for key variables
    plt.figure(figsize=(12, 10))
    
    for i, var in enumerate(key_vars[:min(6, len(key_vars))]):  # Limit to 6 variables
        if i >= 6:  # Safety check
            break
            
        if var not in df.columns:
            continue
            
        plt.subplot(3, 2, i+1)
        try:
            sns.histplot(df[var], kde=True)
            plt.title(f'Distribution of {var}')
            plt.grid(True)
        except Exception as e:
            print(f"Warning: Could not create histogram for {var}: {e}")
    
    plt.tight_layout()
    plt.savefig('visualizations/data_quality/distributions.png')
    plt.close()
    
    print("Created data summary visualizations")

def main():
    """
    Main function to run the data preparation pipeline.
    """
    # Define data paths
    data_paths = {
        'retail': 'data/retail_sales.csv',
        'macro': 'data/macro_indicators.csv',
        'gold': 'data/raw/gold_price.csv',
        'oil': 'data/raw/crude_oil_price.csv',
        'iip': 'data/raw/iip_combined.csv',
        'cpi': 'data/raw/cpi.csv',
        'lending': 'data/raw/lending_rate.csv',
        'wpi': 'data/raw/wpi.csv',
        'repo': 'data/raw/repo_rate.csv'
    }
    
    # Run the pipeline
    prepare_data(data_paths)

def load_and_process_data():
    """
    Wrapper function to load and process the data, expected by pipeline.py.
    Returns a processed dataset with all economic indicators.
    """
    # Define data paths
    data_paths = {
        'retail': 'data/retail_sales.csv',
        'gold': 'data/raw/gold_price_processed.csv',
        'oil': 'data/raw/crude_oil_price_processed.csv',
        'iip': 'data/raw/iip_combined.csv',
        'cpi': 'data/raw/cpi.csv',
        'lending': 'data/raw/lending_rate_processed.csv',
        'wpi': 'data/raw/wpi_processed.csv',
        'repo': 'data/raw/repo_rate_processed.csv',
        'usd_inr': 'data/raw/usd_inr.csv'
    }
    
    # Load raw data
    raw_data = load_raw_data(data_paths)
    
    # Standardize date formats
    dated_data = standardize_date_format(raw_data)
    
    # Handle outliers
    cleaned_data = detect_and_handle_outliers(dated_data)
    
    # Impute missing values
    imputed_data = impute_missing_values(cleaned_data, method='knn')
    
    # Align time series
    aligned_data = align_time_series(imputed_data)
    
    # Merge all sources into a single dataframe
    merged_df = merge_data_sources(aligned_data)
    
    return merged_df

def create_features(raw_data):
    """
    Create features from raw data, expected by pipeline.py.
    
    Parameters:
    -----------
    raw_data : pd.DataFrame
        Raw data with economic indicators
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features
    """
    # Transform variables (create log transformations)
    transformations = {
        'retail_sales': 'log',
        'gold_price': 'log', 
        'oil_price': 'log'
    }
    
    transformed_data = transform_variables({'combined': raw_data}, transformations=transformations)
    transformed_df = transformed_data['combined']
    
    # Make sure there are no missing values
    if transformed_df.isna().any().any():
        print("Handling remaining missing values...")
        transformed_df = transformed_df.interpolate(method='time').bfill().ffill()
    
    # Save the features dataset
    os.makedirs('data/processed', exist_ok=True)
    transformed_df.to_csv('data/processed/engineered_features.csv')
    
    return transformed_df

if __name__ == "__main__":
    main() 
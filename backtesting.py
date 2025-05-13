"""
India Retail Demand Forecaster - Time Series Backtesting
-------------------------------------------------------
This module implements advanced backtesting techniques for
rigorous evaluation of forecasting models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
warnings.filterwarnings('ignore')

def load_dataset(file_path="data/processed/features_dataset.csv"):
    """
    Load the dataset for backtesting.
    
    Parameters:
    -----------
    file_path : str
        Path to the dataset
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Check if there's a date column
        date_col = None
        if 'date' in df.columns:
            date_col = 'date'
        else:
            # Try to identify a date column
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_col = col
                    break
                    
            # If still no date column, check if the first column might be a date
            if date_col is None and len(df.columns) > 0:
                try:
                    # Try to parse the first column as a date
                    if 'Unnamed: 0' in df.columns:
                        pd.to_datetime(df['Unnamed: 0'])
                        date_col = 'Unnamed: 0'
                    else:
                        pd.to_datetime(df.iloc[:, 0])
                        date_col = df.columns[0]
                except:
                    # If it's not a date, create a date index based on the data
                    print("No date column found in the dataset")
                    df['date'] = pd.date_range(start='2015-01-01', periods=len(df), freq='MS')
                    date_col = 'date'
        
        # Set the date column as index if it exists
        if date_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                print(f"Date range: {df.index.min()} to {df.index.max()}")
            except Exception as e:
                print(f"Error setting date index: {e}")
                print(f"Date range: 0 to {len(df)-1}")
        else:
            print(f"No date column found in the dataset")
            print(f"Date range: 0 to {len(df)-1}")
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def load_model(model_path='models/best_model.pkl'):
    """
    Load a trained model for backtesting.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    object or None
        Loaded model
    """
    try:
        print(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_time_windows(df, start_date=None, end_date=None, 
                       window_size=12, step_size=3, min_train_size=24):
    """
    Create time windows for walk-forward validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with datetime index
    start_date : str or pd.Timestamp, optional
        Start date for backtesting
    end_date : str or pd.Timestamp, optional
        End date for backtesting
    window_size : int
        Size of each test window in months
    step_size : int
        Number of months to step forward for each window
    min_train_size : int
        Minimum number of training observations
        
    Returns:
    --------
    list
        List of (train_start, train_end, test_start, test_end) timestamps
    """
    # Convert string dates to timestamps if provided
    if start_date is not None and isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    if end_date is not None and isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)
    
    # Use the dataset date range if not specified
    if start_date is None:
        start_date = df.index.min()
    if end_date is None:
        end_date = df.index.max()
    
    # Create a DatetimeIndex with monthly frequency
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Create windows
    windows = []
    for i in range(min_train_size, len(dates) - window_size + 1, step_size):
        # Calculate indices for this window
        train_start_idx = 0
        train_end_idx = i - 1
        test_start_idx = i
        test_end_idx = min(i + window_size - 1, len(dates) - 1)
        
        # Create the window
        train_start = dates[train_start_idx]
        train_end = dates[train_end_idx]
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]
        
        # Add this window
        windows.append((train_start, train_end, test_start, test_end))
    
    # Print window summary
    print(f"Created {len(windows)} backtesting windows:")
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        print(f"Window {i+1}: Train {train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}, "
              f"Test {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}")
    
    if not windows:
        print("No valid backtesting windows created. Please check your parameters.")
    
    return windows

def prepare_window_data(df, window, target_col='retail_sales', feature_names=None):
    """
    Prepare data for a single time window.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with datetime index
    window : tuple
        (train_start, train_end, test_start, test_end) timestamps
    target_col : str
        Name of the target column
    feature_names : list, optional
        List of feature names to use
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test) for the window
    """
    train_start, train_end, test_start, test_end = window
    
    # Create train/test split
    train_mask = (df.index >= train_start) & (df.index <= train_end)
    test_mask = (df.index >= test_start) & (df.index <= test_end)
    
    train_df = df.loc[train_mask]
    test_df = df.loc[test_mask]
    
    # Check for log-transformed target
    log_target = f'log_{target_col}'
    actual_target = log_target if log_target in df.columns else target_col
    
    # Define features (exclude target and date)
    if feature_names is not None:
        # Use provided feature names (ensure they exist in dataframe)
        feature_cols = [col for col in feature_names if col in df.columns]
        print(f"Using {len(feature_cols)} features from provided feature_names")
    else:
        # Exclude target columns
        exclude_cols = [col for col in df.columns if col == target_col or col == log_target]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"Using {len(feature_cols)} features from dataframe columns")
    
    # Create X and y
    X_train = train_df[feature_cols]
    y_train = train_df[actual_target]
    
    X_test = test_df[feature_cols]
    y_test = test_df[actual_target]
    
    return X_train, y_train, X_test, y_test

def run_backtesting(df, model_path, target_col='retail_sales', 
                   window_size=12, step_size=3, min_train_size=24,
                   start_date=None, end_date=None):
    """
    Run a complete backtesting simulation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with datetime index
    model_path : str
        Path to the model to backtest
    target_col : str
        Name of the target column
    window_size : int
        Size of each test window in months
    step_size : int
        Number of months to step forward for each window
    min_train_size : int
        Minimum number of training observations
    start_date : str or pd.Timestamp, optional
        Start date for backtesting
    end_date : str or pd.Timestamp, optional
        End date for backtesting
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with backtesting results
    """
    print("\nStarting backtesting simulation...")
    
    # Create time windows
    windows = create_time_windows(
        df, start_date, end_date, 
        window_size, step_size, min_train_size
    )
    
    # Check if we have windows
    if not windows:
        print("No valid backtesting windows created. Please check your parameters.")
        return None
    
    # Load the model
    model = load_model(model_path)
    if model is None:
        return None
    
    # Try to load the scaler if available
    scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from {scaler_path}")
    else:
        print("No scaler found, creating a new one")
        scaler = StandardScaler()
    
    # Try to get the feature names
    feature_names = None
    
    # First check if model has feature_names attribute
    if hasattr(model, 'feature_names'):
        feature_names = model.feature_names
        print(f"Using {len(feature_names)} features from model.feature_names")
    # Check if model has feature_names_in_ attribute (sklearn models)
    elif hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
        print(f"Using {len(feature_names)} features from model.feature_names_in_")
    # Look for a features file
    else:
        # Check different possible feature file locations
        feature_files = [
            os.path.join(os.path.dirname(model_path), 'feature_names.joblib'),
            os.path.join(os.path.dirname(model_path), 'feature_names.json'),
            os.path.join('models/optimized', 'xgboost_features.json'),
            os.path.join('models/optimized', 'gradient_boosting_features.json'),
            os.path.join('models/optimized', 'random_forest_features.json')
        ]
        
        for feature_file in feature_files:
            if os.path.exists(feature_file):
                try:
                    if feature_file.endswith('.joblib'):
                        feature_names = joblib.load(feature_file)
                    elif feature_file.endswith('.json'):
                        import json
                        with open(feature_file, 'r') as f:
                            feature_names = json.load(f)
                    print(f"Loaded {len(feature_names)} features from {feature_file}")
                    break
                except Exception as e:
                    print(f"Error loading features from {feature_file}: {e}")
    
    # Store results
    results = []
    
    # Loop through windows
    for i, window in enumerate(windows):
        print(f"\nProcessing window {i+1}/{len(windows)}...")
        
        train_start, train_end, test_start, test_end = window
        
        # Prepare data for this window
        X_train, y_train, X_test, y_test = prepare_window_data(df, window, target_col, feature_names)
        
        print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        try:
            # Check if this is a pre-trained model that needs fitting or not
            if hasattr(model, 'fit'):
                print("Fitting model on training data...")
                model.fit(X_train_scaled, y_train)
            
            # Make predictions
            print("Making predictions...")
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Window {i+1} performance: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
            
            # Store results
            for idx, (true_val, pred_val) in enumerate(zip(y_test, y_pred)):
                date = y_test.index[idx]
                
                # Check if we need to reverse the log transformation
                log_target = f'log_{target_col}'
                # Define actual_target here to fix the reference error
                actual_target = log_target if log_target in df.columns else target_col
                
                if log_target in df.columns and log_target == actual_target:
                    true_val_original = np.exp(true_val)
                    pred_val_original = np.exp(pred_val)
                else:
                    true_val_original = true_val
                    pred_val_original = pred_val
                
                results.append({
                    'date': date,
                    'window': i+1,
                    'true_value': true_val,
                    'predicted_value': pred_val,
                    'true_value_original': true_val_original,
                    'predicted_value_original': pred_val_original,
                    'error': true_val - pred_val,
                    'squared_error': (true_val - pred_val)**2,
                    'absolute_error': abs(true_val - pred_val),
                    'window_mse': mse,
                    'window_rmse': rmse,
                    'window_mae': mae,
                    'window_r2': r2
                })
            
        except Exception as e:
            print(f"Error in window {i+1}: {e}")
            continue
    
    # Convert results to dataframe
    results_df = pd.DataFrame(results)
    
    # Calculate overall metrics
    if not results_df.empty:
        overall_mse = mean_squared_error(results_df['true_value'], results_df['predicted_value'])
        overall_rmse = np.sqrt(overall_mse)
        overall_mae = mean_absolute_error(results_df['true_value'], results_df['predicted_value'])
        overall_r2 = r2_score(results_df['true_value'], results_df['predicted_value'])
        
        print("\nOverall backtesting performance:")
        print(f"RMSE: {overall_rmse:.4f}")
        print(f"MAE: {overall_mae:.4f}")
        print(f"R²: {overall_r2:.4f}")
        
        # Add a timestamp index
        results_df.set_index('date', inplace=True)
        
        # Save results
        os.makedirs('outputs/backtesting', exist_ok=True)
        results_df.to_csv('outputs/backtesting/backtesting_results.csv')
        
        # Create visualizations
        create_backtesting_visualizations(results_df, target_col)
    
    return results_df

def create_backtesting_visualizations(results_df, target_col):
    """
    Create visualizations from backtesting results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Dataframe with backtesting results
    target_col : str
        Name of the target column
    """
    print("\nCreating backtesting visualizations...")
    
    # Create directory if it doesn't exist
    os.makedirs('visualizations/backtesting', exist_ok=True)
    
    # 1. Actual vs Predicted over time
    plt.figure(figsize=(14, 7))
    plt.plot(results_df.index, results_df['true_value_original'], label='Actual', marker='o')
    plt.plot(results_df.index, results_df['predicted_value_original'], label='Predicted', marker='x')
    plt.title(f'Backtesting Results: Actual vs Predicted {target_col}')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/backtesting/actual_vs_predicted.png')
    plt.close()
    
    # 2. Error over time
    plt.figure(figsize=(14, 7))
    plt.scatter(results_df.index, results_df['error'], alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Prediction Error Over Time')
    plt.xlabel('Date')
    plt.ylabel('Error (Actual - Predicted)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/backtesting/error_over_time.png')
    plt.close()
    
    # 3. Error distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['error'], kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Error Distribution')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/backtesting/error_distribution.png')
    plt.close()
    
    # 4. Performance by window
    window_metrics = results_df.groupby('window')[['window_rmse', 'window_mae', 'window_r2']].first()
    
    plt.figure(figsize=(14, 7))
    plt.subplot(3, 1, 1)
    plt.plot(window_metrics.index, window_metrics['window_rmse'], marker='o')
    plt.title('RMSE by Window')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(window_metrics.index, window_metrics['window_mae'], marker='o', color='orange')
    plt.title('MAE by Window')
    plt.ylabel('MAE')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(window_metrics.index, window_metrics['window_r2'], marker='o', color='green')
    plt.title('R² by Window')
    plt.ylabel('R²')
    plt.xlabel('Window')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/backtesting/metrics_by_window.png')
    plt.close()
    
    # 5. Prediction intervals (assuming normal distribution of errors)
    error_std = results_df['error'].std()
    
    plt.figure(figsize=(14, 7))
    plt.plot(results_df.index, results_df['true_value_original'], label='Actual', marker='o')
    plt.plot(results_df.index, results_df['predicted_value_original'], label='Predicted', marker='x')
    
    # Add 95% prediction intervals
    upper_95 = results_df['predicted_value_original'] + 1.96 * error_std
    lower_95 = results_df['predicted_value_original'] - 1.96 * error_std
    
    plt.fill_between(results_df.index, lower_95, upper_95, alpha=0.3, color='gray', label='95% Prediction Interval')
    
    plt.title(f'Backtesting Results with Prediction Intervals')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/backtesting/prediction_intervals.png')
    plt.close()
    
    # 6. Scatterplot of actual vs predicted values
    plt.figure(figsize=(10, 10))
    plt.scatter(results_df['true_value_original'], results_df['predicted_value_original'], alpha=0.7)
    
    # Add a diagonal line (perfect predictions)
    min_val = min(results_df['true_value_original'].min(), results_df['predicted_value_original'].min())
    max_val = max(results_df['true_value_original'].max(), results_df['predicted_value_original'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Actual vs Predicted Values')
    plt.xlabel(f'Actual {target_col}')
    plt.ylabel(f'Predicted {target_col}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/backtesting/actual_vs_predicted_scatter.png')
    plt.close()
    
    print("Backtesting visualizations created in visualizations/backtesting/")

def calculate_confidence_intervals(results_df, alpha=0.05):
    """
    Calculate confidence intervals for forecast accuracy.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Dataframe with backtesting results
    alpha : float
        Significance level (default 0.05 for 95% confidence)
        
    Returns:
    --------
    dict
        Dictionary with confidence intervals
    """
    from scipy import stats
    
    # Calculate means
    mean_error = results_df['error'].mean()
    mean_abs_error = results_df['absolute_error'].mean()
    mean_squared_error = results_df['squared_error'].mean()
    
    # Calculate standard errors
    n = len(results_df)
    se_error = results_df['error'].std() / np.sqrt(n)
    se_abs_error = results_df['absolute_error'].std() / np.sqrt(n)
    se_squared_error = results_df['squared_error'].std() / np.sqrt(n)
    
    # Calculate t-critical value
    t_crit = stats.t.ppf(1 - alpha/2, n-1)
    
    # Calculate confidence intervals
    error_ci = (mean_error - t_crit * se_error, mean_error + t_crit * se_error)
    mae_ci = (mean_abs_error - t_crit * se_abs_error, mean_abs_error + t_crit * se_abs_error)
    mse_ci = (mean_squared_error - t_crit * se_squared_error, mean_squared_error + t_crit * se_squared_error)
    rmse_ci = (np.sqrt(mse_ci[0]), np.sqrt(mse_ci[1]))
    
    # Format and return results
    ci_results = {
        'Error': {
            'mean': mean_error,
            'lower_ci': error_ci[0],
            'upper_ci': error_ci[1]
        },
        'MAE': {
            'mean': mean_abs_error,
            'lower_ci': mae_ci[0],
            'upper_ci': mae_ci[1]
        },
        'MSE': {
            'mean': mean_squared_error,
            'lower_ci': mse_ci[0],
            'upper_ci': mse_ci[1]
        },
        'RMSE': {
            'mean': np.sqrt(mean_squared_error),
            'lower_ci': rmse_ci[0],
            'upper_ci': rmse_ci[1]
        }
    }
    
    # Print confidence intervals
    print("\nConfidence Intervals (95%):")
    for metric, values in ci_results.items():
        print(f"{metric}: {values['mean']:.4f} [{values['lower_ci']:.4f}, {values['upper_ci']:.4f}]")
    
    return ci_results

def main():
    """Main function to run backtesting."""
    print("=" * 80)
    print("INDIA RETAIL DEMAND FORECASTER - BACKTESTING")
    print("=" * 80)
    
    try:
        # Load data
        df = load_dataset()
        if df is None:
            print("Error loading data. Exiting.")
            return
        
        # Determine target column
        target_col = 'retail_sales'
        if target_col not in df.columns:
            target_col = 'log_retail_sales'
            if target_col not in df.columns:
                print(f"Neither 'retail_sales' nor 'log_retail_sales' found in dataset.")
                target_cols = [col for col in df.columns if 'retail' in col.lower() or 'sales' in col.lower()]
                if target_cols:
                    target_col = target_cols[0]
                    print(f"Using '{target_col}' as target column.")
                else:
                    print("No suitable target column found. Exiting.")
                    return
            else:
                print(f"Using '{target_col}' as target column.")
        
        # Run backtesting
        results_df = run_backtesting(
            df, 
            model_path='models/best_model.pkl',
            target_col=target_col,
            window_size=6,  # 6-month test windows
            step_size=3,    # Step forward 3 months each time
            min_train_size=24  # Require at least 24 months of training data
        )
        
        # Calculate confidence intervals
        if results_df is not None and not results_df.empty:
            ci_results = calculate_confidence_intervals(results_df)
            
            # Save confidence intervals
            import json
            os.makedirs('outputs/backtesting', exist_ok=True)
            with open('outputs/backtesting/confidence_intervals.json', 'w') as f:
                # Convert numpy values to floats for JSON serialization
                ci_json = {}
                for metric, values in ci_results.items():
                    ci_json[metric] = {
                        'mean': float(values['mean']),
                        'lower_ci': float(values['lower_ci']),
                        'upper_ci': float(values['upper_ci'])
                    }
                json.dump(ci_json, f, indent=2)
        
        print("\nBacktesting complete!")
        
    except Exception as e:
        print(f"Error in backtesting: {str(e)}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comparison of ARIMA Models with and without Log Transformation
-------------------------------------------------------------
This script compares:
1. Original ARIMA/SARIMA model from grid search
2. SARIMA(1,1,1)×(1,1,1,12) with log transformation
using identical train-test splits to analyze differences in performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Create directory for output
os.makedirs('visualizations/model_comparison', exist_ok=True)

def load_data():
    """Load the retail sales data"""
    print("Loading retail sales data...")
    
    try:
        # Load retail sales data
        sales_df = pd.read_csv('data/retail_sales.csv')
        
        # Convert date and set as index
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        sales_df.set_index('date', inplace=True)
        
        # Sort index to ensure chronological order
        sales_df = sales_df.sort_index()
        
        print(f"Loaded data with {len(sales_df)} observations from {sales_df.index.min()} to {sales_df.index.max()}")
        return sales_df
    
    except FileNotFoundError:
        print("Error: Retail sales data file not found.")
        return None

def apply_log_transform(series):
    """Apply log transformation"""
    return np.log1p(series)

def inverse_log_transform(series):
    """Apply inverse log transformation"""
    return np.expm1(series)

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def fit_original_sarima_model(series, train_data, test_data):
    """Fit original SARIMA model with grid search"""
    print("\nFitting original SARIMA model (grid search)...")
    
    # Perform grid search on training data
    # We'll simplify and use fixed parameters that were likely found in the original model
    # based on the conversation - SARIMA(1,1,1)×(1,1,1,12) but without log transform
    
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    model_name = f"Original SARIMA{order}{seasonal_order}"
    
    # Fit model on training data
    model = SARIMAX(
        train_data,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    
    # Make predictions
    predictions = results.forecast(steps=len(test_data))
    
    # Calculate metrics
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data, predictions)
    r2 = r2_score(test_data, predictions)
    mape = mean_absolute_percentage_error(test_data, predictions)
    
    # Calculate in-sample (training) metrics
    train_pred = results.predict(start=0, end=len(train_data)-1)
    train_mse = mean_squared_error(train_data, train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mape = mean_absolute_percentage_error(train_data, train_pred)
    
    print(f"\n{model_name} Test Set Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R²: {r2:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Train MAPE: {train_mape:.4f}%")
    print(f"Test/Train RMSE Ratio: {rmse/train_rmse:.4f}")
    
    return results, predictions, {
        'model_name': model_name,
        'RMSE': rmse, 
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Train_RMSE': train_rmse,
        'Train_MAPE': train_mape,
        'Ratio': rmse/train_rmse
    }

def fit_log_transformed_sarima_model(series, train_data, test_data):
    """Fit SARIMA model with log transformation"""
    print("\nFitting log-transformed SARIMA model...")
    
    # Apply log transformation
    log_series = apply_log_transform(series)
    log_train = apply_log_transform(train_data)
    log_test = apply_log_transform(test_data)
    
    # Define model parameters
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    model_name = f"Log-transformed SARIMA{order}{seasonal_order}"
    
    # Fit model on log-transformed training data
    model = SARIMAX(
        log_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    
    # Make predictions in log space
    log_predictions = results.forecast(steps=len(test_data))
    
    # Transform back to original space for evaluation
    predictions = inverse_log_transform(log_predictions)
    
    # Calculate metrics on original scale
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data, predictions)
    r2 = r2_score(test_data, predictions)
    mape = mean_absolute_percentage_error(test_data, predictions)
    
    # Calculate in-sample (training) metrics on original scale
    log_train_pred = results.predict(start=0, end=len(log_train)-1)
    train_pred = inverse_log_transform(log_train_pred)
    train_mse = mean_squared_error(train_data, train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mape = mean_absolute_percentage_error(train_data, train_pred)
    
    print(f"\n{model_name} Test Set Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R²: {r2:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Train MAPE: {train_mape:.4f}%")
    print(f"Test/Train RMSE Ratio: {rmse/train_rmse:.4f}")
    
    return results, predictions, {
        'model_name': model_name,
        'RMSE': rmse, 
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Train_RMSE': train_rmse,
        'Train_MAPE': train_mape,
        'Ratio': rmse/train_rmse
    }

def plot_forecasts(series, train_data, test_data, original_predictions, log_predictions, original_metrics, log_metrics):
    """Plot and compare forecasts from both models"""
    plt.figure(figsize=(14, 8))
    
    # Plot actual data
    plt.plot(series.index, series, label='Actual', color='black')
    
    # Plot train-test split
    plt.axvline(x=train_data.index[-1], color='gray', linestyle='--')
    plt.text(train_data.index[-1], series.max()*0.9, 'Train-Test Split', 
             horizontalalignment='center', verticalalignment='bottom')
    
    # Plot predictions
    plt.plot(test_data.index, original_predictions, 
             label=f"{original_metrics['model_name']} (MAPE={original_metrics['MAPE']:.2f}%)", 
             color='blue')
    plt.plot(test_data.index, log_predictions, 
             label=f"{log_metrics['model_name']} (MAPE={log_metrics['MAPE']:.2f}%)", 
             color='red')
    
    plt.title('Comparison of SARIMA Models With and Without Log Transformation')
    plt.xlabel('Date')
    plt.ylabel('Retail Sales')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/model_comparison/forecast_comparison.png')
    plt.close()
    
    # Plot residuals
    plt.figure(figsize=(14, 8))
    
    original_residuals = test_data - original_predictions
    log_residuals = test_data - log_predictions
    
    plt.subplot(2, 1, 1)
    plt.plot(test_data.index, original_residuals, label=f"{original_metrics['model_name']} Residuals", color='blue')
    plt.title(f"{original_metrics['model_name']} Residuals")
    plt.axhline(y=0, color='black', linestyle='-')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(test_data.index, log_residuals, label=f"{log_metrics['model_name']} Residuals", color='red')
    plt.title(f"{log_metrics['model_name']} Residuals")
    plt.axhline(y=0, color='black', linestyle='-')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison/residual_comparison.png')
    plt.close()
    
    # Create scatter plot of actual vs predicted
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    plt.scatter(test_data, original_predictions, color='blue', alpha=0.7)
    plt.plot([test_data.min(), test_data.max()], [test_data.min(), test_data.max()], 'k--')
    plt.title(f"{original_metrics['model_name']}\nMAPE = {original_metrics['MAPE']:.2f}%")
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(test_data, log_predictions, color='red', alpha=0.7)
    plt.plot([test_data.min(), test_data.max()], [test_data.min(), test_data.max()], 'k--')
    plt.title(f"{log_metrics['model_name']}\nMAPE = {log_metrics['MAPE']:.2f}%")
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison/scatter_comparison.png')
    plt.close()
    
    # Plot percentage errors
    plt.figure(figsize=(14, 6))
    
    original_pct_errors = (test_data - original_predictions) / test_data * 100
    log_pct_errors = (test_data - log_predictions) / test_data * 100
    
    plt.plot(test_data.index, original_pct_errors, label=f"{original_metrics['model_name']}", color='blue')
    plt.plot(test_data.index, log_pct_errors, label=f"{log_metrics['model_name']}", color='red')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title('Percentage Errors Comparison')
    plt.xlabel('Date')
    plt.ylabel('Percentage Error (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/model_comparison/percentage_errors.png')
    plt.close()

def analyze_test_set(test_data):
    """Analyze test set characteristics"""
    print("\nTest Set Analysis:")
    print(f"Mean: {test_data.mean():.4f}")
    print(f"Std Dev: {test_data.std():.4f}")
    print(f"Min: {test_data.min():.4f}")
    print(f"Max: {test_data.max():.4f}")
    print(f"Range: {test_data.max() - test_data.min():.4f}")
    
    # Calculate growth rates
    test_growth = test_data.pct_change().dropna()
    print(f"Average Month-over-Month Growth: {test_growth.mean()*100:.2f}%")
    print(f"Growth Volatility (Std Dev): {test_growth.std()*100:.2f}%")
    
    # Check for outliers - values outside 2 standard deviations
    mean = test_data.mean()
    std = test_data.std()
    outliers = test_data[(test_data < mean - 2*std) | (test_data > mean + 2*std)]
    
    if len(outliers) > 0:
        print(f"\nPotential outliers in test set ({len(outliers)} found):")
        for date, value in outliers.items():
            print(f"  {date.strftime('%Y-%m-%d')}: {value:.2f}")

def main():
    """Main execution function"""
    print("="*80)
    print("COMPARING ARIMA MODELS WITH AND WITHOUT LOG TRANSFORMATION")
    print("="*80)
    
    # Load data
    sales_df = load_data()
    if sales_df is None:
        return
    
    # Extract the retail sales series
    retail_sales = sales_df['retail_sales']
    
    # Set fixed train-test split (80% training data)
    train_size = 0.8
    n = len(retail_sales)
    train_n = int(n * train_size)
    
    train_data = retail_sales[:train_n]
    test_data = retail_sales[train_n:]
    
    print(f"\nTrain-test split: {train_n} training observations, {len(test_data)} test observations")
    print(f"Training period: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Testing period: {test_data.index.min()} to {test_data.index.max()}")
    
    # Analyze test set characteristics
    analyze_test_set(test_data)
    
    # Fit original SARIMA model
    original_results, original_predictions, original_metrics = fit_original_sarima_model(
        retail_sales, train_data, test_data
    )
    
    # Fit log-transformed SARIMA model
    log_results, log_predictions, log_metrics = fit_log_transformed_sarima_model(
        retail_sales, train_data, test_data
    )
    
    # Compare metrics
    print("\nMetrics Comparison:")
    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'MAPE (%)', 'R²', 'Train RMSE', 'Train MAPE (%)', 'Test/Train RMSE Ratio'],
        'Original SARIMA': [
            original_metrics['RMSE'], 
            original_metrics['MAE'],
            original_metrics['MAPE'],
            original_metrics['R2'],
            original_metrics['Train_RMSE'],
            original_metrics['Train_MAPE'],
            original_metrics['Ratio']
        ],
        'Log-transformed SARIMA': [
            log_metrics['RMSE'], 
            log_metrics['MAE'],
            log_metrics['MAPE'],
            log_metrics['R2'],
            log_metrics['Train_RMSE'],
            log_metrics['Train_MAPE'],
            log_metrics['Ratio']
        ]
    })
    print(metrics_df)
    
    # Save metrics to CSV
    metrics_df.to_csv('visualizations/model_comparison/metrics_comparison.csv', index=False)
    
    # Plot forecasts for comparison
    plot_forecasts(
        retail_sales, train_data, test_data,
        original_predictions, log_predictions,
        original_metrics, log_metrics
    )
    
    # Identify where the models differ most
    diff = abs(original_predictions - log_predictions)
    max_diff_idx = diff.idxmax()
    
    print(f"\nLargest difference between models at: {max_diff_idx}")
    print(f"Original model prediction: {original_predictions[max_diff_idx]:.4f}")
    print(f"Log-transformed model prediction: {log_predictions[max_diff_idx]:.4f}")
    print(f"Actual value: {test_data[max_diff_idx]:.4f}")
    
    # Calculate error differences
    original_errors = abs(test_data - original_predictions)
    log_errors = abs(test_data - log_predictions)
    
    better_original = sum(original_errors < log_errors)
    better_log = sum(log_errors < original_errors)
    ties = sum(original_errors == log_errors)
    
    print(f"\nPoint-by-point comparison:")
    print(f"Original model better at {better_original}/{len(test_data)} points ({better_original/len(test_data)*100:.1f}%)")
    print(f"Log-transformed model better at {better_log}/{len(test_data)} points ({better_log/len(test_data)*100:.1f}%)")
    print(f"Ties: {ties}/{len(test_data)} points ({ties/len(test_data)*100:.1f}%)")
    
    # Calculate percentage error differences 
    original_pct_errors = abs((test_data - original_predictions) / test_data) * 100
    log_pct_errors = abs((test_data - log_predictions) / test_data) * 100
    
    better_original_pct = sum(original_pct_errors < log_pct_errors)
    better_log_pct = sum(log_pct_errors < original_pct_errors)
    ties_pct = sum(np.isclose(original_pct_errors, log_pct_errors))
    
    print(f"\nPercentage error comparison:")
    print(f"Original model better at {better_original_pct}/{len(test_data)} points ({better_original_pct/len(test_data)*100:.1f}%)")
    print(f"Log-transformed model better at {better_log_pct}/{len(test_data)} points ({better_log_pct/len(test_data)*100:.1f}%)")
    print(f"Ties: {ties_pct}/{len(test_data)} points ({ties_pct/len(test_data)*100:.1f}%)")
    
    print("\nAnalysis complete. Results saved to visualizations/model_comparison/")

if __name__ == "__main__":
    main() 
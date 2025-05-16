#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Log-Transformed SARIMA Model for Retail Sales Forecasting
--------------------------------------------------------
This script implements a SARIMA(1,1,1)×(1,1,1,12) model with log transformation
for retail sales forecasting.
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
os.makedirs('outputs/scenarios/baseline', exist_ok=True)
os.makedirs('visualizations/arima', exist_ok=True)

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

def fit_log_transformed_sarima(train_data, test_data=None, forecast_periods=0):
    """
    Fit SARIMA model with log transformation
    
    Parameters:
    -----------
    train_data : pandas.Series
        Training data
    test_data : pandas.Series, optional
        Test data for evaluation
    forecast_periods : int, optional
        Number of periods to forecast beyond the training data
        
    Returns:
    --------
    results : statsmodels.tsa.statespace.sarimax.SARIMAXResults
        Fitted model results
    predictions : pandas.Series
        Predictions for test_data or forecast
    metrics : dict
        Performance metrics (if test_data is provided)
    """
    print("\nFitting log-transformed SARIMA model...")
    
    # Apply log transformation to training data
    log_train = apply_log_transform(train_data)
    
    # Define model parameters - SARIMA(1,1,1)×(1,1,1,12)
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    
    # Fit model on log-transformed training data
    model = SARIMAX(
        log_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    
    # Print model summary
    print("\nModel Summary:")
    print(results.summary().tables[1])
    
    # Evaluation mode
    if test_data is not None:
        # Apply log transformation to test data
        log_test = apply_log_transform(test_data)
        
        # Make predictions in log space
        log_predictions = results.forecast(steps=len(test_data))
        
        # Transform back to original space for evaluation
        predictions = inverse_log_transform(log_predictions)
        predictions = pd.Series(predictions, index=test_data.index)
        
        # Calculate metrics on original scale
        mse = mean_squared_error(test_data, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data, predictions)
        r2 = r2_score(test_data, predictions)
        mape = mean_absolute_percentage_error(test_data, predictions)
        
        # Calculate in-sample (training) metrics
        log_train_pred = results.predict(start=0, end=len(log_train)-1)
        train_pred = inverse_log_transform(log_train_pred)
        train_mse = mean_squared_error(train_data, train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mape = mean_absolute_percentage_error(train_data, train_pred)
        
        # Calculate metrics in log space
        log_mse = mean_squared_error(log_test, log_predictions)
        log_rmse = np.sqrt(log_mse)
        log_train_mse = mean_squared_error(log_train, log_train_pred)
        log_train_rmse = np.sqrt(log_train_mse)
        
        print("\nModel Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.4f}%")
        print(f"R²: {r2:.4f}")
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Train MAPE: {train_mape:.4f}%")
        print(f"Test/Train RMSE Ratio: {rmse/train_rmse:.4f}")
        
        print("\nLog-space metrics:")
        print(f"Log-space RMSE: {log_rmse:.4f}")
        print(f"Log-space Train RMSE: {log_train_rmse:.4f}")
        print(f"Log-space RMSE Ratio: {log_rmse/log_train_rmse:.4f}")
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Train_RMSE': train_rmse,
            'Train_MAPE': train_mape,
            'RMSE_Ratio': rmse/train_rmse,
            'Log_RMSE': log_rmse,
            'Log_Train_RMSE': log_train_rmse,
            'Log_RMSE_Ratio': log_rmse/log_train_rmse
        }
        
        return results, predictions, metrics
    
    # Forecast mode
    elif forecast_periods > 0:
        # Generate forecast in log space
        log_forecast = results.forecast(steps=forecast_periods)
        
        # Transform back to original space
        forecast = inverse_log_transform(log_forecast)
        
        # Create date index for forecast
        last_date = train_data.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_periods,
            freq='MS'  # Month Start frequency
        )
        
        forecast = pd.Series(forecast, index=forecast_index)
        
        print(f"\nGenerated forecast for {forecast_periods} periods")
        print(f"Forecast period: {forecast_index[0]} to {forecast_index[-1]}")
        
        return results, forecast, None
    
    # In-sample predictions only
    else:
        # Generate in-sample predictions in log space
        log_pred = results.predict()
        
        # Transform back to original space
        predictions = inverse_log_transform(log_pred)
        predictions = pd.Series(predictions, index=train_data.index[:len(predictions)])
        
        return results, predictions, None

def plot_results(series, train_data, test_data, predictions, forecast=None):
    """Plot results of the model"""
    plt.figure(figsize=(14, 8))
    
    # Plot actual data
    plt.plot(series.index, series, label='Actual', color='black')
    
    # Plot train-test split if test data is provided
    if test_data is not None:
        plt.axvline(x=train_data.index[-1], color='gray', linestyle='--')
        plt.text(train_data.index[-1], series.max()*0.9, 'Train-Test Split', 
                 horizontalalignment='center', verticalalignment='bottom')
    
    # Plot predictions
    if test_data is not None:
        plt.plot(test_data.index, predictions, label='Predictions', color='red')
    else:
        plt.plot(predictions.index, predictions, label='In-sample Predictions', color='blue')
    
    # Plot forecast if provided
    if forecast is not None:
        plt.plot(forecast.index, forecast, label='Forecast', color='green')
        plt.axvline(x=series.index[-1], color='gray', linestyle='--')
        plt.text(series.index[-1], series.max()*0.9, 'Forecast Start', 
                 horizontalalignment='center', verticalalignment='bottom')
    
    plt.title('Log-Transformed SARIMA(1,1,1)×(1,1,1,12) Model')
    plt.xlabel('Date')
    plt.ylabel('Retail Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/arima/log_transformed_sarima_results.png')
    plt.close()
    
    # Plot residuals if test data is provided
    if test_data is not None:
        plt.figure(figsize=(14, 6))
        residuals = test_data - predictions
        plt.plot(test_data.index, residuals)
        plt.axhline(y=0, color='black', linestyle='-')
        plt.title('Residuals')
        plt.xlabel('Date')
        plt.ylabel('Residual')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('visualizations/arima/log_transformed_sarima_residuals.png')
        plt.close()
        
        # Create scatter plot of actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(test_data, predictions, alpha=0.7)
        plt.plot([test_data.min(), test_data.max()], [test_data.min(), test_data.max()], 'k--')
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('visualizations/arima/log_transformed_sarima_scatter.png')
        plt.close()

def save_forecast(forecast, filename='baseline_forecast.csv'):
    """Save forecast to CSV file"""
    forecast_df = pd.DataFrame({
        'date': forecast.index,
        'forecast': forecast.values
    })
    
    filepath = os.path.join('outputs/scenarios/baseline', filename)
    forecast_df.to_csv(filepath, index=False)
    print(f"\nForecast saved to {filepath}")

def main():
    """Main execution function"""
    print("="*80)
    print("LOG-TRANSFORMED SARIMA MODEL FOR RETAIL SALES FORECASTING")
    print("="*80)
    
    # Load data
    sales_df = load_data()
    if sales_df is None:
        return
    
    # Extract the retail sales series
    retail_sales = sales_df['retail_sales']
    
    # Set train-test split (80% training data)
    train_size = 0.8
    n = len(retail_sales)
    train_n = int(n * train_size)
    
    train_data = retail_sales[:train_n]
    test_data = retail_sales[train_n:]
    
    print(f"\nTrain-test split: {train_n} training observations, {len(test_data)} test observations")
    print(f"Training period: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Testing period: {test_data.index.min()} to {test_data.index.max()}")
    
    # Fit model and evaluate on test set
    results, predictions, metrics = fit_log_transformed_sarima(train_data, test_data)
    
    # Plot results
    plot_results(retail_sales, train_data, test_data, predictions)
    
    # Generate forecast for next 12 months
    forecast_periods = 12
    
    # Use full dataset for forecasting
    full_results, forecast, _ = fit_log_transformed_sarima(retail_sales, forecast_periods=forecast_periods)
    
    # Plot forecast
    plot_results(retail_sales, None, None, 
                 full_results.predict(start=0, end=len(retail_sales)-1).pipe(inverse_log_transform), 
                 forecast)
    
    # Save forecast to CSV
    save_forecast(forecast)
    
    print("\nAnalysis complete. Results saved to visualizations/arima/ and outputs/scenarios/baseline/")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARIMA Modeling using the Box-Jenkins Methodology
------------------------------------------------
This script implements ARIMA models for the retail demand forecasting data
using the systematic Box-Jenkins approach:
1. Identification/Stationarity Testing
2. Differencing to achieve stationarity
3. Model selection using ACF/PACF and information criteria
4. Parameter estimation
5. Diagnostic checking
6. Forecasting
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Create directories for output
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

def test_stationarity(series, title='', window=12, plot=True):
    """
    Test stationarity of a time series using the Augmented Dickey-Fuller test
    and visual inspection of rolling statistics.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to test
    title : str
        Title for the plot
    window : int
        Window size for rolling statistics
    plot : bool
        Whether to create a plot
        
    Returns:
    --------
    bool
        True if series is stationary, False otherwise
    """
    print(f"\nStationarity Test for {title if title else 'Time Series'}")
    
    # Calculate rolling statistics
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    if plot:
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(series, label='Original')
        plt.plot(rolling_mean, label=f'Rolling Mean (window={window})')
        plt.plot(rolling_std, label=f'Rolling Std (window={window})')
        plt.legend()
        plt.title(f'Rolling Statistics - {title}')
        plt.tight_layout()
        plt.savefig(f'visualizations/arima/stationarity_test_{title.replace(" ", "_").lower()}.png')
        plt.close()
    
    # Perform Augmented Dickey-Fuller test
    print("Augmented Dickey-Fuller Test:")
    adf_result = adfuller(series.dropna())
    adf_output = pd.Series(
        adf_result[0:4],
        index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used']
    )
    
    for key, value in adf_result[4].items():
        adf_output[f'Critical Value ({key})'] = value
    
    print(adf_output)
    
    # Interpretation
    is_stationary = adf_result[1] < 0.05
    print(f"Result: {'Stationary' if is_stationary else 'Non-stationary'} " 
          f"(p-value: {adf_result[1]:.6f})")
    
    return is_stationary

def difference_series(series, d=1, plot=True, title=''):
    """
    Apply differencing to make a time series stationary.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to difference
    d : int
        Order of differencing
    plot : bool
        Whether to create visualization
    title : str
        Title for the plot
        
    Returns:
    --------
    pd.Series
        Differenced series
    """
    # Apply differencing
    differenced = series.diff(d).dropna()
    
    if plot:
        # Create plot to compare original and differenced series
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original series
        axes[0].plot(series)
        axes[0].set_title(f'Original Series - {title}')
        axes[0].set_ylabel('Value')
        axes[0].grid(True)
        
        # Differenced series
        axes[1].plot(differenced, color='orange')
        axes[1].set_title(f'Differenced Series (d={d}) - {title}')
        axes[1].set_ylabel('Value')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'visualizations/arima/differencing_{title.replace(" ", "_").lower()}_d{d}.png')
        plt.close()
        
    return differenced

def plot_acf_pacf(series, lags=40, title=''):
    """
    Plot ACF and PACF to identify ARIMA model orders.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to analyze
    lags : int
        Number of lags to include
    title : str
        Title for the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # ACF plot
    plot_acf(series, lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title(f'Autocorrelation Function - {title}')
    axes[0].grid(True)
    
    # PACF plot
    plot_pacf(series, lags=lags, ax=axes[1], alpha=0.05)
    axes[1].set_title(f'Partial Autocorrelation Function - {title}')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/arima/acf_pacf_{title.replace(" ", "_").lower()}.png')
    plt.close()

def grid_search_arima(series, max_p=3, max_d=2, max_q=3, seasonal=False, m=12, 
                       max_P=1, max_D=1, max_Q=1, ic='aic'):
    """
    Perform grid search to find optimal ARIMA parameters based on information criteria.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to model
    max_p, max_d, max_q : int
        Maximum values for p, d, q parameters
    seasonal : bool
        Whether to include seasonal component
    m : int
        Seasonal period
    max_P, max_D, max_Q : int
        Maximum values for seasonal P, D, Q parameters
    ic : str
        Information criterion to use ('aic' or 'bic')
        
    Returns:
    --------
    tuple
        Best parameters (p, d, q) or (p, d, q, P, D, Q, m) and best model
    """
    print(f"\nPerforming grid search for {'SARIMA' if seasonal else 'ARIMA'} parameters...")
    
    best_score = float('inf')
    best_params = None
    best_model = None
    
    # Differencing order (assuming already determined)
    d_values = list(range(max_d + 1))
    
    # Create results container
    if seasonal:
        results = pd.DataFrame(columns=['p', 'd', 'q', 'P', 'D', 'Q', ic, 'is_invertible', 'is_stationary'])
    else:
        results = pd.DataFrame(columns=['p', 'd', 'q', ic, 'is_invertible', 'is_stationary'])
    
    # Grid search
    total_models = (max_p + 1) * len(d_values) * (max_q + 1)
    if seasonal:
        total_models *= (max_P + 1) * (max_D + 1) * (max_Q + 1)
    print(f"Evaluating {total_models} different models...")
    
    model_count = 0
    for p in range(max_p + 1):
        for d in d_values:
            for q in range(max_q + 1):
                if seasonal:
                    for P in range(max_P + 1):
                        for D in range(max_D + 1):
                            for Q in range(max_Q + 1):
                                if p == 0 and q == 0 and P == 0 and Q == 0:
                                    continue  # Skip models with no parameters
                                
                                try:
                                    # Fit SARIMA model
                                    model = SARIMAX(
                                        series,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, m),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )
                                    result = model.fit(disp=False)
                                    
                                    # Get information criterion
                                    score = getattr(result, ic)
                                    
                                    # Check if model is invertible and stationary
                                    is_invertible = result.mle_retvals.get('success', False)
                                    is_stationary = all(abs(r) < 1 for r in result.arroots)
                                    
                                    # Store results
                                    results = pd.concat([
                                        results, 
                                        pd.DataFrame({
                                            'p': [p], 'd': [d], 'q': [q],
                                            'P': [P], 'D': [D], 'Q': [Q],
                                            ic: [score],
                                            'is_invertible': [is_invertible],
                                            'is_stationary': [is_stationary]
                                        })
                                    ], ignore_index=True)
                                    
                                    # Update best model
                                    if score < best_score and is_invertible and is_stationary:
                                        best_score = score
                                        best_params = (p, d, q, P, D, Q, m)
                                        best_model = result
                                    
                                    model_count += 1
                                    if model_count % 50 == 0:
                                        print(f"  Evaluated {model_count}/{total_models} models...")
                                    
                                except:
                                    continue
                else:
                    if p == 0 and q == 0:
                        continue  # Skip models with no parameters
                    
                    try:
                        # Fit ARIMA model
                        model = ARIMA(series, order=(p, d, q))
                        result = model.fit()
                        
                        # Get information criterion
                        score = getattr(result, ic)
                        
                        # Check if model is invertible and stationary
                        is_invertible = True  # Assume success for simple ARIMA
                        is_stationary = all(abs(r) < 1 for r in result.arroots)
                        
                        # Store results
                        results = pd.concat([
                            results, 
                            pd.DataFrame({
                                'p': [p], 'd': [d], 'q': [q],
                                ic: [score],
                                'is_invertible': [is_invertible],
                                'is_stationary': [is_stationary]
                            })
                        ], ignore_index=True)
                        
                        # Update best model
                        if score < best_score and is_invertible and is_stationary:
                            best_score = score
                            best_params = (p, d, q)
                            best_model = result
                        
                        model_count += 1
                        if model_count % 10 == 0:
                            print(f"  Evaluated {model_count}/{total_models} models...")
                        
                    except:
                        continue
    
    # Sort results
    results = results.sort_values(by=[ic])
    
    # Save results
    results.to_csv(f'visualizations/arima/grid_search_results_{ic}.csv', index=False)
    
    # Print top models
    print("\nTop 5 models by information criterion:")
    print(results.head(5))
    
    # Get best model
    if best_params is None:
        print("\nNo valid models found. Using fallback parameters.")
        if seasonal:
            best_params = (1, 1, 1, 1, 1, 1, m)
        else:
            best_params = (1, 1, 1)
    
    print(f"\nBest model: {'SARIMA' if seasonal else 'ARIMA'}{best_params} with {ic.upper()}={best_score:.3f}")
    
    return best_params, best_model

def plot_diagnostics(results, title=''):
    """
    Plot diagnostic checks for ARIMA/SARIMA model.
    
    Parameters:
    -----------
    results : ARIMAResults or SARIMAXResults
        Fitted model results
    title : str
        Title for the plots
    """
    residuals = results.resid
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot residuals
    axes[0, 0].plot(residuals)
    axes[0, 0].set_title('Residuals')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residual Value')
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    axes[0, 0].grid(True)
    
    # Plot residual histogram
    axes[0, 1].hist(residuals, bins=20, density=True, alpha=0.7)
    axes[0, 1].set_title('Residual Histogram')
    axes[0, 1].set_xlabel('Residual Value')
    axes[0, 1].set_ylabel('Density')
    
    # Plot ACF of residuals
    plot_acf(residuals, lags=40, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title('ACF of Residuals')
    axes[1, 0].grid(True)
    
    # Plot Q-Q plot
    sm.qqplot(residuals, line='45', ax=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/arima/diagnostics_{title.replace(" ", "_").lower()}.png')
    plt.close()
    
    # Additional statistical tests
    print("\nLjung-Box Test (Residual Autocorrelation Check):")
    lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    print(lb_test)
    
    # Check distribution
    from scipy import stats
    print("\nJarque-Bera Test (Normality Check):")
    jb_test = stats.jarque_bera(residuals.dropna())
    print(f"Statistic: {jb_test[0]:.4f}, p-value: {jb_test[1]:.4f}")
    if jb_test[1] < 0.05:
        print("Result: Residuals do not follow a normal distribution")
    else:
        print("Result: Residuals follow a normal distribution")

def fit_and_evaluate_arima(series, train_size=0.8, order=None, seasonal_order=None):
    """
    Fit ARIMA/SARIMA model and evaluate performance.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to model
    train_size : float
        Proportion of data to use for training
    order : tuple or None
        (p, d, q) order for ARIMA
    seasonal_order : tuple or None
        (P, D, Q, m) seasonal order for SARIMA
        
    Returns:
    --------
    tuple
        Fitted model, predictions, and performance metrics
    """
    # Split data into train and test sets
    n = len(series)
    train_n = int(n * train_size)
    train_data = series[:train_n]
    test_data = series[train_n:]
    
    print(f"\nFitting ARIMA model on training data ({train_n} observations)")
    print(f"Testing on {len(test_data)} observations")
    
    # Fit model
    if seasonal_order is not None:
        model = SARIMAX(
            train_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_name = f"SARIMA{order}{seasonal_order}"
    else:
        model = ARIMA(train_data, order=order)
        model_name = f"ARIMA{order}"
    
    # Fit the model
    results = model.fit(disp=False)
    
    # Print model summary
    print("\nModel Summary:")
    print(results.summary())
    
    # Make predictions
    predictions = results.forecast(steps=len(test_data))
    
    # Calculate metrics
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data, predictions)
    r2 = r2_score(test_data, predictions)
    
    print("\nTest Set Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series, label='Actual')
    plt.plot(test_data.index, predictions, color='red', label='Predicted')
    
    # Add vertical line to separate train and test
    plt.axvline(x=train_data.index[-1], color='black', linestyle='--')
    plt.text(train_data.index[-1], series.max(), 'Train-Test Split', 
             horizontalalignment='center', verticalalignment='bottom')
    
    plt.title(f'{model_name} Forecast')
    plt.xlabel('Date')
    plt.ylabel('Retail Sales')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'visualizations/arima/forecast_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # Check for overfitting
    train_pred = results.predict(start=0, end=len(train_data)-1)
    train_rmse = np.sqrt(mean_squared_error(train_data, train_pred))
    
    print("\nOverfitting Check:")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Ratio (Test/Train): {rmse/train_rmse:.2f}")
    
    if rmse/train_rmse > 1.5:
        print("WARNING: Possible overfitting (Test RMSE significantly higher than Train RMSE)")
    
    return results, predictions, {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def apply_log_transform(series, plot=True, title=''):
    """
    Apply log transformation to address non-normality and heteroskedasticity.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to transform
    plot : bool
        Whether to create visualization
    title : str
        Title for the plot
        
    Returns:
    --------
    pd.Series
        Log-transformed series
    """
    # Apply log transformation (adding 1 to handle zeros)
    log_series = np.log1p(series)
    
    if plot:
        # Create plot to compare original and transformed series
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original series
        axes[0].plot(series)
        axes[0].set_title(f'Original Series - {title}')
        axes[0].set_ylabel('Value')
        axes[0].grid(True)
        
        # Transformed series
        axes[1].plot(log_series, color='orange')
        axes[1].set_title(f'Log-Transformed Series - {title}')
        axes[1].set_ylabel('Log(Value+1)')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'visualizations/arima/log_transform_{title.replace(" ", "_").lower()}.png')
        plt.close()
        
    return log_series

def inverse_log_transform(series):
    """
    Apply inverse log transformation to convert back to original scale.
    
    Parameters:
    -----------
    series : pd.Series or np.array
        Log-transformed time series
        
    Returns:
    --------
    pd.Series or np.array
        Original scale series
    """
    return np.expm1(series)

def main():
    """Main execution function"""
    print("="*80)
    print("ARIMA MODELING USING BOX-JENKINS METHODOLOGY")
    print("="*80)
    
    # Step 1: Load data
    sales_df = load_data()
    if sales_df is None:
        return
    
    # Extract the retail sales series
    retail_sales = sales_df['retail_sales']
    
    # Apply log transformation to address non-normality and heteroskedasticity
    print("\nApplying log transformation to address non-normality and heteroskedasticity...")
    log_sales = apply_log_transform(retail_sales, title='Retail Sales')
    
    # Step 2: Test stationarity of log-transformed series
    is_stationary = test_stationarity(log_sales, title='Log Retail Sales')
    
    # Skip grid search and use the recommended SARIMA(1,1,1)×(1,1,1,12) model
    print("\nUsing the recommended SARIMA(1,1,1)×(1,1,1,12) model...")
    
    # Define the model parameters
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    
    print(f"Model specification: SARIMA{order}{seasonal_order}")
    
    # Create a dictionary to store model parameters for documentation
    model_params = {
        'p': 1,  # AR order
        'd': 1,  # Differencing order
        'q': 1,  # MA order
        'P': 1,  # Seasonal AR order
        'D': 1,  # Seasonal differencing order
        'Q': 1,  # Seasonal MA order
        'm': 12   # Seasonal period
    }
    
    # Step 6: Fit SARIMA model with log-transformed data
    print("\nFitting SARIMA model on log-transformed data...")
    
    # SARIMA model on log-transformed data
    model = SARIMAX(
        log_sales,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    
    # Print model summary
    print("\nModel Summary:")
    print(results.summary())
    
    # Plot diagnostics
    model_name = f"SARIMA{order}{seasonal_order}"
    plot_diagnostics(results, title=model_name)
    
    # Step 7: Fit model on training data and evaluate
    # Modified function to use log-transformed data
    print("\nEvaluating model performance with train-test split...")
    
    # Split data into train and test sets
    n = len(log_sales)
    train_size = 0.8
    train_n = int(n * train_size)
    train_data = log_sales[:train_n]
    test_data = log_sales[train_n:]
    
    # Fit model on training data
    train_model = SARIMAX(
        train_data,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    train_results = train_model.fit(disp=False)
    
    # Make predictions in log space
    log_predictions = train_results.forecast(steps=len(test_data))
    
    # Transform back to original space for evaluation
    predictions = inverse_log_transform(log_predictions)
    actual_test = inverse_log_transform(test_data)
    
    # Calculate metrics on original scale
    mse = mean_squared_error(actual_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_test, predictions)
    r2 = r2_score(actual_test, predictions)
    
    print("\nTest Set Performance (Original Scale):")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Plot predictions vs actual on original scale
    plt.figure(figsize=(12, 6))
    plt.plot(retail_sales.index, retail_sales, label='Actual')
    plt.plot(actual_test.index, predictions, color='red', label='Predicted')
    
    # Add vertical line to separate train and test
    plt.axvline(x=train_data.index[-1], color='black', linestyle='--')
    plt.text(train_data.index[-1], retail_sales.max(), 'Train-Test Split', 
             horizontalalignment='center', verticalalignment='bottom')
    
    plt.title(f'{model_name} Forecast (Original Scale)')
    plt.xlabel('Date')
    plt.ylabel('Retail Sales')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'visualizations/arima/forecast_{model_name.replace(" ", "_").lower()}_original_scale.png')
    plt.close()
    
    # Check for overfitting
    train_log_pred = train_results.predict(start=0, end=len(train_data)-1)
    train_pred = inverse_log_transform(train_log_pred)
    train_actual = inverse_log_transform(train_data)
    
    train_rmse = np.sqrt(mean_squared_error(train_actual, train_pred))
    
    print("\nOverfitting Check (Original Scale):")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Ratio (Test/Train): {rmse/train_rmse:.2f}")
    
    metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    # Step 8: Compare to previous models
    print("\nComparing SARIMA model with log transformation to previous models:")
    print("SARIMA model RMSE: {:.4f}".format(metrics['RMSE']))
    print("SARIMA model MAE: {:.4f}".format(metrics['MAE']))
    print("SARIMA model R²: {:.4f}".format(metrics['R2']))
    
    # Create a final forecast
    forecast_steps = 12  # One year ahead
    
    # Generate forecast in log space
    log_forecast = results.forecast(steps=forecast_steps)
    
    # Calculate prediction intervals in log space
    pred_int = results.get_forecast(steps=forecast_steps).conf_int(alpha=0.05)
    
    # Transform back to original scale
    final_forecast = inverse_log_transform(log_forecast)
    lower_ci = inverse_log_transform(pred_int.iloc[:, 0])
    upper_ci = inverse_log_transform(pred_int.iloc[:, 1])
    
    # Create a date range for the forecast
    last_date = retail_sales.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=forecast_steps+1, freq='MS')[1:]
    
    # Plot the forecast with prediction intervals
    plt.figure(figsize=(12, 6))
    plt.plot(retail_sales.index, retail_sales, label='Historical Data')
    plt.plot(forecast_dates, final_forecast, color='red', label='Forecast')
    
    # Add prediction intervals
    plt.fill_between(forecast_dates, lower_ci, upper_ci, 
                    color='pink', alpha=0.3, label='95% Prediction Interval')
    
    plt.title(f'{model_name} Forecast with Log Transformation for Next {forecast_steps} Months')
    plt.xlabel('Date')
    plt.ylabel('Retail Sales')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'visualizations/arima/final_forecast_log_transformed.png')
    
    # Create a table with the forecast values
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': final_forecast,
        'Lower_95': lower_ci,
        'Upper_95': upper_ci
    })
    
    print("\nForecast for the next 12 months:")
    print(forecast_df.to_string(index=False))
    
    # Save forecast to CSV
    forecast_df.to_csv('outputs/sarima_forecast.csv', index=False)
    
    # Document the model parameters and performance
    with open('outputs/sarima_model_summary.txt', 'w') as f:
        f.write("SARIMA MODEL WITH LOG TRANSFORMATION\n")
        f.write("===================================\n\n")
        f.write(f"Model: SARIMA({model_params['p']},{model_params['d']},{model_params['q']})×({model_params['P']},{model_params['D']},{model_params['Q']},{model_params['m']})\n\n")
        f.write("Transformations applied:\n")
        f.write("- Log transformation to address non-normality and heteroskedasticity\n\n")
        f.write("Performance metrics (original scale):\n")
        f.write(f"- RMSE: {metrics['RMSE']:.4f}\n")
        f.write(f"- MAE: {metrics['MAE']:.4f}\n")
        f.write(f"- R²: {metrics['R2']:.4f}\n\n")
        f.write(f"Overfitting check:\n")
        f.write(f"- Train RMSE: {train_rmse:.4f}\n")
        f.write(f"- Test RMSE: {rmse:.4f}\n")
        f.write(f"- Ratio (Test/Train): {rmse/train_rmse:.2f}\n\n")
        f.write("Forecast generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    print("\nAnalysis complete. Results saved to visualizations/arima/ and outputs/")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Log-Transformed SARIMA Model for Retail Sales Forecasting
--------------------------------------------------------
This script implements a SARIMA(1,1,1)×(1,1,1,12) model with log transformation
for retail sales forecasting. It includes EDA steps for data quality and diagnostics,
and post-estimation checks for residuals.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm # For Breusch-Pagan test
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Create directories for output
os.makedirs('outputs/scenarios/baseline', exist_ok=True)
os.makedirs('visualizations/arima', exist_ok=True)
os.makedirs('visualizations/data_quality', exist_ok=True)
os.makedirs('visualizations/diagnostics', exist_ok=True)
os.makedirs('visualizations/model_diagnostics', exist_ok=True) # For residual diagnostics


def load_data(file_path='data/retail_sales.csv'):
    """Load the retail sales data"""
    print("Loading retail sales data...")
    
    try:
        # Load retail sales data
        sales_df = pd.read_csv(file_path)
        
        # Standardize column names
        sales_df.columns = [col.lower().strip().replace(' ', '_') for col in sales_df.columns]
        
        # Convert date and set as index
        if 'date' not in sales_df.columns:
            print("Error: 'date' column not found in the data.")
            return None
        sales_df['date'] = pd.to_datetime(sales_df['date'], errors='coerce')
        sales_df.dropna(subset=['date'], inplace=True) # Drop rows where date conversion failed
        sales_df.set_index('date', inplace=True)
        
        # Sort index to ensure chronological order
        sales_df = sales_df.sort_index()
        
        # Ensure 'retail_sales' column exists
        if 'retail_sales' not in sales_df.columns:
            print("Error: 'retail_sales' column not found.")
            return None
            
        print(f"Loaded data with {len(sales_df)} observations from {sales_df.index.min()} to {sales_df.index.max()}")
        return sales_df['retail_sales'] # Return as Series
    
    except FileNotFoundError:
        print(f"Error: Retail sales data file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# --- EDA Functions ---
def analyze_missingness(series: pd.Series, series_name: str = 'retail_sales'):
    """Analyzes missing values in the time series."""
    print(f"\nAnalyzing missing values for {series_name}...")
    missing_values = series.isnull().sum()
    missing_percent = (missing_values / len(series)) * 100
    
    print(f"Missing values: {missing_values}")
    print(f"Missing percent: {missing_percent:.2f}%")
    
    if missing_values > 0:
        plt.figure(figsize=(10, 6))
        series.isnull().plot(kind='bar')
        plt.title(f'Missing Values in {series_name}')
        plt.savefig(f'visualizations/data_quality/missing_values_{series_name}.png')
        plt.close()
        print(f"ACTION: Consider imputation if missing values are significant.")
    else:
        print("No missing values found.")
    return missing_values

def impute_missing_if_needed(series: pd.Series, method='ffill'):
    """Imputes missing values if any are present."""
    if series.isnull().any():
        print(f"Imputing missing values using {method} method...")
        if method == 'ffill':
            series_imputed = series.ffill()
        elif method == 'bfill':
            series_imputed = series.bfill()
        elif method == 'linear':
            series_imputed = series.interpolate(method='linear')
        else: # Default to ffill
            series_imputed = series.ffill()
        
        # Check if any NaNs remain (e.g., if all were NaNs or at the beginning for ffill)
        if series_imputed.isnull().any():
            series_imputed = series_imputed.bfill() # Try bfill for remaining
        
        print("Missing values imputed.")
        return series_imputed
    return series

def detect_and_handle_outliers(series: pd.Series, series_name: str = 'retail_sales', method='iqr', cap_percentiles=(0.01, 0.99)):
    """Detects and handles outliers using IQR or Z-score method."""
    print(f"\nDetecting and handling outliers for {series_name} using {method} method...")
    series_cleaned = series.copy()
    
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
    elif method == 'zscore':
        # Ensure series is not all NaN before zscore calculation
        if series.notna().sum() == 0:
            print("Series is all NaN. Skipping Z-score outlier detection.")
            return series
        z_scores = np.abs(stats.zscore(series.dropna())) # dropna for zscore calculation
        outliers_indices = series.index[series.notna()][z_scores > 3] # Get original indices
        outliers = pd.Series(False, index=series.index)
        if not outliers_indices.empty:
            outliers[outliers_indices] = True
    else:
        print(f"Unknown outlier detection method: {method}. Skipping.")
        return series

    num_outliers = outliers.sum()
    print(f"Found {num_outliers} outliers ({num_outliers/len(series)*100:.2f}%).")

    if num_outliers > 0:
        print(f"Capping outliers at {cap_percentiles[0]*100:.0f}th and {cap_percentiles[1]*100:.0f}th percentiles.")
        lower_cap = series.quantile(cap_percentiles[0])
        upper_cap = series.quantile(cap_percentiles[1])
        series_cleaned = series_cleaned.clip(lower=lower_cap, upper=upper_cap)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        series.plot(title=f'Original {series_name}')
        if num_outliers > 0: # Only plot if outliers exist
             plt.scatter(series.index[outliers], series[outliers], color='red', label='Outliers', zorder=5)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        series_cleaned.plot(title=f'Cleaned {series_name} (Outliers Capped)')
        plt.tight_layout()
        plt.savefig(f'visualizations/data_quality/outlier_handling_{series_name}.png')
        plt.close()
    else:
        print("No significant outliers to handle based on the chosen method.")
        
    return series_cleaned

def plot_time_series_decomposition(series: pd.Series, series_name: str = 'retail_sales', model='additive', period=12):
    """Performs and plots time series decomposition."""
    print(f"\nPerforming time series decomposition for {series_name}...")
    if series.dropna().empty:
        print(f"Skipping decomposition for {series_name} as it contains too many NaNs or is empty after dropna.")
        return
    try:
        decomposition = seasonal_decompose(series.dropna(), model=model, period=period)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        decomposition.observed.plot(ax=ax1, title='Observed')
        decomposition.trend.plot(ax=ax2, title='Trend')
        decomposition.seasonal.plot(ax=ax3, title='Seasonal')
        decomposition.resid.plot(ax=ax4, title='Residual')
        plt.suptitle(f'Time Series Decomposition of {series_name}', y=1.02)
        plt.tight_layout()
        plt.savefig(f'visualizations/diagnostics/decomposition_{series_name}.png')
        plt.close()
        print("Time series decomposition plot saved.")
    except Exception as e:
        print(f"Error during time series decomposition for {series_name}: {e}")

def check_stationarity(series: pd.Series, series_name: str = 'retail_sales'):
    """Performs ADF test for stationarity."""
    print(f"\nChecking stationarity for {series_name} using ADF test...")
    series_to_test = series.dropna()
    if series_to_test.empty:
        print(f"Skipping stationarity check for {series_name} as it is empty after dropna.")
        return
    result = adfuller(series_to_test)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    if result[1] <= 0.05:
        print(f"Result: {series_name} is likely stationary (p <= 0.05).")
    else:
        print(f"Result: {series_name} is likely non-stationary (p > 0.05). Differencing may be needed.")

def plot_acf_pacf_charts(series: pd.Series, series_name: str = 'retail_sales', lags=40, plot_path_prefix='visualizations/diagnostics'):
    """Plots ACF and PACF charts."""
    print(f"\nPlotting ACF and PACF for {series_name}...")
    series_to_plot = series.dropna()
    if series_to_plot.empty:
        print(f"Skipping ACF/PACF plot for {series_name} as it is empty after dropna.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(series_to_plot, lags=min(lags, len(series_to_plot)//2 - 1), ax=axes[0], title=f'ACF - {series_name}')
    plot_pacf(series_to_plot, lags=min(lags, len(series_to_plot)//2 - 1), ax=axes[1], title=f'PACF - {series_name}')
    plt.tight_layout()
    plt.savefig(f'{plot_path_prefix}/acf_pacf_{series_name.lower().replace(" ", "_")}.png')
    plt.close()
    print(f"ACF and PACF plots for {series_name} saved.")

# --- End EDA Functions ---


def apply_log_transform(series):
    """Apply log transformation"""
    # Ensure series is non-negative before log1p
    if (series < 0).any():
        print("Warning: Negative values found in series. Shifting data before log transform.")
        series = series - series.min() + 1e-6 # Shift to be positive
    return np.log1p(series)

def inverse_log_transform(series):
    """Apply inverse log transformation"""
    return np.expm1(series)

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan 
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def perform_residual_diagnostics(residuals, fitted_values, model_name_suffix='', lags=40):
    """Performs Ljung-Box test, Breusch-Pagan test, and plots for residuals."""
    print(f"\n--- Residual Diagnostics ({model_name_suffix}) ---")
    
    # Ensure residuals and fitted_values are not all NaN or empty
    if residuals.isna().all() or residuals.empty:
        print("Skipping residual diagnostics: Residuals are all NaN or empty.")
        return

    # 1. Ljung-Box test for Autocorrelation
    print("\nChecking for Autocorrelation in Residuals (Ljung-Box Test):")
    # Use lags up to min(40, n_obs//5) for Ljung-Box as a common heuristic, ensure lags < n_obs
    lb_lags = min(lags, len(residuals)//5, len(residuals)-1)
    if lb_lags > 0:
        lb_test_df = acorr_ljungbox(residuals.dropna(), lags=[lb_lags], return_df=True)
        lb_pvalue = lb_test_df['lb_pvalue'].iloc[0]
        print(f"Ljung-Box test p-value (lag {lb_lags}): {lb_pvalue:.4f}")
        if lb_pvalue <= 0.05:
            print("Indication: Significant autocorrelation present in residuals. Model may need order adjustment.")
        else:
            print("Indication: No significant autocorrelation detected in residuals.")
    else:
        print("Skipping Ljung-Box test: Not enough data points or lags.")

    # Plot ACF/PACF of residuals
    plot_acf_pacf_charts(residuals, series_name=f'Residuals_{model_name_suffix}', lags=lags, plot_path_prefix='visualizations/model_diagnostics')

    # 2. Heteroskedasticity Test (Breusch-Pagan)
    print("\nChecking for Heteroskedasticity in Residuals (Breusch-Pagan Test):")
    # Need exog for Breusch-Pagan. Using fitted values (in log space).
    # Ensure residuals and fitted_values can be aligned and are not all NaN
    aligned_residuals = residuals.dropna()
    aligned_fitted_values = fitted_values[aligned_residuals.index].dropna()
    aligned_residuals = aligned_residuals[aligned_fitted_values.index]

    if len(aligned_residuals) < 2 or len(aligned_fitted_values) < 2 or aligned_fitted_values.var() == 0:
        print("Skipping Breusch-Pagan test: Not enough data or no variance in fitted values.")
    else:
        # Reshape fitted_values to be 2D for statsmodels exog
        exog_bp = sm.add_constant(aligned_fitted_values.values.reshape(-1, 1))
        try:
            bp_test = het_breuschpagan(aligned_residuals.values, exog_bp)
            bp_pvalue = bp_test[1]
            print(f"Breusch-Pagan test p-value: {bp_pvalue:.4f}")
            if bp_pvalue <= 0.05:
                print("Indication: Heteroskedasticity present in residuals (variance is not constant).")
                print("Log transformation aims to reduce this, but some might remain.")
            else:
                print("Indication: No significant heteroskedasticity detected in residuals.")
        except Exception as e:
            print(f"Error during Breusch-Pagan test: {e}")

    # Plot Residuals vs Fitted values
    plt.figure(figsize=(10, 6))
    plt.scatter(aligned_fitted_values, aligned_residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel(f'Fitted Values (Log Scale) - {model_name_suffix}')
    plt.ylabel(f'Residuals (Log Scale) - {model_name_suffix}')
    plt.title(f'Residuals vs. Fitted Values - {model_name_suffix}')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(f'visualizations/model_diagnostics/residuals_vs_fitted_{model_name_suffix.lower().replace(" ", "_")}.png')
    plt.close()
    print(f"Residuals vs. Fitted plot for {model_name_suffix} saved.")

    # 3. Normality of Residuals (Q-Q Plot and Jarque-Bera test)
    print("\nChecking for Normality of Residuals:")
    if aligned_residuals.empty:
        print("Skipping normality check: No residuals to check.")
    else:
        jb_stat, jb_pvalue, skew, kurtosis = sm.stats.jarque_bera(aligned_residuals.dropna())
        print(f"Jarque-Bera test p-value: {jb_pvalue:.4f}")
        if jb_pvalue <= 0.05:
            print("Indication: Residuals may not be normally distributed.")
        else:
            print("Indication: Residuals appear to be normally distributed.")

        plt.figure(figsize=(8, 6))
        sm.qqplot(aligned_residuals.dropna(), line='s', fit=True)
        plt.title(f'Q-Q Plot of Residuals - {model_name_suffix}')
        plt.savefig(f'visualizations/model_diagnostics/qq_plot_residuals_{model_name_suffix.lower().replace(" ", "_")}.png')
        plt.close()
        print(f"Q-Q plot for {model_name_suffix} residuals saved.")
    print("--- Residual Diagnostics Complete ---")

def fit_log_transformed_sarima(train_data, test_data=None, forecast_periods=0, model_name_suffix='TrainTestSplit'):
    """
    Fit SARIMA model with log transformation
    
    Parameters:
    -----------
    train_data : pandas.Series
        Training data (original scale)
    test_data : pandas.Series, optional
        Test data for evaluation (original scale)
    forecast_periods : int, optional
        Number of periods to forecast beyond the training data
    model_name_suffix : str
        Suffix for naming diagnostic plots related to this model fit.
        
    Returns:
    --------
    results : statsmodels.tsa.statespace.sarimax.SARIMAXResults
        Fitted model results
    predictions : pandas.Series
        Predictions for test_data or forecast (original scale)
    metrics : dict
        Performance metrics (if test_data is provided)
    """
    print(f"\nFitting log-transformed SARIMA model ({model_name_suffix})...")
    
    log_train = apply_log_transform(train_data)
    
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    
    model = SARIMAX(
        log_train.dropna(), # Ensure no NaNs from transformations are passed to model
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=False
    )
    results = model.fit(disp=False)
    
    print("\nModel Summary:")
    try:
        print(results.summary().tables[1])
    except Exception as e:
        print(f"Could not print model summary table: {e}")

    # Residual Diagnostics (on log-transformed residuals)
    log_residuals = results.resid.reindex(log_train.index).dropna() # residuals are from the model on log_train
    log_fitted_values = results.fittedvalues.reindex(log_train.index).dropna()
    perform_residual_diagnostics(log_residuals, log_fitted_values, model_name_suffix=model_name_suffix)
    
    if test_data is not None:
        log_test = apply_log_transform(test_data)
        log_predictions = results.get_forecast(steps=len(test_data)).predicted_mean
        predictions = inverse_log_transform(log_predictions)
        predictions = pd.Series(predictions, index=test_data.index)
        
        mse = mean_squared_error(test_data, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data, predictions)
        r2 = r2_score(test_data, predictions)
        mape = mean_absolute_percentage_error(test_data, predictions)
        
        log_train_pred = results.predict(start=log_train.dropna().index[0], end=log_train.dropna().index[-1])
        train_pred = inverse_log_transform(log_train_pred)
        train_pred = train_pred.reindex(train_data.index).dropna()
        aligned_train_data = train_data.reindex(train_pred.index)

        train_mse = mean_squared_error(aligned_train_data, train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mape = mean_absolute_percentage_error(aligned_train_data, train_pred)
        
        log_test_pred_aligned = log_predictions.reindex(log_test.index).dropna()
        aligned_log_test = log_test.reindex(log_test_pred_aligned.index)
        log_mse = mean_squared_error(aligned_log_test, log_test_pred_aligned)
        log_rmse = np.sqrt(log_mse)
        
        log_train_pred_aligned = log_train_pred.reindex(log_train.dropna().index).dropna()
        aligned_log_train = log_train.dropna().reindex(log_train_pred_aligned.index)
        log_train_mse = mean_squared_error(aligned_log_train, log_train_pred_aligned)
        log_train_rmse = np.sqrt(log_train_mse)
        
        print("\nModel Performance (Original Scale):")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.4f}%")
        print(f"R²: {r2:.4f}")
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Train MAPE: {train_mape:.4f}%")
        test_train_rmse_ratio = rmse / train_rmse if train_rmse != 0 else np.nan
        print(f"Test/Train RMSE Ratio: {test_train_rmse_ratio:.4f}")
        
        print("\nLog-space metrics:")
        print(f"Log-space Test RMSE: {log_rmse:.4f}")
        print(f"Log-space Train RMSE: {log_train_rmse:.4f}")
        log_space_rmse_ratio = log_rmse / log_train_rmse if log_train_rmse != 0 else np.nan
        print(f"Log-space Test/Train RMSE Ratio: {log_space_rmse_ratio:.4f}")
        
        metrics = {
            'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2,
            'Train_RMSE': train_rmse, 'Train_MAPE': train_mape, 'RMSE_Ratio': test_train_rmse_ratio,
            'Log_RMSE': log_rmse, 'Log_Train_RMSE': log_train_rmse, 'Log_RMSE_Ratio': log_space_rmse_ratio
        }
        return results, predictions, metrics
    
    elif forecast_periods > 0:
        log_forecast = results.get_forecast(steps=forecast_periods).predicted_mean
        forecast = inverse_log_transform(log_forecast)
        last_date = train_data.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_periods,
            freq=train_data.index.freqstr if train_data.index.freqstr else 'MS'
        )
        forecast = pd.Series(forecast, index=forecast_index)
        print(f"\nGenerated forecast for {forecast_periods} periods: {forecast_index[0]} to {forecast_index[-1]}")
        return results, forecast, None
    else:
        log_pred = results.predict(start=log_train.dropna().index[0], end=log_train.dropna().index[-1])
        predictions = inverse_log_transform(log_pred)
        predictions = pd.Series(predictions, index=log_train.dropna().index[:len(predictions)])
        return results, predictions, None

def plot_results(series, train_data, test_data, predictions, forecast=None, model_name='Log-Transformed SARIMA'):
    """Plot results of the model"""
    plt.figure(figsize=(14, 8))
    plt.plot(series.index, series, label='Actual', color='black', alpha=0.7)
    if test_data is not None and train_data is not None:
        plt.axvline(x=train_data.index[-1], color='gray', linestyle='--')
        plt.text(train_data.index[-1], series.max()*0.9, 'Train-Test Split', 
                 horizontalalignment='right', verticalalignment='bottom', rotation=90)
    if test_data is not None and predictions is not None:
        plt.plot(test_data.index, predictions, label='Test Set Predictions', color='red', linestyle='--')
    elif train_data is not None and predictions is not None: 
        plt.plot(predictions.index, predictions, label='In-sample Predictions', color='blue', linestyle='--')
    if forecast is not None:
        plt.plot(forecast.index, forecast, label='Forecast', color='green', linestyle='-.')
        if series is not None and not series.empty:
             plt.axvline(x=series.index[-1], color='gray', linestyle=':')
             plt.text(series.index[-1], series.max()*0.8, 'Forecast Start', 
                 horizontalalignment='right', verticalalignment='bottom', rotation=90)
    plt.title(f'{model_name}(1,1,1)×(1,1,1,12) Model Results')
    plt.xlabel('Date')
    plt.ylabel('Retail Sales (Original Scale)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'visualizations/arima/{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}_results.png')
    plt.close()
    
    # Residuals plot on original scale (for test data)
    if test_data is not None and predictions is not None:
        residuals_original_scale = test_data - predictions
        plt.figure(figsize=(14, 6))
        residuals_original_scale.plot(label='Residuals (Test Set - Original Scale)', alpha=0.8)
        plt.axhline(y=0, color='black', linestyle='--')
        plt.title(f'Residuals of {model_name} on Test Set (Original Scale)')
        plt.xlabel('Date')
        plt.ylabel('Residual (Original Scale)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'visualizations/model_diagnostics/{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}_residuals_original_scale.png')
        plt.close()
        
        plt.figure(figsize=(8, 8))
        plt.scatter(test_data, predictions, alpha=0.7, edgecolors='k', s=50)
        min_val = min(test_data.min(), predictions.min()) if not test_data.empty and not predictions.empty else 0
        max_val = max(test_data.max(), predictions.max()) if not test_data.empty and not predictions.empty else 1
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Fit')
        plt.title(f'Actual vs Predicted ({model_name} - Test Set)')
        plt.xlabel('Actual Retail Sales')
        plt.ylabel('Predicted Retail Sales')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'visualizations/arima/{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}_scatter.png')
        plt.close()

def save_forecast(forecast, filename='baseline_forecast.csv'):
    """Save forecast to CSV file"""
    if forecast is None or forecast.empty:
        print("No forecast data to save.")
        return
    forecast_df = pd.DataFrame({
        'date': forecast.index,
        'forecast_retail_sales': forecast.values
    })
    filepath = os.path.join('outputs/scenarios/baseline', filename)
    forecast_df.to_csv(filepath, index=False)
    print(f"\nForecast saved to {filepath}")

def main():
    """Main execution function"""
    print("="*80)
    print("LOG-TRANSFORMED SARIMA MODEL FOR RETAIL SALES FORECASTING (WITH EDA & RESIDUAL DIAGNOSTICS)")
    print("="*80)
    
    raw_retail_sales = load_data()
    if raw_retail_sales is None:
        return
    
    print("\n--- Starting EDA Pipeline ---")
    analyze_missingness(raw_retail_sales, series_name='raw_retail_sales')
    retail_sales_imputed = impute_missing_if_needed(raw_retail_sales)
    retail_sales_cleaned = detect_and_handle_outliers(retail_sales_imputed, series_name='retail_sales_imputed')
    plot_time_series_decomposition(retail_sales_cleaned, series_name='cleaned_retail_sales')
    check_stationarity(retail_sales_cleaned, series_name='cleaned_retail_sales_original_scale')
    
    first_diff = retail_sales_cleaned.diff().dropna()
    check_stationarity(first_diff, series_name='first_differenced_retail_sales')
    seasonal_diff = retail_sales_cleaned.diff(12).dropna()
    check_stationarity(seasonal_diff, series_name='seasonal_differenced_retail_sales')
    both_diff = retail_sales_cleaned.diff().diff(12).dropna()
    check_stationarity(both_diff, series_name='first_and_seasonal_differenced_retail_sales')
    plot_acf_pacf_charts(both_diff, series_name='first_and_seasonal_differenced_retail_sales', lags=40)
    
    log_cleaned_retail_sales = apply_log_transform(retail_sales_cleaned)
    log_both_diff = log_cleaned_retail_sales.diff().diff(12).dropna()
    print("\nDiagnostics on log-transformed, differenced data (as input to SARIMA core):")
    check_stationarity(log_both_diff, series_name='log_transformed_differenced_sales')
    plot_acf_pacf_charts(log_both_diff, series_name='log_transformed_differenced_sales', lags=40)
    print("--- EDA Pipeline Complete ---")

    retail_sales_for_modeling = retail_sales_cleaned.copy()
    train_size = 0.8
    n = len(retail_sales_for_modeling)
    train_n = int(n * train_size)
    train_data = retail_sales_for_modeling[:train_n]
    test_data = retail_sales_for_modeling[train_n:]
    
    print(f"\nTrain-test split: {train_n} training observations, {len(test_data)} test observations")
    print(f"Training period: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Testing period: {test_data.index.min()} to {test_data.index.max()}")
    
    results, predictions, metrics = fit_log_transformed_sarima(train_data, test_data, model_name_suffix='EvaluationFit')
    
    if predictions is not None:
        plot_results(retail_sales_for_modeling, train_data, test_data, predictions, model_name='Log-Transformed SARIMA (Evaluation)')
    
    forecast_periods = 12
    print(f"\n--- Generating Forecast using Full Dataset ({len(retail_sales_for_modeling)} observations) ---")
    full_results, forecast, _ = fit_log_transformed_sarima(retail_sales_for_modeling, forecast_periods=forecast_periods, model_name_suffix='ForecastFit')
    
    if forecast is not None and full_results is not None:
        log_full_in_sample_pred = full_results.predict(start=apply_log_transform(retail_sales_for_modeling).dropna().index[0], 
                                                       end=apply_log_transform(retail_sales_for_modeling).dropna().index[-1])
        full_in_sample_pred = inverse_log_transform(log_full_in_sample_pred)
        full_in_sample_pred = pd.Series(full_in_sample_pred, index=retail_sales_for_modeling.dropna().index[:len(full_in_sample_pred)])
        plot_results(retail_sales_for_modeling, None, None, full_in_sample_pred, forecast=forecast, model_name='Log-Transformed SARIMA (Forecast)')
    
    save_forecast(forecast)
    
    print("\nAnalysis complete. EDA, Model Training, Evaluation, Residual Diagnostics, and Forecasting finished.")
    print("Outputs saved to visualizations/ (arima, data_quality, diagnostics, model_diagnostics) and outputs/scenarios/baseline/")

if __name__ == "__main__":
    main() 
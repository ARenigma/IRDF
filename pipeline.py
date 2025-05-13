"""
India Retail Demand Forecaster - Pipeline
-----------------------------------------
This module implements the complete pipeline for the retail demand forecasting model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import project modules
import data_preparation
import feature_selection
import hyperparameter_optimization
import backtesting
import ensemble_model
import data_diagnostics

def create_output_directories():
    """Create necessary output directories."""
    directories = [
        'data/processed',
        'models',
        'models/optimized',
        'outputs',
        'outputs/scenarios',
        'outputs/backtesting',
        'visualizations',
        'visualizations/data_cleaning',
        'visualizations/data_quality',
        'visualizations/feature_selection',
        'visualizations/optimization',
        'visualizations/backtesting',
        'visualizations/ensemble',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def generate_forecast(periods=12):
    """
    Generates a forecast for future retail demand using the best model.
    
    Parameters:
    -----------
    periods : int
        Number of periods to forecast
        
    Returns:
    --------
    pd.DataFrame
        Forecast results
    """
    print(f"\nGenerating forecast for next {periods} periods...")
    
    # Check if we have advanced forecasting capabilities
    has_advanced_forecasting = True
    try:
        import statsmodels.api as sm
        import tensorflow as tf
    except ImportError:
        has_advanced_forecasting = False
        print("Advanced forecasting dependencies not available")
        print("Using simple forecasting methods")
    
    # Load the features dataset for forecasting
    try:
        features_df = pd.read_csv('data/processed/features_dataset.csv')
        
        # Handle date columns properly
        if 'index' in features_df.columns and not 'date' in features_df.columns:
            try:
                features_df['date'] = pd.to_datetime(features_df['index'])
                print("Using 'index' column as date")
            except:
                print("Warning: Could not convert 'index' column to datetime")
        
        # Try to parse the date column if it exists
        if 'date' in features_df.columns:
            try:
                features_df['date'] = pd.to_datetime(features_df['date'])
                features_df.set_index('date', inplace=True)
                print("Set 'date' as index for forecasting")
            except Exception as e:
                print(f"Warning: Could not parse date column: {e}")
        
    except FileNotFoundError:
        try:
            # Try complete dataset if features dataset is not available
            features_df = pd.read_csv('data/processed/complete_dataset.csv')
            
            # Handle date columns
            if 'date' in features_df.columns:
                try:
                    features_df['date'] = pd.to_datetime(features_df['date'])
                    features_df.set_index('date', inplace=True)
                except Exception as e:
                    print(f"Warning: Could not parse date column: {e}")
        except FileNotFoundError:
            try:
                # Try loading retail sales at minimum
                features_df = pd.read_csv('data/retail_sales.csv')
                
                # Handle date columns
                if 'date' in features_df.columns:
                    try:
                        features_df['date'] = pd.to_datetime(features_df['date'])
                        features_df.set_index('date', inplace=True)
                    except Exception as e:
                        print(f"Warning: Could not parse date column: {e}")
            except FileNotFoundError:
                print("No dataset found for forecasting. Please run data preparation first.")
                return None
    
    # Make sure we have the target column
    if 'retail_sales' not in features_df.columns:
        print("Retail sales column not found in dataset.")
        return None
    
    # Get the last date in the dataset
    last_date = features_df.index.max()
    
    # Generate forecast dates - use last day of month format consistently
    # This ensures we get proper last day of each month
    forecast_dates = []
    next_month_date = last_date
    for _ in range(periods):
        # Move to next month
        if next_month_date.month == 12:
            next_month = 1
            next_year = next_month_date.year + 1
        else:
            next_month = next_month_date.month + 1
            next_year = next_month_date.year
        
        # Get the last day of the next month
        if next_month == 12:
            # Last day of December is always 31
            next_month_end = pd.Timestamp(year=next_year, month=next_month, day=31)
        else:
            # For other months, get the last day by getting the first day of the following month and subtracting one day
            if next_month == 12:
                following_month_first_day = pd.Timestamp(year=next_year+1, month=1, day=1)
            else:
                following_month_first_day = pd.Timestamp(year=next_year, month=next_month+1, day=1)
            next_month_end = following_month_first_day - pd.Timedelta(days=1)
        
        forecast_dates.append(next_month_end)
        next_month_date = next_month_end
    
    # Try to load models
    try:
        # Try to load the ensemble model first
        model = joblib.load('models/optimized/ensemble_model.pkl')
        model_type = 'ensemble'
        print("Using ensemble model for forecasting")
    except FileNotFoundError:
        try:
            # Try to load the best model
            model = joblib.load('models/best_model.pkl')
            model_type = 'best_model'
            print("Using best model for forecasting")
        except FileNotFoundError:
            # Fallback to simple method
            model_type = 'simple'
            print("No trained model found, using simple trend-based forecasting")
    
    if model_type in ['ensemble', 'best_model']:
        try:
            # Load scaler
            scaler = joblib.load('models/optimized/scaler.pkl')
            
            # Create a dataframe for the forecast periods
            last_features = pd.DataFrame(index=range(periods))
            
            # Add date column
            last_features['date'] = forecast_dates
            
            # Use the last known values for all features
            last_row = features_df.iloc[-1]
            feature_cols = [col for col in features_df.columns if col != 'retail_sales']
            
            for col in feature_cols:
                last_features[col] = last_row[col]
            
            # Scale the features
            X_forecast = last_features[feature_cols]
            X_forecast_scaled = scaler.transform(X_forecast)
            
            # Make predictions
            forecast_values = model.predict(X_forecast_scaled)
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'retail_sales_forecast': forecast_values
            })
            
            # Add confidence intervals (simple approach)
            error_std = 0.05  # Assume 5% standard deviation
            forecast_df['lower_bound'] = forecast_df['retail_sales_forecast'] * (1 - 1.96 * error_std)
            forecast_df['upper_bound'] = forecast_df['retail_sales_forecast'] * (1 + 1.96 * error_std)
        except Exception as e:
            print(f"Error making model-based forecast: {e}")
            # Fallback to simple method
            model_type = 'simple'
    
    if model_type == 'simple':
        # Simple forecasting method based on historical patterns
        print("Using simple trend-based forecasting")
        
        # Get the retail sales history
        retail_sales = features_df['retail_sales']
        
        # Calculate average growth rate over the last year (or available data)
        periods_to_use = min(12, len(retail_sales) - 1)
        if periods_to_use > 0:
            # Calculate growth rate from periods_to_use months ago to now
            start_value = retail_sales.iloc[-(periods_to_use + 1)]
            end_value = retail_sales.iloc[-1]
            
            if start_value > 0:  # Avoid division by zero
                total_growth = (end_value / start_value) - 1
                avg_monthly_growth = (1 + total_growth) ** (1/periods_to_use) - 1
            else:
                # Default growth if start value is zero or negative
                avg_monthly_growth = 0.02  # 2% per month
        else:
            # Default growth rate if not enough data
            avg_monthly_growth = 0.02  # 2% per month
        
        # Calculate seasonality factors if we have at least 13 months of data
        has_seasonality = len(retail_sales) >= 13
        if has_seasonality:
            # Calculate month-to-month seasonality factors
            seasonality_factors = {}
            for month in range(1, 13):
                # Find values for this month
                month_values = [val for date, val in zip(features_df.index, retail_sales) if date.month == month]
                if month_values:
                    seasonality_factors[month] = sum(month_values) / len(month_values)
            
            # Normalize seasonality factors
            if seasonality_factors:
                avg_factor = sum(seasonality_factors.values()) / len(seasonality_factors)
                seasonality_factors = {m: f / avg_factor for m, f in seasonality_factors.items()}
        
        # Generate forecast values
        last_value = retail_sales.iloc[-1]
        forecast_values = []
        
        for i, date in enumerate(forecast_dates):
            # Apply growth
            if i == 0:
                base_forecast = last_value * (1 + avg_monthly_growth)
            else:
                base_forecast = forecast_values[-1] * (1 + avg_monthly_growth)
            
            # Apply seasonality if available
            if has_seasonality and date.month in seasonality_factors:
                forecast_value = base_forecast * seasonality_factors[date.month]
            else:
                forecast_value = base_forecast
            
            forecast_values.append(forecast_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'retail_sales_forecast': forecast_values
        })
        
        # Add simple confidence intervals (±10% increasing over time)
        lower_bounds = []
        upper_bounds = []
        for i, value in enumerate(forecast_values):
            # Uncertainty increases with time
            uncertainty = 0.05 + (i * 0.01)  # 5% initially, increasing by 1% each month
            lower_bounds.append(value * (1 - uncertainty))
            upper_bounds.append(value * (1 + uncertainty))
        
        forecast_df['lower_bound'] = lower_bounds
        forecast_df['upper_bound'] = upper_bounds
    
    # Convert date to string with standard format for better CSV compatibility
    forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')
    
    # Print a summary of the forecast
    print("\nForecast Summary:")
    print(f"Last known value: {retail_sales.iloc[-1]:.2f}")
    print(f"Forecast for next {periods} months:")
    for i, row in forecast_df.iterrows():
        print(f"{row['date']}: {row['retail_sales_forecast']:.2f}")
    
    # Save forecast to CSV
    os.makedirs('outputs', exist_ok=True)
    forecast_df.to_csv('outputs/retail_sales_forecast.csv', index=False)
    
    # Visualize forecast
    plt.figure(figsize=(12, 6))
    
    # Plot historical data (last 24 periods or all if less)
    periods_to_plot = min(24, len(features_df))
    historical = features_df['retail_sales'].iloc[-periods_to_plot:]
    plt.plot(historical.index, historical.values, 'b-', label='Historical Retail Sales')
    
    # Convert forecast dates back to datetime for plotting
    forecast_dates_dt = pd.to_datetime(forecast_df['date'])
    
    # Plot forecast
    plt.plot(forecast_dates_dt, forecast_df['retail_sales_forecast'], 'r--', label='Forecast')
    
    # Plot confidence intervals if available
    if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
        plt.fill_between(
            forecast_dates_dt,
            forecast_df['lower_bound'],
            forecast_df['upper_bound'],
            color='pink', alpha=0.3,
            label='Confidence Interval'
        )
    
    plt.title('Retail Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Retail Sales')
    plt.legend()
    plt.grid(True)
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/retail_sales_forecast.png')
    plt.close()
    
    print(f"Forecast generated for {periods} months ahead")
    return forecast_df

def run_pipeline(steps=None):
    """
    Run the complete pipeline or specific steps.
    
    Parameters:
    -----------
    steps : list of str or None
        List of steps to run, or None to run all steps
    """
    # Set up timing
    start_time = time.time()
    
    # Create output directories
    create_output_directories()
    
    # Define all available steps
    all_steps = [
        'data_preparation',
        'data_diagnostics',
        'feature_selection',
        'hyperparameter_optimization',
        'backtesting',
        'ensemble_model',
        'forecast_scenarios',
        'insights'
    ]
    
    # If no steps specified, run all
    if steps is None:
        steps = all_steps
    
    # Validate steps
    for step in steps:
        if step not in all_steps:
            print(f"Warning: Unknown step '{step}'. Skipping.")
    
    # Filter valid steps
    steps = [step for step in steps if step in all_steps]
    
    print("=" * 80)
    print("INDIA RETAIL DEMAND FORECASTING - PIPELINE")
    print("=" * 80)
    print(f"Running pipeline steps: {', '.join(steps)}")
    print("-" * 80)
    
    # Step 1: Data Preparation
    if 'data_preparation' in steps:
        print("\nSTEP 1: DATA PREPARATION")
        try:
            data_preparation.main()
            print("Data preparation completed successfully.")
        except Exception as e:
            print(f"Error in data preparation: {e}")
            return
    
    # Step 2: Data Diagnostics
    if 'data_diagnostics' in steps:
        print("\nSTEP 2: DATA DIAGNOSTICS")
        try:
            data_diagnostics.main()
            print("Data diagnostics completed successfully.")
        except Exception as e:
            print(f"Error in data diagnostics: {e}")
    
    # Step 3: Feature Selection
    if 'feature_selection' in steps:
        print("\nSTEP 3: FEATURE SELECTION")
        try:
            feature_selection.main()
            print("Feature selection completed successfully.")
        except Exception as e:
            print(f"Error in feature selection: {e}")
            return
    
    # Step 4: Hyperparameter Optimization
    if 'hyperparameter_optimization' in steps:
        print("\nSTEP 4: HYPERPARAMETER OPTIMIZATION")
        try:
            hyperparameter_optimization.main()
            print("Hyperparameter optimization completed successfully.")
        except Exception as e:
            print(f"Error in hyperparameter optimization: {e}")
            return
    
    # Step 5: Backtesting
    if 'backtesting' in steps:
        print("\nSTEP 5: BACKTESTING")
        try:
            backtesting.main()
            print("Backtesting completed successfully.")
        except Exception as e:
            print(f"Error in backtesting: {e}")
    
    # Step 6: Ensemble Model
    if 'ensemble_model' in steps:
        print("\nSTEP 6: ENSEMBLE MODELING")
        try:
            ensemble_model.main()
            print("Ensemble modeling completed successfully.")
            
            # Copy the ensemble model to the best_model location
            os.makedirs('models', exist_ok=True)
            try:
                ensemble_path = 'models/optimized/ensemble_model.pkl'
                best_model_path = 'models/best_model.pkl'
                joblib.dump(joblib.load(ensemble_path), best_model_path)
                print(f"Copied ensemble model to {best_model_path}")
            except Exception as e:
                print(f"Warning: Could not copy ensemble model: {e}")
                
        except Exception as e:
            print(f"Error in ensemble modeling: {e}")
    
    # Step 7: Forecast Scenarios
    if 'forecast_scenarios' in steps:
        print("\nSTEP 7: FORECAST SCENARIOS")
        try:
            # Import here to avoid circular imports
            import forecast_scenarios
            forecast_scenarios.main()
            print("Forecast scenarios completed successfully.")
        except Exception as e:
            print(f"Error in forecast scenarios: {e}")
    
    # Step 8: Insights
    if 'insights' in steps:
        print("\nSTEP 8: INSIGHTS")
        try:
            # Import here to avoid circular imports
            import insights
            insights.main()
            print("Insights generation completed successfully.")
        except Exception as e:
            print(f"Error in insights generation: {e}")
    
    # Finish
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED IN {elapsed_time:.2f} SECONDS")
    print("=" * 80)
    
    # Save pipeline completion record
    pipeline_record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'steps_executed': steps,
        'execution_time_seconds': elapsed_time,
        'status': 'completed'
    }
    
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/pipeline_record.json', 'w') as f:
        json.dump(pipeline_record, f, indent=2)
    
def main():
    """Main function to run the pipeline."""
    # Parse command line arguments (if any)
    import sys
    
    if len(sys.argv) > 1:
        # If arguments provided, run specific steps
        steps = sys.argv[1:]
        run_pipeline(steps)
    else:
        # Run the complete pipeline
        run_pipeline()

if __name__ == "__main__":
    main() 
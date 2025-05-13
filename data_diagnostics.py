"""
India Retail Demand Forecaster - Data Diagnostics
-------------------------------------------------
This module provides diagnostic functions to identify and analyze statistical
issues in the retail demand forecasting model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
from pathlib import Path

def diagnose_model_performance():
    """
    Analyze the model performance summary to identify issues with R^2 values
    """
    print("Analyzing model performance metrics...")
    
    try:
        # Load model performance summary
        summary_path = Path('outputs/model_performance_summary.csv')
        if not summary_path.exists():
            print(f"Error: Model performance summary not found at {summary_path}")
            return None
            
        performance_df = pd.read_csv(summary_path)
        
        # Display performance metrics
        print("\nModel Performance Metrics:")
        print(performance_df)
        
        # Calculate average R²
        avg_r2 = performance_df['R²'].mean()
        print(f"\nAverage R² across models: {avg_r2:.4f}")
        
        # Interpret R² values
        if avg_r2 < 0.3:
            print("\nPROBLEM: Very low R² values indicate the models explain very little of the variance in retail sales.")
            print("Possible causes:")
            print("1. Missing important predictor variables")
            print("2. Non-linear relationships not captured by the models")
            print("3. High noise-to-signal ratio in the data")
            print("4. Incorrect feature engineering")
            print("5. Insufficient data points relative to features (overfitting)")
        elif avg_r2 < 0.7:
            print("\nCAUTION: Moderate R² values suggest the models explain some but not most of the variance.")
            print("Consider feature engineering improvements or additional data sources.")
        else:
            print("\nGOOD: High R² values indicate the models explain most of the variance in retail sales.")
        
        # Create visualization of model performance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='R²', data=performance_df)
        plt.title('R² Values by Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/diagnostics/r2_comparison.png')
        plt.close()
        
        return performance_df
    
    except Exception as e:
        print(f"Error in model performance analysis: {e}")
        return None

def check_multicollinearity(features_path='data/processed/features_dataset.csv', target_col='retail_sales'):
    """
    Check for multicollinearity issues in the dataset using VIF (Variance Inflation Factor)
    """
    print("\nChecking for multicollinearity...")
    
    try:
        # Load dataset
        df = pd.read_csv(features_path)
        
        # Handle date/index columns
        for col in ['date', 'index']:
            if col in df.columns:
                try:
                    if col == 'index':
                        # Try to convert to datetime for proper filtering
                        df['date'] = pd.to_datetime(df[col])
                    else:
                        df[col] = pd.to_datetime(df[col])
                    # Remove from further analysis
                    df = df.drop(columns=[col])
                except:
                    print(f"Warning: Could not convert {col} to datetime, dropping from analysis")
                    df = df.drop(columns=[col])
        
        # Also drop any other non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            print(f"Dropping non-numeric columns: {non_numeric_cols}")
            df = df.drop(columns=non_numeric_cols)
        
        # Check and handle NaN and infinite values
        if df.isna().any().any() or np.isinf(df.values).any():
            print("Found NaN or infinite values in the data. Applying imputation.")
            # Replace inf with NaN first
            df = df.replace([np.inf, -np.inf], np.nan)
            # Then fill NaN with column means
            df = df.fillna(df.mean())
        
        # Separate features from target
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
        else:
            log_target = f"log_{target_col}"
            if log_target in df.columns:
                X = df.drop(columns=[log_target])
            else:
                # No clear target, use all numeric columns
                X = df
        
        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_data = vif_data.sort_values('VIF', ascending=False)
        
        print("\nVariance Inflation Factors (VIF):")
        print(vif_data)
        
        # Interpret VIF values
        high_vif_features = vif_data[vif_data["VIF"] > 10]
        if not high_vif_features.empty:
            print("\nPROBLEM: High multicollinearity detected!")
            print(f"Features with VIF > 10 ({len(high_vif_features)} features):")
            for _, row in high_vif_features.iterrows():
                print(f"  - {row['Feature']}: VIF = {row['VIF']:.2f}")
            print("\nRecommendations:")
            print("1. Remove highly correlated features")
            print("2. Create composite features through PCA or factor analysis")
            print("3. Use regularization techniques (Ridge, Lasso) to handle multicollinearity")
        else:
            print("\nGOOD: No severe multicollinearity detected (all VIF values < 10)")
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        corr = X.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        os.makedirs('visualizations/diagnostics', exist_ok=True)
        plt.savefig('visualizations/diagnostics/correlation_heatmap.png')
        plt.close()
        
        return vif_data
    
    except Exception as e:
        print(f"Error in multicollinearity analysis: {e}")
        return None

def check_heteroskedasticity(features_path='data/processed/features_dataset.csv', target_col='retail_sales'):
    """
    Check for heteroskedasticity in the model residuals
    """
    print("\nChecking for heteroskedasticity...")
    
    try:
        # Load dataset
        df = pd.read_csv(features_path)
        
        # Handle date/index columns
        for col in ['date', 'index']:
            if col in df.columns:
                try:
                    if col == 'index':
                        # Try to convert to datetime for proper filtering
                        df['date'] = pd.to_datetime(df[col])
                    else:
                        df[col] = pd.to_datetime(df[col])
                    # Remove from further analysis
                    df = df.drop(columns=[col])
                except:
                    print(f"Warning: Could not convert {col} to datetime, dropping from analysis")
                    df = df.drop(columns=[col])
        
        # Also drop any other non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            print(f"Dropping non-numeric columns: {non_numeric_cols}")
            df = df.drop(columns=non_numeric_cols)
        
        # Check and handle NaN and infinite values
        if df.isna().any().any() or np.isinf(df.values).any():
            print("Found NaN or infinite values in the data. Applying imputation.")
            # Replace inf with NaN first
            df = df.replace([np.inf, -np.inf], np.nan)
            # Then fill NaN with column means
            df = df.fillna(df.mean())
        
        # Check for target column or its log-transformed version
        if target_col in df.columns:
            actual_target = target_col
        else:
            log_target = f"log_{target_col}"
            if log_target in df.columns:
                actual_target = log_target
                print(f"Using log-transformed target: {log_target}")
            else:
                print(f"Target column '{target_col}' not found in dataset")
                return None
        
        # Separate features from target
        y = df[actual_target]
        X = df.drop(columns=[actual_target])
        
        # Add constant to X for statsmodels
        X = sm.add_constant(X)
        
        # Fit OLS model
        model = sm.OLS(y, X).fit()
        
        # Calculate residuals
        residuals = model.resid
        
        # Breusch-Pagan test for heteroskedasticity
        bp_test = het_breuschpagan(residuals, X)
        bp_pvalue = bp_test[1]
        
        print(f"\nBreusch-Pagan test p-value: {bp_pvalue:.4f}")
        
        if bp_pvalue < 0.05:
            print("PROBLEM: Heteroskedasticity detected (p < 0.05)")
            print("Residuals have non-constant variance, which can affect standard errors and hypothesis tests.")
            print("Recommendations:")
            print("1. Transform the target variable (log, square root, etc.)")
            print("2. Use robust standard errors (HC3)")
            print("3. Use weighted least squares")
            print("4. Apply variance-stabilizing transformations to features")
        else:
            print("GOOD: No significant heteroskedasticity detected")
        
        # Plot residuals vs. fitted values
        plt.figure(figsize=(10, 6))
        plt.scatter(model.fittedvalues, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted Values')
        plt.savefig('visualizations/diagnostics/heteroskedasticity.png')
        plt.close()
        
        return {
            'bp_statistic': bp_test[0],
            'bp_pvalue': bp_test[1],
            'residuals': residuals,
            'fitted_values': model.fittedvalues
        }
    
    except Exception as e:
        print(f"Error in heteroskedasticity analysis: {e}")
        return None

def check_serial_correlation(features_path='data/processed/features_dataset.csv', target_col='retail_sales'):
    """
    Check for serial correlation in time series data
    """
    print("\nChecking for serial correlation...")
    
    try:
        # Load dataset
        df = pd.read_csv(features_path)
        
        # Handle date/index columns for time series
        date_col = None
        for col in ['date', 'index']:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_col = col
                    df = df.sort_values(by=date_col)  # Sort by date for proper time series analysis
                except:
                    print(f"Warning: Could not convert {col} to datetime")
        
        # Drop non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        if date_col and date_col in non_numeric_cols:
            non_numeric_cols.remove(date_col)  # Keep date column
        
        if non_numeric_cols:
            print(f"Dropping non-numeric columns: {non_numeric_cols}")
            df = df.drop(columns=non_numeric_cols)
        
        # Check and handle NaN and infinite values
        if df.isna().any().any() or np.isinf(df.values).any():
            print("Found NaN or infinite values in the data. Applying imputation.")
            # Replace inf with NaN first
            df = df.replace([np.inf, -np.inf], np.nan)
            # Then fill NaN with column means
            df = df.fillna(df.mean())
        
        # Check for target column or its log-transformed version
        if target_col in df.columns:
            actual_target = target_col
        else:
            log_target = f"log_{target_col}"
            if log_target in df.columns:
                actual_target = log_target
                print(f"Using log-transformed target: {log_target}")
            else:
                print(f"Target column '{target_col}' not found in dataset")
                return None
        
        # Separate features from target
        y = df[actual_target]
        X = df.drop(columns=[actual_target])
        if date_col and date_col in X.columns:
            X = X.drop(columns=[date_col])
        
        # Add constant to X for statsmodels
        X = sm.add_constant(X)
        
        # Fit OLS model
        model = sm.OLS(y, X).fit()
        
        # Calculate residuals
        residuals = model.resid
        
        # Durbin-Watson test
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(residuals)
        
        print(f"Durbin-Watson statistic: {dw_stat:.4f}")
        
        if dw_stat < 1.5:
            print("PROBLEM: Positive serial correlation detected (DW < 1.5)")
            print("Residuals have significant positive correlation, affecting model validity.")
        elif dw_stat > 2.5:
            print("PROBLEM: Negative serial correlation detected (DW > 2.5)")
            print("Residuals have significant negative correlation.")
        else:
            print("GOOD: No significant serial correlation detected (DW between 1.5 and 2.5)")
        
        # Ljung-Box test
        lb_test = acorr_ljungbox(residuals, lags=[1], return_df=False)
        lb_pvalue = lb_test[1][0]
        
        print(f"Ljung-Box test p-value (lag 1): {lb_pvalue:.4f}")
        
        if lb_pvalue < 0.05:
            print("PROBLEM: Serial correlation detected from Ljung-Box test (p < 0.05)")
            print("Recommendations:")
            print("1. Include lagged variables in the model")
            print("2. Use ARIMA or time series specific models")
            print("3. Apply first differencing to the data")
            print("4. Use HAC standard errors")
        
        # Autocorrelation plot
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 1, 1)
        plot_acf(residuals, lags=20, alpha=0.05, title='Autocorrelation Function (ACF)')
        
        plt.subplot(2, 1, 2)
        plot_pacf(residuals, lags=20, alpha=0.05, title='Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        plt.savefig('visualizations/diagnostics/serial_correlation.png')
        plt.close()
        
        return {
            'durbin_watson': dw_stat,
            'ljung_box_stat': lb_test[0][0],
            'ljung_box_pvalue': lb_pvalue,
            'residuals': residuals
        }
    
    except Exception as e:
        print(f"Error in serial correlation analysis: {e}")
        return None

def diagnose_forecast_pattern(forecast_path='outputs/retail_sales_forecast.csv'):
    """
    Analyze why forecasts show the same values for all months
    """
    print("\nAnalyzing forecast pattern issues...")
    
    try:
        # Load forecast data
        df = pd.read_csv(forecast_path)
        
        # Ensure 'date' column is properly formatted
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Get forecast column name
        forecast_col = [col for col in df.columns if 'forecast' in col.lower() or 'predict' in col.lower()]
        if not forecast_col:
            print("Error: No forecast column found in the data")
            return None
            
        forecast_col = forecast_col[0]
        
        # Analyze variance in forecasts
        forecast_std = df[forecast_col].std()
        forecast_min = df[forecast_col].min()
        forecast_max = df[forecast_col].max()
        forecast_range = forecast_max - forecast_min
        
        print(f"Forecast standard deviation: {forecast_std:.4f}")
        print(f"Forecast range: {forecast_range:.4f} ({forecast_min:.2f} to {forecast_max:.2f})")
        
        if forecast_std < 0.1 or forecast_range < 1.0:
            print("\nPROBLEM: Forecasts show almost identical values across all periods.")
            print("Possible causes:")
            print("1. Model is not utilizing time-dependent features correctly")
            print("2. Baseline forecasting method is being used instead of the trained model")
            print("3. Model is overfit to a flat pattern in the training data")
            print("4. Bug in the forecast_scenarios.py implementation")
            print("5. Seasonal effects are not being captured")
        else:
            print("\nForecasts show appropriate variation across periods.")
        
        # Plot forecast
        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df[forecast_col], marker='o')
        plt.title('Forecast Pattern Analysis')
        plt.xlabel('Date')
        plt.ylabel('Forecast Value')
        plt.grid(True)
        plt.savefig('visualizations/diagnostics/forecast_pattern.png')
        plt.close()
        
        return df
    
    except Exception as e:
        print(f"Error in forecast pattern analysis: {e}")
        return None

def diagnose_scenario_comparison(scenario_comparison_path='outputs/scenarios/scenario_comparison.csv'):
    """
    Analyze why scenarios show unexpected relationships to the baseline
    """
    print("\nAnalyzing scenario comparison issues...")
    
    try:
        # Load scenario comparison data
        df = pd.read_csv(scenario_comparison_path)
        
        # Ensure 'date' column is properly formatted
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Get baseline column
        if 'baseline' not in df.columns:
            print("Error: Baseline scenario not found in comparison data")
            return None
        
        # Compare scenarios to baseline
        scenario_cols = [col for col in df.columns if col != 'date']
        
        # Calculate percent differences from baseline for each scenario at the end
        last_row = df.iloc[-1]
        baseline_value = last_row['baseline']
        
        scenario_diffs = {}
        for col in scenario_cols:
            if col != 'baseline':
                scenario_value = last_row[col]
                pct_diff = ((scenario_value - baseline_value) / baseline_value) * 100
                scenario_diffs[col] = pct_diff
        
        # Print scenario differences
        print("\nScenario differences from baseline (final forecast point):")
        for scenario, diff in scenario_diffs.items():
            print(f"{scenario}: {diff:.2f}%")
        
        # Check for unexpected scenario behaviors
        high_growth_diff = scenario_diffs.get('high_growth', 0)
        recession_diff = scenario_diffs.get('recession', 0)
        
        if high_growth_diff < 0:
            print("\nPROBLEM: High Growth scenario shows lower retail demand than Baseline.")
            print("This contradicts economic theory - high GDP growth should increase retail demand.")
            print("Possible causes:")
            print("1. Sign error in the economic impact weights")
            print("2. Implementation error in the scenario adjustments")
            print("3. Error in the baseline parameter assumptions")
            print("4. Incorrect weight application for other factors (inflation, interest rates)")
        
        if recession_diff > 0:
            print("\nPROBLEM: Recession scenario shows higher retail demand than Baseline.")
            print("This contradicts economic theory - economic contractions should decrease retail demand.")
            print("Possible causes:")
            print("1. Sign error in the economic impact weights")
            print("2. Implementation error in the scenario adjustments")
            print("3. Gold price effect might be overriding GDP impacts inappropriately")
            print("4. Possible error in cumulative effect calculation")
        
        # Plot scenario comparisons
        plt.figure(figsize=(12, 8))
        for col in scenario_cols:
            plt.plot(df['date'], df[col], marker='o', label=col)
        
        plt.title('Scenario Comparison Analysis')
        plt.xlabel('Date')
        plt.ylabel('Retail Sales')
        plt.legend()
        plt.grid(True)
        plt.savefig('visualizations/diagnostics/scenario_comparison_analysis.png')
        plt.close()
        
        return {
            'scenario_diffs': scenario_diffs,
            'df': df
        }
    
    except Exception as e:
        print(f"Error in scenario comparison analysis: {e}")
        return None

def main():
    """
    Run all diagnostic checks
    """
    print("="*80)
    print("INDIA RETAIL DEMAND FORECASTER - DIAGNOSTICS ANALYSIS")
    print("="*80)
    
    # Create diagnostics visualization directory
    os.makedirs('visualizations/diagnostics', exist_ok=True)
    
    # Run all diagnostics
    diagnose_model_performance()
    check_multicollinearity()
    check_heteroskedasticity()
    check_serial_correlation()
    diagnose_forecast_pattern()
    diagnose_scenario_comparison()
    
    print("\n"+"="*80)
    print("DIAGNOSTICS ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main() 
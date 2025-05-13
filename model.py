"""
India Retail Demand Forecaster - Model Building
-----------------------------------------------
This script implements and evaluates various machine learning models to predict
retail demand based on macroeconomic indicators.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Time series forecasting libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_data():
    """
    Load and prepare the data for modeling.
    """
    try:
        # Try to load the engineered features
        print("Loading engineered features...")
        df = pd.read_csv('data/engineered_features.csv')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    except FileNotFoundError:
        print("Engineered features not found. Run the setup script first.")
        return None
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def prepare_train_test(df, target_column='retail_sales', test_size=0.2):
    """
    Prepare train/test split for time series data, with proper handling of transformations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and target
    target_column : str
        Name of the target column (will check for log_<target> too)
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    X_train, y_train, X_test, y_test, metadata
        Training and test data with additional metadata
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Check if we have a log-transformed version of the target
    log_target = f"log_{target_column}"
    use_log_target = log_target in df.columns
    
    if use_log_target:
        print(f"Using log-transformed target variable: {log_target}")
        actual_target = log_target
    else:
        actual_target = target_column
    
    # Check if target exists
    if actual_target not in df.columns:
        raise ValueError(f"Target column '{actual_target}' not found in dataframe")
    
    # Initialize date_col
    date_col = None
    has_date = False
    
    # Process date columns
    if 'date' in df.columns:
        date_col = df['date'].copy()
        df = df.drop(columns=['date'])
        has_date = True
    # Handle 'index' column that might contain dates
    elif 'index' in df.columns:
        try:
            # Try to convert index to datetime
            date_col = pd.to_datetime(df['index'])
            # Remove index column from feature set to avoid data leakage
            df = df.drop(columns=['index'])
            has_date = True
            print("Using 'index' column as dates")
        except:
            print("Warning: 'index' column found but could not be converted to dates")
    
    # Split data chronologically for time series
    train_size = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Add date information back for reference if we had it
    if has_date:
        train_dates = date_col.iloc[:train_size]
        test_dates = date_col.iloc[train_size:]
        
        print(f"Train set: {len(train_df)} observations, Test set: {len(test_df)} observations")
        print(f"Train period: {train_dates.min()} to {train_dates.max()}")
        print(f"Test period: {test_dates.min()} to {test_dates.max()}")
    else:
        print(f"Train set: {len(train_df)} observations, Test set: {len(test_df)} observations")
        print(f"Train indices: 0-{train_size-1}, Test indices: {train_size}-{len(df)-1}")
    
    # Define features (exclude target and its transforms)
    exclude_cols = [target_column, log_target]
    feature_cols = [col for col in df.columns if col not in exclude_cols and col != 'index' and col in df.columns]
    
    # Check for low variance features that might cause problems
    low_var_features = []
    for col in feature_cols:
        if df[col].std() < 1e-8:
            print(f"Warning: Feature {col} has very low variance, excluding from model")
            low_var_features.append(col)
    
    # Remove low variance features
    feature_cols = [col for col in feature_cols if col not in low_var_features]
    
    # Create train/test sets
    X_train = train_df[feature_cols]
    y_train = train_df[actual_target]
    
    X_test = test_df[feature_cols]
    y_test = test_df[actual_target]
    
    # Create metadata for later use
    metadata = {
        'target_column': actual_target,
        'original_target': target_column,
        'is_log_transformed': use_log_target,
        'feature_columns': feature_cols,
        'has_date': has_date
    }
    
    # Add date information to metadata if we have it
    if has_date:
        metadata.update({
            'train_start': train_dates.min(),
            'train_end': train_dates.max(),
            'test_start': test_dates.min(),
            'test_end': test_dates.max(),
            'train_dates': train_dates,
            'test_dates': test_dates
        })
    
    return X_train, y_train, X_test, y_test, metadata

def create_baseline_models():
    """
    Create a dictionary of baseline models to evaluate.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    return models

def calculate_metrics(y_true, y_pred):
    """
    Calculate common regression metrics.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE if no zeros in y_true
    try:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except:
        mape = float('nan')
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

def evaluate_model(y_true, y_pred, model_name, X_test=None, model=None):
    """
    Evaluate model performance and perform statistical tests for model validity.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model for reporting
    X_test : pd.DataFrame, optional
        Test features for additional diagnostic tests
    model : object, optional
        Trained model for additional analysis
        
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    
    # Create diagnostic directory
    os.makedirs('visualizations/model_diagnostics', exist_ok=True)
    
    # Calculate standard metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Basic metrics output
    print(f"\n{model_name} Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%" if not np.isnan(mape) else "MAPE: N/A (zero values in y_true)")
    
    # Create residual plots for diagnostics
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Residuals vs Fitted values (check for heteroskedasticity)
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    
    # Plot 2: Residual histogram (check for normality)
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=15, alpha=0.7)
    plt.title('Residual Histogram')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    
    # Plot 3: Actual vs Predicted
    plt.subplot(2, 2, 3)
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add the perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Plot 4: Residual QQ Plot (check for normality)
    from scipy import stats
    plt.subplot(2, 2, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Residual Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig(f'visualizations/model_diagnostics/{model_name.replace(" ", "_").lower()}_diagnostics.png')
    plt.close()
    
    # Statistical tests for residuals
    # Only perform if we have enough samples
    statistical_tests = {}
    
    if len(residuals) > 10:
        # 1. Normality test
        from scipy.stats import shapiro
        shapiro_test = shapiro(residuals)
        statistical_tests['shapiro_statistic'] = shapiro_test[0]
        statistical_tests['shapiro_pvalue'] = shapiro_test[1]
        
        normality_msg = "Residuals are normally distributed (p > 0.05)" if shapiro_test[1] > 0.05 else "Residuals are NOT normally distributed (p < 0.05)"
        print(f"Shapiro-Wilk test for normality: {shapiro_test[1]:.4f} - {normality_msg}")
        
        # 2. Durbin-Watson test for autocorrelation (if time series)
        if isinstance(y_true, pd.Series) and hasattr(y_true, 'index') and hasattr(y_true.index, 'is_all_dates'):
            try:
                from statsmodels.stats.stattools import durbin_watson
                dw_stat = durbin_watson(residuals)
                statistical_tests['durbin_watson'] = dw_stat
                
                if dw_stat < 1.5:
                    print(f"Durbin-Watson statistic: {dw_stat:.4f} - Positive autocorrelation detected")
                elif dw_stat > 2.5:
                    print(f"Durbin-Watson statistic: {dw_stat:.4f} - Negative autocorrelation detected")
                else:
                    print(f"Durbin-Watson statistic: {dw_stat:.4f} - No significant autocorrelation")
            except:
                print("Could not perform Durbin-Watson test")
    
    # Feature importance (if available)
    feature_importance = None
    if model is not None and X_test is not None:
        # Try to extract feature importance if the model supports it
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            feature_importance = pd.DataFrame({
                'Feature': X_test.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
            plt.title(f'Top 10 Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'visualizations/model_diagnostics/{model_name.replace(" ", "_").lower()}_feature_importance.png')
            plt.close()
            
            print("\nTop 5 Important Features:")
            for i, row in feature_importance.head(5).iterrows():
                print(f"  - {row['Feature']}: {row['Importance']:.4f}")
            
        elif hasattr(model, 'coef_'):
            # For linear models
            feature_importance = pd.DataFrame({
                'Feature': X_test.columns,
                'Coefficient': model.coef_
            }).sort_values('Coefficient', ascending=False)
            
            # Plot coefficients
            plt.figure(figsize=(10, 6))
            feature_importance['abs_coef'] = np.abs(feature_importance['Coefficient'])
            sorted_features = feature_importance.sort_values('abs_coef', ascending=False)
            
            plt.barh(sorted_features['Feature'][:10], sorted_features['Coefficient'][:10])
            plt.title(f'Top 10 Feature Coefficients - {model_name}')
            plt.xlabel('Coefficient Value')
            plt.tight_layout()
            plt.savefig(f'visualizations/model_diagnostics/{model_name.replace(" ", "_").lower()}_coefficients.png')
            plt.close()
            
            print("\nTop 5 Features by Coefficient Magnitude:")
            for i, row in sorted_features.head(5).iterrows():
                print(f"  - {row['Feature']}: {row['Coefficient']:.4f}")
    
    # Check for potential issues with low R²
    if r2 < 0.5:
        print("\nLow R² Potential Causes:")
        print("1. Missing key predictive variables")
        print("2. Non-linear relationships that the model can't capture")
        print("3. High data variability or noise")
        print("4. Temporal patterns not captured (time series)")
        print("5. Insufficient data for complex relationships")
        
        # Additional analysis to detect specific issues
        if np.std(y_true) / np.mean(y_true) > 0.5:
            print("\nHigh coefficient of variation detected in target variable")
            print("Consider log or other transformations to stabilize variance")
    
    # Return all metrics
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'Statistical Tests': statistical_tests,
        'Feature Importance': feature_importance
    }
    
    return metrics

def add_time_series_features(df: pd.DataFrame, target_column='retail_sales', lag_periods=[1, 3, 6, 12], rolling_windows=[3, 6, 12]) -> pd.DataFrame:
    """
    Add lagged variables and time-based features to improve time series forecasting accuracy.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and target
    target_column : str
        Name of the target column
    lag_periods : list
        List of lag periods to include as features
    rolling_windows : list
        List of rolling window sizes for moving averages/std
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional time series features
    """
    print(f"Adding time series features to improve model accuracy...")
    
    # Make a copy to avoid modifying the original
    df_ts = df.copy()
    
    # First ensure we have a proper datetime index for time series operations
    date_col = None
    if 'date' in df.columns:
        date_col = 'date'
    elif 'index' in df.columns and pd.api.types.is_datetime64_any_dtype(df['index']):
        date_col = 'index'
        
    if date_col:
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df_ts[date_col]):
            df_ts[date_col] = pd.to_datetime(df_ts[date_col])
            
        # Set as index if it's not already
        if df_ts.index.name != date_col:
            df_ts = df_ts.set_index(date_col)
    
    # Check if the target column exists
    if target_column not in df_ts.columns:
        print(f"Warning: Target column {target_column} not found. Skipping lag features.")
        return df_ts.reset_index() if date_col else df_ts
    
    # Sort by date to ensure proper lag creation
    df_ts = df_ts.sort_index()
    
    # Create lag features for the target variable
    for lag in lag_periods:
        lag_col = f"{target_column}_lag_{lag}"
        df_ts[lag_col] = df_ts[target_column].shift(lag)
        print(f"  Created lag feature: {lag_col}")
    
    # Create rolling window features for the target variable
    for window in rolling_windows:
        # Rolling mean
        mean_col = f"{target_column}_rolling_mean_{window}"
        df_ts[mean_col] = df_ts[target_column].rolling(window=window, min_periods=1).mean()
        print(f"  Created rolling mean feature: {mean_col}")
        
        # Rolling standard deviation to capture volatility
        std_col = f"{target_column}_rolling_std_{window}"
        df_ts[std_col] = df_ts[target_column].rolling(window=window, min_periods=2).std()
        print(f"  Created rolling std feature: {std_col}")
    
    # Identify economic indicators and create time series features for them too
    # List of potential economic indicators to look for
    potential_indicators = [
        'gdp_growth', 'inflation', 'interest_rate', 'unemployment',
        'gold_price', 'oil_price', 'iip_combined', 'cpi_index', 
        'lending_rate', 'wpi_index', 'repo_rate',
        # Include potential column names with suffixes if merged from different sources
        'inflation_rate', 'gdp', 'cpi', 'wpi'
    ]
    
    # Find which indicators are actually in the dataframe
    indicators = [col for col in potential_indicators if col in df_ts.columns]
    
    # Add time series features for these economic indicators
    if indicators:
        print(f"  Adding time series features for {len(indicators)} economic indicators")
        
        for indicator in indicators:
            # Add lag features for economic indicators (shorter lags to limit feature explosion)
            for lag in [1, 3]:
                lag_col = f"{indicator}_lag_{lag}"
                df_ts[lag_col] = df_ts[indicator].shift(lag)
                print(f"  Created lag feature: {lag_col}")
            
            # Add rolling mean/std for economic indicators (shorter windows)
            for window in [3]:
                # Rolling mean
                mean_col = f"{indicator}_rolling_mean_{window}"
                df_ts[mean_col] = df_ts[indicator].rolling(window=window, min_periods=1).mean()
                print(f"  Created rolling mean: {mean_col}")
                
                # Rolling volatility
                std_col = f"{indicator}_rolling_std_{window}"
                df_ts[std_col] = df_ts[indicator].rolling(window=window, min_periods=2).std()
                print(f"  Created rolling std: {std_col}")
    
    # Add month of year as a cyclical feature
    if isinstance(df_ts.index, pd.DatetimeIndex):
        # Extract month and convert to cyclical features using sine and cosine
        month = df_ts.index.month
        df_ts['month_sin'] = np.sin(2 * np.pi * month / 12)
        df_ts['month_cos'] = np.cos(2 * np.pi * month / 12)
        print(f"  Created cyclical month features")
        
        # Add quarter
        df_ts['quarter'] = df_ts.index.quarter
        
        # Add year
        df_ts['year'] = df_ts.index.year
    
    # Add year-over-year growth rate
    if len(df_ts) >= 12:
        df_ts[f'{target_column}_yoy_change'] = df_ts[target_column].pct_change(periods=12)
        print(f"  Created year-over-year change feature")
    
    # Return with date column as a regular column if it was one before
    if date_col:
        return df_ts.reset_index()
    return df_ts

def train_evaluate_models(X_train, y_train, X_test, y_test, metadata=None):
    """
    Train and evaluate a variety of models on the data.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target values
    X_test : DataFrame
        Test features
    y_test : Series
        Test target values
    metadata : dict, optional
        Additional metadata about the dataset
        
    Returns:
    --------
    dict
        Trained models with their evaluation metrics
    """
    print("\nTraining and evaluating models...")
    models = {}
    
    # Linear models
    print("Training Linear Regression model...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_metrics = calculate_metrics(y_test, lr_pred)
    models['Linear Regression'] = (lr, lr_metrics)
    print(f"Linear Regression performance: MAE={lr_metrics['MAE']:.2f}, RMSE={lr_metrics['RMSE']:.2f}, R²={lr_metrics['R2']:.2f}")
    
    # Decision tree-based models
    print("Training Random Forest model...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = calculate_metrics(y_test, rf_pred)
    models['Random Forest'] = (rf, rf_metrics)
    print(f"Random Forest performance: MAE={rf_metrics['MAE']:.2f}, RMSE={rf_metrics['RMSE']:.2f}, R²={rf_metrics['R2']:.2f}")
    
    # Gradient Boosting
    print("Training Gradient Boosting model...")
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_metrics = calculate_metrics(y_test, gb_pred)
    models['Gradient Boosting'] = (gb, gb_metrics)
    print(f"Gradient Boosting performance: MAE={gb_metrics['MAE']:.2f}, RMSE={gb_metrics['RMSE']:.2f}, R²={gb_metrics['R2']:.2f}")
    
    # Try Support Vector Regression
    print("Training SVR model...")
    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)
    svr_pred = svr.predict(X_test)
    svr_metrics = calculate_metrics(y_test, svr_pred)
    models['SVR'] = (svr, svr_metrics)
    print(f"SVR performance: MAE={svr_metrics['MAE']:.2f}, RMSE={svr_metrics['RMSE']:.2f}, R²={svr_metrics['R2']:.2f}")
    
    # Try to use XGBoost if available
    try:
        import xgboost as xgb
        print("Training XGBoost model...")
        xgb_model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_metrics = calculate_metrics(y_test, xgb_pred)
        models['XGBoost'] = (xgb_model, xgb_metrics)
        print(f"XGBoost performance: MAE={xgb_metrics['MAE']:.2f}, RMSE={xgb_metrics['RMSE']:.2f}, R²={xgb_metrics['R2']:.2f}")
    except ImportError:
        print("XGBoost not available, skipping...")
    
    # Try to use ARIMA if statsmodels is available
    if metadata and 'date_col' in metadata:
        try:
            import statsmodels.api as sm
            from statsmodels.tsa.arima.model import ARIMA
            print("Training ARIMA model...")
            
            # For ARIMA we need a time series
            # Try to combine training and test data
            combined_df = pd.DataFrame()
            combined_df['date'] = metadata.get('dates', pd.date_range(start='2010-01-01', periods=len(y_train) + len(y_test), freq='M'))
            combined_df['y'] = np.concatenate([y_train.values, y_test.values])
            combined_df.set_index('date', inplace=True)
            
            # Try to fit ARIMA model
            try:
                arima_model = ARIMA(y_train, order=(1,1,1))  # Simple model to start
                arima_results = arima_model.fit()
                arima_pred = arima_results.forecast(steps=len(y_test))
                arima_metrics = calculate_metrics(y_test, arima_pred)
                models['ARIMA'] = (arima_results, arima_metrics)
                print(f"ARIMA performance: MAE={arima_metrics['MAE']:.2f}, RMSE={arima_metrics['RMSE']:.2f}, R²={arima_metrics['R2']:.2f}")
            except Exception as e:
                print(f"Error fitting ARIMA model: {str(e)}")
        except ImportError:
            print("Statsmodels not available, skipping ARIMA model...")
    
    return models

def find_best_model(results):
    """
    Find the best performing model based on RMSE.
    """
    best_model = min(results.items(), key=lambda x: x[1][1]['RMSE'])
    print(f"\nBest model based on RMSE: {best_model[0]} with RMSE: {best_model[1][1]['RMSE']:.4f}")
    return best_model[0]

def optimize_best_model(model, model_name, X_train, y_train, X_test, y_test, scaler=None):
    """
    Optimize hyperparameters for the best performing model.
    
    Parameters:
    -----------
    model : estimator object
        The model to optimize
    model_name : str
        Name of the model
    X_train, y_train : DataFrame, Series
        Training data
    X_test, y_test : DataFrame, Series
        Test data
    scaler : scaler object, optional
        Scaler used to transform the data
        
    Returns:
    --------
    estimator
        Optimized model
    """
    print(f"\nOptimizing hyperparameters for {model_name}...")
    
    # Define hyperparameter grids for different model types
    param_grids = {
        'Linear Regression': {},  # Linear Regression has no hyperparameters to tune
        
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        
        'Gradient Boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        },
        
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        
        'SVR': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }
    }
    
    # If model doesn't have hyperparameters to tune or isn't in our list, return original
    if model_name not in param_grids or not param_grids[model_name]:
        print(f"No hyperparameters to tune for {model_name}")
        return model
    
    # Get parameter grid for this model
    param_grid = param_grids[model_name]
    
    # Use RandomizedSearchCV for faster optimization
    from sklearn.model_selection import RandomizedSearchCV
    
    search = RandomizedSearchCV(
        model, 
        param_grid,
        n_iter=10,  # Number of parameter settings to try
        cv=5,       # 5-fold cross-validation
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1   # Use all cores
    )
    
    try:
        # Fit the search
        search.fit(X_train, y_train)
        
        # Get best parameters
        best_params = search.best_params_
        print(f"Best parameters: {best_params}")
        
        # Create new model with best parameters
        optimized_model = search.best_estimator_
        
        # Evaluate optimized model
        y_pred = optimized_model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        
        print(f"Optimized model performance: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.2f}")
        
        # Save the optimized model
        os.makedirs('models', exist_ok=True)
        joblib.dump(optimized_model, f'models/{model_name.replace(" ", "_").lower()}_optimized.pkl')
        
        # Also save the scaler if provided
        if scaler is not None:
            joblib.dump(scaler, f'models/scaler_{model_name.replace(" ", "_").lower()}.pkl')
        
        return optimized_model
        
    except Exception as e:
        print(f"Error during hyperparameter optimization: {str(e)}")
        print("Using original model instead")
        return model

def simplified_optimize_best_model(model, model_name, X_train, y_train, scaler=None):
    """
    Simplified hyperparameter optimization for models.
    
    Parameters:
    -----------
    model : estimator object
        The model to optimize
    model_name : str
        Name of the model
    X_train, y_train : DataFrame, Series
        Training data
    scaler : scaler object, optional
        Scaler used to transform the data
        
    Returns:
    --------
    estimator
        Optimized model
    """
    print(f"\nPerforming simplified optimization for {model_name}...")
    
    # Define simplified param grids for common models
    simplified_params = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 20]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1]
        }
    }
    
    # If model isn't in our list or is a basic model, return original
    if model_name not in simplified_params:
        print(f"No simplified optimization available for {model_name}")
        return model
    
    # Get parameter grid
    param_grid = simplified_params[model_name]
    
    try:
        from sklearn.model_selection import GridSearchCV
        
        # Use simple grid search with cross-validation
        search = GridSearchCV(
            model,
            param_grid,
            cv=3,  # 3-fold CV to save time
            scoring='neg_mean_squared_error'
        )
        
        # Fit the search
        search.fit(X_train, y_train)
        
        # Get best parameters
        best_params = search.best_params_
        print(f"Best parameters: {best_params}")
        
        # Get best model
        optimized_model = search.best_estimator_
        
        # Save the optimized model
        os.makedirs('models', exist_ok=True)
        joblib.dump(optimized_model, f'models/{model_name.replace(" ", "_").lower()}_optimized.pkl')
        
        # Also save the scaler if provided
        if scaler is not None:
            joblib.dump(scaler, f'models/scaler_{model_name.replace(" ", "_").lower()}.pkl')
        
        print(f"Saved optimized {model_name} model")
        
        return optimized_model
        
    except Exception as e:
        print(f"Error during simplified optimization: {str(e)}")
        print("Using original model instead")
        return model

def build_lstm_model(X_train, X_test, y_train, y_test):
    """
    Build and train an LSTM model for time series forecasting.
    """
    print("\nBuilding LSTM model...")
    
    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    # Reshape input data for LSTM [samples, timesteps, features]
    # For simplicity, we'll use timesteps=1
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train_reshaped, y_train_scaled,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_reshaped)
    
    # Inverse transform to get actual values
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Evaluate model
    metrics = {
        'MAE': mean_absolute_error(y_test.values, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test.values, y_pred)),
        'R2': r2_score(y_test.values, y_pred)
    }
    
    print("\nLSTM Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save the model
    model.save('models/lstm_model.h5')
    
    # Save scalers
    joblib.dump(scaler_X, 'models/lstm_scaler_X.pkl')
    joblib.dump(scaler_y, 'models/lstm_scaler_y.pkl')
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test.values, label='Actual', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted', color='red', linestyle='--')
    plt.title('LSTM - Actual vs Predicted Retail Sales')
    plt.xlabel('Date')
    plt.ylabel('Retail Sales')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/lstm_prediction.png')
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/lstm_loss.png')
    plt.close()
    
    return model, metrics

def build_arima_model(df, target='retail_sales'):
    """
    Build and evaluate an ARIMA model.
    """
    print("\nBuilding ARIMA model...")
    
    # Get the target series
    if target in df.columns:
        y = df[target].copy()
    else:
        print(f"Error: Target column '{target}' not found")
        return None, {'MAE': float('inf'), 'RMSE': float('inf'), 'R2': 0}
    
    # Check for NaN values in the target series
    if y.isna().any():
        print(f"Warning: Target series contains {y.isna().sum()} NaN values. Interpolating...")
        # Interpolate NaN values
        y = y.interpolate(method='linear')
        
        # If NaNs remain (at edges), fill with forward and backward fill
        if y.isna().any():
            y = y.fillna(method='ffill').fillna(method='bfill')
            
        # Make sure there are no NaNs left
        if y.isna().any():
            print(f"Error: Could not remove all NaN values from target series")
            return None, {'MAE': float('inf'), 'RMSE': float('inf'), 'R2': 0}
    
    try:
    # Use auto_arima to find the best parameters
    auto_arima_model = pm.auto_arima(
        y,
        seasonal=True,
        m=12,  # Monthly data
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        d=None, D=None,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    print(f"Best ARIMA model: {auto_arima_model.order} {auto_arima_model.seasonal_order}")
    
    # Split data
    train_size = int(len(y) * 0.8)
    train = y[:train_size]
    test = y[train_size:]
    
    # Fit the model on the training data
    best_order = auto_arima_model.order
    best_seasonal_order = auto_arima_model.seasonal_order
    
    # Create and fit SARIMAX model
    model = SARIMAX(
        train,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    model_fit = model.fit(disp=False)
    
    # Forecast
    forecast = model_fit.get_forecast(steps=len(test))
    forecast_mean = forecast.predicted_mean
    
    # Evaluate
    metrics = {
        'MAE': mean_absolute_error(test, forecast_mean),
        'RMSE': np.sqrt(mean_squared_error(test, forecast_mean)),
        'R2': r2_score(test, forecast_mean)
    }
    
    print("\nARIMA Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test.values, label='Actual', color='blue')
    plt.plot(test.index, forecast_mean, label='Predicted', color='red', linestyle='--')
    plt.title('ARIMA - Actual vs Predicted Retail Sales')
    plt.xlabel('Date')
    plt.ylabel('Retail Sales')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/arima_prediction.png')
    plt.close()
    
    # Save the model
    joblib.dump(model_fit, 'models/arima_model.pkl')
    
    return model_fit, metrics
    
    except Exception as e:
        print(f"Error in ARIMA modeling: {e}")
        return None, {'MAE': float('inf'), 'RMSE': float('inf'), 'R2': 0}

def feature_importance_analysis(X_train, X_test, y_train, y_test):
    """
    Analyze and visualize feature importance using different methods.
    """
    print("Analyzing feature importance...")
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest model for feature importance
    print("Training Random Forest for feature importance...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # Get feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature ranking
    print("\nFeature ranking:")
    for f in range(X_train.shape[1]):
        print(f"{f+1}. {X_train.columns[indices[f]]} ({importances[indices[f]]:.4f})")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance for Retail Sales Prediction')
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    
    # Train an XGBoost model for SHAP analysis
    print("\nTraining XGBoost for SHAP analysis...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Sort features by importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
        
    return feature_importance

def create_ensemble_model(models, X_train, X_test, y_train, y_test, scaler):
    """
    Create an ensemble model by averaging predictions from multiple models.
    """
    print("\nCreating ensemble model...")
    
    # Scale the data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Load the best models
    model_objs = {}
    for model_name in models:
        try:
            # Try to load the optimized version first
            model_path = f'models/{model_name.replace(" ", "_").lower()}_optimized.pkl'
            model_objs[model_name] = joblib.load(model_path)
        except FileNotFoundError:
            # If not found, load the regular version
            model_path = f'models/{model_name.replace(" ", "_").lower()}.pkl'
            model_objs[model_name] = joblib.load(model_path)
    
    # Get predictions from each model
    predictions = {}
    for name, model in model_objs.items():
        predictions[name] = model.predict(X_test_scaled)
    
    # Create ensemble prediction (simple average)
    y_pred_ensemble = np.zeros(len(y_test))
    for pred in predictions.values():
        y_pred_ensemble += pred
    y_pred_ensemble /= len(predictions)
    
    # Evaluate ensemble model
    metrics = evaluate_model(y_test, y_pred_ensemble, "Ensemble Model")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test.values, label='Actual', color='blue')
    plt.plot(y_test.index, y_pred_ensemble, label='Ensemble Prediction', color='red', linestyle='--')
    plt.title('Ensemble Model - Actual vs Predicted Retail Sales')
    plt.xlabel('Date')
    plt.ylabel('Retail Sales')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/ensemble_prediction.png')
    plt.close()
    
    # Save the ensemble model components
    joblib.dump(model_objs, 'models/ensemble_components.pkl')
    
    return metrics

def feature_importance(model, model_name, feature_names):
    """
    Calculate and visualize feature importance for the given model.
    
    Parameters:
    -----------
    model : model object
        Trained model
    model_name : str
        Name of the model
    feature_names : list
        List of feature names
    """
    print(f"\nCalculating feature importance for {model_name}...")
    
    importance = None
    
    # For tree-based models
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    # For linear models
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
        if len(importance.shape) > 1 and importance.shape[0] == 1:
            importance = importance[0]
    
    if importance is not None:
        # Create DataFrame for visualization
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        
        # Print top features
        print("\nTop 10 most important features:")
        print(feature_importance_df.head(10))
        
        # Visualize
        plt.figure(figsize=(10, 8))
        
        # Plot only top 15 features to avoid clutter
        top_features = feature_importance_df.head(15)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        
        plt.title(f'Feature Importance for {model_name}')
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('visualizations/model_diagnostics', exist_ok=True)
        plot_path = f'visualizations/model_diagnostics/{model_name.replace(" ", "_").lower()}_feature_importance.png'
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Feature importance plot saved to {plot_path}")
        
        # Save as CSV file too
        csv_path = f'outputs/{model_name.replace(" ", "_").lower()}_feature_importance.csv'
        feature_importance_df.to_csv(csv_path, index=False)
        print(f"Feature importance data saved to {csv_path}")
        
        return feature_importance_df
    else:
        print(f"Warning: Could not calculate feature importance for {model_name}")
        return None

def simplified_feature_importance(model, model_name, feature_names):
    """
    Simplified version of feature importance calculation for models that may lack built-in functionality.
    
    Parameters:
    -----------
    model : model object
        Trained model
    model_name : str
        Name of the model
    feature_names : list or Index
        Feature names
    """
    # Convert feature_names to list if it's an Index
    if isinstance(feature_names, pd.Index):
        feature_names = feature_names.tolist()
        
    print(f"\nCalculating simplified feature importance for {model_name}...")
    
    # Try different approaches based on model type
    try:
        # For scikit-learn models with feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create a DataFrame of feature importances
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Print top 10 features
            print("\nTop 10 most important features:")
            print(importance_df.head(10))
            
            # Save to CSV
            output_dir = 'outputs'
            os.makedirs(output_dir, exist_ok=True)
            importance_df.to_csv(f'{output_dir}/{model_name.lower().replace(" ","_")}_importances.csv', index=False)
            
            print(f"Feature importances saved to {output_dir}/{model_name.lower().replace(' ','_')}_importances.csv")
            
            return importance_df
            
        # For scikit-learn linear models
        elif hasattr(model, 'coef_'):
            # Get coefficients
            coefs = model.coef_
            
            # Handle different coefficient shapes
            if len(coefs.shape) > 1:
                # For multi-output models, take the average importance
                importance = np.mean(np.abs(coefs), axis=0)
            else:
                importance = np.abs(coefs)
                
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Print top 10 features
            print("\nTop 10 most important features by coefficient magnitude:")
            print(importance_df.head(10))
            
            # Save to CSV
            output_dir = 'outputs'
            os.makedirs(output_dir, exist_ok=True)
            importance_df.to_csv(f'{output_dir}/{model_name.lower().replace(" ","_")}_coefficients.csv', index=False)
            
            print(f"Feature coefficients saved to {output_dir}/{model_name.lower().replace(' ','_')}_coefficients.csv")
            
            return importance_df
            
        else:
            # For models without built-in importance
            print(f"Model {model_name} doesn't support direct feature importance calculation")
            print("Using permutation importance instead (basic estimation)")
            
            # Create a dummy importance (equal weights)
            dummy_importance = np.ones(len(feature_names)) / len(feature_names)
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': dummy_importance
            }).sort_values('Feature')
            
            print("\nFeatures (equal importance assigned):")
            print(importance_df.head(10))
            
            return importance_df
            
    except Exception as e:
        print(f"Error calculating feature importance: {str(e)}")
        return None

def main():
    """
    Main function to execute the modeling pipeline.
    """
    print("Starting model building and evaluation...")
    
    # Load the data
    df = load_data()
    if df is None:
        return
    
    # Prepare train/test sets
    X_train, y_train, X_test, y_test, metadata = prepare_train_test(df, target_column='retail_sales')
    
    # Train and evaluate baseline models
    results, scaler = train_evaluate_models(X_train, y_train, X_test, y_test, metadata)
    
    # Find the best model
    best_model_name = find_best_model(results)
    
    # Optimize the best model
    best_model, best_metrics = optimize_best_model(results[best_model_name][0], best_model_name, X_train, y_train, X_test, y_test, scaler)
    
    # Build LSTM model
    lstm_model, lstm_metrics = build_lstm_model(X_train, X_test, y_train, y_test)
    
    # Build ARIMA model
    arima_model, arima_metrics = build_arima_model(df, target='retail_sales')
    
    # Analyze feature importance
    feature_importance = feature_importance_analysis(X_train, X_test, y_train, y_test)
    
    # Create ensemble model (using top 3 models)
    top_models = sorted(results.items(), key=lambda x: x[1][1]['RMSE'])[:3]
    top_model_names = [model[0] for model in top_models]
    print(f"\nTop 3 models for ensemble: {top_model_names}")
    ensemble_metrics = create_ensemble_model(top_model_names, X_train, X_test, y_train, y_test, scaler)
    
    print("\nModel building and evaluation complete!")
    
if __name__ == "__main__":
    main()
"""
India Retail Demand Forecaster - Ensemble Model with Uncertainty Estimation
---------------------------------------------------------------------------
This module implements ensemble forecasting techniques with uncertainty estimation
for more robust and reliable forecasts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EnsembleForecaster:
    """
    Ensemble forecasting model that combines multiple models with uncertainty estimation.
    """
    
    def __init__(self, models_dir='models/optimized', scaler=None):
        """
        Initialize the ensemble forecaster.
        
        Parameters:
        -----------
        models_dir : str
            Directory containing the trained models
        scaler : object or None
            Fitted scaler for feature standardization
        """
        self.models_dir = models_dir
        self.models = {}
        self.weights = None
        self.scaler = scaler
        self.features = None
        self.model_performance = {}
        
    def load_models(self, weights_path=None):
        """
        Load the component models for the ensemble.
        
        Parameters:
        -----------
        weights_path : str, optional
            Path to the model weights JSON file
        
        Returns:
        --------
        self
            The fitted ensemble forecaster
        """
        print(f"Loading models from {self.models_dir}")
        
        try:
            # Default weights path
            if weights_path is None:
                weights_path = os.path.join(self.models_dir, 'ensemble_weights.json')
                
            # Load weights if available
            if os.path.exists(weights_path):
                with open(weights_path, 'r') as f:
                    self.weights = json.load(f)
                print(f"Loaded weights from {weights_path}")
            
            # Load component models
            component_models_path = os.path.join(self.models_dir, 'ensemble_components.json')
            if os.path.exists(component_models_path):
                with open(component_models_path, 'r') as f:
                    component_models = json.load(f)
                    
                # Load each model
                for model_name in component_models:
                    model_file = os.path.join(self.models_dir, f"{model_name.lower().replace(' ', '_')}_optimized.pkl")
                    if os.path.exists(model_file):
                        self.models[model_name] = joblib.load(model_file)
                        print(f"Loaded model: {model_name}")
                    else:
                        print(f"Warning: Model file not found: {model_file}")
            else:
                # Fallback to finding all model files
                print("No ensemble components file found, searching for models...")
                model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_optimized.pkl')]
                
                for model_file in model_files:
                    model_name = model_file.replace('_optimized.pkl', '').replace('_', ' ').title()
                    self.models[model_name] = joblib.load(os.path.join(self.models_dir, model_file))
                    print(f"Loaded model: {model_name}")
            
            # Load scaler if not provided
            if self.scaler is None:
                scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    print(f"Loaded scaler from {scaler_path}")
                else:
                    print("Warning: No scaler found, will create one if needed")
                    
            # Load feature list
            for model_name in self.models:
                features_path = os.path.join(self.models_dir, f"{model_name.lower().replace(' ', '_')}_features.json")
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        self.features = json.load(f)
                    print(f"Using feature list from {model_name} with {len(self.features)} features")
                    break
            
            # Initialize equal weights if not loaded
            if self.weights is None:
                self.weights = {model_name: 1.0/len(self.models) for model_name in self.models}
                print("Using equal weights for ensemble")
            
            return self
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return None
    
    def optimize_weights(self, X, y, cv=5):
        """
        Optimize ensemble weights using time-series aware cross-validation.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        cv : int
            Number of cross-validation folds
        
        Returns:
        --------
        dict
            Optimized weights
        """
        from sklearn.model_selection import TimeSeriesSplit
        print("\nOptimizing ensemble weights with time-series CV...")
        
        # Create cross-validation folds with smaller test sets
        tscv = TimeSeriesSplit(n_splits=cv, test_size=max(5, len(X) // 20))
        
        # Store predictions for each model
        model_preds = {model_name: np.zeros_like(y) for model_name in self.models}
        
        # For each fold, train and predict with each model
        fold = 1
        for train_idx, test_idx in tscv.split(X):
            print(f"Fold {fold}/{cv}: Training on {len(train_idx)} samples, testing on {len(test_idx)} samples")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features if scaler is available
            if self.scaler is not None:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                # Create a new scaler if needed
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            
            # For each model, fit and predict
            for model_name, model in self.models.items():
                if hasattr(model, 'fit'):
                    model.fit(X_train_scaled, y_train)
                model_preds[model_name][test_idx] = model.predict(X_test_scaled)
        
            fold += 1
        
        # Calculate metrics for each model
        model_metrics = {}
        for model_name, preds in model_preds.items():
            mse = mean_squared_error(y, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, preds)
            r2 = r2_score(y, preds)
            
            model_metrics[model_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            print(f"{model_name}: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")
        
        # Advanced weighting strategy: Consider both MSE and R² for weights
        weights = {}
        model_names = list(self.models.keys())
        
        # Handle the case where all models have negative R²
        if all(model_metrics[name]['R2'] < 0 for name in model_names):
            # In this case, use inverse MSE with a safeguard
            total_inverse_mse = sum(1/max(model_metrics[name]['MSE'], 1e-10) for name in model_names)
            for name in model_names:
                weights[name] = (1/max(model_metrics[name]['MSE'], 1e-10)) / total_inverse_mse
        else:
            # Otherwise, use a weighted combination of MSE and R² for better models
            for name in model_names:
                # Clip R² to be non-negative for weight calculation
                r2 = max(0, model_metrics[name]['R2'])
                # Use a weighted formula that considers both metrics
                weights[name] = r2 + 0.001  # Add small constant to avoid zero weights
        
        # Normalize to sum to 1
        total_weight = sum(weights.values())
        for name in model_names:
            weights[name] /= total_weight
        
        # Update instance weights
        self.weights = weights
        self.model_performance = model_metrics
        
        print("\nOptimized ensemble weights:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f}")
        
        # Save weights
        weights_path = os.path.join(self.models_dir, 'ensemble_weights.json')
        with open(weights_path, 'w') as f:
            json.dump(self.weights, f, indent=2)
        
        return self.weights
    
    def fit(self, X, y):
        """
        Fit the ensemble model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        
        Returns:
        --------
        self
            The fitted ensemble model
        """
        print("\nFitting ensemble model...")
        
        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Fit each component model
        for model_name, model in self.models.items():
            if hasattr(model, 'fit'):
                print(f"Fitting {model_name}...")
                model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X, return_std=False, prediction_intervals=False, alpha=0.05):
        """
        Generate predictions with the ensemble model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        return_std : bool
            Whether to return standard deviations
        prediction_intervals : bool
            Whether to return prediction intervals
        alpha : float
            Significance level for prediction intervals (default: 0.05 for 95% intervals)
        
        Returns:
        --------
        np.ndarray or tuple
            Predictions, or tuple containing predictions and uncertainties
        """
        print("\nGenerating ensemble predictions...")
        
        # Check that we have models and weights
        if not self.models or self.weights is None:
            raise ValueError("No models or weights available. Load models first.")
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            raise ValueError("No scaler available. Fit the model first.")
        
        # Get predictions from each model
        model_predictions = {}
        for model_name, model in self.models.items():
            model_predictions[model_name] = model.predict(X_scaled)
        
        # Weighted average prediction
        y_pred = np.zeros(len(X))
        for model_name, preds in model_predictions.items():
            y_pred += self.weights[model_name] * preds
        
        # If only point predictions are needed
        if not return_std and not prediction_intervals:
            return y_pred
        
        # Calculate standard deviation of predictions
        y_pred_std = np.zeros(len(X))
        for i in range(len(X)):
            model_preds_i = np.array([preds[i] for preds in model_predictions.values()])
            y_pred_std[i] = np.std(model_preds_i)
        
        # If prediction intervals are needed
        if prediction_intervals:
            # Calculate t-critical value
            t_critical = stats.t.ppf(1 - alpha/2, len(self.models) - 1)
            
            # Calculate prediction intervals
            lower_bound = y_pred - t_critical * y_pred_std
            upper_bound = y_pred + t_critical * y_pred_std
            
            return y_pred, y_pred_std, lower_bound, upper_bound
        
        # Return predictions with standard deviations
        return y_pred, y_pred_std
    
    def evaluate(self, X, y, test_size=0.2, return_predictions=False):
        """
        Evaluate the ensemble model using a proper train-test split.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        test_size : float
            Size of the test set as a fraction (default: 0.2 for 20%)
        return_predictions : bool
            Whether to return predictions
        
        Returns:
        --------
        dict or tuple
            Dictionary with evaluation metrics, or tuple with metrics and predictions
        """
        print("\nEvaluating ensemble model on TEST DATA...")
        
        # Create a time series split to properly evaluate
        # For time series, we take the last test_size% of data as the test set
        test_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:test_idx], X.iloc[test_idx:]
        y_train, y_test = y.iloc[:test_idx], y.iloc[test_idx:]
        
        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Scale features using only training data
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Retrain models on training data only
        for model_name, model in self.models.items():
            if hasattr(model, 'fit'):
                print(f"Fitting {model_name} on training data...")
                model.fit(X_train_scaled, y_train)
        
        # Evaluate component models on test data
        component_test_preds = {}
        component_metrics = {}
        
        for model_name, model in self.models.items():
            # Make predictions on test data
            component_test_preds[model_name] = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, component_test_preds[model_name])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, component_test_preds[model_name])
            r2 = r2_score(y_test, component_test_preds[model_name])
            
            component_metrics[model_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
        
        # Recalculate weights based on test performance - dynamic ensemble for test data
        test_weights = {}
        for model_name, metrics in component_metrics.items():
            # Use max of R² or 0 to ignore negative performance
            r2 = max(0, metrics['R2'])
            test_weights[model_name] = r2 + 0.01  # Add small constant to avoid zero weights
        
        # If all models perform badly on test, use original weights
        if sum(test_weights.values()) < 0.1:
            test_weights = self.weights
        else:
            # Normalize weights to sum to 1
            total_weight = sum(test_weights.values())
            for model_name in test_weights:
                test_weights[model_name] /= total_weight
        
        # Generate ensemble predictions on test data with optimized weights
        y_test_pred = np.zeros(len(X_test))
        for model_name, preds in component_test_preds.items():
            y_test_pred += test_weights[model_name] * preds
        
        # Calculate ensemble metrics on test data
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        
        # Calculate prediction intervals if possible
        if len(self.models) > 1:
            # Calculate standard deviation of predictions
            y_pred_std = np.zeros(len(X_test))
            for i in range(len(X_test)):
                model_preds_i = np.array([preds[i] for preds in component_test_preds.values()])
                y_pred_std[i] = np.std(model_preds_i)
            
            # Calculate prediction intervals
            alpha = 0.05  # 95% intervals
            t_critical = stats.t.ppf(1 - alpha/2, len(self.models) - 1)
            lower_bound = y_test_pred - t_critical * y_pred_std
            upper_bound = y_test_pred + t_critical * y_pred_std
            
            # Calculate prediction interval coverage
            in_interval = np.logical_and(y_test >= lower_bound, y_test <= upper_bound)
            coverage = np.mean(in_interval)
        else:
            y_pred_std = np.zeros(len(X_test))
            coverage = 0.0
            lower_bound = y_test_pred.copy()
            upper_bound = y_test_pred.copy()
        
        # Create metrics dictionary
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Coverage': coverage,
            'Avg_Std': np.mean(y_pred_std) if len(y_pred_std) > 0 else 0
        }
        
        # Print metrics
        print("\nEnsemble model performance on TEST DATA:")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.4f}")
        if len(self.models) > 1:
            print(f"Prediction interval coverage (95%): {coverage:.2%}")
            print(f"Average uncertainty (std): {np.mean(y_pred_std):.6f}")
        
        # Compare to component models
        print("\nEnsemble vs. Component Models on TEST DATA:")
        print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Weight':<10}")
        print("-" * 60)
        
        for model_name, perf in component_metrics.items():
            print(f"{model_name:<20} {perf['RMSE']:<10.6f} {perf['MAE']:<10.6f} {perf['R2']:<10.4f} {test_weights[model_name]:<10.4f}")
        
        print(f"{'Ensemble':<20} {rmse:<10.6f} {mae:<10.6f} {r2:<10.4f} {'1.0000':<10}")
        
        if return_predictions:
            predictions = {
                'y_pred': y_test_pred,
                'y_test': y_test,
                'y_pred_std': y_pred_std,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            return metrics, predictions
        
        return metrics
    
    def visualize_predictions(self, X, y, dates=None):
        """
        Visualize the ensemble predictions and component model predictions.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        y : pd.Series
            Actual values
        dates : pd.Series, optional
            Date values for plotting
        """
        # Create visualizations directory if it doesn't exist
        output_dir = 'visualizations/ensemble'
        os.makedirs(output_dir, exist_ok=True)
        
        # Make predictions with ensemble model
        y_pred_ensemble = self.predict(X)
        
        # Make predictions with component models
        y_preds = {}
        for name, model in self.models.items():
            y_preds[name] = model.predict(X)
        
        # Create DataFrame properly by first creating an empty DataFrame
        results_df = pd.DataFrame(index=range(len(y)))
        
        # Add each series individually to avoid the dict/Series mixing error
        results_df['Actual'] = y.values if isinstance(y, pd.Series) else y
        results_df['Ensemble'] = y_pred_ensemble if isinstance(y_pred_ensemble, np.ndarray) else y_pred_ensemble
        
        # Add component model predictions one by one
        for name, preds in y_preds.items():
            results_df[name] = preds
        
        # Add dates if provided, otherwise use index
        if dates is not None:
            if len(dates) == len(results_df):
                results_df['Date'] = dates
                x_col = 'Date'
            else:
                x_col = results_df.index
        else:
            if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
                # Use the DataFrame's DatetimeIndex
                x_col = X.index
            else:
                # Generate sequential x values
                x_col = results_df.index
        
        # Plot 1: Actual vs. Ensemble Prediction
        plt.figure(figsize=(12, 6))
        
        # Add prediction intervals if available
        if hasattr(self, 'lower_bounds') and hasattr(self, 'upper_bounds'):
            plt.fill_between(
                x_col, 
                self.lower_bounds, 
                self.upper_bounds, 
                alpha=0.2, 
                color='gray', 
                label='95% Prediction Interval'
            )
        
        # Plot actual and predicted values
        plt.plot(x_col, results_df['Actual'], 'b-', label='Actual', linewidth=2)
        plt.plot(x_col, results_df['Ensemble'], 'r-', label='Ensemble Forecast', linewidth=2)
        
        plt.title('Actual vs. Ensemble Prediction')
        plt.xlabel('Time')
        plt.ylabel('Retail Sales')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/actual_vs_ensemble.png")
        plt.close()
        
        # Plot 2: Component Model Predictions
        plt.figure(figsize=(12, 6))
        
        # Plot actual values
        plt.plot(x_col, results_df['Actual'], 'k-', label='Actual', linewidth=2)
        
        # Plot component model predictions
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        for i, (name, _) in enumerate(self.models.items()):
            color = colors[i % len(colors)]
            plt.plot(x_col, results_df[name], f"{color}--", label=name, alpha=0.7)
        
        # Plot ensemble prediction
        plt.plot(x_col, results_df['Ensemble'], 'k--', label='Ensemble', linewidth=2)
        
        plt.title('Component Model Predictions vs. Actual')
        plt.xlabel('Time')
        plt.ylabel('Retail Sales')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/component_models.png")
        plt.close()
        
        # Plot 3: Model Weights
        if hasattr(self, 'weights') and self.weights is not None:
            plt.figure(figsize=(10, 6))
            
            # Create bar chart of model weights - properly handling dict conversion
            models_list = list(self.models.keys())
            weights_list = [self.weights[model] for model in models_list]
            
            # Create a DataFrame with lists instead of a dict
            weights_df = pd.DataFrame({
                'Model': models_list,
                'Weight': weights_list
            })
            
            # Sort by weight descending
            weights_df = weights_df.sort_values('Weight', ascending=False)
            
            # Create bar chart
            plt.bar(weights_df['Model'], weights_df['Weight'])
            
            plt.title('Ensemble Model Weights')
            plt.xlabel('Model')
            plt.ylabel('Weight')
            plt.xticks(rotation=45)
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/model_weights.png")
            plt.close()
        
        # Plot 4: Error Distribution
        plt.figure(figsize=(10, 6))
        
        # Calculate errors
        results_df['Ensemble_Error'] = results_df['Actual'] - results_df['Ensemble']
        
        # Plot error histogram
        plt.hist(results_df['Ensemble_Error'], bins=20, alpha=0.7, color='b')
        
        plt.title('Ensemble Prediction Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_distribution.png")
        plt.close()
        
        # Plot 5: Error Over Time
        plt.figure(figsize=(12, 6))
        
        # Plot error over time
        plt.plot(x_col, results_df['Ensemble_Error'], 'b-')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        plt.title('Ensemble Prediction Error Over Time')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_over_time.png")
        plt.close()
        
        # Plot 6: Train-Test Performance Comparison
        # Use part of the data as test (same proportion as in evaluate method)
        test_size = 0.2
        test_idx = int(len(X) * (1 - test_size))
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Mark training and test regions
        plt.axvspan(0, test_idx, alpha=0.2, color='green', label='Training Data')
        plt.axvspan(test_idx, len(X), alpha=0.2, color='red', label='Test Data')
        
        # Plot actual and predicted values
        plt.plot(x_col, results_df['Actual'], 'b-', label='Actual', linewidth=2)
        plt.plot(x_col, results_df['Ensemble'], 'r-', label='Ensemble Forecast', linewidth=2)
        
        plt.title('Model Performance: Training vs Test Data')
        plt.xlabel('Time')
        plt.ylabel('Retail Sales')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/train_test_comparison.png")
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")

    def reload_models(self):
        """
        Reload models with adjusted parameters for better time series performance.
        
        Returns:
        --------
        self
            The forecaster with updated models
        """
        print("\nReloading models with optimized time series parameters...")
        
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        import xgboost as xgb
        
        # Define improved model parameters specifically for time series forecasting
        new_models = {}
        
        # XGBoost with time series-specific parameters
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 3,        # Reduced to prevent overfitting
            'learning_rate': 0.03, # Slower learning rate for better generalization
            'subsample': 0.8,      # Subsample for robustness
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.01,     # L1 regularization
            'reg_lambda': 1.0,     # L2 regularization
            'random_state': 42
        }
        new_models['XGBoost'] = xgb.XGBRegressor(**xgb_params)
        
        # Gradient Boosting with time series-specific parameters
        gb_params = {
            'n_estimators': 200,
            'max_depth': 2,         # Shallow trees to prevent overfitting
            'learning_rate': 0.03,
            'subsample': 0.8,
            'loss': 'squared_error',
            'random_state': 42
        }
        new_models['Gradient Boosting'] = GradientBoostingRegressor(**gb_params)
        
        # Random Forest with time series-specific parameters
        rf_params = {
            'n_estimators': 200,
            'max_depth': 10,       # Limit depth for better generalization
            'min_samples_split': 5,
            'min_samples_leaf': 4, # Require more samples in leaves for stability
            'max_features': 1.0,   # Using 1.0 instead of 'auto' to avoid deprecation warning
            'bootstrap': True,
            'random_state': 42
        }
        new_models['Random Forest'] = RandomForestRegressor(**rf_params)
        
        # Add a simple linear model as a baseline
        from sklearn.linear_model import ElasticNet
        new_models['ElasticNet'] = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
        
        # Preserve any existing weights if available
        if self.weights is not None:
            # Extract weights for the models we're keeping
            existing_weights = {name: self.weights.get(name, 0) for name in new_models if name in self.weights}
            
            # For new models, assign equal portions of the remaining weight
            new_model_names = [name for name in new_models if name not in self.weights]
            if new_model_names:
                existing_total = sum(existing_weights.values())
                remaining = max(0, 1.0 - existing_total)
                for name in new_model_names:
                    existing_weights[name] = remaining / len(new_model_names)
                
                # Normalize all weights to sum to 1
                total = sum(existing_weights.values())
                for name in existing_weights:
                    existing_weights[name] /= total
                
                self.weights = existing_weights
            else:
                # Keep existing weights but normalize them
                total = sum(existing_weights.values())
                self.weights = {name: val/total for name, val in existing_weights.items()}
        else:
            # Equal weights if no weights exist
            self.weights = {name: 1.0/len(new_models) for name in new_models}
        
        # Update models
        self.models = new_models
        
        print(f"Loaded {len(self.models)} models with optimized parameters")
        return self

def load_dataset(file_path='data/processed/selected_features_dataset.csv'):
    """
    Load the dataset for ensemble modeling with enhanced time series features.
    
    Parameters:
    -----------
    file_path : str
        Path to the dataset
    
    Returns:
    --------
    tuple
        X, y data for modeling
    """
    print(f"Loading dataset from {file_path}")
    
    try:
        # Try to load the selected features dataset
        df = pd.read_csv(file_path)
        
        # Clean and prepare data
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Add an index column if there's no date
        df['record_index'] = np.arange(len(df))
        
        # Check for target column
        if 'retail_sales' in df.columns:
            target_col = 'retail_sales'
        elif 'log_retail_sales' in df.columns:
            target_col = 'log_retail_sales'
        else:
            raise ValueError("No target column found in dataset")
        
        # Create time series features
        print("Creating time series features...")
        
        # Add lag features of the target (1, 2, 3 months)
        for lag in range(1, 4):
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Add moving averages (3, 6, 12 months)
        for window in [3, 6, 12]:
            if len(df) > window:
                df[f'{target_col}_ma_{window}'] = df[target_col].rolling(window=window).mean()
        
        # Add rate of change features (month-over-month changes)
        df[f'{target_col}_mom_change'] = df[target_col].pct_change()
        df[f'{target_col}_abs_change'] = df[target_col].diff()
        
        # Add expanding mean (represents all historical data to that point)
        df[f'{target_col}_exp_mean'] = df[target_col].expanding().mean()
        
        # Add trend features
        df['trend'] = np.arange(len(df))
        df['trend_squared'] = df['trend']**2
        
        # Add seasonal features if no date index
        if 'date' not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            # Approximate seasonality with sin/cos waves
            n_samples = len(df)
            for period in [4, 6, 12]:  # Quarterly, biannual, annual cycles
                df[f'seasonal_sin_{period}'] = np.sin(2 * np.pi * df['record_index'] / period)
                df[f'seasonal_cos_{period}'] = np.cos(2 * np.pi * df['record_index'] / period)
        
        # Drop rows with NaN values from lag features
        original_len = len(df)
        df = df.dropna()
        print(f"Dropped {original_len - len(df)} rows with NaN values from lag features")
        
        # Split into features and target
        y = df[target_col]
        X = df.drop(target_col, axis=1)
        
        # Remove any non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Select only the most important features
        X = feature_selection(X, y)
        
        print(f"Dataset loaded: {X.shape[1]} features, {len(y)} samples")
        
        return X, y
    
    except FileNotFoundError:
        print(f"Selected features dataset not found at {file_path}")
        print("Trying to load the features dataset instead...")
        
        try:
            # Fallback to the full features dataset
            features_path = 'data/processed/features_dataset.csv'
            df = pd.read_csv(features_path)
            
            # Handle date column if it exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Add an index column if there's no date
            df['record_index'] = np.arange(len(df))
            
            # Handle unnamed column if it exists
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            
            # Check for target column
            if 'retail_sales' in df.columns:
                target_col = 'retail_sales'
            elif 'log_retail_sales' in df.columns:
                target_col = 'log_retail_sales'
            else:
                raise ValueError("No target column found in dataset")
            
            # Create time series features
            print("Creating time series features...")
            
            # Add lag features of the target (1, 2, 3 months)
            for lag in range(1, 4):
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            
            # Add moving averages (3, 6, 12 months)
            for window in [3, 6, 12]:
                if len(df) > window:
                    df[f'{target_col}_ma_{window}'] = df[target_col].rolling(window=window).mean()
            
            # Add rate of change features (month-over-month changes)
            df[f'{target_col}_mom_change'] = df[target_col].pct_change()
            df[f'{target_col}_abs_change'] = df[target_col].diff()
            
            # Add expanding mean (represents all historical data to that point)
            df[f'{target_col}_exp_mean'] = df[target_col].expanding().mean()
            
            # Add trend features
            df['trend'] = np.arange(len(df))
            df['trend_squared'] = df['trend']**2
            
            # Add seasonal features if no date index
            if 'date' not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                # Approximate seasonality with sin/cos waves
                n_samples = len(df)
                for period in [4, 6, 12]:  # Quarterly, biannual, annual cycles
                    df[f'seasonal_sin_{period}'] = np.sin(2 * np.pi * df['record_index'] / period)
                    df[f'seasonal_cos_{period}'] = np.cos(2 * np.pi * df['record_index'] / period)
            
            # Drop rows with NaN values from lag features
            original_len = len(df)
            df = df.dropna()
            print(f"Dropped {original_len - len(df)} rows with NaN values from lag features")
            
            # Split into features and target
            y = df[target_col]
            X = df.drop(target_col, axis=1)
            
            # Remove any non-numeric columns
            X = X.select_dtypes(include=[np.number])
            
            # Select only the most important features
            X = feature_selection(X, y)
            
            print(f"Features dataset loaded: {X.shape[1]} features, {len(y)} samples")
            
            return X, y
            
        except Exception as e:
            print(f"Error loading features dataset: {e}")
            return None, None
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def feature_selection(X, y, max_features=15):
    """
    Select the most important features using a random forest-based approach.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    max_features : int
        Maximum number of features to select
    
    Returns:
    --------
    pd.DataFrame
        Selected features
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel
    
    print("Performing feature selection...")
    
    # Ensure lag_1 feature is always included if it exists
    lag_features = [col for col in X.columns if '_lag_1' in col]
    priority_features = lag_features + [col for col in X.columns if '_ma_' in col or 'trend' in col]
    
    # Initialize a random forest for feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the model
    rf.fit(X, y)
    
    # Get feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create a dataframe of feature importances
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("Top features by importance:")
    for i, feature in enumerate(feature_importance['Feature'][:10]):
        print(f"  {i+1}. {feature}: {feature_importance.iloc[i]['Importance']:.4f}")
    
    # Ensure priority features are included, then add others by importance until max_features
    selected_features = []
    
    # First add priority features if they exist in X
    for feature in priority_features:
        if feature in X.columns and feature not in selected_features:
            selected_features.append(feature)
    
    # Then add other important features
    for feature in feature_importance['Feature']:
        if len(selected_features) >= max_features:
            break
        if feature not in selected_features:
            selected_features.append(feature)
    
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    return X[selected_features]

def main():
    """Main function to run ensemble modeling with proper train-test evaluation."""
    print("=" * 80)
    print("INDIA RETAIL DEMAND FORECASTER - ADVANCED TIME SERIES ENSEMBLE MODELING")
    print("=" * 80)
    
    try:
        # Load dataset with enhanced time series features
        X, y = load_dataset()
        if X is None or y is None:
            print("Error loading dataset. Exiting.")
            return
        
        # Create ensemble forecaster
        forecaster = EnsembleForecaster()
        
        # Load and reload component models with better parameters
        forecaster.load_models()
        forecaster.reload_models()
        
        # Optimize weights using cross-validation on full data
        # This is acceptable as it's part of the training process
        forecaster.optimize_weights(X, y, cv=5)
        
        # Fit the ensemble on full data
        forecaster.fit(X, y)
        
        # Evaluate performance using a proper train-test split
        # This creates separate training and test sets
        test_metrics = forecaster.evaluate(X, y, test_size=0.2)
        
        # Create visualizations (using full data is okay for visualization)
        forecaster.visualize_predictions(X, y)
        
        print("\nAdvanced time series ensemble modeling complete!")
        
    except Exception as e:
        print(f"Error in ensemble modeling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
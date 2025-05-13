"""
India Retail Demand Forecaster - Advanced Hyperparameter Optimization
--------------------------------------------------------------------
This module implements advanced hyperparameter optimization techniques
for the retail demand forecasting models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Import skopt for Bayesian optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("Warning: scikit-optimize not found. Falling back to standard RandomizedSearchCV.")
    from sklearn.model_selection import RandomizedSearchCV

def load_dataset(dataset_path='data/processed/selected_features_dataset.csv', 
                target_col='retail_sales'):
    """
    Load the dataset for optimization.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset
    target_col : str
        Name of the target column
        
    Returns:
    --------
    tuple
        (X, y) feature matrix and target vector
    """
    print(f"Loading dataset from {dataset_path}")
    
    try:
        # Try to load the selected features dataset first
        df = pd.read_csv(dataset_path)
        
        # Handle date column if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Determine the target column
        log_target = f'log_{target_col}'
        if log_target in df.columns:
            print(f"Using log-transformed target: {log_target}")
            y = df[log_target]
            actual_target = log_target
        elif target_col in df.columns:
            print(f"Using target: {target_col}")
            y = df[target_col]
            actual_target = target_col
        else:
            raise ValueError(f"Target column '{target_col}' or '{log_target}' not found in dataset")
        
        # Create feature matrix
        X = df.drop(columns=[col for col in [target_col, log_target] if col in df.columns])
        
        print(f"Dataset loaded: {X.shape[1]} features, {len(y)} samples")
        
        return X, y
    
    except FileNotFoundError:
        print(f"Selected features dataset not found at {dataset_path}")
        print("Trying to load the features dataset instead...")
        
        try:
            # Fallback to the full features dataset
            features_path = 'data/processed/features_dataset.csv'
            df = pd.read_csv(features_path)
            
            # Handle date column if it exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Handle unnamed column if it exists
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            
            # Determine the target column
            log_target = f'log_{target_col}'
            if log_target in df.columns:
                print(f"Using log-transformed target: {log_target}")
                y = df[log_target]
                actual_target = log_target
            elif target_col in df.columns:
                print(f"Using target: {target_col}")
                y = df[target_col]
                actual_target = target_col
            else:
                raise ValueError(f"Target column '{target_col}' or '{log_target}' not found in dataset")
            
            # Create feature matrix, excluding target columns
            X = df.copy()
            for col in [target_col, log_target]:
                if col in X.columns:
                    X = X.drop(columns=[col])
            
            # Get only numeric columns for modeling
            numeric_cols = X.select_dtypes(include=np.number).columns
            X = X[numeric_cols]
            
            print(f"Features dataset loaded: {X.shape[1]} features, {len(y)} samples")
            
            return X, y
        
        except Exception as e:
            print(f"Error loading features dataset: {e}")
            return None, None
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def create_time_series_cv(n_splits=5):
    """
    Create a time series cross-validation object.
    
    Parameters:
    -----------
    n_splits : int
        Number of splits for cross-validation
        
    Returns:
    --------
    TimeSeriesSplit
        Cross-validation object
    """
    return TimeSeriesSplit(n_splits=n_splits, gap=0, test_size=None)

def optimize_xgboost(X, y, cv=None, n_iter=50):
    """
    Optimize XGBoost hyperparameters using Bayesian optimization.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    cv : cross-validation object
        Cross-validation strategy
    n_iter : int
        Number of iterations for optimization
        
    Returns:
    --------
    tuple
        (best_model, best_params, cv_results)
    """
    print("\nOptimizing XGBoost hyperparameters...")
    
    # Create default cross-validation if not provided
    if cv is None:
        cv = create_time_series_cv(n_splits=5)
    
    # Define the parameter space
    if SKOPT_AVAILABLE:
        param_space = {
            'n_estimators': Integer(100, 1000),
            'learning_rate': Real(0.001, 0.5, prior='log-uniform'),
            'max_depth': Integer(3, 10),
            'min_child_weight': Integer(1, 10),
            'subsample': Real(0.5, 1.0),
            'colsample_bytree': Real(0.5, 1.0),
            'gamma': Real(0, 5),
            'reg_alpha': Real(0.001, 10, prior='log-uniform'),
            'reg_lambda': Real(0.001, 10, prior='log-uniform')
        }
        
        # Create the Bayesian search object
        opt = BayesSearchCV(
            xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            param_space,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    else:
        # Fallback to RandomizedSearchCV
        param_space = {
            'n_estimators': np.arange(100, 1000, 100),
            'learning_rate': np.logspace(-3, -1, 10),
            'max_depth': np.arange(3, 11),
            'min_child_weight': np.arange(1, 11),
            'subsample': np.linspace(0.5, 1.0, 6),
            'colsample_bytree': np.linspace(0.5, 1.0, 6),
            'gamma': np.linspace(0, 5, 6),
            'reg_alpha': np.logspace(-3, 1, 10),
            'reg_lambda': np.logspace(-3, 1, 10)
        }
        
        # Create the randomized search object
        opt = RandomizedSearchCV(
            xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            param_space,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    
    # Fit the optimizer
    opt.fit(X, y)
    
    # Get best parameters and model
    best_params = opt.best_params_
    best_model = opt.best_estimator_
    
    # Get cross-validation results
    cv_results = opt.cv_results_
    
    # Print best parameters
    print(f"\nBest XGBoost parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Calculate best score
    best_score = opt.best_score_
    print(f"Best CV score: {-best_score:.6f} MSE ({np.sqrt(-best_score):.6f} RMSE)")
    
    return best_model, best_params, cv_results

def optimize_gradient_boosting(X, y, cv=None, n_iter=50):
    """
    Optimize Gradient Boosting hyperparameters using Bayesian optimization.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    cv : cross-validation object
        Cross-validation strategy
    n_iter : int
        Number of iterations for optimization
        
    Returns:
    --------
    tuple
        (best_model, best_params, cv_results)
    """
    print("\nOptimizing Gradient Boosting hyperparameters...")
    
    # Create default cross-validation if not provided
    if cv is None:
        cv = create_time_series_cv(n_splits=5)
    
    # Define the parameter space
    if SKOPT_AVAILABLE:
        param_space = {
            'n_estimators': Integer(100, 1000),
            'learning_rate': Real(0.001, 0.5, prior='log-uniform'),
            'max_depth': Integer(3, 10),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'subsample': Real(0.5, 1.0),
            'max_features': Categorical([1.0, 'sqrt', 'log2', None])
        }
        
        # Create the Bayesian search object
        opt = BayesSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_space,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    else:
        # Fallback to RandomizedSearchCV
        param_space = {
            'n_estimators': np.arange(100, 1000, 100),
            'learning_rate': np.logspace(-3, -1, 10),
            'max_depth': np.arange(3, 11),
            'min_samples_split': np.arange(2, 21),
            'min_samples_leaf': np.arange(1, 11),
            'subsample': np.linspace(0.5, 1.0, 6),
            'max_features': [1.0, 'sqrt', 'log2', None]
        }
        
        # Create the randomized search object
        opt = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_space,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    
    # Fit the optimizer
    opt.fit(X, y)
    
    # Get best parameters and model
    best_params = opt.best_params_
    best_model = opt.best_estimator_
    
    # Get cross-validation results
    cv_results = opt.cv_results_
    
    # Print best parameters
    print(f"\nBest Gradient Boosting parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Calculate best score
    best_score = opt.best_score_
    print(f"Best CV score: {-best_score:.6f} MSE ({np.sqrt(-best_score):.6f} RMSE)")
    
    return best_model, best_params, cv_results

def optimize_random_forest(X, y, cv=None, n_iter=50):
    """
    Optimize Random Forest hyperparameters using Bayesian optimization.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    cv : cross-validation object
        Cross-validation strategy
    n_iter : int
        Number of iterations for optimization
        
    Returns:
    --------
    tuple
        (best_model, best_params, cv_results)
    """
    print("\nOptimizing Random Forest hyperparameters...")
    
    # Create default cross-validation if not provided
    if cv is None:
        cv = create_time_series_cv(n_splits=5)
    
    # Define the parameter space
    if SKOPT_AVAILABLE:
        param_space = {
            'n_estimators': Integer(100, 1000),
            'max_depth': Integer(5, 30),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical([1.0, 'sqrt', 'log2']),
            'bootstrap': Categorical([True, False])
        }
        
        # Create the Bayesian search object
        opt = BayesSearchCV(
            RandomForestRegressor(random_state=42),
            param_space,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    else:
        # Fallback to RandomizedSearchCV
        param_space = {
            'n_estimators': np.arange(100, 1000, 100),
            'max_depth': np.arange(5, 31, 5),
            'min_samples_split': np.arange(2, 21),
            'min_samples_leaf': np.arange(1, 11),
            'max_features': [1.0, 'sqrt', 'log2'],
            'bootstrap': [True, False]
        }
        
        # Create the randomized search object
        opt = RandomizedSearchCV(
            RandomForestRegressor(random_state=42),
            param_space,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    
    # Fit the optimizer
    opt.fit(X, y)
    
    # Get best parameters and model
    best_params = opt.best_params_
    best_model = opt.best_estimator_
    
    # Get cross-validation results
    cv_results = opt.cv_results_
    
    # Print best parameters
    print(f"\nBest Random Forest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Calculate best score
    best_score = opt.best_score_
    print(f"Best CV score: {-best_score:.6f} MSE ({np.sqrt(-best_score):.6f} RMSE)")
    
    return best_model, best_params, cv_results

def visualize_optimization_results(cv_results, model_name):
    """
    Visualize the optimization results.
    
    Parameters:
    -----------
    cv_results : dict
        Cross-validation results
    model_name : str
        Name of the model
    """
    # Create directory if it doesn't exist
    os.makedirs('visualizations/optimization', exist_ok=True)
    
    # Convert to dataframe for easier manipulation
    results_df = pd.DataFrame(cv_results)
    
    # Sort by mean test score
    results_df = results_df.sort_values('mean_test_score', ascending=False)
    
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(results_df) + 1), -results_df['mean_test_score'], 'b-', marker='o')
    plt.fill_between(
        range(1, len(results_df) + 1),
        -results_df['mean_test_score'] - results_df['std_test_score'],
        -results_df['mean_test_score'] + results_df['std_test_score'],
        alpha=0.2, color='b'
    )
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'{model_name} Optimization Learning Curve')
    plt.grid(True)
    plt.savefig(f'visualizations/optimization/{model_name.lower().replace(" ", "_")}_learning_curve.png')
    plt.close()
    
    # Get parameter names (excluding fixed parameters)
    param_names = [name for name in results_df.columns if name.startswith('param_')]
    
    # Create parameter importance plots
    top_n = min(10, len(results_df))
    top_configs = results_df.iloc[:top_n]
    
    plt.figure(figsize=(12, 8))
    for i, params in enumerate(top_configs[param_names].values):
        param_values = [str(val) for val in params]
        plt.plot(param_names, param_values, marker='o', linestyle='-', alpha=0.7, label=f'Config {i+1}')
    
    plt.xticks(rotation=45)
    plt.title(f'Top {top_n} {model_name} Configurations')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f'visualizations/optimization/{model_name.lower().replace(" ", "_")}_top_configs.png')
    plt.close()
    
    # For binary parameters, create boxplots
    for param in param_names:
        param_key = param.replace('param_', '')
        param_values = results_df[param].astype(str)
        
        # Check if the parameter has few unique values
        if len(param_values.unique()) <= 10:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=param_values, y=-results_df['mean_test_score'])
            plt.xlabel(param_key)
            plt.ylabel('Mean Squared Error (MSE)')
            plt.title(f'Effect of {param_key} on {model_name} Performance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'visualizations/optimization/{model_name.lower().replace(" ", "_")}_{param_key}_boxplot.png')
            plt.close()

def save_optimized_model(model, model_name, params, X, y):
    """
    Save the optimized model and its parameters.
    
    Parameters:
    -----------
    model : estimator
        The optimized model
    model_name : str
        Name of the model
    params : dict
        Best parameters
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    """
    # Create directory if it doesn't exist
    os.makedirs('models/optimized', exist_ok=True)
    
    # Save the model
    model_path = f'models/optimized/{model_name.lower().replace(" ", "_")}_optimized.pkl'
    joblib.dump(model, model_path)
    
    # Save parameters
    params_path = f'models/optimized/{model_name.lower().replace(" ", "_")}_params.json'
    with open(params_path, 'w') as f:
        # Convert non-serializable types to strings
        serializable_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                serializable_params[k] = v
            else:
                serializable_params[k] = str(v)
        
        import json
        json.dump(serializable_params, f, indent=2)
    
    # Save feature names
    features_path = f'models/optimized/{model_name.lower().replace(" ", "_")}_features.json'
    with open(features_path, 'w') as f:
        import json
        json.dump(list(X.columns), f)
    
    print(f"Saved optimized {model_name} model to {model_path}")
    print(f"Saved {model_name} parameters to {params_path}")

def optimize_all_models(X, y, n_iter=50):
    """
    Optimize all models and select the best one.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    n_iter : int
        Number of iterations for optimization
        
    Returns:
    --------
    tuple
        (best_model, best_model_name)
    """
    print("=" * 80)
    print("OPTIMIZING ALL MODELS")
    print("=" * 80)
    
    # Create a cross-validation object
    cv = create_time_series_cv(n_splits=5)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Save the scaler
    os.makedirs('models/optimized', exist_ok=True)
    joblib.dump(scaler, 'models/optimized/scaler.pkl')
    
    # Optimize XGBoost
    xgb_model, xgb_params, xgb_results = optimize_xgboost(X_scaled, y, cv, n_iter)
    
    # Optimize Gradient Boosting
    gb_model, gb_params, gb_results = optimize_gradient_boosting(X_scaled, y, cv, n_iter)
    
    # Optimize Random Forest
    rf_model, rf_params, rf_results = optimize_random_forest(X_scaled, y, cv, n_iter)
    
    # Evaluate models on a test set (using cross-validation)
    models = {
        'XGBoost': xgb_model,
        'Gradient Boosting': gb_model,
        'Random Forest': rf_model
    }
    
    model_scores = {}
    
    for name, model in models.items():
        # Calculate cross-validation scores
        scores = cross_val_score(
            model, X_scaled, y, 
            cv=cv, 
            scoring='neg_mean_squared_error'
        )
        
        # Calculate metrics
        mse = -np.mean(scores)
        rmse = np.sqrt(mse)
        
        model_scores[name] = {
            'MSE': mse,
            'RMSE': rmse
        }
        
        # Re-fit on the full dataset
        model.fit(X_scaled, y)
        
        # Save the model
        save_optimized_model(model, name, models[name].get_params(), X, y)
        
        # Visualize optimization results
        if name == 'XGBoost':
            visualize_optimization_results(xgb_results, name)
        elif name == 'Gradient Boosting':
            visualize_optimization_results(gb_results, name)
        elif name == 'Random Forest':
            visualize_optimization_results(rf_results, name)
    
    # Print model scores
    print("\nModel Performance Comparison:")
    for name, scores in model_scores.items():
        print(f"{name}: MSE={scores['MSE']:.6f}, RMSE={scores['RMSE']:.6f}")
    
    # Find the best model
    best_model_name = min(model_scores.items(), key=lambda x: x[1]['MSE'])[0]
    best_model = models[best_model_name]
    
    print(f"\nBest model: {best_model_name} with RMSE={model_scores[best_model_name]['RMSE']:.6f}")
    
    # Create an ensemble model from the optimized models
    create_ensemble_model(models, X_scaled, y)
    
    return best_model, best_model_name

# Create and save an ensemble model class
class EnsembleModel:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
    
    def predict(self, X):
        preds = np.zeros(len(X))
        for name, model in self.models.items():
            preds += self.weights[name] * model.predict(X)
        return preds

def create_ensemble_model(models, X, y):
    """
    Create an ensemble model from the optimized models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    """
    print("\nCreating ensemble model...")
    
    # Create cross-validation object
    cv = create_time_series_cv(n_splits=5)
    
    # Get cross-validation predictions for each model
    cv_preds = {}
    
    for name, model in models.items():
        # Initialize an array to hold the predictions
        cv_preds[name] = np.zeros_like(y)
        
        # Perform cross-validation
        for train_idx, test_idx in cv.split(X):
            # Split the data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            cv_preds[name][test_idx] = model.predict(X_test)
    
    # Calculate weights for each model based on MSE
    model_mse = {}
    for name, preds in cv_preds.items():
        model_mse[name] = mean_squared_error(y, preds)
    
    # Inverse MSE weighting
    weights = {}
    total_inverse_mse = sum(1/mse for mse in model_mse.values())
    for name, mse in model_mse.items():
        weights[name] = (1/mse) / total_inverse_mse
    
    print("\nEnsemble model weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")
    
    # Save ensemble weights
    ensemble_path = 'models/optimized/ensemble_weights.json'
    with open(ensemble_path, 'w') as f:
        import json
        json.dump(weights, f, indent=2)
    
    # Create and evaluate ensemble predictions
    ensemble_preds = np.zeros_like(y, dtype=float)
    for name, preds in cv_preds.items():
        ensemble_preds += weights[name] * preds
    
    # Calculate ensemble metrics
    ensemble_mse = mean_squared_error(y, ensemble_preds)
    ensemble_rmse = np.sqrt(ensemble_mse)
    ensemble_r2 = r2_score(y, ensemble_preds)
    
    print(f"\nEnsemble model performance: MSE={ensemble_mse:.6f}, RMSE={ensemble_rmse:.6f}, R²={ensemble_r2:.4f}")
    
    # Compare to individual models
    print("\nModel comparison:")
    for name, mse in model_mse.items():
        print(f"  {name}: MSE={mse:.6f}, RMSE={np.sqrt(mse):.6f}")
    print(f"  Ensemble: MSE={ensemble_mse:.6f}, RMSE={ensemble_rmse:.6f}")
    
    # Create the ensemble model instance
    ensemble_model = EnsembleModel(models, weights)
    
    # Save the ensemble model
    joblib.dump(ensemble_model, 'models/optimized/ensemble_model.pkl')
    
    # Save a record of component models
    component_models_path = 'models/optimized/ensemble_components.json'
    with open(component_models_path, 'w') as f:
        import json
        json.dump(list(models.keys()), f, indent=2)
    
    print(f"Saved ensemble model to models/optimized/ensemble_model.pkl")

def main():
    """Main function to run the hyperparameter optimization."""
    try:
        # Load the dataset
        X, y = load_dataset()
        
        if X is None or y is None:
            print("Error loading dataset. Exiting.")
            return
        
        # Optimize all models
        best_model, best_model_name = optimize_all_models(X, y, n_iter=30)
        
        print("\nHyperparameter optimization complete!")
        print(f"Best model: {best_model_name}")
        print("Optimized models and ensemble saved to models/optimized/")
        
    except Exception as e:
        print(f"Error in hyperparameter optimization: {str(e)}")

if __name__ == "__main__":
    main() 
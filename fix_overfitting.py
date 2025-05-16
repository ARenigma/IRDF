#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to implement fixes for overfitting in the models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Create directory for results
os.makedirs('visualizations/overfitting_fixed', exist_ok=True)

def calculate_metrics(y_true, y_pred):
    """Calculate standard regression metrics"""
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics

def load_sample_data():
    """Load the sample data for evaluation"""
    print("Loading sample data...")
    
    # Check if retail_sales.csv exists
    try:
        # Load retail sales data
        sales_df = pd.read_csv('data/retail_sales.csv')
        
        # Load macro indicators data
        macro_df = pd.read_csv('data/macro_indicators.csv')
        
        # Convert dates and set as index
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        macro_df['date'] = pd.to_datetime(macro_df['date'])
        
        sales_df.set_index('date', inplace=True)
        macro_df.set_index('date', inplace=True)
        
        # Merge datasets
        merged_df = sales_df.join(macro_df, how='inner')
        
        # Drop any rows with missing values for simplicity
        merged_df = merged_df.dropna()
        
        # Create basic features - lag features for retail sales
        for lag in range(1, 4):
            merged_df[f'sales_lag_{lag}'] = merged_df['retail_sales'].shift(lag)
        
        # Drop rows with NaN from lag creation
        merged_df = merged_df.dropna()
        
        # Create target variable (y) and features (X)
        y = merged_df['retail_sales']
        X = merged_df.drop('retail_sales', axis=1)
        
        print(f"Loaded and prepared data with {X.shape[0]} samples and {X.shape[1]} features")
        return X, y
    
    except FileNotFoundError:
        print("Error: Required data files not found.")
        return None, None

def select_features(X, y, n_features=None):
    """Select the most important features to reduce model complexity"""
    print("Performing feature selection...")
    
    if n_features is None:
        # Heuristic: sqrt of sample size is a good starting point
        n_features = min(int(np.sqrt(len(X))), X.shape[1])
    
    print(f"Selecting top {n_features} features from {X.shape[1]} total features")
    
    # Use a simple RFE with a stable model for feature selection
    selector = RFE(estimator=Ridge(alpha=1.0), n_features_to_select=n_features)
    selector.fit(X, y)
    
    # Get selected features
    selected_features = X.columns[selector.support_]
    print("Selected features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i}. {feature}")
    
    return X[selected_features], selected_features

def create_fixed_models():
    """Create models with reduced complexity to address overfitting"""
    print("Creating fixed models with reduced complexity...")
    
    fixed_models = {
        # Regularized linear models
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5),
        
        # Simplified tree-based models
        'Random Forest (Fixed)': RandomForestRegressor(
            n_estimators=50,             # Reduced number of trees
            max_depth=5,                 # Limited depth
            min_samples_leaf=5,          # Require more samples per leaf
            min_samples_split=10,        # Require more samples to split
            max_features='sqrt',         # Use only sqrt(n_features) features
            random_state=42
        ),
        
        'Gradient Boosting (Fixed)': GradientBoostingRegressor(
            n_estimators=50,             # Reduced number of trees
            learning_rate=0.05,          # Smaller learning rate
            max_depth=3,                 # Limited depth
            min_samples_leaf=5,          # Require more samples per leaf
            min_samples_split=10,        # Require more samples to split
            max_features='sqrt',         # Use only sqrt(n_features) features
            subsample=0.8,               # Use only 80% of samples per tree
            random_state=42
        ),
    }
    
    return fixed_models

def evaluate_model_with_cv(model, X, y, model_name, cv=5):
    """
    Evaluate model using time series cross-validation to get a more
    reliable estimate of performance.
    """
    print(f"Evaluating {model_name} with {cv}-fold time series cross-validation...")
    
    # Create time series split
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Store metrics for each fold
    train_metrics = []
    test_metrics = []
    
    # Store predictions for later visualization
    all_preds = []
    all_trues = []
    
    # Perform cross-validation
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        fold_train_metrics = calculate_metrics(y_train, y_train_pred)
        fold_test_metrics = calculate_metrics(y_test, y_test_pred)
        
        # Store metrics
        train_metrics.append(fold_train_metrics)
        test_metrics.append(fold_test_metrics)
        
        # Store predictions
        all_preds.extend(y_test_pred)
        all_trues.extend(y_test.values)
        
        # Print fold results
        print(f"  Fold {i+1} - Train RMSE: {fold_train_metrics['RMSE']:.2f}, "
              f"Test RMSE: {fold_test_metrics['RMSE']:.2f}, "
              f"Test R²: {fold_test_metrics['R2']:.4f}")
    
    # Calculate average metrics
    avg_train_metrics = {
        metric: np.mean([fold[metric] for fold in train_metrics])
        for metric in train_metrics[0].keys()
    }
    
    avg_test_metrics = {
        metric: np.mean([fold[metric] for fold in test_metrics])
        for metric in test_metrics[0].keys()
    }
    
    # Calculate metric std devs
    std_test_metrics = {
        metric: np.std([fold[metric] for fold in test_metrics])
        for metric in test_metrics[0].keys()
    }
    
    # Assess overfitting
    rmse_increase = ((avg_test_metrics['RMSE'] - avg_train_metrics['RMSE']) / 
                      avg_train_metrics['RMSE']) * 100
    r2_drop = avg_train_metrics['R2'] - avg_test_metrics['R2']
    
    overfitting_threshold = 20  # % increase in error metrics
    r2_drop_threshold = 0.2     # absolute drop in R2
    
    if rmse_increase > overfitting_threshold or r2_drop > r2_drop_threshold:
        overfitting_status = "STILL OVERFITTING"
    elif rmse_increase > overfitting_threshold/2 or r2_drop > r2_drop_threshold/2:
        overfitting_status = "MILD OVERFITTING"
    else:
        overfitting_status = "NO SIGNIFICANT OVERFITTING"
    
    # Print summary
    print(f"\nAverage Cross-Validation Results for {model_name}:")
    print(f"  {'Metric':<6} {'Train':<12} {'Test':<12} {'Difference':<12} {'% Increase':<12}")
    print(f"  {'-'*50}")
    
    for metric in ['RMSE', 'MAE', 'R2']:
        diff = avg_test_metrics[metric] - avg_train_metrics[metric]
        pct = ((avg_test_metrics[metric] - avg_train_metrics[metric]) / 
                abs(avg_train_metrics[metric])) * 100 if avg_train_metrics[metric] != 0 else float('inf')
        
        if metric == 'R2':  # For R2, decrease is bad
            pct = -pct
            
        print(f"  {metric:<6} {avg_train_metrics[metric]:<12.6f} {avg_test_metrics[metric]:<12.6f} "
              f"{diff:<+12.6f} {pct:<+12.2f}%")
    
    print(f"  ASSESSMENT: {overfitting_status}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.scatter(all_trues, all_preds, alpha=0.5)
    
    # Add the perfect prediction line
    min_val = min(min(all_trues), min(all_preds))
    max_val = max(max(all_trues), max(all_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'{model_name}: Actual vs Predicted (CV)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig(f'visualizations/overfitting_fixed/{model_name.lower().replace(" ", "_")}_cv_predictions.png')
    plt.close()
    
    # Store results for comparison
    results = {
        'model_name': model_name,
        'avg_train_metrics': avg_train_metrics,
        'avg_test_metrics': avg_test_metrics,
        'std_test_metrics': std_test_metrics,
        'rmse_increase': rmse_increase,
        'r2_drop': r2_drop,
        'overfitting_status': overfitting_status
    }
    
    return results

def main():
    """Main function to implement fixes for overfitting"""
    print("\n" + "="*80)
    print("FIXING MODEL OVERFITTING")
    print("="*80)
    
    # Load sample data
    X, y = load_sample_data()
    if X is None or y is None:
        return
    
    # Step 1: Feature Selection to reduce dimensionality
    X_selected, selected_features = select_features(X, y)
    
    # Step 2: Create fixed models with reduced complexity
    fixed_models = create_fixed_models()
    
    # Step 3: Evaluate models with cross-validation
    results = []
    for model_name, model in fixed_models.items():
        print("\n" + "-"*70)
        print(f"Evaluating {model_name}")
        print("-"*70)
        
        # Evaluate with cross-validation
        model_results = evaluate_model_with_cv(model, X_selected, y, model_name)
        results.append({
            'Model': model_results['model_name'],
            'Train RMSE': model_results['avg_train_metrics']['RMSE'],
            'Test RMSE': model_results['avg_test_metrics']['RMSE'],
            'Test RMSE Std': model_results['std_test_metrics']['RMSE'],
            'RMSE % Increase': model_results['rmse_increase'],
            'Train R2': model_results['avg_train_metrics']['R2'],
            'Test R2': model_results['avg_test_metrics']['R2'],
            'Test R2 Std': model_results['std_test_metrics']['R2'],
            'R2 Drop': model_results['r2_drop'],
            'Overfitting Status': model_results['overfitting_status']
        })
    
    # Create summary report
    results_df = pd.DataFrame(results)
    
    # Sort by test RMSE (best performing models first)
    results_df = results_df.sort_values('Test RMSE')
    
    print("\n" + "="*80)
    print("MODEL COMPARISON AFTER OVERFITTING REDUCTION")
    print("="*80)
    print(results_df[['Model', 'Train RMSE', 'Test RMSE', 'RMSE % Increase', 
                     'Train R2', 'Test R2', 'Overfitting Status']].to_string(index=False))
    
    # Save results
    results_df.to_csv('visualizations/overfitting_fixed/fixed_models_comparison.csv', index=False)
    
    # Create comparative visualization
    plt.figure(figsize=(12, 10))
    
    # Plot RMSE comparison with error bars
    plt.subplot(2, 1, 1)
    models = results_df['Model']
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, results_df['Train RMSE'], width, label='Train', color='blue', alpha=0.7)
    plt.bar(x + width/2, results_df['Test RMSE'], width, label='Test', color='red', alpha=0.7,
            yerr=results_df['Test RMSE Std'], capsize=5)
    
    plt.xticks(x, models, rotation=45, ha='right')
    plt.title('RMSE: Training vs Test (with std dev)')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(axis='y')
    
    # Plot R2 comparison with error bars
    plt.subplot(2, 1, 2)
    plt.bar(x - width/2, results_df['Train R2'], width, label='Train', color='blue', alpha=0.7)
    plt.bar(x + width/2, results_df['Test R2'], width, label='Test', color='red', alpha=0.7,
            yerr=results_df['Test R2 Std'], capsize=5)
    
    plt.xticks(x, models, rotation=45, ha='right')
    plt.title('R²: Training vs Test (with std dev)')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('visualizations/overfitting_fixed/fixed_models_comparison.png')
    plt.close()
    
    # Find the best model
    best_model_row = results_df.loc[results_df['Test RMSE'].idxmin()]
    best_model_name = best_model_row['Model']
    
    print(f"\nBest model: {best_model_name}")
    print(f"  Test RMSE: {best_model_row['Test RMSE']:.4f} ± {best_model_row['Test RMSE Std']:.4f}")
    print(f"  Test R²: {best_model_row['Test R2']:.4f} ± {best_model_row['Test R2 Std']:.4f}")
    print(f"  Overfitting Status: {best_model_row['Overfitting Status']}")
    
    print(f"\nAnalysis complete. Reports and visualizations saved to visualizations/overfitting_fixed/")

if __name__ == "__main__":
    main() 
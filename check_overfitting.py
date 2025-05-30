#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to check for overfitting in the trained models.
This script loads trained models, evaluates them on both training and test sets,
and provides visualizations to detect overfitting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Create directory for overfitting checks
os.makedirs('visualizations/overfitting_check', exist_ok=True)

def calculate_metrics(y_true, y_pred):
    """Calculate standard regression metrics"""
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics

def load_data():
    """Load the data for evaluation"""
    # Look for processed data files
    print("Loading data...")
    try:
        # Try to load previously processed data
        X = pd.read_csv('data/processed/X_processed.csv', index_col=0)
        y = pd.read_csv('data/processed/y_processed.csv', index_col=0).iloc[:, 0]
        print(f"Loaded processed data with {X.shape[0]} samples and {X.shape[1]} features")
        return X, y
    except FileNotFoundError:
        print("Error: Processed data files not found. Run data_preparation.py first.")
        return None, None

def load_models():
    """Load the trained models"""
    print("Loading models...")
    models = {}
    
    # Try to load optimized models from models/optimized directory
    try:
        model_files = os.listdir('models/optimized')
        for file in model_files:
            if file.endswith('.pkl') and not file.startswith('scaler'):
                model_name = file.replace('.pkl', '').replace('_', ' ')
                model_path = os.path.join('models/optimized', file)
                models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name}")
    except FileNotFoundError:
        print("Warning: No optimized models found.")
    
    # Also try to load individual model files in models directory
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl') and not f.startswith('scaler')]
    for file in model_files:
        model_name = file.replace('.pkl', '').replace('_', ' ')
        model_path = os.path.join('models', file)
        models[model_name] = joblib.load(model_path)
        print(f"Loaded {model_name}")
    
    return models

def plot_learning_curve(model, X, y, model_name, cv=5):
    """
    Generate a learning curve for a model to check for overfitting.
    A learning curve shows training and validation scores for different training set sizes.
    """
    print(f"Generating learning curve for {model_name}...")
    
    # Create CV split (time series aware)
    cv_splitter = TimeSeriesSplit(n_splits=cv)
    
    # Define training sizes
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, 
        cv=cv_splitter, 
        train_sizes=train_sizes,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Calculate statistics
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.title(f'Learning Curve ({model_name})')
    plt.xlabel('Training examples')
    plt.ylabel('Mean Squared Error')
    plt.grid()
    
    plt.fill_between(train_sizes, 
                    train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, 
                    alpha=0.1, color="r")
    plt.fill_between(train_sizes, 
                    test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, 
                    alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f'visualizations/overfitting_check/{model_name.replace(" ", "_").lower()}_learning_curve.png')
    plt.close()

def evaluate_train_test_performance(model, X, y, model_name):
    """
    Evaluate model on both training and test sets to check for overfitting.
    A large gap between training and test performance indicates overfitting.
    """
    print(f"Evaluating train/test performance for {model_name}...")
    
    # Time-based split (last 20% for testing)
    test_size = 0.2
    test_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X.iloc[:test_idx], X.iloc[test_idx:]
    y_train, y_test = y.iloc[:test_idx], y.iloc[test_idx:]
    
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Make predictions on train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Print metrics comparison
    metrics_diff = {
        key: test_metrics[key] - train_metrics[key] 
        for key in train_metrics.keys()
    }
    
    print(f"  {'Metric':<6} {'Train':<12} {'Test':<12} {'Difference':<12} {'% Increase':<12}")
    print(f"  {'-'*50}")
    
    for metric in ['RMSE', 'MAE', 'R2']:
        pct_increase = ((test_metrics[metric] - train_metrics[metric]) / abs(train_metrics[metric])) * 100 if train_metrics[metric] != 0 else float('inf')
        
        if metric == 'R2':  # For R2, decrease is bad
            pct_increase = -pct_increase
            
        print(f"  {metric:<6} {train_metrics[metric]:<12.6f} {test_metrics[metric]:<12.6f} {metrics_diff[metric]:<+12.6f} {pct_increase:<+12.2f}%")
    
    # Overfitting assessment
    overfitting_threshold = 20  # % increase in error metrics
    r2_drop_threshold = 0.2     # absolute drop in R2
    
    rmse_increase = ((test_metrics['RMSE'] - train_metrics['RMSE']) / train_metrics['RMSE']) * 100
    r2_drop = train_metrics['R2'] - test_metrics['R2']
    
    if rmse_increase > overfitting_threshold or r2_drop > r2_drop_threshold:
        overfitting_status = "SEVERE OVERFITTING DETECTED"
    elif rmse_increase > overfitting_threshold/2 or r2_drop > r2_drop_threshold/2:
        overfitting_status = "MODERATE OVERFITTING DETECTED"
    else:
        overfitting_status = "NO SIGNIFICANT OVERFITTING"
    
    print(f"  ASSESSMENT: {overfitting_status}")
    
    # Create visual comparison
    metrics_comparison = pd.DataFrame({
        'Train': [train_metrics['RMSE'], train_metrics['MAE'], train_metrics['R2']],
        'Test': [test_metrics['RMSE'], test_metrics['MAE'], test_metrics['R2']]
    }, index=['RMSE', 'MAE', 'R2'])
    
    # Plot metrics comparison
    plt.figure(figsize=(10, 6))
    ax = metrics_comparison.loc[['RMSE', 'MAE']].plot(kind='bar', figsize=(10, 6))
    plt.title(f'Training vs Test Error Metrics ({model_name})')
    plt.ylabel('Error Value')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f'visualizations/overfitting_check/{model_name.replace(" ", "_").lower()}_error_comparison.png')
    plt.close()
    
    # Plot R2 comparison separately
    plt.figure(figsize=(10, 6))
    metrics_comparison.loc[['R2']].plot(kind='bar', figsize=(10, 6))
    plt.title(f'Training vs Test R² Score ({model_name})')
    plt.ylabel('R² Score')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f'visualizations/overfitting_check/{model_name.replace(" ", "_").lower()}_r2_comparison.png')
    plt.close()
    
    # Plot predictions
    plt.figure(figsize=(12, 6))
    
    # Plot train predictions
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(f'Training Set: Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Plot test predictions
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(f'Test Set: Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    plt.tight_layout()
    plt.savefig(f'visualizations/overfitting_check/{model_name.replace(" ", "_").lower()}_prediction_comparison.png')
    plt.close()
    
    return train_metrics, test_metrics, overfitting_status

def analyze_model_complexity(model, model_name):
    """
    Analyze the complexity of a model to provide recommendations for reducing overfitting.
    """
    print(f"Analyzing model complexity for {model_name}...")
    
    recommendations = []
    
    # Model-specific complexity analysis
    if hasattr(model, 'n_estimators'):
        n_trees = model.n_estimators
        print(f"  Number of trees: {n_trees}")
        if n_trees > 100:
            recommendations.append(f"Reduce the number of trees (current: {n_trees})")
    
    if hasattr(model, 'max_depth') and model.max_depth is not None:
        depth = model.max_depth
        print(f"  Maximum tree depth: {depth}")
        if depth > 5:
            recommendations.append(f"Reduce the maximum tree depth (current: {depth})")
    
    if hasattr(model, 'min_samples_leaf'):
        min_leaf = model.min_samples_leaf
        print(f"  Minimum samples per leaf: {min_leaf}")
        if min_leaf < 5:
            recommendations.append(f"Increase minimum samples per leaf (current: {min_leaf})")
    
    if hasattr(model, 'min_samples_split'):
        min_split = model.min_samples_split
        print(f"  Minimum samples to split: {min_split}")
        if min_split < 5:
            recommendations.append(f"Increase minimum samples to split (current: {min_split})")
    
    if hasattr(model, 'alpha'):  # For regularized models
        alpha = model.alpha
        print(f"  Regularization alpha: {alpha}")
        if alpha < 0.5:
            recommendations.append(f"Increase regularization strength (current alpha: {alpha})")
    
    # General recommendations
    if not recommendations:
        recommendations = [
            "Use cross-validation to tune hyperparameters",
            "Add regularization (L1 or L2) to reduce model complexity",
            "Reduce model complexity by using simpler model architecture",
            "Increase the size of the training dataset if possible",
            "Feature engineering or selection to reduce dimensionality"
        ]
    
    print("Recommendations to reduce overfitting:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    return recommendations

def main():
    """Main function to check for overfitting in models"""
    print("\n" + "="*50)
    print("MODEL OVERFITTING ANALYSIS")
    print("="*50)
    
    # Load data and models
    X, y = load_data()
    if X is None or y is None:
        return
    
    models = load_models()
    if not models:
        print("No models found for analysis.")
        return
    
    # Scale features if a scaler is available
    try:
        scaler = joblib.load('models/optimized/scaler.pkl')
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        print("Applied feature scaling using saved scaler")
    except FileNotFoundError:
        try:
            scaler = joblib.load('models/scaler.pkl')
            X_scaled = pd.DataFrame(
                scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
            print("Applied feature scaling using saved scaler")
        except FileNotFoundError:
            X_scaled = X
            print("No scaler found, using original features")
    
    # Results summary
    results = []
    
    # Analyze each model
    for model_name, model in models.items():
        print("\n" + "-"*50)
        print(f"Analyzing {model_name}")
        print("-"*50)
        
        # Generate learning curve
        try:
            plot_learning_curve(model, X_scaled, y, model_name)
        except Exception as e:
            print(f"Error generating learning curve: {str(e)}")
        
        # Evaluate train/test performance
        try:
            train_metrics, test_metrics, overfitting_status = evaluate_train_test_performance(
                model, X_scaled, y, model_name
            )
            
            # Store results
            results.append({
                'Model': model_name,
                'Train RMSE': train_metrics['RMSE'],
                'Test RMSE': test_metrics['RMSE'],
                'RMSE Diff': test_metrics['RMSE'] - train_metrics['RMSE'],
                'RMSE % Increase': ((test_metrics['RMSE'] - train_metrics['RMSE']) / train_metrics['RMSE']) * 100,
                'Train R2': train_metrics['R2'],
                'Test R2': test_metrics['R2'],
                'R2 Diff': train_metrics['R2'] - test_metrics['R2'],
                'Overfitting Status': overfitting_status
            })
            
            # If overfitting is detected, provide recommendations
            if "OVERFITTING" in overfitting_status:
                analyze_model_complexity(model, model_name)
        
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
    
    # Create summary report
    if results:
        results_df = pd.DataFrame(results)
        
        # Sort by overfitting severity (RMSE % increase)
        results_df = results_df.sort_values('RMSE % Increase', ascending=False)
        
        print("\n" + "="*50)
        print("OVERFITTING ANALYSIS SUMMARY")
        print("="*50)
        print(results_df[['Model', 'Train RMSE', 'Test RMSE', 'RMSE % Increase', 'Train R2', 'Test R2', 'Overfitting Status']].to_string(index=False))
        
        # Save summary to file
        results_df.to_csv('visualizations/overfitting_check/overfitting_summary.csv', index=False)
        
        # Create comparative visualization
        plt.figure(figsize=(12, 8))
        
        # Plot RMSE comparison
        plt.subplot(2, 1, 1)
        for i, model in enumerate(results_df['Model']):
            plt.bar(i-0.2, results_df.loc[results_df['Model']==model, 'Train RMSE'].values[0], width=0.4, label='Train' if i==0 else "", color='blue')
            plt.bar(i+0.2, results_df.loc[results_df['Model']==model, 'Test RMSE'].values[0], width=0.4, label='Test' if i==0 else "", color='red')
        
        plt.xticks(range(len(results_df['Model'])), results_df['Model'], rotation=45, ha='right')
        plt.title('RMSE: Training vs Test')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(axis='y')
        
        # Plot R2 comparison
        plt.subplot(2, 1, 2)
        for i, model in enumerate(results_df['Model']):
            plt.bar(i-0.2, results_df.loc[results_df['Model']==model, 'Train R2'].values[0], width=0.4, label='Train' if i==0 else "", color='blue')
            plt.bar(i+0.2, results_df.loc[results_df['Model']==model, 'Test R2'].values[0], width=0.4, label='Test' if i==0 else "", color='red')
        
        plt.xticks(range(len(results_df['Model'])), results_df['Model'], rotation=45, ha='right')
        plt.title('R²: Training vs Test')
        plt.ylabel('R² Score')
        plt.legend()
        plt.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig('visualizations/overfitting_check/all_models_comparison.png')
        plt.close()
        
        print(f"\nAnalysis complete. Reports and visualizations saved to visualizations/overfitting_check/")

if __name__ == "__main__":
    main() 
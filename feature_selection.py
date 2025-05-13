"""
India Retail Demand Forecaster - Feature Selection
-------------------------------------------------
This module implements various feature selection techniques to 
address multicollinearity and improve model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os
import warnings
warnings.filterwarnings('ignore')

def calculate_vif(X):
    """
    Calculate Variance Inflation Factors to identify multicollinearity.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features dataframe
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with VIF values for each feature
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    
    # Ensure X contains only numeric values and replace infinity with large values
    X_clean = X.copy()
    for col in X_clean.columns:
        # Replace infinite values with large finite values
        X_clean[col] = X_clean[col].replace([np.inf, -np.inf], np.nan)
        # Fill NaN values with column mean
        if X_clean[col].isnull().any():
            X_clean[col] = X_clean[col].fillna(X_clean[col].mean())
    
    # Calculate VIF
    vif_data["VIF"] = [variance_inflation_factor(X_clean.values, i) for i in range(X_clean.shape[1])]
    
    # Sort by VIF
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    return vif_data

def reduce_multicollinearity_vif(X, threshold=10, max_features=None):
    """
    Iteratively remove features with high VIF to reduce multicollinearity.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features dataframe
    threshold : float
        VIF threshold for removal
    max_features : int, optional
        Maximum number of features to keep
        
    Returns:
    --------
    list
        List of selected feature names
    """
    print(f"Starting with {X.shape[1]} features")
    
    # Make a copy of the dataframe
    X_reduced = X.copy()
    
    # Initialize tracking variables
    features_to_remove = []
    initial_feature_count = X_reduced.shape[1]
    
    # Calculate initial VIF values
    vif_data = calculate_vif(X_reduced)
    print(f"Initial max VIF: {vif_data['VIF'].max():.2f} for feature '{vif_data['Feature'].iloc[0]}'")
    
    # Iteratively remove features with highest VIF above threshold
    iteration = 1
    while (vif_data['VIF'].max() > threshold) and (X_reduced.shape[1] > 1):
        # If we've reached the max_features limit, stop
        if max_features is not None and X_reduced.shape[1] <= max_features:
            print(f"Reached target of {max_features} features, stopping removal")
            break
            
        # Get the feature with the highest VIF
        feature_to_remove = vif_data['Feature'].iloc[0]
        features_to_remove.append(feature_to_remove)
        
        # Remove the feature
        X_reduced = X_reduced.drop(columns=[feature_to_remove])
        
        # Recalculate VIF values
        vif_data = calculate_vif(X_reduced)
        
        # Print progress every 5 iterations
        if iteration % 5 == 0 or X_reduced.shape[1] <= 10:
            print(f"Iteration {iteration}: Removed '{feature_to_remove}', {X_reduced.shape[1]} features remaining")
            
            if not vif_data.empty:
                print(f"  Current max VIF: {vif_data['VIF'].max():.2f} for feature '{vif_data['Feature'].iloc[0]}'")
        
        iteration += 1
    
    # Report final results
    print(f"\nFinal feature set: {X_reduced.shape[1]} features (removed {initial_feature_count - X_reduced.shape[1]})")
    print(f"Final max VIF: {vif_data['VIF'].max():.2f}" if not vif_data.empty else "No features remaining")
    
    return X_reduced.columns.tolist()

def select_features_rfe(X, y, n_features_to_select=None, step=1):
    """
    Select features using Recursive Feature Elimination (RFE).
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features dataframe
    y : pd.Series
        Target variable
    n_features_to_select : int, optional
        Number of features to select
    step : int
        Number of features to remove at each iteration
        
    Returns:
    --------
    list
        List of selected feature names
    """
    # If n_features_to_select is not provided, use 1/3 of available features
    if n_features_to_select is None:
        n_features_to_select = max(1, X.shape[1] // 3)
    
    print(f"Performing RFE to select {n_features_to_select} features")
    
    # Initialize an estimator
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Create an RFE object
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step, verbose=1)
    
    # Fit the RFE
    rfe.fit(X, y)
    
    # Get the selected features
    selected_features = X.columns[rfe.support_].tolist()
    
    # Print the ranking of features
    feature_ranking = pd.DataFrame({
        'Feature': X.columns,
        'Ranking': rfe.ranking_
    }).sort_values('Ranking')
    
    print("\nFeature Rankings (lower is better):")
    print(feature_ranking.head(10))
    
    return selected_features

def select_features_pca(X, variance_threshold=0.95, scaler=None, return_components=False):
    """
    Reduce dimensionality using PCA.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features dataframe
    variance_threshold : float
        Minimum cumulative variance to retain
    scaler : object, optional
        Fitted scaler to use for standardization
    return_components : bool
        Whether to return the PCA components
        
    Returns:
    --------
    pd.DataFrame or tuple
        Transformed features dataframe, or tuple of (df, pca, scaler)
    """
    print(f"Performing PCA to retain {variance_threshold*100:.1f}% of variance")
    
    # Create a scaler if not provided
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Initialize PCA
    pca = PCA()
    
    # Fit and transform the data
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Determine number of components to keep
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    print(f"Selected {n_components} principal components out of {X.shape[1]}")
    print(f"Explained variance: {cumulative_variance[n_components-1]*100:.2f}%")
    
    # Create a dataframe with the selected components
    component_names = [f'PC{i+1}' for i in range(n_components)]
    X_reduced = pd.DataFrame(X_pca[:, :n_components], columns=component_names, index=X.index)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, alpha=0.6)
    plt.step(range(1, len(cumulative_variance)+1), cumulative_variance, where='mid', label='Cumulative Explained Variance')
    plt.axhline(y=variance_threshold, linestyle='--', color='r', label=f'{variance_threshold*100}% Variance Threshold')
    plt.axvline(x=n_components, linestyle='--', color='k', label=f'Selected Components: {n_components}')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend(loc='best')
    
    # Create directory if it doesn't exist
    os.makedirs('visualizations/feature_selection', exist_ok=True)
    plt.savefig('visualizations/feature_selection/pca_explained_variance.png')
    plt.close()
    
    # Plot feature contributions to the first two principal components
    plt.figure(figsize=(12, 10))
    loading_matrix = pca.components_.T[:, :2]  # Get loadings for first two components
    
    # Create a dataframe for easier plotting
    loadings_df = pd.DataFrame(loading_matrix, columns=['PC1', 'PC2'], index=X.columns)
    
    # Plot PC contributions as a heatmap
    sns.heatmap(loadings_df.iloc[:20], cmap='coolwarm', annot=True, fmt='.2f')
    plt.title('Feature Loadings for First Two Principal Components')
    plt.tight_layout()
    plt.savefig('visualizations/feature_selection/pca_feature_loadings.png')
    plt.close()
    
    if return_components:
        return X_reduced, pca, scaler
    
    return X_reduced

def select_features_importance(X, y, threshold=0.01):
    """
    Select features based on feature importance from RandomForest.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features dataframe
    y : pd.Series
        Target variable
    threshold : float
        Importance threshold for selection
        
    Returns:
    --------
    list
        List of selected feature names
    """
    print("Selecting features based on importance scores")
    
    # Train a random forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a dataframe of feature importances
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Select features above the threshold
    selected_features = feature_importance[feature_importance['Importance'] > threshold]['Feature'].tolist()
    
    print(f"Selected {len(selected_features)} features based on importance threshold {threshold}")
    print("Top 10 features by importance:")
    print(feature_importance.head(10))
    
    # Visualize feature importances
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Feature Importance')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('visualizations/feature_selection', exist_ok=True)
    plt.savefig('visualizations/feature_selection/feature_importance.png')
    plt.close()
    
    return selected_features

def create_selected_features_dataset(X, y, selected_features, output_path=None):
    """
    Create a dataset with only the selected features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features dataframe
    y : pd.Series
        Target variable
    selected_features : list
        List of selected feature names
    output_path : str, optional
        Path to save the dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset with selected features and target
    """
    # Create a dataframe with selected features and target
    X_selected = X[selected_features].copy()
    
    # Add the target variable
    df_selected = X_selected.copy()
    df_selected[y.name] = y
    
    # Save the dataset if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_selected.to_csv(output_path, index=True)
        print(f"Saved selected features dataset to {output_path}")
    
    return df_selected

def optimal_feature_selection(X, y, output_path=None):
    """
    Perform optimal feature selection by combining multiple methods.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features dataframe
    y : pd.Series
        Target variable
    output_path : str, optional
        Path to save the selected features dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset with selected features and target
    """
    print("=" * 80)
    print("PERFORMING OPTIMAL FEATURE SELECTION")
    print("=" * 80)
    
    # Make sure X contains only numeric data and handle any non-numeric columns
    numeric_cols = X.select_dtypes(include=np.number).columns
    X_numeric = X[numeric_cols].copy()
    
    # Handle any infinite values
    for col in X_numeric.columns:
        # Replace infinite values with column mean or zeros
        mask_inf = np.isinf(X_numeric[col])
        if mask_inf.any():
            # Replace with mean of non-infinite values or zero if all are infinite
            non_inf_values = X_numeric.loc[~mask_inf, col]
            if len(non_inf_values) > 0:
                replace_value = non_inf_values.mean()
            else:
                replace_value = 0
            X_numeric.loc[mask_inf, col] = replace_value
    
    # Step 1: Calculate initial VIF and remove highly multicollinear features
    print("\nSTEP 1: Removing highly multicollinear features")
    vif_selected = reduce_multicollinearity_vif(X_numeric, threshold=50, max_features=30)
    X_vif = X_numeric[vif_selected]
    
    # Step 2: Select features based on importance
    print("\nSTEP 2: Selecting features based on importance")
    importance_selected = select_features_importance(X_vif, y, threshold=0.01)
    
    # Step 3: Use RFE for final feature selection
    print("\nSTEP 3: Performing Recursive Feature Elimination")
    n_features = min(20, len(importance_selected))
    final_selected = select_features_rfe(X[importance_selected], y, n_features_to_select=n_features)
    
    # Create final dataset
    print("\nCreating final selected features dataset")
    final_dataset = create_selected_features_dataset(X, y, final_selected, output_path)
    
    print("\nFinal selected features:")
    for feature in final_selected:
        print(f"- {feature}")
    
    print(f"\nReduced from {X.shape[1]} to {len(final_selected)} features")
    
    return final_dataset, final_selected

def main():
    """Main function to execute feature selection."""
    try:
        # Load the features dataset
        print("Loading features dataset...")
        features_df = pd.read_csv('data/processed/features_dataset.csv')
        
        # Handle date column if present
        if 'date' in features_df.columns:
            features_df['date'] = pd.to_datetime(features_df['date'])
            features_df.set_index('date', inplace=True)
        
        # If there's an unnamed index column, drop it
        if 'Unnamed: 0' in features_df.columns:
            features_df = features_df.drop(columns=['Unnamed: 0'])
        
        # Identify target variable
        target_col = 'retail_sales'
        log_target = f'log_{target_col}'
        
        if log_target in features_df.columns:
            print(f"Using log-transformed target: {log_target}")
            y = features_df[log_target]
            actual_target = log_target
        elif target_col in features_df.columns:
            print(f"Using target: {target_col}")
            y = features_df[target_col]
            actual_target = target_col
        else:
            raise ValueError(f"Target column '{target_col}' or '{log_target}' not found in dataset")
        
        # Create feature matrix, drop the target columns
        X = features_df.copy()
        if target_col in X.columns:
            X = X.drop(columns=[target_col])
        if log_target in X.columns:
            X = X.drop(columns=[log_target])
        
        print(f"Dataset loaded: {X.shape[1]} features, {len(y)} samples")
        
        # Perform optimal feature selection
        selected_df, selected_features = optimal_feature_selection(
            X, y, output_path='data/processed/selected_features_dataset.csv'
        )
        
        # Save the selected features list
        import json
        os.makedirs('models', exist_ok=True)
        with open('models/selected_features.json', 'w') as f:
            json.dump(selected_features, f)
        
        print("Feature selection complete!")
        
    except Exception as e:
        print(f"Error in feature selection: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
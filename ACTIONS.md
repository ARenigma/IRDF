# India Retail Demand Forecaster - Sequential Actions

This document outlines the step-by-step workflow of the India Retail Demand Forecaster project, explaining how the system processes data, builds models, and generates forecasts.

## 1. Data Collection and Preparation

### 1.1. Raw Data Collection
- Economic indicators are collected from various sources
- Files are stored in `data/raw/` directory
- Includes gold prices, crude oil prices, GDP, inflation metrics, interest rates, etc.

### 1.2. Data Quality Assessment
- `cleanup.py` scans data quality with `scan_data_quality()`
- Quality report identifies missing values, outliers, and inconsistencies
- Report is stored in `data/data_quality_report.csv`

### 1.3. Data Cleaning
- Missing values are imputed using appropriate methods:
  - Forward/backward fill for time series
  - Mean/median imputation for other variables
  - KNN imputation for complex relationships
- Outliers are detected and capped at 3 standard deviations
- Date formats are standardized across all datasets
- Cleaned data is stored in `data/processed/` directory

### 1.4. Date Consistency Verification
- `verify_date_consistency()` ensures temporal alignment across datasets
- Discrepancies in date ranges are identified and addressed
- A consistent time period is established for all indicators

## 2. Feature Engineering

### 2.1. Data Integration
- `features.py` merges cleaned datasets on date with `merge_all_signals()`
- Aligns all economic indicators to the same time periods
- Creates a unified dataset for modeling

### 2.2. Lag Feature Creation
- `generate_lag_features()` creates lagged versions of indicators
- Captures delayed effects (1, 3, 6 months)
- Example: GDP growth from 3 months ago affecting current retail sales

### 2.3. Rolling Statistics
- `generate_rolling_features()` creates moving window calculations
- Includes rolling means (trends) and standard deviations (volatility)
- Windows of 3, 6, and 12 months are typically used

### 2.4. Date Feature Extraction
- `generate_date_features()` extracts calendar-based patterns
- Creates cyclical encodings of month and quarter (sine/cosine)
- Adds indicators for special periods (festivals, budget announcements)

### 2.5. Final Dataset Preparation
- `prepare_features_and_target()` finalizes the modeling dataset
- Normalizes features using StandardScaler or MinMaxScaler
- Performs last-stage imputation if needed
- Creates a chronological train-test split (typically 80/20)
- Stores the prepared dataset in `data/processed/features_dataset.csv`

## 3. Model Training and Evaluation

### 3.1. Model Training
- `model.py` trains multiple forecasting models:
  - **Linear Models**: Linear Regression, Ridge, Lasso, ElasticNet
  - **Tree-based Models**: Random Forest, Gradient Boosting, XGBoost
  - **SVR**: Support Vector Regression with RBF kernel
  - **Time Series Models**: ARIMA/SARIMAX
  - **Deep Learning**: LSTM neural networks

### 3.2. Model Evaluation
- Each model is evaluated on the test set using:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Square Error)
  - R² (R-squared)
- Results are compared in a performance summary table
- Performance metrics are stored in `outputs/model_performance_summary.csv`

### 3.3. Best Model Selection
- `find_best_model()` identifies the best performer (typically by RMSE)
- The best model is selected for further optimization

### 3.4. Hyperparameter Optimization
- `optimize_best_model()` performs grid search with cross-validation
- Uses TimeSeriesSplit to maintain temporal order
- Identifies optimal parameters for the best model

### 3.5. Feature Importance Analysis
- `feature_importance_analysis()` identifies most predictive indicators
- Uses permutation importance or model-specific measures
- Results are visualized and saved to `visualizations/feature_importance.png`
- Detailed report is saved to `outputs/feature_importance.csv`

### 3.6. Ensemble Creation
- Top-performing models are combined into an ensemble
- Predictions are averaged to improve robustness
- Final ensemble model is saved to `models/ensemble_components.pkl`

## 4. Forecasting

### 4.1. Model Application
- The best model(s) are used to generate future forecasts
- Default forecast period is 12 months
- Specialized handling for different model types (ML vs. time series)

### 4.2. Confidence Interval Calculation
- Uncertainty is quantified with confidence intervals
- Intervals widen with forecast horizon to reflect increasing uncertainty
- For ensembles, combines uncertainty from multiple models

### 4.3. Forecast Visualization
- Historical and forecasted values are plotted together
- Confidence intervals are shown as shaded regions
- Visualization is saved to `visualizations/retail_sales_forecast.png`

### 4.4. Forecast Export
- Results are exported to `outputs/retail_sales_forecast.csv`
- Includes point forecasts and confidence intervals
- Formatted for easy integration with other systems

## 5. Insight Generation

### 5.1. Seasonal Pattern Analysis
- Monthly and quarterly patterns are identified
- Peak and trough months are determined
- Agricultural and festival season effects are quantified

### 5.2. Correlation Analysis
- Relationships between economic indicators and retail sales are measured
- Top positive and negative correlations are ranked
- Key drivers of retail demand are identified

### 5.3. Lead-Lag Relationship Analysis
- Time delays between indicator changes and retail impacts are measured
- Leading indicators with predictive power are highlighted
- Optimal forecast horizons for different indicators are determined

### 5.4. Insight Reporting
- Key findings are compiled into a comprehensive report
- Visualizations illustrate important relationships
- Results are saved to `outputs/key_insights.txt`

## 6. Scenario Analysis

### 6.1. Scenario Definition
- `forecast_scenarios.py` defines various economic scenarios:
  - **Baseline**: Continuation of current trends
  - **High Growth**: Accelerated economic growth (7-8% GDP)
  - **Stagflation**: Low growth with high inflation
  - **Recession**: Economic contraction
  - **Gold Boom**: Sharp increase in gold prices

### 6.2. Scenario Configuration
- Each scenario specifies trajectories for key indicators:
  - GDP growth
  - Inflation
  - Interest rates
  - Gold price YoY change
  - Oil price YoY change

### 6.3. Scenario Forecasting
- `ScenarioAnalysis` class generates forecasts for each scenario
- Uses the previously trained models
- Adjusts inputs according to scenario parameters

### 6.4. Scenario Comparison
- `compare_scenarios()` contrasts results across scenarios
- Side-by-side visualization shows different trajectories
- Percentage differences from baseline are calculated
- Results are stored in `outputs/scenarios/scenario_comparison.csv`

### 6.5. Scenario Visualization
- Each scenario gets dedicated visualizations:
  - Retail sales forecast
  - Economic indicator paths
  - Commodity price trajectories
- Visualizations are saved to `outputs/scenarios/<scenario_name>/`

## 7. Pipeline Orchestration

### 7.1. Pipeline Execution
- `pipeline.py` coordinates the entire workflow
- `main()` function controls the execution sequence
- Command-line arguments customize the run:
  - `--force-rebuild`: Rebuilds datasets from scratch
  - `--forecast-periods`: Sets forecast horizon
  - `--run-scenarios`: Activates scenario analysis

### 7.2. Error Handling
- Comprehensive try-except blocks manage failures
- Graceful degradation when components fail
- Informative error messages guide troubleshooting

### 7.3. Dependency Management
- Checks for required and optional libraries
- Falls back to simpler models when advanced dependencies are missing
- Example: Using sklearn models when TensorFlow is unavailable

### 7.4. Output Organization
- Results are systematically organized into directories:
  - `data/processed/`: Processed datasets
  - `models/`: Saved model files
  - `outputs/`: Forecasts and insights
  - `visualizations/`: Charts and plots
  - `outputs/scenarios/`: Scenario analysis results

### 7.5. Completion Reporting
- Summary of completed steps is provided
- Any warnings or issues are highlighted
- Execution times for major components are reported

## Running the Complete Pipeline

To execute the full sequence of actions:

```bash
python pipeline.py --force-rebuild --forecast-periods 12 --run-scenarios
```

This will perform all steps from data preparation through modeling, forecasting, insight generation, and scenario analysis, creating a comprehensive analysis of India's retail demand forecasts based on macroeconomic signals. 
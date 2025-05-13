# India Retail Demand Forecaster - Usage Guide

This guide provides practical instructions for running, modifying, and extending the India Retail Demand Forecaster system.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Macro_Model
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Note: Some advanced features require optional dependencies like TensorFlow and pmdarima. The system will gracefully fall back to simpler methods if these are not available.

## Running the System

### Basic Pipeline Execution

To run the complete pipeline from data preparation to forecasting:

```bash
python pipeline.py
```

This will:
- Load and prepare data from the `data/` directory
- Train and evaluate multiple prediction models
- Generate forecasts for future retail demand
- Create visualizations and insights in the `outputs/` directory

### Command-Line Options

The pipeline supports several command-line options:

```bash
python pipeline.py --force-rebuild --forecast-periods 24 --run-scenarios
```

- `--force-rebuild`: Forces rebuilding the dataset even if processed data exists
- `--forecast-periods`: Sets the number of months to forecast (default: 12)
- `--run-scenarios`: Runs economic scenario analysis after the main pipeline

### Running Individual Components

You can also run individual modules separately:

#### Data Cleaning
```bash
python cleanup.py
```

#### Model Training
```bash
python model.py
```

#### Scenario Analysis
```bash
python forecast_scenarios.py
```

## Working with Data

### Data Organization

The system expects data to be organized as follows:

- `data/raw/`: Raw input data files (CSV, Excel)
- `data/processed/`: Cleaned and processed data files

### Adding New Data Sources

To add a new economic indicator:

1. Place the raw data file in `data/raw/`
2. Update the `data_paths` dictionary in `pipeline.py` to include the new file
3. Run the pipeline with the `--force-rebuild` flag

### Data Format Requirements

Each data file should contain at minimum:
- A date column (usually named 'date')
- One or more numeric columns containing the economic indicators

## Customizing the System

### Creating Custom Scenarios

You can create custom economic scenarios in `forecast_scenarios.py`:

1. Open `forecast_scenarios.py`
2. Add a new scenario definition in the `run_predefined_scenarios()` method:

```python
# Example: Adding a new "Recovery" scenario
scenarios['recovery'] = self.define_scenario(
    name='Recovery',
    description='Post-recession recovery with gradual improvement',
    gdp_growth=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
    inflation=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2],
    interest_rate=[3.0, 3.0, 3.0, 3.25, 3.25, 3.5, 3.5, 3.75, 3.75, 4.0, 4.0, 4.25],
    gold_price_yoy=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0],
    oil_price_yoy=[-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 12.0, 10.0, 8.0]
)
```

### Adding New Models

To add a new prediction model:

1. Open `model.py`
2. Add your model to the `create_baseline_models()` function:

```python
models["My New Model"] = MyNewModelClass(param1=value1, param2=value2)
```

3. Ensure your model follows the scikit-learn API (with `fit()` and `predict()` methods)

### Feature Engineering

To add new features:

1. Open `features.py`
2. Add your feature generation function
3. Call it from `run_data_preparation_pipeline()` in `pipeline.py`

Example of a new feature generation function:

```python
def generate_volatility_features(df: pd.DataFrame, columns: List[str], window: int = 30) -> pd.DataFrame:
    """
    Generates features measuring the volatility of key indicators.
    """
    result_df = df.copy()
    
    for col in columns:
        if col in result_df.columns:
            # Calculate rolling standard deviation as volatility measure
            result_df[f"{col}_volatility_{window}d"] = result_df[col].rolling(window=window).std()
    
    return result_df
```

## Troubleshooting

### Common Issues

#### Missing Dependencies
If you see warnings about missing dependencies:
```
Warning: Some modeling dependencies are missing
Falling back to basic sklearn models
```

Install the optional dependencies:
```bash
pip install tensorflow statsmodels pmdarima xgboost
```

#### Data Not Found
If you see errors about missing data files:
```
Error: No dataset found for forecasting. Please run data preparation first.
```

Ensure you have placed data files in the correct location (`data/raw/`) and run with `--force-rebuild`.

#### NaN Values in Results
If you encounter NaN values in model outputs, try:
1. Inspecting the raw data for quality issues
2. Running `cleanup.py` to preprocess the data
3. Using a different imputation strategy in `features.py`

### Logs and Outputs

- Check the console output for warnings and errors
- Examine the generated files in `outputs/` for detailed results
- Review visualizations in `visualizations/` to identify data issues

## Advanced Usage

### Integrating with Other Systems

The modular design allows for integration with other systems:

- Export forecasts to CSV for use in other applications
- Call the Python modules from other scripts
- Use the saved models for real-time predictions

### Batch Processing

For batch processing of multiple scenarios:

```python
from forecast_scenarios import ScenarioAnalysis, compare_scenarios

# Initialize the analyzer
analyzer = ScenarioAnalysis(load_best_model=True)

# Define custom scenarios
scenarios = {}
scenarios['custom1'] = analyzer.define_scenario(...)
scenarios['custom2'] = analyzer.define_scenario(...)

# Run all scenarios
results = {}
for name, scenario in scenarios.items():
    results[name] = analyzer.run_scenario(scenario)

# Compare results
summary = compare_scenarios(results)
```

## Performance Optimization

If you're working with large datasets or many scenarios:

1. Use a subset of features to speed up model training
2. Reduce the number of cross-validation folds in hyperparameter optimization
3. Limit the number of models in the ensemble
4. Use parallel processing where available

## Getting Help

For more detailed information:
- Read the comprehensive documentation in `LEARNINGS.md`
- Examine the code comments in each Python module
- Check the executive summary in `EXECUTIVE_SUMMARY.md` 
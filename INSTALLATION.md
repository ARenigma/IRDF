# Installation Guide

This guide will help you set up the India Retail Demand Forecaster project environment.

## System Requirements

- Python 3.9+ 
- 4GB+ RAM recommended for model training
- Internet connection (for initial data downloads)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/ARenigma/IRDF
cd IRDF
```

### 2. Create a Virtual Environment (Recommended)

#### Using venv (Python's built-in module)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Using conda
```bash
conda create -n macro_model python=3.10
conda activate macro_model
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter issues with TensorFlow installation, refer to the [official TensorFlow installation guide](https://www.tensorflow.org/install) for your specific platform.

### 4. API Keys Setup

This project requires an API key from FRED (Federal Reserve Economic Data). 

1. Obtain a free API key from [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Create a file named `.env` in the project root with the following content:
   ```
   FRED_API_KEY=your_api_key_here
   ```

Alternatively, place your API key in the `FRED api key.txt` file.

### 5. Initial Data Setup

Run the project setup script to fetch and prepare initial data:

```bash
python project_setup.py
```

This will create the necessary directory structure and download initial datasets.

## Troubleshooting

### Common Issues

1. **TensorFlow Installation Problems**
   - For Windows: Try using `pip install tensorflow==2.15.0`
   - For Apple Silicon Macs: Install tensorflow-macos instead

2. **Missing Dependencies**
   - If you encounter "module not found" errors, install the specific package: `pip install <package_name>`

3. **Data Download Errors**
   - Ensure your internet connection is active
   - Check your FRED API key is correctly set up
   - Some data sources might be temporarily unavailable; in this case, try again later

4. **Memory Errors During Model Training**
   - Reduce batch sizes or model complexity in the configuration files
   - Close other memory-intensive applications during training

## Running the Pipeline

Once installed, you can run the complete forecasting pipeline with:

```bash
python pipeline.py
```

To run specific components:

```bash
# Data preparation only
python data_preparation.py

# Feature selection
python feature_selection.py

# Hyperparameter optimization
python hyperparameter_optimization.py

# Backtesting
python backtesting.py

# Scenario analysis
python forecast_scenarios.py
```

## Development Environment

For development work on this project, we recommend using:

- VS Code with Python and Jupyter extensions
- PyCharm with data science plugins
- Jupyter Lab for interactive exploration

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
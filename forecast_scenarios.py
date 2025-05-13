"""
India Retail Demand Forecaster - Scenario Analysis
-------------------------------------------------
This module allows testing different economic scenarios 
and their impact on retail demand forecasts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import json
warnings.filterwarnings('ignore')

# Modified import to avoid circular dependency
# from pipeline import run_data_preparation_pipeline, run_modeling_pipeline, generate_forecast
# Import pipeline separately only when needed
import pipeline
from insights import load_data_sources, merge_needed_signals

class ScenarioAnalysis:
    """
    Class for creating and analyzing different economic scenarios
    and their impact on retail demand forecasts.
    """
    
    def __init__(self, load_best_model: bool = True, model_path: str = 'models/best_model.pkl'):
        """
        Initialize the scenario analysis model.
        
        Parameters:
        -----------
        load_best_model : bool
            Whether to load the best model from disk
        model_path : str
            Path to the saved best model
        """
        self.model = None
        self.scaler = None
        self.last_data_date = None
        self.forecast_start_date = None
        self.feature_names = None
        self.log_retail_sales_available = False
        self.retail_sales_available = False
        
        # Initialize predefined scenarios dictionary
        self.predefined_scenarios = {}
        
        # Create directory for scenario outputs
        os.makedirs('outputs/scenarios', exist_ok=True)
        
        # Load models if requested
        if load_best_model and os.path.exists(model_path):
            try:
                # Before loading the model, preprocess feature datasets to understand available columns
                try:
                    features_df = pd.read_csv('data/processed/features_dataset.csv')
                    self.log_retail_sales_available = 'log_retail_sales' in features_df.columns
                    self.retail_sales_available = 'retail_sales' in features_df.columns
                    print(f"Feature dataset contains: log_retail_sales={self.log_retail_sales_available}, retail_sales={self.retail_sales_available}")
                except Exception as e:
                    print(f"Error checking feature dataset: {e}")
                
                # If we have a potential conflict, modify the model we're loading
                if self.log_retail_sales_available and not self.retail_sales_available:
                    # We have log_retail_sales but not retail_sales - need to handle this mismatch
                    print("Model will need to convert log_retail_sales to retail_sales")
                    self._handle_feature_mismatch(model_path)
                else:
                    # Standard loading
                    self.model = joblib.load(model_path)
                    print(f"Loaded best model from {model_path}")
                
                # Load scaler
                self.scaler = joblib.load('models/scaler.pkl')
                
                # Load the most recent data to determine forecast start date
                try:
                    # First try loading with parse_dates parameter
                    features_df = pd.read_csv('data/processed/features_dataset.csv', parse_dates=['date'])
                    if 'date' in features_df.columns:
                        self.last_data_date = features_df['date'].max()
                        # Ensure it's a datetime object
                        if not isinstance(self.last_data_date, pd.Timestamp):
                            self.last_data_date = pd.to_datetime(self.last_data_date)
                    else:
                        # Try to find a date in the first column
                        try:
                            first_col = features_df.columns[0]
                            features_df[first_col] = pd.to_datetime(features_df[first_col])
                            self.last_data_date = features_df[first_col].max()
                        except:
                            # Set default date if no date column found
                            print("No date column found, using current date as reference")
                            self.last_data_date = pd.Timestamp.now().normalize()
                except Exception as e:
                    print(f"Error parsing dates: {e}")
                    # Fallback to current date
                    self.last_data_date = pd.Timestamp.now().normalize()
                
                # Set forecast start date to the first day of the next month
                if self.last_data_date is not None:
                    if isinstance(self.last_data_date, pd.Timestamp):
                        # Calculate the first day of the next month
                        year = self.last_data_date.year + (self.last_data_date.month == 12)
                        month = (self.last_data_date.month % 12) + 1
                        self.forecast_start_date = pd.Timestamp(year=year, month=month, day=1)
                    else:
                        # Fallback if last_data_date is not a timestamp
                        self.forecast_start_date = pd.Timestamp.now().normalize() + pd.DateOffset(months=1)
                        self.forecast_start_date = pd.Timestamp(
                            year=self.forecast_start_date.year,
                            month=self.forecast_start_date.month,
                            day=1
                        )
                else:
                    # Default forecast start date if no last_data_date
                    now = pd.Timestamp.now()
                    self.forecast_start_date = pd.Timestamp(
                        year=now.year + (now.month == 12),
                        month=(now.month % 12) + 1,
                        day=1
                    )
                
                # Get feature names
                if os.path.exists('models/feature_names.joblib'):
                    self.feature_names = joblib.load('models/feature_names.joblib')
                else:
                    # Infer feature names from the most recent data
                    X_cols = [col for col in features_df.columns if col != 'date' and col != 'retail_sales']
                    self.feature_names = X_cols
                    joblib.dump(X_cols, 'models/feature_names.joblib')
                
                print(f"Model ready for scenario analysis. Forecast start date: {self.forecast_start_date}")
            except Exception as e:
                print(f"Error loading models: {e}")
                print("Please run the full pipeline first to generate models.")
        else:
            print("No model loaded. Will use the pipeline to generate forecasts for each scenario.")
    
    def define_scenario(self, 
                       name: str,
                       description: str,
                       forecast_periods: int = 12,
                       gdp_growth: Optional[List[float]] = None,
                       inflation: Optional[List[float]] = None, 
                       interest_rate: Optional[List[float]] = None,
                       gold_price_yoy: Optional[List[float]] = None,
                       oil_price_yoy: Optional[List[float]] = None) -> Dict:
        """
        Define a custom economic scenario.
        
        Parameters:
        -----------
        name : str
            Name of the scenario
        description : str
            Description of the scenario
        forecast_periods : int
            Number of periods to forecast
        gdp_growth : List[float], optional
            GDP growth rates for forecast periods
        inflation : List[float], optional
            Inflation rates for forecast periods
        interest_rate : List[float], optional
            Interest rates for forecast periods
        gold_price_yoy : List[float], optional
            Gold price YoY change for forecast periods
        oil_price_yoy : List[float], optional
            Oil price YoY change for forecast periods
            
        Returns:
        --------
        Dict
            Scenario definition
        """
        # Create dates for forecast periods
        if self.forecast_start_date is None:
            # Set a default forecast start date if not loaded from existing data
            self.forecast_start_date = datetime.now().replace(day=1)  # First day of current month
        
        forecast_dates = pd.date_range(
            start=self.forecast_start_date,
            periods=forecast_periods,
            freq='M'
        )
        
        # Initialize scenario data
        scenario = {
            'name': name,
            'description': description,
            'forecast_periods': forecast_periods,
            'forecast_dates': forecast_dates,
            'data': pd.DataFrame({'date': forecast_dates})
        }
        
        # Add scenario parameters if provided
        params = {
            'gdp_growth': gdp_growth,
            'inflation': inflation,
            'interest_rate': interest_rate,
            'gold_price_yoy': gold_price_yoy,
            'oil_price_yoy': oil_price_yoy
        }
        
        for param_name, param_values in params.items():
            if param_values is not None:
                # Ensure the list is the right length
                if len(param_values) < forecast_periods:
                    # Extend with the last value if list is too short
                    param_values = param_values + [param_values[-1]] * (forecast_periods - len(param_values))
                elif len(param_values) > forecast_periods:
                    # Truncate if list is too long
                    param_values = param_values[:forecast_periods]
                
                scenario['data'][param_name] = param_values
        
        return scenario
    
    def run_predefined_scenarios(self) -> Dict[str, Dict]:
        """
        Run a set of predefined economic scenarios.
        
        Returns:
        --------
        Dict[str, Dict]
            Results for each scenario
        """
        print("Running predefined scenarios...")
        
        results = {}
        
        # Define the baseline scenario - current trajectory
        baseline = self.define_scenario(
            name='baseline',
            description='Current economic trajectory with no major shocks',
            forecast_periods=12,
            gdp_growth=[5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8], # Steady growth
            inflation=[4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5], # Stable inflation
            interest_rate=[6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25], # Stable rates
            gold_price_yoy=[8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0], # Steady gold appreciation
            oil_price_yoy=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0] # Steady oil prices
        )
        
        # Store baseline in predefined_scenarios dict
        self.predefined_scenarios['baseline'] = baseline
        
        # High Growth Scenario - IMPROVED
        high_growth = self.define_scenario(
            name='high_growth',
            description='Robust economic growth with controlled inflation',
            forecast_periods=12,
            # Stronger GDP growth, especially in early periods for momentum
            gdp_growth=[7.5, 7.8, 8.2, 8.5, 8.6, 8.7, 8.5, 8.2, 8.0, 7.8, 7.6, 7.5], 
            # More moderate inflation that lags GDP growth
            inflation=[4.6, 4.7, 4.8, 5.0, 5.2, 5.3, 5.3, 5.2, 5.0, 4.8, 4.7, 4.6], 
            # More gradual interest rate increases
            interest_rate=[6.25, 6.25, 6.5, 6.5, 6.75, 6.75, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0], 
            # Stable gold prices due to economic confidence
            gold_price_yoy=[7.0, 7.0, 6.5, 6.0, 5.5, 5.0, 5.0, 5.0, 5.5, 6.0, 6.5, 7.0], 
            # More moderate oil price increases due to supply keeping up with demand
            oil_price_yoy=[5.0, 6.0, 7.0, 8.0, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5] 
        )
        
        # Store in predefined_scenarios dict
        self.predefined_scenarios['high_growth'] = high_growth
        
        # Stagflation Scenario - IMPROVED
        stagflation = self.define_scenario(
            name='stagflation',
            description='Slow growth with persistent high inflation',
            forecast_periods=12,
            # More realistic slowing growth pattern
            gdp_growth=[4.0, 3.5, 3.0, 2.5, 2.0, 1.8, 1.5, 1.3, 1.2, 1.0, 1.0, 1.0], 
            # Higher inflation that persists longer
            inflation=[6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 9.8, 10.0, 10.0, 9.8, 9.5], 
            # More aggressive tightening to combat inflation
            interest_rate=[6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5], 
            # High gold prices as investors seek inflation hedge
            gold_price_yoy=[12.0, 15.0, 18.0, 20.0, 22.0, 24.0, 22.0, 20.0, 18.0, 16.0, 14.0, 12.0], 
            # Supply constraints driving oil prices higher
            oil_price_yoy=[15.0, 18.0, 22.0, 25.0, 28.0, 30.0, 28.0, 25.0, 22.0, 20.0, 18.0, 15.0] 
        )
        
        # Store in predefined_scenarios dict
        self.predefined_scenarios['stagflation'] = stagflation
        
        # Recession Scenario - IMPROVED
        recession = self.define_scenario(
            name='recession',
            description='Economic contraction with deflationary pressure',
            forecast_periods=12,
            # More severe contraction in the middle periods
            gdp_growth=[3.0, 1.5, 0.0, -1.0, -2.5, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5], 
            # Lower inflation due to demand destruction
            inflation=[4.0, 3.5, 3.0, 2.0, 1.0, 0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5], 
            # More aggressive rate cuts to stimulate economy
            interest_rate=[6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 3.0, 3.0, 3.0, 3.5, 4.0], 
            # Flight to safety in early recession, normalizing later
            gold_price_yoy=[15.0, 18.0, 20.0, 22.0, 20.0, 18.0, 15.0, 12.0, 10.0, 8.0, 7.0, 6.0], 
            # Significant decline due to demand destruction
            oil_price_yoy=[-5.0, -10.0, -15.0, -20.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 2.0, 3.0] 
        )
        
        # Store in predefined_scenarios dict
        self.predefined_scenarios['recession'] = recession
        
        # Gold Boom Scenario - IMPROVED
        gold_boom = self.define_scenario(
            name='gold_boom',
            description='Sharp rise in gold prices with moderate economic growth',
            forecast_periods=12,
            # Steady but moderate growth
            gdp_growth=[5.5, 5.5, 5.5, 5.4, 5.3, 5.2, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], 
            # Slightly elevated inflation
            inflation=[5.0, 5.5, 6.0, 6.0, 6.0, 6.0, 5.8, 5.5, 5.3, 5.0, 5.0, 5.0], 
            # Moderate tightening to address inflation
            interest_rate=[6.25, 6.5, 6.75, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0], 
            # Very sharp rise in gold due to global uncertainty
            gold_price_yoy=[20.0, 25.0, 30.0, 35.0, 40.0, 35.0, 30.0, 25.0, 20.0, 18.0, 15.0, 12.0], 
            # Moderate oil price increases
            oil_price_yoy=[5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 6.0, 6.0, 5.0, 5.0, 4.0, 4.0] 
        )
        
        # Store in predefined_scenarios dict
        self.predefined_scenarios['gold_boom'] = gold_boom
        
        # Run all scenarios
        scenarios = [baseline, high_growth, stagflation, recession, gold_boom]
        
        for scenario in scenarios:
            scenario_name = scenario['name']
            print(f"\nRunning scenario: {scenario_name}")
            
            scenario_result = self.run_scenario(scenario)
            results[scenario_name] = scenario_result
            
            print(f"Scenario {scenario_name} complete. Average forecast: {scenario_result['forecast']['predicted_retail_sales'].mean():.2f}")
        
        return results
    
    def run_scenario(self, scenario: Dict) -> Dict:
        """
        Run a scenario and generate forecasts.
        
        Parameters:
        -----------
        scenario : Dict
            Scenario definition
            
        Returns:
        --------
        Dict
            Scenario results including forecast
        """
        print(f"Running scenario: {scenario['name']}")
        
        # Expand scenario features to match model expectations
        expanded_scenario = self._expand_scenario_features(scenario)
        
        # Generate baseline forecast
        if self.model is not None and self.scaler is not None and self.feature_names is not None:
            forecast_df = self._forecast_with_loaded_model(expanded_scenario)
        else:
            forecast_df = self._forecast_with_pipeline(expanded_scenario)
        
        # Apply scenario-specific adjustments
        adjusted_forecast_df = self._apply_scenario_adjustments(forecast_df, scenario)
        
        # Create visualizations
        self._visualize_scenario({
            'name': scenario['name'],
            'description': scenario['description'],
            'forecast': adjusted_forecast_df,
            'forecast_periods': scenario.get('forecast_periods', 12)
        })
        
        # Return results
        results = {
            'name': scenario['name'],
            'description': scenario['description'],
            'forecast': adjusted_forecast_df,
            'forecast_periods': scenario.get('forecast_periods', 12),
            'scenario_params': {k: v for k, v in scenario.items() 
                              if k not in ['name', 'description', 'forecast_periods']}
        }
        
        return results
    
    def load_best_model(self, model_path: str = 'models/best_model.pkl'):
        """
        Load the trained model for forecasting

        Parameters:
        -----------
        model_path : str
            Path to the saved model
        """
        try:
            print(f"Loading best model from {model_path}")
            
            # Use the feature mismatch handler to load the model with feature name mapping
            self._handle_feature_mismatch(model_path)
            
            # Load the scaler
            scaler_path = 'models/scaler.pkl'
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                # Try alternative paths
                scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                else:
                    print("Warning: No scaler found, using raw features")
                    self.scaler = None
            
            # Load feature dataset to check for log vs regular retail sales
            try:
                features_df = pd.read_csv('data/processed/features_dataset.csv')
                has_log_retail = 'log_retail_sales' in features_df.columns
                has_retail = 'retail_sales' in features_df.columns
                print(f"Feature dataset contains: log_retail_sales={has_log_retail}, retail_sales={has_retail}")
            except Exception as e:
                print(f"Error checking features: {e}")
            
            print("Model ready for scenario analysis.", end=" ")
            
            # Set forecast date
            self._set_forecast_date()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.scaler = None

    def _forecast_with_loaded_model(self, scenario: Dict) -> pd.DataFrame:
        """
        Generate forecasts using the loaded model.
        
        Parameters:
        -----------
        scenario : Dict
            Scenario definition
            
        Returns:
        --------
        pd.DataFrame
            Forecast results
        """
        # Create a copy of the scenario data
        scenario_df = scenario['data'].copy()
        
        try:
            # Get the features dataset to use as a reference
            features_df = pd.read_csv('data/processed/features_dataset.csv')
            
            # Create forecast dataframe with the scenario dates
            forecast_df = pd.DataFrame(index=range(len(scenario_df)))
            forecast_df['date'] = scenario_df['date']
            
            # Get the latest features for fallback values
            if 'date' in features_df.columns:
                features_df['date'] = pd.to_datetime(features_df['date'])
            last_features = features_df.iloc[-1].copy() if not features_df.empty else pd.Series()
            
            # Create the input feature dataframe
            X_forecast = pd.DataFrame(index=range(len(scenario_df)))
            
            # First add all columns from scenario_df (excluding date)
            for col in scenario_df.columns:
                if col != 'date':
                    X_forecast[col] = scenario_df[col].values
            
            # Add any missing columns from features_df
            for col in features_df.columns:
                if col != 'date' and col not in X_forecast.columns:
                    X_forecast[col] = last_features.get(col, 0.0)
            
            # Apply scaling if needed
            if self.scaler is not None:
                # Only scale the features that the model expects
                # Our wrapper model will handle any needed column renaming
                if self.feature_names is not None:
                    # Filter to only include the columns the scaler was trained on
                    scaling_cols = [col for col in self.feature_names if col in X_forecast.columns]
                    if len(scaling_cols) > 0:
                        X_scaled = self.scaler.transform(X_forecast[scaling_cols])
                        # Put the scaled values back into a DataFrame
                        X_forecast_scaled = pd.DataFrame(X_scaled, columns=scaling_cols)
                        # Add any columns that weren't scaled
                        for col in X_forecast.columns:
                            if col not in scaling_cols:
                                X_forecast_scaled[col] = X_forecast[col].values
                    else:
                        # If no columns to scale, use the original
                        X_forecast_scaled = X_forecast
                else:
                    # If no feature names, try to scale everything
                    X_forecast_scaled = pd.DataFrame(
                        self.scaler.transform(X_forecast),
                        columns=X_forecast.columns
                    )
            else:
                # No scaling needed
                X_forecast_scaled = X_forecast
            
            # Make predictions
            # Our FeatureTransformerModel wrapper will handle feature name mismatches
            predictions = self.model.predict(X_forecast_scaled)
            
            # Add predictions to the forecast dataframe
            forecast_df['predicted_retail_sales'] = predictions
            
            return forecast_df
                
        except Exception as e:
            print(f"Error in forecasting: {e}")
            # Use a simple fallback if there's an error
            forecast_df = scenario_df.copy()
            base_value = 4350  # Default value
            growth_rate = 0.005  # Monthly growth rate
            forecast_df['predicted_retail_sales'] = [base_value * (1 + growth_rate * i) for i in range(1, len(scenario_df) + 1)]
            return forecast_df

    def _apply_economic_impact(self, 
                         forecast_df: pd.DataFrame, 
                         scenario_params: Dict, 
                         baseline_params: Dict = None) -> pd.DataFrame:
        """
        Core function to apply economic scenario impacts on retail forecasts.
        Both _forecast_with_pipeline and _apply_scenario_adjustments will call this.
        
        Parameters:
        -----------
        forecast_df : pd.DataFrame
            Base forecast dataframe containing retail sales predictions
        scenario_params : Dict
            Dictionary containing scenario parameter values (GDP, inflation, etc.)
        baseline_params : Dict, optional
            Baseline parameters to compare against. If None, will use default values.
            
        Returns:
        --------
        pd.DataFrame
            Adjusted forecast with scenario effects applied
        """
        # Make a copy of the dataframe to avoid modifying the original
        forecast_df = forecast_df.copy()
        scenario_name = scenario_params.get('name', 'custom')
        
        # Skip for baseline scenario
        if scenario_name == 'baseline':
            forecast_df['scenario'] = scenario_name
            return forecast_df
            
        # Define impact weights of different economic factors
        # These weights represent how much a 1 percentage point change affects retail sales
        weights = {
            'gdp_growth': 0.8,       # GDP has strong positive correlation
            'inflation': -0.2,       # Inflation has negative impact
            'interest_rate': -0.15,  # Interest rates have negative impact
            'gold_price_yoy': 0.15,  # Gold prices have moderate positive impact in India
            'oil_price_yoy': -0.1    # Oil prices have small negative impact
        }
        
        # Define baseline values if not provided
        default_baseline = {
            'gdp_growth': 5.8,      # 5.8% baseline GDP growth
            'inflation': 4.5,       # 4.5% baseline inflation
            'interest_rate': 6.25,  # 6.25% baseline interest rate
            'gold_price_yoy': 8.0,  # 8% baseline gold price growth
            'oil_price_yoy': 2.0    # 2% baseline oil price growth
        }
        
        # Use provided baseline or default
        baseline = baseline_params if baseline_params else default_baseline
        
        # Extract scenario data 
        scenario_data = scenario_params.get('data', {})
        if not isinstance(scenario_data, dict) and hasattr(scenario_data, 'to_dict'):
            # Convert pandas DataFrame to dict if needed
            scenario_data = scenario_data.to_dict('list')
        
        # Store all parameter values by period for use in calculations
        scenario_values_by_period = {}
        for param in weights.keys():
            if param in scenario_data:
                # Extract values, handling both dict and DataFrame formats
                if isinstance(scenario_data[param], list):
                    scenario_values_by_period[param] = scenario_data[param]
                else:
                    scenario_values_by_period[param] = scenario_data[param].tolist()
        
        # Apply impact for each time period and parameter
        for i in range(len(forecast_df)):
            # Start with the baseline factor (no change)
            period_adjustment = 1.0
            
            for param, weight in weights.items():
                # Skip if parameter not in scenario
                if param not in scenario_values_by_period:
                    continue
                    
                # Get scenario and baseline values for this parameter
                if i < len(scenario_values_by_period[param]):
                    scenario_value = scenario_values_by_period[param][i]
                    baseline_value = baseline.get(param, 0)
                    
                    # Skip if values are missing
                    if scenario_value is None:
                        continue
                    
                    # Calculate difference from baseline
                    param_diff = scenario_value - baseline_value
                    
                    # Apply modified weight based on scenario conditions
                    modified_weight = weight
                    
                    # Special case: Reduce gold price impact during recession with negative GDP
                    if param == 'gold_price_yoy' and scenario_name == 'recession':
                        if 'gdp_growth' in scenario_values_by_period and i < len(scenario_values_by_period['gdp_growth']):
                            gdp_value = scenario_values_by_period['gdp_growth'][i]
                            if gdp_value < 0:
                                # Reduce gold impact when GDP is negative
                                modified_weight = weight * 0.5
                                print(f"  Reducing gold price impact for period {i} (GDP={gdp_value}): weight={modified_weight}")
                    
                    # Calculate impact factor
                    # Divide by 100 to convert percentage points to decimal impact
                    impact_factor = 1.0 + (param_diff * modified_weight / 100)
                    
                    # Accumulate the impact
                    period_adjustment *= impact_factor
                    
                    # Debug output
                    print(f"  Period {i}: {param}={scenario_value:.1f} vs baseline={baseline_value:.1f}, diff={param_diff:+.1f}")
                    print(f"    Impact: {impact_factor:.4f}, Cumulative: {period_adjustment:.4f}")
            
            # Apply the accumulated adjustment to this period's forecast
            orig_value = forecast_df.iloc[i]['predicted_retail_sales']
            new_value = orig_value * period_adjustment
            forecast_df.iloc[i, forecast_df.columns.get_loc('predicted_retail_sales')] = new_value
        
        # Add scenario name
        forecast_df['scenario'] = scenario_name
        
        return forecast_df

    def _forecast_with_pipeline(self, scenario: Dict) -> pd.DataFrame:
        """
        Generate forecasts using the pipeline when no model is loaded.
        
        Parameters:
        -----------
        scenario : Dict
            Scenario definition
            
        Returns:
        --------
        pd.DataFrame
            Forecast results
        """
        # Generate base forecast
        forecast_results = pipeline.generate_forecast(periods=scenario['forecast_periods'])
        
        # Check what type of object was returned and extract forecast_df
        if isinstance(forecast_results, pd.DataFrame):
            forecast_df = forecast_results.copy()
        elif isinstance(forecast_results, dict) and 'forecast_df' in forecast_results:
            forecast_df = forecast_results['forecast_df'].copy() 
        elif forecast_results is None:
            # Create a dummy forecast dataframe
            last_date = self.forecast_start_date if self.forecast_start_date else pd.Timestamp.now()
            forecast_dates = pd.date_range(
                start=last_date,
                periods=scenario['forecast_periods'],
                freq='M'
            )
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'predicted_retail_sales': [4350.30] * len(forecast_dates)  # Use a default value
            })
        else:
            raise ValueError(f"Unexpected result from forecast generation: {type(forecast_results)}")
        
        # Handle different column naming conventions
        # Check if the forecast values column exists, rename if needed
        if 'retail_sales_forecast' in forecast_df.columns:
            forecast_df = forecast_df.rename(columns={'retail_sales_forecast': 'predicted_retail_sales'})
        elif 'forecast' in forecast_df.columns:
            forecast_df = forecast_df.rename(columns={'forecast': 'predicted_retail_sales'})
        
        # If we still don't have the right column, create it
        if 'predicted_retail_sales' not in forecast_df.columns:
            # Try to find any column that might contain the forecast values
            numeric_cols = forecast_df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0 and numeric_cols[0] != 'date':
                # Use the first numeric column that's not a date
                forecast_df = forecast_df.rename(columns={numeric_cols[0]: 'predicted_retail_sales'})
            else:
                # If no suitable column found, create a dummy one
                forecast_df['predicted_retail_sales'] = [4350.30] * len(forecast_df)
        
        # Add scenario-specific data
        for param in ['gdp_growth', 'inflation', 'interest_rate', 'gold_price_yoy', 'oil_price_yoy']:
            if param in scenario['data']:
                forecast_df[param] = scenario['data'][param].values
        
        # Get baseline parameters for comparison
        baseline_scenario = self.predefined_scenarios.get('baseline', {})
        baseline_params = None
        
        if baseline_scenario:
            # Extract baseline parameter values
            baseline_params = {
                'gdp_growth': baseline_scenario['data']['gdp_growth'].iloc[0] if 'gdp_growth' in baseline_scenario['data'] else 5.8,
                'inflation': baseline_scenario['data']['inflation'].iloc[0] if 'inflation' in baseline_scenario['data'] else 4.5,
                'interest_rate': baseline_scenario['data']['interest_rate'].iloc[0] if 'interest_rate' in baseline_scenario['data'] else 6.25,
                'gold_price_yoy': baseline_scenario['data']['gold_price_yoy'].iloc[0] if 'gold_price_yoy' in baseline_scenario['data'] else 8.0,
                'oil_price_yoy': baseline_scenario['data']['oil_price_yoy'].iloc[0] if 'oil_price_yoy' in baseline_scenario['data'] else 2.0
            }
        
        # Use the unified economic impact function
        forecast_df = self._apply_economic_impact(forecast_df, scenario, baseline_params)
        
        return forecast_df

    def _apply_scenario_adjustments(self, forecast_df: pd.DataFrame, scenario: Dict) -> pd.DataFrame:
        """
        Apply scenario-specific adjustments to the forecast values.
        
        Parameters:
        -----------
        forecast_df : pd.DataFrame
            Base forecast with dates and sales
        scenario : Dict
            Scenario definition
            
        Returns:
        --------
        pd.DataFrame
            Adjusted forecast with scenario effects
        """
        print(f"Applying scenario adjustments for: {scenario['name']}")
        
        # Ensure correct column naming
        if 'predicted_retail_sales' not in forecast_df.columns and 'retail_sales_forecast' in forecast_df.columns:
            forecast_df['predicted_retail_sales'] = forecast_df['retail_sales_forecast']
        
        # Make sure dates are properly formatted
        if 'date' in forecast_df.columns and not pd.api.types.is_datetime64_any_dtype(forecast_df['date']):
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        # Get the baseline scenario for comparison
        baseline_scenario = self.predefined_scenarios.get('baseline', {})
        baseline_params = None
        
        if baseline_scenario:
            # Extract baseline parameter values
            baseline_params = {
                'gdp_growth': baseline_scenario['data']['gdp_growth'].iloc[0] if 'gdp_growth' in baseline_scenario['data'] else 5.8,
                'inflation': baseline_scenario['data']['inflation'].iloc[0] if 'inflation' in baseline_scenario['data'] else 4.5,
                'interest_rate': baseline_scenario['data']['interest_rate'].iloc[0] if 'interest_rate' in baseline_scenario['data'] else 6.25,
                'gold_price_yoy': baseline_scenario['data']['gold_price_yoy'].iloc[0] if 'gold_price_yoy' in baseline_scenario['data'] else 8.0,
                'oil_price_yoy': baseline_scenario['data']['oil_price_yoy'].iloc[0] if 'oil_price_yoy' in baseline_scenario['data'] else 2.0
            }
        
        # Use the unified economic impact function
        return self._apply_economic_impact(forecast_df, scenario, baseline_params)

    def _visualize_scenario(self, scenario: Dict) -> None:
        """
        Create visualizations for a scenario.
        
        Parameters:
        -----------
        scenario : Dict
            Scenario definition and results
        """
        forecast_df = scenario['forecast']
        scenario_name = scenario['name']
        
        # Create scenario directory
        scenario_dir = f"outputs/scenarios/{scenario_name.lower().replace(' ', '_')}"
        os.makedirs(scenario_dir, exist_ok=True)
        
        # 1. Retail sales forecast
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_df['date'], forecast_df['predicted_retail_sales'], 'r-', marker='o')
        plt.title(f'Retail Sales Forecast - {scenario_name} Scenario')
        plt.xlabel('Date')
        plt.ylabel('Retail Sales')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{scenario_dir}/retail_forecast.png")
        plt.close()
        
        # 2. Key indicators plot
        key_indicators = ['gdp_growth', 'inflation', 'interest_rate']
        available_indicators = [ind for ind in key_indicators if ind in forecast_df.columns]
        
        if available_indicators:
            plt.figure(figsize=(12, 6))
            for indicator in available_indicators:
                plt.plot(forecast_df['date'], forecast_df[indicator], marker='o', label=indicator)
            plt.title(f'Key Economic Indicators - {scenario_name} Scenario')
            plt.xlabel('Date')
            plt.ylabel('Value (%)')
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{scenario_dir}/economic_indicators.png")
            plt.close()
        
        # 3. Commodity prices plot
        commodity_indicators = ['gold_price_yoy', 'oil_price_yoy']
        available_commodities = [ind for ind in commodity_indicators if ind in forecast_df.columns]
        
        if available_commodities:
            plt.figure(figsize=(12, 6))
            for indicator in available_commodities:
                plt.plot(forecast_df['date'], forecast_df[indicator], marker='o', label=indicator)
            plt.title(f'Commodity Price Changes - {scenario_name} Scenario')
            plt.xlabel('Date')
            plt.ylabel('YoY Change (%)')
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{scenario_dir}/commodity_prices.png")
            plt.close()
        
        # Save forecast data
        forecast_df.to_csv(f"{scenario_dir}/forecast_data.csv", index=False)
        
        # Summary statistics
        summary = {
            'name': scenario_name,
            'description': scenario['description'],
            'periods': scenario['forecast_periods'],
            'avg_retail_sales': forecast_df['predicted_retail_sales'].mean(),
            'max_retail_sales': forecast_df['predicted_retail_sales'].max(),
            'min_retail_sales': forecast_df['predicted_retail_sales'].min(),
            'end_retail_sales': forecast_df['predicted_retail_sales'].iloc[-1]
        }
        
        # Save as text file
        with open(f"{scenario_dir}/summary.txt", 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

    def _handle_feature_mismatch(self, model_path):
        """
        Handle feature name mismatch between model feature expectations and available data.
        This method creates a wrapped model that can handle the feature name differences.
        
        Parameters:
        -----------
        model_path : str
            Path to the model to be loaded
        """
        try:
            # Load the original model
            original_model = joblib.load(model_path)
            
            # Get feature names from the original model
            if hasattr(original_model, 'feature_names_in_'):
                # For sklearn models that keep track of feature names
                self.feature_names = list(original_model.feature_names_in_)
                print(f"Using {len(self.feature_names)} feature names from model.feature_names_in_")
            elif os.path.exists('models/feature_names.joblib'):
                # Try to load from saved feature names
                self.feature_names = joblib.load('models/feature_names.joblib')
                print(f"Using {len(self.feature_names)} feature names from models/feature_names.joblib")
            else:
                # Infer from model params
                # This is a fallback and might not work for all models
                print("Warning: Could not find feature names in model, using inference")
                features_df = pd.read_csv('data/processed/features_dataset.csv')
                # Exclude obvious non-feature columns
                self.feature_names = [col for col in features_df.columns 
                                     if col not in ['date', 'log_retail_sales', 'retail_sales']]
                print(f"Inferred {len(self.feature_names)} features from dataset")
            
            # Create a wrapper model class to handle the feature name conversion
            class FeatureTransformerModel:
                def __init__(self, base_model, feature_names):
                    self.base_model = base_model
                    self.feature_names = feature_names
                    print(f"FeatureTransformerModel initialized with {len(feature_names)} feature names")
                    
                    # Create mappings for different feature naming patterns
                    # This handles both direct replacements and suffix mappings
                    self.direct_replacements = {
                        'retail_sales': 'log_retail_sales',
                        'retail_sales_lag_1': 'log_retail_sales_lag_1',
                        'retail_sales_lag_2': 'log_retail_sales_lag_2',
                        'retail_sales_lag_3': 'log_retail_sales_lag_3',
                        'retail_sales_ma_3': 'log_retail_sales_ma_3',
                        'retail_sales_ma_6': 'log_retail_sales_ma_6',
                        'retail_sales_ma_12': 'log_retail_sales_ma_12',
                        'retail_sales_mom_change': 'log_retail_sales_mom_change',
                        'retail_sales_abs_change': 'log_retail_sales_abs_change',
                        'retail_sales_exp_mean': 'log_retail_sales_exp_mean',
                        'lending_rate': 'interest_rate_lending',
                        'interest_rate': 'interest_rate_repo'
                    }
                    
                    # Create suffix mappings (when features have been renamed with suffixes)
                    self.suffix_mappings = {
                        '_gold': '',
                        '_oil': '',
                        '_lending': '',
                        '_iip': '',
                        '_cpi': '',
                        '_macro': ''
                    }
                    
                def transform_features(self, X):
                    """Transform input features to match what the model expects"""
                    if not isinstance(X, pd.DataFrame):
                        return X
                    
                    X_transformed = X.copy()
                    missing_columns = []
                    
                    # Print a comparison of available vs. needed features
                    print(f"Input dataframe has {len(X_transformed.columns)} columns.")
                    print(f"Model requires {len(self.feature_names)} specific features.")
                    
                    # Debug: show a sample of input columns and required features
                    print(f"Input columns (sample): {list(X_transformed.columns)[:5]}...")
                    print(f"Required features (sample): {self.feature_names[:5]}...")
                    
                    # First try direct replacements
                    for target, source in self.direct_replacements.items():
                        if target in self.feature_names and target not in X_transformed.columns and source in X_transformed.columns:
                            print(f"Mapping {source} to {target}")
                            X_transformed[target] = X_transformed[source]
                    
                    # Then try suffix mappings
                    for feature in self.feature_names:
                        if feature not in X_transformed.columns:
                            # Check if any of the suffix mappings can help
                            found = False
                            for suffix, replacement in self.suffix_mappings.items():
                                # Check if adding suffix helps
                                if feature + suffix in X_transformed.columns:
                                    print(f"Mapping {feature + suffix} to {feature}")
                                    X_transformed[feature] = X_transformed[feature + suffix]
                                    found = True
                                    break
                                # Check if replacing suffix helps
                                elif suffix and feature.endswith(suffix) and feature.replace(suffix, replacement) in X_transformed.columns:
                                    print(f"Mapping {feature.replace(suffix, replacement)} to {feature}")
                                    X_transformed[feature] = X_transformed[feature.replace(suffix, replacement)]
                                    found = True
                                    break
                            
                            if not found:
                                missing_columns.append(feature)
                    
                    # Check for _retail_sales mappings for features like lag variables
                    if 'log_retail_sales' in X_transformed.columns and 'retail_sales' not in X_transformed.columns:
                        # For each column that starts with retail_sales_ and is needed
                        for feature in self.feature_names:
                            if feature.startswith('retail_sales_') and feature not in X_transformed.columns:
                                # Try to find a corresponding log_retail_sales_ feature
                                log_feature = 'log_' + feature
                                if log_feature in X_transformed.columns:
                                    print(f"Mapping {log_feature} to {feature}")
                                    X_transformed[feature] = X_transformed[log_feature]
                                    if feature in missing_columns:
                                        missing_columns.remove(feature)
                    
                    # Fill any remaining missing columns with zeros
                    for col in missing_columns:
                        print(f"Warning: Could not find mapping for {col}, using zero")
                        X_transformed[col] = 0.0
                    
                    # Confirm we have all needed features now
                    missing_after = [f for f in self.feature_names if f not in X_transformed.columns]
                    if missing_after:
                        print(f"Still missing features after transformation: {missing_after}")
                    else:
                        print("All required features are now available")
                    
                    return X_transformed
                    
                def predict(self, X):
                    """
                    Transform the input data and make predictions
                    """
                    # Transform features to match what the model expects
                    X_transformed = self.transform_features(X)
                    
                    # Extract just the columns the model needs
                    if isinstance(X_transformed, pd.DataFrame):
                        # Make sure we have all needed columns
                        for feat in self.feature_names:
                            if feat not in X_transformed.columns:
                                X_transformed[feat] = 0.0
                        
                        # Select only the needed columns in the right order
                        print(f"Selecting {len(self.feature_names)} features for model prediction")
                        X_for_model = X_transformed[self.feature_names]
                    else:
                        X_for_model = X_transformed
                    
                    # Make predictions with the base model
                    return self.base_model.predict(X_for_model)
                
            # Create the wrapped model and store it
            self.model = FeatureTransformerModel(original_model, self.feature_names)
            print("Created feature transformer model to handle feature name mismatches")
                
        except Exception as e:
            print(f"Error handling feature mismatch: {e}")
            # Fallback to standard loading
            self.model = joblib.load(model_path)

    def _expand_scenario_features(self, scenario):
        """
        Expand the scenario data to include all features required by the model.
        
        Parameters:
        -----------
        scenario : Dict
            Scenario definition
            
        Returns:
        --------
        Dict
            Updated scenario with expanded features
        """
        # Get the original scenario data
        original_data = scenario['data'].copy()
        
        try:
            # Load the training dataset to get all feature names
            features_df = pd.read_csv('data/processed/features_dataset.csv')
            
            # Get the last row of features as defaults
            if 'date' in features_df.columns:
                features_df['date'] = pd.to_datetime(features_df['date'])
            
            last_features = features_df.iloc[-1].copy() if not features_df.empty else pd.Series()
            
            # Create expanded dataframe with all required columns
            expanded_df = pd.DataFrame(index=range(len(original_data)))
            
            # First copy all scenario-defined columns
            for col in original_data.columns:
                expanded_df[col] = original_data[col].values
            
            # Copy date column and ensure it's datetime
            if 'date' in expanded_df.columns:
                expanded_df['date'] = pd.to_datetime(expanded_df['date'])
                
                # Add seasonal features that depend on the date
                expanded_df['month'] = expanded_df['date'].dt.month
                expanded_df['month_sin'] = np.sin(2 * np.pi * expanded_df['month']/12)
                expanded_df['month_cos'] = np.cos(2 * np.pi * expanded_df['month']/12)
                expanded_df['quarter'] = expanded_df['date'].dt.quarter
                expanded_df['year'] = expanded_df['date'].dt.year
            
            # Create mappings for expanded interest rate calculations
            if 'interest_rate' in expanded_df.columns:
                # Map scenario's interest_rate to more specific rates
                expanded_df['interest_rate_repo'] = expanded_df['interest_rate']
                expanded_df['interest_rate_lending'] = expanded_df['interest_rate'] + 2.0  # Lending rate is typically higher
                
                # Create averaged versions
                expanded_df['lending_rate_lending'] = expanded_df['interest_rate_lending']
                expanded_df['lending_rate_3m_avg_lending'] = expanded_df['interest_rate_lending']
                expanded_df['lending_rate_6m_avg_lending'] = expanded_df['interest_rate_lending']
                
                # Add mom changes
                expanded_df['lending_rate_mom_lending'] = 0.0  # Default to no change
            
            # Handle oil price derivatives if oil_price_yoy exists
            if 'oil_price_yoy' in expanded_df.columns:
                # Set base oil price
                last_oil_price = last_features.get('oil_price', 80.0)
                oil_prices = []
                
                # Calculate oil prices from yoy changes
                for i, yoy in enumerate(expanded_df['oil_price_yoy']):
                    if i == 0:
                        # First month uses last known price as reference
                        new_price = last_oil_price * (1 + yoy/100)
                    else:
                        # Subsequent months use 12 months ago as reference
                        if i >= 12:
                            new_price = oil_prices[i-12] * (1 + yoy/100)
                        else:
                            # If we don't have 12 months of history yet
                            new_price = last_oil_price * (1 + yoy/100)
                    oil_prices.append(new_price)
                
                expanded_df['oil_price'] = oil_prices
                expanded_df['oil_price_oil'] = expanded_df['oil_price']
                
                # Calculate averages
                expanded_df['oil_price_30d_avg_oil'] = expanded_df['oil_price']
                expanded_df['oil_price_90d_avg_oil'] = expanded_df['oil_price']
                expanded_df['oil_price_180d_avg_oil'] = expanded_df['oil_price']
                
                # Calculate momentum
                expanded_df['oil_momentum_oil'] = 0.0
                expanded_df['oil_volatility_1m_oil'] = 2.0  # Default volatility
                expanded_df['oil_volatility_3m_oil'] = 4.0  # Default volatility
            
            # Handle gold price derivatives if gold_price_yoy exists
            if 'gold_price_yoy' in expanded_df.columns:
                # Set base gold price
                last_gold_price = last_features.get('gold_price', 1800.0)
                gold_prices = []
                
                # Calculate gold prices from yoy changes
                for i, yoy in enumerate(expanded_df['gold_price_yoy']):
                    if i == 0:
                        # First month uses last known price as reference
                        new_price = last_gold_price * (1 + yoy/100)
                    else:
                        # Subsequent months use 12 months ago as reference
                        if i >= 12:
                            new_price = gold_prices[i-12] * (1 + yoy/100)
                        else:
                            # If we don't have 12 months of history yet
                            new_price = last_gold_price * (1 + yoy/100)
                    gold_prices.append(new_price)
                
                expanded_df['gold_price'] = gold_prices
                expanded_df['gold_price_gold'] = expanded_df['gold_price']
                
                # Calculate averages
                expanded_df['gold_price_30d_avg_gold'] = expanded_df['gold_price']
                expanded_df['gold_price_90d_avg_gold'] = expanded_df['gold_price']
                expanded_df['gold_price_180d_avg_gold'] = expanded_df['gold_price']
                
                # Calculate momentum and volatility
                expanded_df['gold_momentum_gold'] = 0.0
                expanded_df['gold_volatility_1m_gold'] = 1.5  # Default volatility
                expanded_df['gold_volatility_3m_gold'] = 3.0  # Default volatility
                expanded_df['gold_trend_gold'] = 1.0  # Default uptrend
            
            # Add all required columns from the training dataset if not already in the expanded df
            for col in features_df.columns:
                if col != 'date' and col not in expanded_df.columns:
                    if col in last_features:
                        expanded_df[col] = last_features[col]
                    else:
                        expanded_df[col] = 0.0  # Default value
            
            # Update the scenario data with expanded features
            scenario['data'] = expanded_df
            
            return scenario
            
        except Exception as e:
            print(f"Error expanding scenario features: {e}")
            return scenario

    def _set_forecast_date(self):
        """
        Sets the forecast start date based on the last date in the features dataset
        """
        try:
            # Try to load the features dataset
            features_df = pd.read_csv('data/processed/features_dataset.csv')
            
            if 'date' in features_df.columns:
                features_df['date'] = pd.to_datetime(features_df['date'])
                self.last_data_date = features_df['date'].max()
                
                # Set forecast start date to the first day of the next month after last data
                if isinstance(self.last_data_date, pd.Timestamp):
                    year = self.last_data_date.year + (self.last_data_date.month == 12)
                    month = (self.last_data_date.month % 12) + 1
                    self.forecast_start_date = pd.Timestamp(year=year, month=month, day=1)
                else:
                    # Fallback
                    self.forecast_start_date = pd.Timestamp.now().replace(day=1) + pd.DateOffset(months=1)
            else:
                # Default if no date column
                self.forecast_start_date = pd.Timestamp.now().replace(day=1) + pd.DateOffset(months=1)
            
            print(f"Forecast start date: {self.forecast_start_date}")
            
        except Exception as e:
            print(f"Error parsing dates: {e}")
            # Default to first day of next month from now
            self.forecast_start_date = pd.Timestamp.now().replace(day=1) + pd.DateOffset(months=1)

def compare_scenarios(scenarios_results: Dict[str, Dict], 
                     output_dir: str = 'outputs/scenarios'):
    """
    Compare multiple scenarios side by side.
    
    Parameters:
    -----------
    scenarios_results : Dict[str, Dict]
        Dictionary of scenario results from run_predefined_scenarios
    output_dir : str
        Directory to save comparison outputs
    """
    print("\nComparing scenarios...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect retail sales forecasts from all scenarios
    all_forecasts = pd.DataFrame()
    
    for scenario_name, scenario in scenarios_results.items():
        temp_df = scenario['forecast'][['date', 'predicted_retail_sales']].copy()
        temp_df.rename(columns={'predicted_retail_sales': scenario_name}, inplace=True)
        
        if all_forecasts.empty:
            all_forecasts = temp_df.copy()
        else:
            all_forecasts = pd.merge(all_forecasts, temp_df, on='date', how='outer')
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    
    for scenario_name in scenarios_results.keys():
        plt.plot(all_forecasts['date'], all_forecasts[scenario_name], marker='o', label=scenario_name)
    
    plt.title('Retail Sales Forecast - Scenario Comparison')
    plt.xlabel('Date')
    plt.ylabel('Retail Sales')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scenario_comparison.png")
    plt.close()
    
    # Save comparison data
    all_forecasts.to_csv(f"{output_dir}/scenario_comparison.csv", index=False)
    
    # Create a summary table
    summary_table = pd.DataFrame(index=scenarios_results.keys())
    
    summary_table['Avg Retail Sales'] = [scenario['forecast']['predicted_retail_sales'].mean() 
                                       for scenario in scenarios_results.values()]
    summary_table['Max Retail Sales'] = [scenario['forecast']['predicted_retail_sales'].max() 
                                      for scenario in scenarios_results.values()]
    summary_table['Min Retail Sales'] = [scenario['forecast']['predicted_retail_sales'].min() 
                                      for scenario in scenarios_results.values()]
    summary_table['End Retail Sales'] = [scenario['forecast']['predicted_retail_sales'].iloc[-1] 
                                      for scenario in scenarios_results.values()]
    
    # Calculate percent difference from baseline
    if 'baseline' in summary_table.index:
        baseline_end = summary_table.loc['baseline', 'End Retail Sales']
        summary_table['% Diff from Baseline'] = (
            (summary_table['End Retail Sales'] - baseline_end) / baseline_end * 100
        ).round(2)
    
    # Save summary table
    summary_table.to_csv(f"{output_dir}/scenario_summary.csv")
    
    # Print summary
    print("\nScenario Comparison Summary:")
    print(summary_table)
    
    return summary_table

def main():
    """Main function to run scenario analysis."""
    print("Running scenario analysis...")
    
    # Initialize scenario analysis
    scenario_analyzer = ScenarioAnalysis(load_best_model=True)
    
    # Run predefined scenarios
    scenarios_results = scenario_analyzer.run_predefined_scenarios()
    
    # Compare scenarios
    summary = compare_scenarios(scenarios_results)
    
    print("\nScenario analysis complete. Results saved to outputs/scenarios/")
    
    return scenarios_results, summary

if __name__ == "__main__":
    main() 
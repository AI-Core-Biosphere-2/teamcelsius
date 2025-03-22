# simulation.py
import pandas as pd
from statsmodels.tsa.api import VAR

def train_var_model(data, maxlags=15):
    """
    Trains a VAR model on the given DataFrame.
    Assumes data is indexed by DateTime.
    Returns the fitted VAR model.
    """
    # Ensure the index is DateTime
    data = data.set_index("DateTime")
    
    # Optional: Difference data if non-stationary (skipped here for brevity)
    model = VAR(data)
    # Select lag order based on AIC; fall back to lag=1 if selection fails.
    lag_order_results = model.select_order(maxlags)
    lag = lag_order_results.aic if lag_order_results.aic is not None else 1
    results = model.fit(lag)
    return results

def simulate_scenario(data, var_results, adjustments, steps=24):
    """
    Simulates a scenario given:
      - data: original DataFrame (with DateTime column)
      - var_results: fitted VAR model
      - adjustments: dict of variable adjustments (e.g., {"RainForest_MountainTower_Temp": 2.0})
      - steps: forecast horizon (default 24 hours)
    
    Returns a forecast DataFrame.
    """
    # Use the last 'k_ar' observations for forecasting.
    data_indexed = data.set_index("DateTime")
    last_obs = data_indexed.iloc[-var_results.k_ar:]
    modified_obs = last_obs.copy()
    
    # Apply adjustments to the last observation row.
    for var, change in adjustments.items():
        if var in modified_obs.columns:
            modified_obs.iloc[-1, modified_obs.columns.get_loc(var)] += change
    
    # Forecast future states.
    forecast = var_results.forecast(modified_obs.values, steps=steps)
    forecast_index = pd.date_range(start=data["DateTime"].iloc[-1], periods=steps+1, freq='H')[1:]
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=last_obs.columns)
    return forecast_df

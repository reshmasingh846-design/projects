import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timesfm
torch.set_float32_matmul_precision("high")

# --- 1. Generate Synthetic Bike Rental Data ---
def create_bike_rental_data():
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
    n_days = len(dates)

    trend = np.linspace(start=150, stop=350, num=n_days)
    yearly_seasonality = 180 * (1 + np.sin(2 * np.pi * dates.dayofyear / 365.25 - np.pi/2))
    weekly_seasonality = 120 * (dates.dayofweek >= 5)
    noise = np.random.normal(0, 40, n_days)
    rentals = trend + yearly_seasonality + weekly_seasonality + noise
    rentals = np.maximum(0, rentals).astype(int)  # Ensure no negative rentals
    temp = 15 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365.25 - np.pi/2) + np.random.randn(n_days) * 2
    is_weekend = (dates.dayofweek >= 5).astype(int)
    df = pd.DataFrame({
        'rentals': rentals,
        'temperature': temp,
        'is_weekend': is_weekend
    }, index=dates)

    return df

# --- 2. Prepare Data ---
rental_df = create_bike_rental_data()
time_series_data = rental_df['rentals'].values
horizon_len = 90  # Forecast the next 90 days
historical_data = time_series_data[:-horizon_len]
true_future_values = time_series_data[-horizon_len:]

# --- 3. Initialize TimesFM 2.5 PyTorch Model ---
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

# --- 4. Generate Forecast ---
point_forecast, quantile_forecast = model.forecast(
    horizon=horizon_len,
    inputs=[historical_data],  # Single time series as input
)
forecast_values = point_forecast[0]  # Extract the forecast for our single series

# --- 5. Visualize the Results ---
dates = rental_df.index
plt.figure(figsize=(15, 7))
plt.plot(dates[:-horizon_len], historical_data, label="Historical Data", color="black")
plt.plot(dates[-horizon_len:], true_future_values, label="True Future Values", color="blue", linestyle='--')
plt.plot(dates[-horizon_len:], forecast_values, label="TimesFM 2.5 Forecast", color="red")
plt.fill_between(dates[-horizon_len:],quantile_forecast[0, :, 1], quantile_forecast[0, :, 9], alpha=0.2, color='red', label='80% Prediction Interval')
plt.legend()
plt.title("TimesFM 2.5 Zero-Shot Forecast for Daily Bike Rentals")
plt.xlabel("Date")
plt.ylabel("Number of Rentals")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- 6. Calculate Metrics ---
rmse = np.sqrt(np.mean((true_future_values - forecast_values)**2))
mae = np.mean(np.abs(true_future_values - forecast_values))

print(f"Zero-Shot Forecast RMSE: {rmse:.2f}")
print(f"Zero-Shot Forecast MAE: {mae:.2f}")
print(f"\nPoint forecast shape: {point_forecast.shape}")
print(f"Quantile forecast shape: {quantile_forecast.shape}")
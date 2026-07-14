# Hybrid TimesFM + LSTM Residual Forecast

This delivers a 1-hour-ahead hybrid model for `chaps_demo.csv`.

## What It Does

- Cleans `-` numeric values to `0`.
- Parses `Datetime` as day-first timestamps.
- Aggregates duplicate timestamps to one hourly row.
- Uses `transaction_count` as the default target because `dr_amount` and `cr_amount` are all zero in the provided file.
- Runs TimesFM as the zero-shot base forecast.
- Trains a small PyTorch LSTM on sampled TimesFM residuals.
- Produces:
  - `prepared_hourly_data.csv`
  - `training_residuals.csv`
  - `validation_forecasts.csv`
  - `next_1h_forecast.csv`
  - `summary.json`

## Install

Run this in the Python environment where you use TimesFM:

```powershell
pip install "timesfm[torch]" pandas numpy
```

## Run

```powershell
python .\hybrid_timesfm_lstm_residual.py `
  --input "C:\Users\reshm\OneDrive\Documents\work\timesfm\timesfm-forecasting\chaps_demo.csv" `
  --target-col transaction_count `
  --validation-hours 168 `
  --residual-samples 256 `
  --output-dir "C:\Users\reshm\OneDrive\Documents\work\timesfm\timesfm-forecasting\hybrid_chaps_output"
```

## Faster First Test

Use fewer residual samples to confirm the full stack loads:

```powershell
python .\hybrid_timesfm_lstm_residual.py `
  --residual-samples 24 `
  --epochs 20 `
  --output-dir ".\hybrid_quick_test"
```

## Model Formula

```text
TimesFM base:         y_hat_t+1 = TimesFM(history)
Residual label:       e_t+1 = actual_t+1 - y_hat_t+1
LSTM correction:      e_hat_t+1 = LSTM(recent target history, calendar features, base forecast)
Hybrid forecast:      final_t+1 = max(0, y_hat_t+1 + e_hat_t+1)
```


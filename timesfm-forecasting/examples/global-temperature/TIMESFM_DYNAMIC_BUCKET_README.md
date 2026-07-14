# TimesFM dynamic bucket forecasting

This workflow uses `Input_data.csv` with:

- timestamp column: `TransactionDate`
- target column: `AccountBalance`

The script is:

```text
C:\Users\reshm\Documents\Codex\2026-05-17\files-mentioned-by-the-user-input\timesfm_dynamic_bucket_forecast.py
```

## Install/runtime

Run it from a Python environment that has TimesFM installed:

```bash
pip install "timesfm[torch]" pandas numpy
```

## Forecast 5 minutes ahead

This uses 5-minute buckets and forecasts 1 bucket ahead.

```bash
python timesfm_dynamic_bucket_forecast.py --bucket 5min --forecast-length 5min
```

## Forecast 1 hour ahead with 5-minute detail

This forecasts 12 future 5-minute buckets.

```bash
python timesfm_dynamic_bucket_forecast.py --bucket 5min --forecast-length 1h
```

## Forecast 1 hour ahead with hourly buckets

This aggregates the data hourly and forecasts 1 future hourly bucket.

```bash
python timesfm_dynamic_bucket_forecast.py --bucket 1h --forecast-length 1h
```

## What the code is doing

1. Loads the CSV and parses `TransactionDate` as day-first timestamps.
2. Buckets irregular rows into a regular time series using `--bucket`.
3. Aggregates values inside each bucket using `--agg mean` by default.
4. Fills missing buckets using time interpolation by default.
5. Learns an additive seasonality profile from train history only:
   - day of week + hour + minute
   - fallback to hour + minute
   - fallback to hour
6. Subtracts that seasonal pattern from the context.
7. Sends the deseasonalized residual series into TimesFM.
8. Adds the future seasonal pattern back to the TimesFM forecast.
9. Evaluates chronological train/validation/test splits:
   - `70:15:15`
   - `80:10:10`
   - `60:20:20`
10. Runs rolling walk-forward validation from the later part of the series.
11. Saves metrics and forecast CSV files.

## Outputs

By default outputs are written next to `Input_data.csv`:

```text
C:\Users\reshm\OneDrive\Documents\work\timesfm\timesfm-forecasting\examples\global-temperature\output_dynamic_bucket
```

Main files:

- `prepared_<bucket>.csv`: regularized time series used by TimesFM.
- `split_metrics.csv`: validation/test metrics for each split.
- `split_forecasts.csv`: actual vs forecast rows for split evaluation.
- `walk_forward_metrics.csv`: metrics for rolling-origin validation.
- `walk_forward_forecasts.csv`: actual vs forecast rows for each walk.
- `future_forecast.csv`: forecast beyond the last timestamp.
- `summary.json`: machine-readable run summary.

## Useful options

```bash
python timesfm_dynamic_bucket_forecast.py ^
  --bucket 5min ^
  --forecast-length 1h ^
  --seasonality additive ^
  --splits 70:15:15,80:10:10,60:20:20 ^
  --max-walks 12
```

Use `--seasonality none` to compare against plain TimesFM.

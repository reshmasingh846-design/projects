#!/usr/bin/env python3
"""Train/test style 1-hour forecasting for Input_data.csv with TimesFM.

TimesFM is a pretrained forecasting model, so this script does not fit model
weights on your data. Instead, it creates a train/test split, forecasts the
held-out 1-hour test window from the training context, reports test metrics,
then forecasts the next 1 hour using the full dataset.

Input columns expected by default:
  - TransactionDate
  - AccountBalance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = (
    r"C:\Users\reshm\OneDrive\Documents\work\timesfm\timesfm-forecasting"
    r"\examples\global-temperature\Input_data.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast AccountBalance 1 hour ahead with TimesFM."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--date-col",
        default="TransactionDate",
        help="Timestamp column name.",
    )
    parser.add_argument(
        "--target-col",
        default="AccountBalance",
        help="Numeric column to forecast.",
    )
    parser.add_argument(
        "--freq",
        default="1min",
        help="Regular time frequency for modeling. Use 1min for 1-minute data.",
    )
    parser.add_argument(
        "--horizon-minutes",
        type=int,
        default=60,
        help="Forecast horizon in minutes. Default is 60 for 1 hour ahead.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for outputs. Default: <input folder>/output_1h.",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=1024,
        help="Maximum context points to use for TimesFM.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="TimesFM per-core batch size.",
    )
    parser.add_argument(
        "--model-id",
        default="google/timesfm-2.5-200m-pytorch",
        help="Hugging Face TimesFM model id.",
    )
    return parser.parse_args()


def load_and_prepare(
    input_path: Path,
    date_col: str,
    target_col: str,
    freq: str,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    missing = [col for col in [date_col, target_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found columns: {list(df.columns)}")

    df = df[[date_col, target_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[date_col, target_col])
    df = df.sort_values(date_col)

    # Multiple rows can land in the same minute; average them before resampling.
    regular = (
        df.set_index(date_col)[target_col]
        .groupby(level=0)
        .mean()
        .resample(freq)
        .mean()
        .interpolate(method="time")
        .ffill()
        .bfill()
        .to_frame(name=target_col)
    )
    regular.index.name = date_col
    regular = regular.reset_index()

    if len(regular) <= 60:
        raise ValueError(
            f"Need more than 60 regularized rows for a 1-hour test split; got {len(regular)}."
        )

    return regular


def load_timesfm(model_id: str, context_len: int, horizon_len: int, batch_size: int):
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_len,
            max_horizon=horizon_len,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            per_core_batch_size=batch_size,
        )
    )
    return model


def run_forecast(model, values: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    point, quantiles = model.forecast(
        horizon=horizon,
        inputs=[values.astype(np.float32)],
    )
    return point[0], quantiles[0]


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = np.where(np.abs(actual) < 1e-8, np.nan, np.abs(actual))
    mape = float(np.nanmean(np.abs(error) / denom) * 100)
    return {"mae": mae, "rmse": rmse, "mape_percent": mape}


def make_result_frame(
    timestamps: pd.Series,
    point: np.ndarray,
    quantiles: np.ndarray,
    actual: np.ndarray | None = None,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "timestamp": timestamps,
            "forecast": point,
            "q10": quantiles[:, 1],
            "q20": quantiles[:, 2],
            "q50": quantiles[:, 5],
            "q80": quantiles[:, 8],
            "q90": quantiles[:, 9],
        }
    )
    if actual is not None:
        result.insert(1, "actual", actual)
        result["error"] = result["actual"] - result["forecast"]
    return result


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "output_1h"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and regularizing data...")
    data = load_and_prepare(input_path, args.date_col, args.target_col, args.freq)
    horizon = args.horizon_minutes

    train = data.iloc[:-horizon].copy()
    test = data.iloc[-horizon:].copy()

    print(f"Regular rows: {len(data)}")
    print(f"Train rows:   {len(train)}")
    print(f"Test rows:    {len(test)} ({horizon} minutes)")
    print(f"Date range:   {data[args.date_col].min()} to {data[args.date_col].max()}")

    print("Loading TimesFM...")
    model = load_timesfm(
        model_id=args.model_id,
        context_len=args.context_len,
        horizon_len=horizon,
        batch_size=args.batch_size,
    )

    print("Forecasting held-out test hour...")
    test_point, test_quantiles = run_forecast(
        model,
        train[args.target_col].to_numpy(),
        horizon,
    )
    actual = test[args.target_col].to_numpy()
    test_metrics = metrics(actual, test_point)

    test_output = make_result_frame(
        timestamps=test[args.date_col].reset_index(drop=True),
        point=test_point,
        quantiles=test_quantiles,
        actual=actual,
    )
    test_output.to_csv(output_dir / "timesfm_test_1h_forecast.csv", index=False)

    print("Forecasting next 1 hour from all data...")
    future_point, future_quantiles = run_forecast(
        model,
        data[args.target_col].to_numpy(),
        horizon,
    )
    last_timestamp = data[args.date_col].iloc[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.tseries.frequencies.to_offset(args.freq),
        periods=horizon,
        freq=args.freq,
    )
    future_output = make_result_frame(
        timestamps=pd.Series(future_timestamps),
        point=future_point,
        quantiles=future_quantiles,
    )
    future_output.to_csv(output_dir / "timesfm_future_1h_forecast.csv", index=False)

    summary = {
        "input": str(input_path),
        "target_column": args.target_col,
        "frequency": args.freq,
        "horizon_minutes": horizon,
        "regular_rows": int(len(data)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_metrics": test_metrics,
        "outputs": {
            "test_forecast": str(output_dir / "timesfm_test_1h_forecast.csv"),
            "future_forecast": str(output_dir / "timesfm_future_1h_forecast.csv"),
        },
    }
    with open(output_dir / "timesfm_1h_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTest metrics")
    print(f"MAE:  {test_metrics['mae']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAPE: {test_metrics['mape_percent']:.2f}%")
    print("\nSaved outputs:")
    print(output_dir / "timesfm_test_1h_forecast.csv")
    print(output_dir / "timesfm_future_1h_forecast.csv")
    print(output_dir / "timesfm_1h_summary.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dynamic-bucket TimesFM forecasting for Input_data.csv.

What this script does:
1. Loads the irregular transaction time series.
2. Buckets it to a regular cadence such as 5 minutes or 1 hour.
3. Fills missing buckets so TimesFM receives an evenly spaced series.
4. Optionally removes recurring seasonality before forecasting.
5. Runs chronological train/validation/test splits.
6. Runs rolling walk-forward validation.
7. Forecasts the next requested period from all available data.

TimesFM is a pretrained zero-shot model. "Train" here means "history used as
context"; this script does not fine-tune TimesFM model weights.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = (
    r"C:\Users\reshm\OneDrive\Documents\work\timesfm\timesfm-forecasting"
    r"\examples\global-temperature\Input_data.csv"
)


@dataclass(frozen=True)
class SeasonalityProfile:
    """Additive calendar pattern learned from the training context."""

    enabled: bool
    global_mean: float
    by_week_slot: dict[str, float]
    by_time_slot: dict[str, float]
    by_hour: dict[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TimesFM forecast with dynamic 5-minute/1-hour bucketing, seasonality, splits, and walk-forward validation."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to Input_data.csv.")
    parser.add_argument("--date-col", default="TransactionDate", help="Timestamp column.")
    parser.add_argument("--target-col", default="AccountBalance", help="Value column to forecast.")
    parser.add_argument(
        "--bucket",
        default="5min",
        help="Modeling bucket size. Examples: 5min, 15min, 1h. Use 5min for 5-minute forecasts or 1h for hourly forecasts.",
    )
    parser.add_argument(
        "--forecast-length",
        default="1h",
        help="How far ahead to forecast. Examples: 5min, 1h, 2h, 1d.",
    )
    parser.add_argument(
        "--agg",
        choices=["mean", "sum", "last"],
        default="mean",
        help="How to aggregate multiple observations inside a bucket.",
    )
    parser.add_argument(
        "--fill",
        choices=["interpolate", "ffill", "zero"],
        default="interpolate",
        help="How to fill buckets with no source row after resampling.",
    )
    parser.add_argument(
        "--seasonality",
        choices=["additive", "none"],
        default="additive",
        help="Additive removes a learned calendar pattern, forecasts residuals, then adds the pattern back.",
    )
    parser.add_argument(
        "--splits",
        default="70:15:15,80:10:10,60:20:20",
        help="Chronological train:validation:test percentage splits to evaluate.",
    )
    parser.add_argument(
        "--walk-start-ratio",
        type=float,
        default=0.70,
        help="Walk-forward starts after this fraction of the series.",
    )
    parser.add_argument(
        "--walk-step",
        default=None,
        help="Distance between rolling origins. Default equals forecast-length.",
    )
    parser.add_argument(
        "--max-walks",
        type=int,
        default=12,
        help="Maximum rolling forecast windows to run, to keep runtime practical.",
    )
    parser.add_argument("--context-len", type=int, default=1024, help="Maximum TimesFM context points.")
    parser.add_argument("--batch-size", type=int, default=16, help="TimesFM batch size.")
    parser.add_argument(
        "--model-id",
        default="google/timesfm-2.5-200m-pytorch",
        help="Hugging Face TimesFM model id.",
    )
    parser.add_argument(
        "--fallback-model-id",
        default="google/timesfm-1.0-200m-pytorch",
        help="Older TimesFM checkpoint to try if the 2.5 API is unavailable.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <input folder>/output_dynamic_bucket.",
    )
    return parser.parse_args()


def fixed_timedelta(freq: str) -> pd.Timedelta:
    """Convert fixed pandas frequencies such as 5min or 1h to Timedelta."""

    offset = pd.tseries.frequencies.to_offset(freq)
    try:
        return pd.Timedelta(offset.nanos, unit="ns")
    except ValueError as exc:
        raise ValueError(
            f"Frequency {freq!r} is not fixed-width. Use values like 5min, 15min, or 1h."
        ) from exc


def horizon_steps(bucket: str, forecast_length: str) -> int:
    bucket_delta = fixed_timedelta(bucket)
    forecast_delta = pd.Timedelta(forecast_length)
    if forecast_delta <= pd.Timedelta(0):
        raise ValueError("--forecast-length must be positive.")
    return int(math.ceil(forecast_delta / bucket_delta))


def load_and_bucket(args: argparse.Namespace) -> pd.DataFrame:
    """Read CSV, parse timestamps, aggregate to the requested bucket, and fill gaps."""

    raw = pd.read_csv(args.input)
    missing = [c for c in [args.date_col, args.target_col] if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Available columns: {list(raw.columns)}")

    df = raw[[args.date_col, args.target_col]].copy()
    df[args.date_col] = pd.to_datetime(df[args.date_col], dayfirst=True, errors="coerce")
    df[args.target_col] = pd.to_numeric(df[args.target_col], errors="coerce")
    df = df.dropna(subset=[args.date_col, args.target_col]).sort_values(args.date_col)

    series = df.set_index(args.date_col)[args.target_col]
    if args.agg == "mean":
        regular = series.resample(args.bucket).mean()
    elif args.agg == "sum":
        regular = series.resample(args.bucket).sum(min_count=1)
    else:
        regular = series.resample(args.bucket).last()

    observed = regular.notna().astype(int)
    if args.fill == "interpolate":
        regular = regular.interpolate(method="time").ffill().bfill()
    elif args.fill == "ffill":
        regular = regular.ffill().bfill()
    else:
        regular = regular.fillna(0.0)

    out = regular.to_frame(name=args.target_col)
    out["was_observed_bucket"] = observed.reindex(out.index).fillna(0).astype(int)
    out = out.reset_index().rename(columns={args.date_col: "timestamp"})
    return out


def add_calendar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["day_of_week"] = ts.dt.dayofweek
    out["hour"] = ts.dt.hour
    out["minute"] = ts.dt.minute
    out["week_slot"] = (
        out["day_of_week"].astype(str) + "_" + out["hour"].astype(str) + "_" + out["minute"].astype(str)
    )
    out["time_slot"] = out["hour"].astype(str) + "_" + out["minute"].astype(str)
    return out


def fit_seasonality(train: pd.DataFrame, target_col: str, enabled: bool) -> SeasonalityProfile:
    """Learn a simple additive weekly/intraday seasonal effect from train only."""

    if not enabled:
        return SeasonalityProfile(False, 0.0, {}, {}, {})

    train = add_calendar_columns(train)
    global_mean = float(train[target_col].mean())

    by_week = (train.groupby("week_slot")[target_col].mean() - global_mean).to_dict()
    by_time = (train.groupby("time_slot")[target_col].mean() - global_mean).to_dict()
    by_hour = (train.groupby("hour")[target_col].mean() - global_mean).to_dict()

    return SeasonalityProfile(
        enabled=True,
        global_mean=global_mean,
        by_week_slot={str(k): float(v) for k, v in by_week.items()},
        by_time_slot={str(k): float(v) for k, v in by_time.items()},
        by_hour={int(k): float(v) for k, v in by_hour.items()},
    )


def seasonal_effect(frame: pd.DataFrame, profile: SeasonalityProfile) -> np.ndarray:
    """Map timestamps to learned seasonal effects with sensible fallbacks."""

    if not profile.enabled:
        return np.zeros(len(frame), dtype=np.float32)

    data = add_calendar_columns(frame)
    effects: list[float] = []
    for row in data.itertuples(index=False):
        effects.append(
            profile.by_week_slot.get(
                row.week_slot,
                profile.by_time_slot.get(row.time_slot, profile.by_hour.get(int(row.hour), 0.0)),
            )
        )
    return np.asarray(effects, dtype=np.float32)


def load_timesfm(model_id: str, fallback_model_id: str, context_len: int, max_horizon: int, batch_size: int):
    """Load TimesFM 2.5 when available, otherwise fall back to the older API."""

    import timesfm

    if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        model.compile(
            timesfm.ForecastConfig(
                max_context=context_len,
                max_horizon=max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
                per_core_batch_size=batch_size,
            )
        )
        return model, "timesfm_2p5"

    try:
        hparams = timesfm.TimesFmHparams(context_len=context_len, horizon_len=max_horizon)
    except TypeError:
        # Some older TimesFM builds accept horizon_len but not context_len.
        hparams = timesfm.TimesFmHparams(horizon_len=max_horizon)
    checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=fallback_model_id)
    return timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint), "timesfm_1x"


def raw_timesfm_forecast(model: Any, api_kind: str, values: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Call either TimesFM API shape and return point plus quantile array."""

    values = values.astype(np.float32)
    if api_kind == "timesfm_2p5":
        point, quantiles = model.forecast(horizon=horizon, inputs=[values])
    else:
        point, quantiles = model.forecast([values], freq=[0])
    return np.asarray(point[0])[:horizon], np.asarray(quantiles[0])[:horizon]


def forecast_with_optional_seasonality(
    model: Any,
    api_kind: str,
    context: pd.DataFrame,
    future_timestamps: pd.Series,
    target_col: str,
    horizon: int,
    use_seasonality: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast target values, using train-only additive seasonality if requested."""

    profile = fit_seasonality(context, target_col, enabled=use_seasonality)
    context_effect = seasonal_effect(context, profile)
    future_frame = pd.DataFrame({"timestamp": future_timestamps})
    future_effect = seasonal_effect(future_frame, profile)

    context_values = context[target_col].to_numpy(dtype=np.float32)
    model_input = context_values - context_effect
    point, quantiles = raw_timesfm_forecast(model, api_kind, model_input, horizon)

    point = point + future_effect
    if quantiles.ndim == 2 and len(quantiles) == len(future_effect):
        quantiles = quantiles + future_effect[:, None]
    return point, quantiles


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    denom = np.where(np.abs(actual) < 1e-8, np.nan, np.abs(actual))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_percent": float(np.nanmean(np.abs(error) / denom) * 100),
        "bias": float(np.mean(predicted - actual)),
    }


def quantile_columns(quantiles: np.ndarray) -> dict[str, np.ndarray]:
    """Expose common quantiles without assuming every TimesFM version labels them equally."""

    if quantiles.ndim != 2 or quantiles.shape[1] == 0:
        return {}

    cols: dict[str, np.ndarray] = {}
    labels = ["q00", "q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]
    for i in range(min(quantiles.shape[1], len(labels))):
        cols[labels[i]] = quantiles[:, i]
    return cols


def result_frame(
    timestamps: pd.Series,
    actual: np.ndarray | None,
    point: np.ndarray,
    quantiles: np.ndarray,
    split_name: str,
    origin_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "split": split_name,
            "timestamp": pd.to_datetime(timestamps).to_numpy(),
            "forecast": point,
        }
    )
    if origin_timestamp is not None:
        out.insert(1, "origin_timestamp", origin_timestamp)
    if actual is not None:
        out.insert(3, "actual", actual)
        out["error"] = out["actual"] - out["forecast"]
    for name, values in quantile_columns(quantiles).items():
        out[name] = values
    return out


def parse_splits(text: str) -> list[tuple[int, int, int]]:
    splits: list[tuple[int, int, int]] = []
    for chunk in text.split(","):
        parts = [int(x) for x in chunk.strip().split(":")]
        if len(parts) != 3 or sum(parts) != 100:
            raise ValueError(f"Split {chunk!r} must look like 70:15:15 and sum to 100.")
        splits.append((parts[0], parts[1], parts[2]))
    return splits


def evaluate_splits(
    data: pd.DataFrame,
    model: Any,
    api_kind: str,
    args: argparse.Namespace,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate multiple chronological train/validation/test splits."""

    forecast_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for train_pct, val_pct, test_pct in parse_splits(args.splits):
        n = len(data)
        train_end = int(n * train_pct / 100)
        val_end = int(n * (train_pct + val_pct) / 100)
        split_label = f"{train_pct}_{val_pct}_{test_pct}"

        windows = [
            ("validation", data.iloc[:train_end], data.iloc[train_end : min(train_end + horizon, val_end)]),
            ("test", data.iloc[:val_end], data.iloc[val_end : min(val_end + horizon, n)]),
        ]

        for window_name, context, target in windows:
            if len(context) < 2 or len(target) == 0:
                continue
            h = len(target)
            point, quantiles = forecast_with_optional_seasonality(
                model=model,
                api_kind=api_kind,
                context=context,
                future_timestamps=target["timestamp"].reset_index(drop=True),
                target_col=args.target_col,
                horizon=h,
                use_seasonality=args.seasonality == "additive",
            )
            actual = target[args.target_col].to_numpy(dtype=np.float32)
            row_metrics = metrics(actual, point)
            row_metrics.update(
                {
                    "split": split_label,
                    "window": window_name,
                    "train_rows": int(len(context)),
                    "evaluated_rows": int(h),
                    "start": str(target["timestamp"].iloc[0]),
                    "end": str(target["timestamp"].iloc[-1]),
                }
            )
            metric_rows.append(row_metrics)
            forecast_rows.append(
                result_frame(
                    timestamps=target["timestamp"].reset_index(drop=True),
                    actual=actual,
                    point=point,
                    quantiles=quantiles,
                    split_name=f"{split_label}_{window_name}",
                )
            )

    return pd.DataFrame(metric_rows), pd.concat(forecast_rows, ignore_index=True)


def walk_forward_validation(
    data: pd.DataFrame,
    model: Any,
    api_kind: str,
    args: argparse.Namespace,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin validation: repeatedly forecast the next horizon from history so far."""

    step = horizon_steps(args.bucket, args.walk_step) if args.walk_step else horizon
    start = max(2, int(len(data) * args.walk_start_ratio))
    origins = list(range(start, len(data) - horizon + 1, step))[: args.max_walks]

    forecast_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for i, origin in enumerate(origins, start=1):
        context = data.iloc[:origin]
        target = data.iloc[origin : origin + horizon]
        point, quantiles = forecast_with_optional_seasonality(
            model=model,
            api_kind=api_kind,
            context=context,
            future_timestamps=target["timestamp"].reset_index(drop=True),
            target_col=args.target_col,
            horizon=len(target),
            use_seasonality=args.seasonality == "additive",
        )
        actual = target[args.target_col].to_numpy(dtype=np.float32)
        row_metrics = metrics(actual, point)
        row_metrics.update(
            {
                "walk": i,
                "origin_timestamp": str(data["timestamp"].iloc[origin - 1]),
                "train_rows": int(len(context)),
                "evaluated_rows": int(len(target)),
            }
        )
        metric_rows.append(row_metrics)
        forecast_rows.append(
            result_frame(
                timestamps=target["timestamp"].reset_index(drop=True),
                actual=actual,
                point=point,
                quantiles=quantiles,
                split_name="walk_forward",
                origin_timestamp=data["timestamp"].iloc[origin - 1],
            )
        )

    if not metric_rows:
        return pd.DataFrame(), pd.DataFrame()
    return pd.DataFrame(metric_rows), pd.concat(forecast_rows, ignore_index=True)


def future_forecast(
    data: pd.DataFrame,
    model: Any,
    api_kind: str,
    args: argparse.Namespace,
    horizon: int,
) -> pd.DataFrame:
    offset = pd.tseries.frequencies.to_offset(args.bucket)
    future_timestamps = pd.date_range(
        start=pd.Timestamp(data["timestamp"].iloc[-1]) + offset,
        periods=horizon,
        freq=args.bucket,
    )
    point, quantiles = forecast_with_optional_seasonality(
        model=model,
        api_kind=api_kind,
        context=data,
        future_timestamps=pd.Series(future_timestamps),
        target_col=args.target_col,
        horizon=horizon,
        use_seasonality=args.seasonality == "additive",
    )
    return result_frame(
        timestamps=pd.Series(future_timestamps),
        actual=None,
        point=point,
        quantiles=quantiles,
        split_name="future",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "output_dynamic_bucket"
    output_dir.mkdir(parents=True, exist_ok=True)

    horizon = horizon_steps(args.bucket, args.forecast_length)
    data = load_and_bucket(args)
    if len(data) <= horizon * 3:
        raise ValueError(
            f"Need more data after bucketing. Got {len(data)} rows and horizon {horizon}; "
            "try a larger bucket like 1h or a shorter forecast length."
        )

    prepared_path = output_dir / f"prepared_{args.bucket}.csv"
    data.to_csv(prepared_path, index=False)

    print("Prepared data")
    print(f"  rows:          {len(data)}")
    print(f"  date range:    {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"  bucket:        {args.bucket}")
    print(f"  horizon:       {horizon} steps ({args.forecast_length})")
    print(f"  seasonality:   {args.seasonality}")

    print("Loading TimesFM...")
    model, api_kind = load_timesfm(
        model_id=args.model_id,
        fallback_model_id=args.fallback_model_id,
        context_len=args.context_len,
        max_horizon=horizon,
        batch_size=args.batch_size,
    )
    print(f"  api:           {api_kind}")

    print("Evaluating train/validation/test splits...")
    split_metrics, split_forecasts = evaluate_splits(data, model, api_kind, args, horizon)
    split_metrics_path = output_dir / "split_metrics.csv"
    split_forecasts_path = output_dir / "split_forecasts.csv"
    split_metrics.to_csv(split_metrics_path, index=False)
    split_forecasts.to_csv(split_forecasts_path, index=False)

    print("Running walk-forward validation...")
    walk_metrics, walk_forecasts = walk_forward_validation(data, model, api_kind, args, horizon)
    walk_metrics_path = output_dir / "walk_forward_metrics.csv"
    walk_forecasts_path = output_dir / "walk_forward_forecasts.csv"
    walk_metrics.to_csv(walk_metrics_path, index=False)
    walk_forecasts.to_csv(walk_forecasts_path, index=False)

    print("Forecasting the next period from all data...")
    future = future_forecast(data, model, api_kind, args, horizon)
    future_path = output_dir / "future_forecast.csv"
    future.to_csv(future_path, index=False)

    summary = {
        "input": str(input_path),
        "date_col": args.date_col,
        "target_col": args.target_col,
        "bucket": args.bucket,
        "forecast_length": args.forecast_length,
        "horizon_steps": horizon,
        "aggregation": args.agg,
        "fill_method": args.fill,
        "seasonality": args.seasonality,
        "timesfm_api": api_kind,
        "regular_rows": int(len(data)),
        "date_range": {
            "start": str(data["timestamp"].min()),
            "end": str(data["timestamp"].max()),
        },
        "split_metrics": split_metrics.to_dict(orient="records"),
        "walk_forward_average_metrics": walk_metrics[["mae", "rmse", "mape_percent", "bias"]].mean().to_dict()
        if not walk_metrics.empty
        else {},
        "outputs": {
            "prepared_data": str(prepared_path),
            "split_metrics": str(split_metrics_path),
            "split_forecasts": str(split_forecasts_path),
            "walk_forward_metrics": str(walk_metrics_path),
            "walk_forward_forecasts": str(walk_forecasts_path),
            "future_forecast": str(future_path),
        },
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)

    print("\nSaved outputs")
    for path in [
        prepared_path,
        split_metrics_path,
        split_forecasts_path,
        walk_metrics_path,
        walk_forecasts_path,
        future_path,
        summary_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()

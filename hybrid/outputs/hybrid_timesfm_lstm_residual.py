#!/usr/bin/env python3
"""Hybrid TimesFM + LSTM residual correction for 1-hour-ahead forecasting.

Pipeline:
1. Load and hourly-regularize the input CSV.
2. Use TimesFM as a zero-shot one-step-ahead base forecaster.
3. Train a small PyTorch LSTM to predict the base forecast residual:
      residual = actual_next_hour - timesfm_forecast_next_hour
4. Forecast the next hour as:
      hybrid = timesfm_forecast + lstm_residual_correction

Recommended install in your TimesFM environment:
    pip install "timesfm[torch]" pandas numpy

Example:
    python hybrid_timesfm_lstm_residual.py ^
      --input "C:\\Users\\reshm\\OneDrive\\Documents\\work\\timesfm\\timesfm-forecasting\\chaps_demo.csv" ^
      --target-col transaction_count ^
      --validation-hours 168 ^
      --residual-samples 256
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
    r"\chaps_demo.csv"
)


@dataclass
class Scale:
    mean: float
    std: float

    def transform(self, x: np.ndarray | float) -> np.ndarray | float:
        return (x - self.mean) / self.std

    def inverse_residual(self, x: np.ndarray | float) -> np.ndarray | float:
        return x * self.std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast one hour ahead with TimesFM plus LSTM residual correction."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to input CSV.")
    parser.add_argument("--date-col", default="Datetime", help="Timestamp column.")
    parser.add_argument(
        "--target-col",
        default="transaction_count",
        help="Column to forecast. For the provided CSV, transaction_count is useful; "
        "dr_amount and cr_amount are all zero after cleaning.",
    )
    parser.add_argument("--freq", default="1h", help="Regular modeling frequency.")
    parser.add_argument(
        "--agg",
        choices=["sum", "mean"],
        default="sum",
        help="How to aggregate duplicate timestamps.",
    )
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--validation-hours", type=int, default=168)
    parser.add_argument(
        "--residual-samples",
        type=int,
        default=256,
        help="Maximum sampled TimesFM windows for residual training.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--model-id", default="google/timesfm-2.5-200m-pytorch")
    parser.add_argument(
        "--fallback-model-id",
        default="google/timesfm-1.0-200m-pytorch",
        help="Used only if the installed timesfm package exposes the older API.",
    )
    parser.add_argument(
        "--output-dir",
        default="hybrid_output",
        help="Output directory for prepared data, validation forecasts, and final forecast.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only clean and summarize data; do not load TimesFM or train LSTM.",
    )
    return parser.parse_args()


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"-": 0, "": np.nan}), errors="coerce").fillna(0.0)


def load_hourly_data(path: Path, date_col: str, target_col: str, freq: str, agg: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [date_col, target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")

    work = df[[date_col, target_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], dayfirst=True, errors="coerce")
    work[target_col] = clean_numeric(work[target_col])
    work = work.dropna(subset=[date_col]).sort_values(date_col)

    grouped = work.groupby(date_col, as_index=True)[target_col]
    hourly = grouped.sum() if agg == "sum" else grouped.mean()
    hourly = hourly.resample(freq).sum() if agg == "sum" else hourly.resample(freq).mean()
    hourly = hourly.interpolate(method="time").ffill().bfill()

    out = hourly.to_frame("y").reset_index().rename(columns={date_col: "timestamp"})
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    return out


def load_timesfm(model_id: str, fallback_model_id: str, context_len: int, batch_size: int) -> tuple[Any, str]:
    import timesfm

    if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        model.compile(
            timesfm.ForecastConfig(
                max_context=context_len,
                max_horizon=1,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
                per_core_batch_size=batch_size,
            )
        )
        return model, "timesfm_2p5"

    hparams_kwargs = {"horizon_len": 1}
    try:
        hparams_kwargs["context_len"] = context_len
        hparams = timesfm.TimesFmHparams(**hparams_kwargs)
    except TypeError:
        hparams_kwargs.pop("context_len", None)
        hparams = timesfm.TimesFmHparams(**hparams_kwargs)
    checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=fallback_model_id)
    return timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint), "timesfm_legacy"


def timesfm_one_step(model: Any, api_kind: str, history: np.ndarray) -> tuple[float, dict[str, float]]:
    history = np.asarray(history, dtype=np.float32)
    if api_kind == "timesfm_2p5":
        point, quantiles = model.forecast(horizon=1, inputs=[history])
    else:
        point, quantiles = model.forecast([history], freq=[0])

    forecast = float(np.asarray(point)[0, 0])
    q = np.asarray(quantiles)
    q_values: dict[str, float] = {}
    if q.ndim == 3 and q.shape[-1] >= 10:
        q_values = {
            "q10": float(q[0, 0, 1]),
            "q50": float(q[0, 0, 5]),
            "q90": float(q[0, 0, 9]),
        }
    return max(0.0, forecast), q_values


def cyclical_time_features(ts: pd.Timestamp) -> np.ndarray:
    hour_angle = 2.0 * math.pi * ts.hour / 24.0
    dow_angle = 2.0 * math.pi * ts.dayofweek / 7.0
    return np.array(
        [math.sin(hour_angle), math.cos(hour_angle), math.sin(dow_angle), math.cos(dow_angle)],
        dtype=np.float32,
    )


def sequence_features(df: pd.DataFrame, end_exclusive: int, lookback: int, scale: Scale) -> np.ndarray:
    window = df.iloc[end_exclusive - lookback : end_exclusive]
    y_scaled = scale.transform(window["y"].to_numpy(dtype=np.float32)).reshape(-1, 1)
    time_feats = np.vstack([cyclical_time_features(ts) for ts in window["timestamp"]])
    return np.hstack([y_scaled, time_feats]).astype(np.float32)


def sample_training_indices(start: int, stop: int, max_samples: int) -> np.ndarray:
    all_indices = np.arange(start, stop)
    if len(all_indices) <= max_samples:
        return all_indices
    return np.linspace(start, stop - 1, max_samples, dtype=int)


def make_lstm_model(input_size: int, static_size: int, hidden_size: int):
    import torch
    from torch import nn

    class ResidualLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden_size + static_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
            _, (hidden, _) = self.lstm(seq)
            combined = torch.cat([hidden[-1], static], dim=1)
            return self.head(combined).squeeze(-1)

    return ResidualLSTM()


def train_residual_lstm(
    x_seq: np.ndarray,
    x_static: np.ndarray,
    y_resid_scaled: np.ndarray,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    model = make_lstm_model(x_seq.shape[-1], x_static.shape[-1], hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()

    dataset = TensorDataset(
        torch.tensor(x_seq, dtype=torch.float32),
        torch.tensor(x_static, dtype=torch.float32),
        torch.tensor(y_resid_scaled, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    final_loss = np.nan
    for _ in range(epochs):
        losses = []
        for seq_batch, static_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(seq_batch, static_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        final_loss = float(np.mean(losses))
    return model, final_loss


def predict_residual(model: Any, seq: np.ndarray, static: np.ndarray, scale: Scale) -> float:
    import torch

    model.eval()
    with torch.no_grad():
        pred_scaled = model(
            torch.tensor(seq[None, :, :], dtype=torch.float32),
            torch.tensor(static[None, :], dtype=torch.float32),
        ).numpy()[0]
    return float(scale.inverse_residual(pred_scaled))


def build_samples(
    df: pd.DataFrame,
    indices: np.ndarray,
    model: Any,
    api_kind: str,
    context_len: int,
    lookback: int,
    scale: Scale,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    values = df["y"].to_numpy(dtype=np.float32)
    x_seq, x_static, y_resid_scaled, rows = [], [], [], []

    for pred_idx in indices:
        context_start = max(0, pred_idx - context_len)
        base, q_values = timesfm_one_step(model, api_kind, values[context_start:pred_idx])
        actual = float(values[pred_idx])
        residual = actual - base
        next_feats = cyclical_time_features(df.loc[pred_idx, "timestamp"])
        static = np.concatenate([[float(scale.transform(base))], next_feats]).astype(np.float32)

        x_seq.append(sequence_features(df, pred_idx, lookback, scale))
        x_static.append(static)
        y_resid_scaled.append(float(residual / scale.std))
        rows.append(
            {
                "timestamp": df.loc[pred_idx, "timestamp"],
                "actual": actual,
                "timesfm_forecast": base,
                "residual": residual,
                **q_values,
            }
        )

    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(x_static, dtype=np.float32),
        np.asarray(y_resid_scaled, dtype=np.float32),
        pd.DataFrame(rows),
    )


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    err = actual - predicted
    denom = np.where(np.abs(actual) < 1e-8, np.nan, np.abs(actual))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape_percent": float(np.nanmean(np.abs(err) / denom) * 100.0),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_hourly_data(input_path, args.date_col, args.target_col, args.freq, args.agg)
    prepared_path = output_dir / "prepared_hourly_data.csv"
    df.to_csv(prepared_path, index=False)

    if len(df) < args.lookback + args.validation_hours + 10:
        raise ValueError(
            f"Not enough rows after hourly preparation. Got {len(df)}, need at least "
            f"{args.lookback + args.validation_hours + 10}."
        )

    summary: dict[str, Any] = {
        "input": str(input_path),
        "target_col": args.target_col,
        "rows_hourly": int(len(df)),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "prepared_data": str(prepared_path),
    }

    if args.dry_run:
        summary["dry_run"] = True
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    validation_start = len(df) - args.validation_hours
    train_end = validation_start
    train_values = df.iloc[:train_end]["y"].to_numpy(dtype=np.float32)
    scale = Scale(mean=float(np.mean(train_values)), std=float(np.std(train_values) or 1.0))

    model, api_kind = load_timesfm(args.model_id, args.fallback_model_id, args.context_len, args.batch_size)
    min_pred_idx = max(args.lookback, 1)
    train_indices = sample_training_indices(
        min_pred_idx,
        train_end,
        min(args.residual_samples, max(1, train_end - min_pred_idx)),
    )

    x_seq, x_static, y_resid_scaled, train_residuals = build_samples(
        df, train_indices, model, api_kind, args.context_len, args.lookback, scale
    )
    residual_model, train_loss = train_residual_lstm(
        x_seq,
        x_static,
        y_resid_scaled,
        args.hidden_size,
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )

    val_indices = np.arange(validation_start, len(df))
    val_seq, val_static, _, validation = build_samples(
        df, val_indices, model, api_kind, args.context_len, args.lookback, scale
    )
    corrections = np.array(
        [predict_residual(residual_model, val_seq[i], val_static[i], scale) for i in range(len(val_seq))]
    )
    validation["residual_correction"] = corrections
    validation["hybrid_forecast"] = np.maximum(
        0.0, validation["timesfm_forecast"].to_numpy(dtype=float) + corrections
    )
    validation_path = output_dir / "validation_forecasts.csv"
    validation.to_csv(validation_path, index=False)

    values = df["y"].to_numpy(dtype=np.float32)
    base_next, q_next = timesfm_one_step(
        model, api_kind, values[max(0, len(values) - args.context_len) :]
    )
    next_ts = df["timestamp"].iloc[-1] + pd.tseries.frequencies.to_offset(args.freq)
    next_static = np.concatenate(
        [[float(scale.transform(base_next))], cyclical_time_features(next_ts)]
    ).astype(np.float32)
    next_seq = sequence_features(df, len(df), args.lookback, scale)
    next_correction = predict_residual(residual_model, next_seq, next_static, scale)
    next_hybrid = max(0.0, base_next + next_correction)

    next_forecast = pd.DataFrame(
        [
            {
                "timestamp": next_ts,
                "timesfm_forecast": base_next,
                "residual_correction": next_correction,
                "hybrid_forecast": next_hybrid,
                **q_next,
            }
        ]
    )
    next_path = output_dir / "next_1h_forecast.csv"
    next_forecast.to_csv(next_path, index=False)

    summary.update(
        {
            "timesfm_api": api_kind,
            "context_len": args.context_len,
            "lookback": args.lookback,
            "validation_hours": args.validation_hours,
            "residual_training_samples": int(len(train_indices)),
            "lstm_final_train_loss": train_loss,
            "timesfm_validation_metrics": regression_metrics(
                validation["actual"].to_numpy(dtype=float),
                validation["timesfm_forecast"].to_numpy(dtype=float),
            ),
            "hybrid_validation_metrics": regression_metrics(
                validation["actual"].to_numpy(dtype=float),
                validation["hybrid_forecast"].to_numpy(dtype=float),
            ),
            "training_residuals": str(output_dir / "training_residuals.csv"),
            "validation_forecasts": str(validation_path),
            "next_1h_forecast": str(next_path),
        }
    )
    train_residuals.to_csv(output_dir / "training_residuals.csv", index=False)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(next_forecast.to_string(index=False))


if __name__ == "__main__":
    main()

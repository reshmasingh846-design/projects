"""
1-hour-ahead multi-model forecasting for CHAPS intraday payments, loaded
through the office `data.py` module (fs_gtsy_mde package) instead of the
local demo CSV used by chaps_multimodel_forecast.py.

Run this on the machine/SSH host where the `fs_gtsy_mde` package and its venv
live (e.g. fraasstratc7.de.db.com). It imports `data.load_intraday`, which
auto-detects .xlsx/.parquet/.csv and validates the (date, hour, dr_amount,
cr_amount) schema, then aggregates to an hourly series the same way the demo
script does.

If `import data` fails, either:
  1. Run from a directory where `data.py` is importable (e.g. the
     fs_gtsy_mde/src checkout), or
  2. Put its src dir on PYTHONPATH before running:
       export PYTHONPATH="/home/singres/src/fs_gtsy_mde/src:$PYTHONPATH"
     (adjust FS_GTSY_MDE_SRC below if your checkout lives elsewhere).

Required packages (install in the office venv if missing):
    pip install pandas numpy statsmodels scikit-learn xgboost torch matplotlib openpyxl

Models (same as chaps_multimodel_forecast.py):
  1. Exponential Smoothing (ETS / Holt-Winters, additive, daily seasonality)
  2. SARIMAX (ARIMA + seasonal + calendar exogenous regressors)
  3. XGBoost (lag + rolling + calendar features)
  4. LSTM (PyTorch, sequence-to-one)

Usage:
    python chaps_multimodel_forecast_live.py
    python chaps_multimodel_forecast_live.py --input /path/to/other_file.xlsx
    python chaps_multimodel_forecast_live.py --target dr_amount --test-hours 336
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Make the office `data.py` module importable, then import it.
# ---------------------------------------------------------------------------
FS_GTSY_MDE_SRC = "/home/singres/src/fs_gtsy_mde/src"
if FS_GTSY_MDE_SRC not in sys.path:
    sys.path.insert(0, FS_GTSY_MDE_SRC)

try:
    from data import CREDIT_COL, DATE_COL, DEBIT_COL, HOUR_COL, load_intraday
except ImportError as exc:
    raise ImportError(
        "Could not import 'data' (the fs_gtsy_mde intraday loader). Make sure "
        "you're running inside the project's venv and that its src dir is on "
        f"PYTHONPATH, e.g.:\n"
        f'  export PYTHONPATH="{FS_GTSY_MDE_SRC}:$PYTHONPATH"\n'
        "Edit FS_GTSY_MDE_SRC at the top of this script if your checkout is "
        "somewhere else."
    ) from exc

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DAILY_SEASONAL_PERIOD = 24
LAGS = [1, 2, 3, 24, 48, 168]
EXOG_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
SEQ_LEN = 24
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-model CHAPS forecast using the live fs_gtsy_mde data.py loader."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to the intraday file (.xlsx/.parquet/.csv). "
        "Defaults to data.py's default_intraday_path().",
    )
    parser.add_argument(
        "--target",
        default="total_amount",
        choices=["total_amount", "dr_amount", "cr_amount"],
        help="Column to forecast (total_amount = dr_amount + cr_amount).",
    )
    parser.add_argument("--test-hours", type=int, default=24 * 14, help="Hours held out for backtest.")
    parser.add_argument("--output-dir", default="chaps_forecast_output_live")
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip the PyTorch LSTM model (use if torch isn't installed on this host).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------
def load_hourly_series(input_path: str | None) -> pd.DataFrame:
    raw = load_intraday(input_path)  # columns: date, hour, dr_amount, cr_amount

    dt = pd.to_datetime(raw[DATE_COL]) + pd.to_timedelta(raw[HOUR_COL], unit="h")
    work = raw.assign(Datetime=dt)

    agg = (
        work.groupby("Datetime")
        .agg(dr_amount=(DEBIT_COL, "sum"), cr_amount=(CREDIT_COL, "sum"))
        .sort_index()
    )
    agg["total_amount"] = agg["dr_amount"] + agg["cr_amount"]

    full_idx = pd.date_range(agg.index.min(), agg.index.max(), freq="h")
    agg = agg.reindex(full_idx).fillna(0.0)
    agg.index.name = "Datetime"
    return agg


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek  # 0=Mon .. 6=Sun
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df


def add_lag_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.copy()
    for lag in LAGS:
        df[f"lag_{lag}"] = df[target].shift(lag)
    df["roll_mean_24"] = df[target].shift(1).rolling(24).mean()
    df["roll_std_24"] = df[target].shift(1).rolling(24).std()
    df["roll_mean_168"] = df[target].shift(1).rolling(168).mean()
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(actual: pd.Series, pred: pd.Series) -> dict:
    actual, pred = actual.align(pred, join="inner")
    err = actual - pred
    mae = err.abs().mean()
    rmse = np.sqrt((err**2).mean())
    nonzero = actual != 0
    if nonzero.sum() > 0:
        smape = (
            2 * err[nonzero].abs() / (actual[nonzero].abs() + pred[nonzero].abs())
        ).mean() * 100
        mae_nonzero = err[nonzero].abs().mean()
    else:
        smape = np.nan
        mae_nonzero = np.nan
    bias = err.mean()
    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE_%": smape,
        "MAE_nonzero_hours": mae_nonzero,
        "Bias": bias,
        "n_test": len(actual),
    }


# ---------------------------------------------------------------------------
# Model 1: Exponential Smoothing (daily-refit rolling backtest)
# ---------------------------------------------------------------------------
def run_ets_backtest(data: pd.DataFrame, target: str, test_start: int) -> pd.Series:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    y = data[target].astype(float)
    test_index = data.index[test_start:]
    n_test = len(test_index)

    preds = np.empty(n_test)
    step = 24
    pos = 0
    while pos < n_test:
        train_end = test_start + pos
        y_train = y.iloc[:train_end]
        model = ETSModel(
            y_train,
            error="add",
            trend=None,
            seasonal="add",
            seasonal_periods=DAILY_SEASONAL_PERIOD,
        )
        fit = model.fit(disp=False)
        horizon = min(step, n_test - pos)
        fc = fit.forecast(horizon)
        preds[pos : pos + horizon] = fc.values
        pos += horizon

    preds = np.clip(preds, 0, None)
    return pd.Series(preds, index=test_index, name="ETS")


# ---------------------------------------------------------------------------
# Model 2: SARIMAX (true 1-step-ahead walk-forward via Kalman append)
# ---------------------------------------------------------------------------
def run_sarimax_backtest(data: pd.DataFrame, target: str, test_start: int) -> pd.Series:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = data[target].astype(float)
    exog = data[EXOG_COLS]

    y_train = y.iloc[:test_start]
    exog_train = exog.iloc[:test_start]
    test_index = data.index[test_start:]

    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=(2, 0, 1),
        seasonal_order=(1, 0, 1, DAILY_SEASONAL_PERIOD),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)

    preds = []
    current = fit
    for ts in test_index:
        exog_next = exog.loc[[ts]]
        fc = current.get_forecast(1, exog=exog_next)
        preds.append(fc.predicted_mean.iloc[0])
        current = current.append(y.loc[[ts]], exog=exog_next, refit=False)

    preds = np.clip(np.array(preds), 0, None)
    return pd.Series(preds, index=test_index, name="SARIMAX")


# ---------------------------------------------------------------------------
# Model 3: XGBoost (trained once, true 1-step-ahead via real lag features)
# ---------------------------------------------------------------------------
def run_xgb_backtest(data: pd.DataFrame, target: str, test_start: int) -> pd.Series:
    from xgboost import XGBRegressor

    feat = add_lag_features(data, target)
    feature_cols = [f"lag_{lag}" for lag in LAGS] + [
        "roll_mean_24",
        "roll_std_24",
        "roll_mean_168",
    ] + EXOG_COLS
    feat = feat.dropna(subset=feature_cols)

    split_ts = data.index[test_start]
    train_feat = feat[feat.index < split_ts]
    test_feat = feat[feat.index >= split_ts]

    X_train, y_train = train_feat[feature_cols], train_feat[target]
    X_test = test_feat[feature_cols]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 0, None)
    return pd.Series(preds, index=test_feat.index, name="XGBoost")


# ---------------------------------------------------------------------------
# Model 4: LSTM (PyTorch, trained once, true 1-step-ahead via real sequences)
# ---------------------------------------------------------------------------
def run_lstm_backtest(data: pd.DataFrame, target: str, test_start: int, epochs: int = 30) -> pd.Series:
    import torch
    import torch.nn as nn

    torch.manual_seed(RANDOM_STATE)

    df = data.copy()
    mean = df[target].iloc[:test_start].mean()
    std = df[target].iloc[:test_start].std()
    df["value_scaled"] = (df[target] - mean) / std

    feature_cols = ["value_scaled"] + EXOG_COLS
    values = df[feature_cols].values.astype(np.float32)
    targets = df["value_scaled"].values.astype(np.float32)

    X, y, idx = [], [], []
    for i in range(SEQ_LEN, len(df)):
        X.append(values[i - SEQ_LEN : i])
        y.append(targets[i])
        idx.append(df.index[i])
    X = np.array(X)
    y = np.array(y)
    idx = pd.DatetimeIndex(idx)

    split_ts = data.index[test_start]
    train_mask = idx < split_ts
    X_train, y_train = X[train_mask], y[train_mask]
    X_test = X[~train_mask]
    test_idx = idx[~train_mask]

    class Net(nn.Module):
        def __init__(self, n_features, hidden=64, layers=2):
            super().__init__()
            self.lstm = nn.LSTM(
                n_features, hidden, num_layers=layers, batch_first=True, dropout=0.1
            )
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    net = Net(n_features=X_train.shape[-1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossfn = nn.MSELoss()

    Xtr = torch.tensor(X_train)
    ytr = torch.tensor(y_train).unsqueeze(-1)
    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            pred = net(xb)
            loss = lossfn(pred, yb)
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        preds_scaled = net(torch.tensor(X_test)).squeeze(-1).numpy()
    preds = preds_scaled * std + mean
    preds = np.clip(preds, 0, None)
    return pd.Series(preds, index=test_idx, name="LSTM")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_model(actual: pd.Series, pred: pd.Series, name: str, target: str, out_path: Path):
    actual_a, pred_a = actual.align(pred, join="inner")
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(actual_a.index, actual_a.values, label="Actual", color="#333333", linewidth=1.2)
    ax.plot(pred_a.index, pred_a.values, label=f"{name} forecast", color="#e15759", linewidth=1.1, alpha=0.9)
    ax.set_title(f"{name}: 1-hour-ahead backtest -- {target} (last {len(actual_a)} hours)")
    ax.set_xlabel("Datetime")
    ax.set_ylabel(target)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_zoom(actual: pd.Series, preds: dict, target: str, out_path: Path, days: int = 4):
    end = actual.index.max()
    start = end - pd.Timedelta(hours=24 * days)
    fig, ax = plt.subplots(figsize=(14, 5))
    a = actual.loc[start:end]
    ax.plot(a.index, a.values, label="Actual", color="#222222", linewidth=1.8)
    colors = {"ETS": "#4e79a7", "SARIMAX": "#f28e2b", "XGBoost": "#59a14f", "LSTM": "#e15759"}
    for name, p in preds.items():
        p_a = p.loc[start:end]
        ax.plot(p_a.index, p_a.values, label=name, color=colors.get(name), linewidth=1.3, alpha=0.9)
    ax.set_title(f"All models -- last {days} days of backtest (zoomed) -- {target}")
    ax.set_xlabel("Datetime")
    ax.set_ylabel(target)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_metric_bars(metrics_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, col in zip(axes, ["MAE", "RMSE", "sMAPE_%"]):
        ax.bar(metrics_df.index, metrics_df[col], color=["#4e79a7", "#f28e2b", "#59a14f", "#e15759"])
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Backtest error comparison (14-day holdout, 1-hour-ahead)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = args.target

    t_start = time.time()
    print(f"Loading & preprocessing data via data.load_intraday(input={args.input!r}) ...")
    data = load_hourly_series(args.input)
    data = add_calendar_features(data)
    print(f"  hourly series: {len(data)} rows, {data.index.min()} -> {data.index.max()}")
    print(f"  target column: {target}")

    n = len(data)
    test_start = n - args.test_hours
    if test_start <= SEQ_LEN:
        raise ValueError(
            f"Not enough history ({n} hours) for a {args.test_hours}-hour backtest. "
            "Use a smaller --test-hours."
        )
    actual_test = data[target].iloc[test_start:]
    print(f"  train: {test_start} hours | test (backtest): {args.test_hours} hours "
          f"({data.index[test_start]} -> {data.index[-1]})")

    preds = {}
    n_models = 3 if args.skip_lstm else 4

    print(f"\n[1/{n_models}] Exponential Smoothing (ETS, daily-refit rolling backtest) ...")
    t0 = time.time()
    preds["ETS"] = run_ets_backtest(data, target, test_start)
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"\n[2/{n_models}] SARIMAX (ARIMA + seasonal + calendar exog, true 1-step-ahead) ...")
    t0 = time.time()
    preds["SARIMAX"] = run_sarimax_backtest(data, target, test_start)
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"\n[3/{n_models}] XGBoost (lag + rolling + calendar features) ...")
    t0 = time.time()
    preds["XGBoost"] = run_xgb_backtest(data, target, test_start)
    print(f"  done in {time.time()-t0:.1f}s")

    if not args.skip_lstm:
        print(f"\n[4/{n_models}] LSTM (PyTorch, sequence-to-one) ...")
        t0 = time.time()
        preds["LSTM"] = run_lstm_backtest(data, target, test_start, epochs=30)
        print(f"  done in {time.time()-t0:.1f}s")

    # ---------------- metrics ----------------
    print("\nComputing backtest metrics ...")
    rows = {name: compute_metrics(actual_test, p) for name, p in preds.items()}
    metrics_df = pd.DataFrame(rows).T[
        ["MAE", "RMSE", "sMAPE_%", "MAE_nonzero_hours", "Bias", "n_test"]
    ]
    metrics_df = metrics_df.sort_values("RMSE")
    print(metrics_df.round(3))
    metrics_df.to_csv(out_dir / "backtest_metrics.csv")

    # ---------------- predictions dump ----------------
    pred_df = pd.DataFrame({"actual": actual_test})
    for name, p in preds.items():
        pred_df[name] = p
    pred_df.to_csv(out_dir / "backtest_predictions.csv")

    # ---------------- plots ----------------
    print("\nGenerating plots ...")
    for name, p in preds.items():
        plot_model(actual_test, p, name, target, out_dir / f"backtest_{name.lower()}.png")
    plot_zoom(actual_test, preds, target, out_dir / "backtest_zoom_last4days.png", days=4)
    plot_metric_bars(metrics_df, out_dir / "backtest_metric_comparison.png")

    print(f"\nAll outputs written to: {out_dir.resolve()}")
    print(f"Total runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()

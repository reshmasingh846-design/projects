"""
1-hour-ahead multi-model forecasting for CHAPS intraday payments (dr_amount /
cr_amount / total_amount), loaded through the office data pipeline.

Models:
  1. Exponential Smoothing (ETS / Holt-Winters, additive, daily seasonality)
  2. SARIMAX (ARIMA + seasonal + calendar exogenous regressors)
  3. XGBoost (lag + rolling + calendar features)
  4. LSTM (PyTorch, sequence-to-one)

Target handling:
  dr_amount / cr_amount follow a fixed ledger sign convention (debits <= 0,
  credits >= 0) rather than being genuinely mixed-sign. Rather than modelling
  the signed series directly, we:
    1. detect the sign convention from the training data,
    2. model log1p(|target|)  (always >= 0, heavy tail compressed),
    3. invert with expm1() and reapply the detected sign at the very end.
  This also fixes a bug in the original script: np.clip(preds, 0, None) was
  applied even when the target (dr_amount) is always <= 0, which floored
  every prediction toward zero and made the model structurally unable to
  predict a realistic debit value.

Backtest design (walk-forward, out-of-sample, last 14 days held out):
  - SARIMAX: true 1-step-ahead. Fit once on the training window, then for
    every test hour we forecast h=1 and then push the *actual* observed
    value into the filter (Kalman `append`, refit=False) before moving to
    the next hour. No refitting of parameters during the test window.
  - ETS: statsmodels' ETSModel has no incremental-update API, so we do a
    daily-refit rolling backtest: refit on all history available at the
    start of each test day, then generate a 24-hour-ahead path for that
    day.
  - XGBoost / LSTM: trained once on the training window. For every test
    hour, features/sequences are built from the *true* historical series
    (never from the model's own prior predictions), so these are also
    genuine 1-step-ahead evaluations.
"""
import sys
sys.path.insert(0, '/home/singres/src/fs_gtsy_mde/src')
import time
import warnings
import os
from collections.abc import Iterator
from pathlib import Path

from common import logger
from common.paths import data_dir

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path(data_dir("intraday")) / "hourly_Chaps_pmts3.xlsx"

OUT_DIR = Path("/home/singres/src/fs_gtsy_mde/src/tsfm/Trial/forecast_Chaps_pmts3_output")

OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "dr_amount"  # dr_amount | cr_amount | total_amount
TEST_HOURS = 24 * 14  # last 14 days held out for backtest
DAILY_SEASONAL_PERIOD = 24
LAGS = [1, 2, 3, 24, 48, 168]
EXOG_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_business_hour"]
SEQ_LEN = 24
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_hourly_series() -> pd.DataFrame:
    """Load the already-hourly-aggregated CHAPS sheet (Datetime, dr_amount,
    cr_amount, transaction_count, ...) and reindex to a continuous hourly
    range with zeros filled in for missing hours.

    If your real loader does raw-leg aggregation instead, keep that version
    -- everything below only needs a DataFrame indexed by Datetime with
    dr_amount/cr_amount columns.
    """
    raw = pd.read_excel(DATA_PATH)
    raw["Datetime"] = pd.to_datetime(raw["Datetime"])
    agg = raw.set_index("Datetime").sort_index()

    for col in ["dr_amount", "cr_amount", "transaction_count"]:
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors="coerce")

    agg["total_amount"] = agg["dr_amount"].fillna(0) + agg["cr_amount"].fillna(0)

    full_idx = pd.date_range(agg.index.min(), agg.index.max(), freq="h")
    agg = agg.reindex(full_idx).fillna(0.0)
    agg.index.name = "Datetime"
    return agg


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek  # 0=Mon .. 6=Sun
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_business_hour"] = df["hour"].between(6, 17).astype(int)
    df["day"] = df.index.day
    df["month"] = df.index.month
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
# Sign-aware log transform (handles dr_amount's fixed negative convention)
# ---------------------------------------------------------------------------
def determine_sign(raw: pd.Series) -> float:
    """Detect whether the target's nonzero values are predominantly negative
    (e.g. dr_amount, a debit-convention column) or positive."""
    nz = raw[raw != 0]
    if len(nz) == 0:
        return 1.0
    return -1.0 if (nz < 0).mean() > 0.5 else 1.0


def to_model_target(raw: pd.Series, sign: float) -> pd.Series:
    magnitude = (raw * sign).clip(lower=0)
    return np.log1p(magnitude)


def invert_predictions(preds: pd.Series, sign: float) -> pd.Series:
    magnitude = np.clip(np.expm1(preds), 0, None)
    return magnitude * sign


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(actual: pd.Series, pred: pd.Series) -> dict:
    actual, pred = actual.align(pred, join="inner")
    err = actual - pred
    mae = err.abs().mean()
    rmse = np.sqrt((err**2).mean())
    denom = actual.abs().sum()
    wape = (err.abs().sum() / denom * 100) if denom > 0 else np.nan
    nonzero = actual != 0
    if nonzero.sum() > 0:
        smape = (
            2
            * err[nonzero].abs()
            / (actual[nonzero].abs() + pred[nonzero].abs())
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
        "WAPE_%": wape,
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
    for i, ts in enumerate(test_index):
        exog_next = exog.loc[[ts]]
        fc = current.get_forecast(1, exog=exog_next)
        preds.append(fc.predicted_mean.iloc[0])
        current = current.append(y.loc[[ts]], exog=exog_next, refit=False)

    preds = np.array(preds)
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
    preds = model.predict(X_test)
    return pd.Series(preds, index=test_feat.index, name="XGBoost")


# ---------------------------------------------------------------------------
# Model 4: LSTM (PyTorch, trained once, true 1-step-ahead via real sequences)
# ---------------------------------------------------------------------------
class LSTMForecaster:
    def __init__(self, n_features, hidden=64, layers=2):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    n_features, hidden, num_layers=layers, batch_first=True, dropout=0.1
                )
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        self.net = _Net()

    def __call__(self, x):
        return self.net(x)


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
    for ax, col in zip(axes, ["MAE", "RMSE", "WAPE_%"]):
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
    t_start = time.time()
    print("Loading & preprocessing data ...")
    data = load_hourly_series()
    data = add_calendar_features(data)
    print(f"  hourly series: {len(data)} rows, {data.index.min()} -> {data.index.max()}")

    n = len(data)
    test_start = n - TEST_HOURS
    actual_test = data[TARGET].iloc[test_start:]

    raw_target = data[TARGET].astype(float)
    sign = determine_sign(raw_target)
    data["model_target"] = to_model_target(raw_target, sign)
    print(f"  sign convention detected for '{TARGET}': "
          f"{'negative (debit-style)' if sign < 0 else 'positive'}")

    print(f"  train: {test_start} hours | test (backtest): {TEST_HOURS} hours "
          f"({data.index[test_start]} -> {data.index[-1]})")

    preds = {}

    print("\n[1/4] Exponential Smoothing (ETS, daily-refit rolling backtest) ...")
    t0 = time.time()
    preds["ETS"] = run_ets_backtest(data, "model_target", test_start)
    print(f"  done in {time.time()-t0:.1f}s")

    print("\n[2/4] SARIMAX (ARIMA + seasonal + calendar exog, true 1-step-ahead) ...")
    t0 = time.time()
    preds["SARIMAX"] = run_sarimax_backtest(data, "model_target", test_start)
    print(f"  done in {time.time()-t0:.1f}s")

    print("\n[3/4] XGBoost (lag + rolling + calendar features) ...")
    t0 = time.time()
    preds["XGBoost"] = run_xgb_backtest(data, "model_target", test_start)
    print(f"  done in {time.time()-t0:.1f}s")

    print("\n[4/4] LSTM (PyTorch, sequence-to-one) ...")
    t0 = time.time()
    preds["LSTM"] = run_lstm_backtest(data, "model_target", test_start, epochs=30)
    print(f"  done in {time.time()-t0:.1f}s")

    # ---------------- invert log1p + sign back to real units ----------------
    preds = {name: invert_predictions(p, sign) for name, p in preds.items()}

    # ---------------- metrics ----------------
    print("\nComputing backtest metrics ...")
    rows = {}
    for name, p in preds.items():
        rows[name] = compute_metrics(actual_test, p)
    metrics_df = pd.DataFrame(rows).T[
        ["MAE", "RMSE", "sMAPE_%", "WAPE_%", "MAE_nonzero_hours", "Bias", "n_test"]
    ]
    metrics_df = metrics_df.sort_values("RMSE")
    print(metrics_df.round(3))
    metrics_df.to_csv(OUT_DIR / "backtest_metrics.csv")

    # ---------------- predictions dump ----------------
    pred_df = pd.DataFrame({"actual": actual_test})
    for name, p in preds.items():
        pred_df[name] = p
    pred_df.to_csv(OUT_DIR / "backtest_predictions.csv")

    # ---------------- plots ----------------
    print("\nGenerating plots ...")
    for name, p in preds.items():
        plot_model(actual_test, p, name, TARGET, OUT_DIR / f"backtest_{name.lower()}.png")
    plot_zoom(actual_test, preds, TARGET, OUT_DIR / "backtest_zoom_last4days.png", days=4)
    plot_metric_bars(metrics_df, OUT_DIR / "backtest_metric_comparison.png")

    print(f"\nAll outputs written to: {OUT_DIR}")
    print(f"Total runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()

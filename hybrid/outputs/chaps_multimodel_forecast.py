"""
1-hour-ahead multi-model forecasting for CHAPS hourly transaction data.

Models:
  1. Exponential Smoothing (ETS / Holt-Winters, additive, daily seasonality)
  2. SARIMAX (ARIMA + seasonal + calendar exogenous regressors)
  3. XGBoost (lag + rolling + calendar features)
  4. LSTM (PyTorch, sequence-to-one)

Target: transaction_count per hour (sum of the debit-leg and credit-leg rows
for that hour in the raw CHAPS export).

Backtest design (walk-forward, out-of-sample, last 14 days held out):
  - SARIMAX: true 1-step-ahead. Fit once on the training window, then for
    every test hour we forecast h=1 and then push the *actual* observed
    value into the filter (Kalman `append`, refit=False) before moving to
    the next hour. No refitting of parameters during the test window.
  - ETS: statsmodels' ETSModel has no incremental-update API, so we do a
    daily-refit rolling backtest: refit on all history available at the
    start of each test day, then generate a 24-hour-ahead path for that
    day. This is a heavier-but-standard compromise for classical
    exponential smoothing backtests.
  - XGBoost / LSTM: trained once on the training window. For every test
    hour, features/sequences are built from the *true* historical series
    (never from the model's own prior predictions), so these are also
    genuine 1-step-ahead evaluations.
"""

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
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path(
    r"C:\Users\reshm\OneDrive\Documents\work\timesfm\timesfm-forecasting\chaps_demo.csv"
)
OUT_DIR = Path(
    r"C:\Users\reshm\OneDrive\Documents\work\timesfm\hybrid\outputs\chaps_forecast_output"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "transaction_count"
TEST_HOURS = 24 * 14  # last 14 days held out for backtest
DAILY_SEASONAL_PERIOD = 24
LAGS = [1, 2, 3, 24, 48, 168]
EXOG_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
SEQ_LEN = 24
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------
def clean_amount(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace("-", "0")
    return s.astype(float).abs()


def load_hourly_series() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d-%m-%Y %H:%M")
    df["dr_amount"] = clean_amount(df["dr_amount"])
    df["cr_amount"] = clean_amount(df["cr_amount"])

    agg = (
        df.groupby("Datetime")
        .agg(
            transaction_count=("transaction_count", "sum"),
            dr_amount=("dr_amount", "sum"),
            cr_amount=("cr_amount", "sum"),
        )
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
    for i, ts in enumerate(test_index):
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
    preds = np.clip(preds, 0, None)
    return pd.Series(preds, index=test_idx, name="LSTM")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_model(actual: pd.Series, pred: pd.Series, name: str, out_path: Path):
    actual_a, pred_a = actual.align(pred, join="inner")
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(actual_a.index, actual_a.values, label="Actual", color="#333333", linewidth=1.2)
    ax.plot(pred_a.index, pred_a.values, label=f"{name} forecast", color="#e15759", linewidth=1.1, alpha=0.9)
    ax.set_title(f"{name}: 1-hour-ahead backtest — transaction_count (last {len(actual_a)} hours)")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("transaction_count")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_zoom(actual: pd.Series, preds: dict, out_path: Path, days: int = 4):
    end = actual.index.max()
    start = end - pd.Timedelta(hours=24 * days)
    fig, ax = plt.subplots(figsize=(14, 5))
    a = actual.loc[start:end]
    ax.plot(a.index, a.values, label="Actual", color="#222222", linewidth=1.8)
    colors = {"ETS": "#4e79a7", "SARIMAX": "#f28e2b", "XGBoost": "#59a14f", "LSTM": "#e15759"}
    for name, p in preds.items():
        p_a = p.loc[start:end]
        ax.plot(p_a.index, p_a.values, label=name, color=colors.get(name), linewidth=1.3, alpha=0.9)
    ax.set_title(f"All models — last {days} days of backtest (zoomed)")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("transaction_count")
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
    t_start = time.time()
    print("Loading & preprocessing data ...")
    data = load_hourly_series()
    data = add_calendar_features(data)
    print(f"  hourly series: {len(data)} rows, {data.index.min()} -> {data.index.max()}")

    n = len(data)
    test_start = n - TEST_HOURS
    actual_test = data[TARGET].iloc[test_start:]
    print(f"  train: {test_start} hours | test (backtest): {TEST_HOURS} hours "
          f"({data.index[test_start]} -> {data.index[-1]})")

    preds = {}

    print("\n[1/4] Exponential Smoothing (ETS, daily-refit rolling backtest) ...")
    t0 = time.time()
    preds["ETS"] = run_ets_backtest(data, TARGET, test_start)
    print(f"  done in {time.time()-t0:.1f}s")

    print("\n[2/4] SARIMAX (ARIMA + seasonal + calendar exog, true 1-step-ahead) ...")
    t0 = time.time()
    preds["SARIMAX"] = run_sarimax_backtest(data, TARGET, test_start)
    print(f"  done in {time.time()-t0:.1f}s")

    print("\n[3/4] XGBoost (lag + rolling + calendar features) ...")
    t0 = time.time()
    preds["XGBoost"] = run_xgb_backtest(data, TARGET, test_start)
    print(f"  done in {time.time()-t0:.1f}s")

    print("\n[4/4] LSTM (PyTorch, sequence-to-one) ...")
    t0 = time.time()
    preds["LSTM"] = run_lstm_backtest(data, TARGET, test_start, epochs=30)
    print(f"  done in {time.time()-t0:.1f}s")

    # ---------------- metrics ----------------
    print("\nComputing backtest metrics ...")
    rows = {}
    for name, p in preds.items():
        rows[name] = compute_metrics(actual_test, p)
    metrics_df = pd.DataFrame(rows).T[
        ["MAE", "RMSE", "sMAPE_%", "MAE_nonzero_hours", "Bias", "n_test"]
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
        plot_model(actual_test, p, name, OUT_DIR / f"backtest_{name.lower()}.png")
    plot_zoom(actual_test, preds, OUT_DIR / "backtest_zoom_last4days.png", days=4)
    plot_metric_bars(metrics_df, OUT_DIR / "backtest_metric_comparison.png")

    print(f"\nAll outputs written to: {OUT_DIR}")
    print(f"Total runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()

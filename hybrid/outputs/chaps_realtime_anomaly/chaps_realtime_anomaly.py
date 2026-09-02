"""
CHAPS Real-Time Anomaly Detection (production)

Hourly CHAPS transaction-count / dr_amount / cr_amount series. Every signal
below is computed walk-forward (point-in-time, no look-ahead) so the same
code path that produces this backtest can be run hour-by-hour in production:
each new hour is scored using only data that was available up to and
including that hour.

The raw series has strong hour-of-day and day-of-week seasonality (near-zero
overnight, high at midday) -- a naive global z-score would flag every
morning ramp-up as an anomaly. So the statistical signals first deseasonalize:
a causal seasonal baseline (rolling median of the last N occurrences of the
same hour-of-day / weekend-vs-weekday slot, using only *past* occurrences)
is subtracted from transaction_count before any distance-based statistic is
computed.

Five independent signals, each a real-time-capable detector:
  1. Seasonal Z-score   - residual (actual - seasonal baseline) standardised
                           by a trailing rolling std of that residual.
  2. EWMA control chart - exponentially-weighted mean of the residual vs.
                           its asymptotic control limits; reacts faster than
                           a rolling z-score to a sustained small drift,
                           which is the classic reason EWMA charts are used
                           for real-time process monitoring.
  3. CUSUM structural break - Page's CUSUM on the standardised residual,
                           with reset-on-signal (so a breach is an event,
                           not a sticky level) and a pre-alarm band that
                           fires before the full threshold is crossed --
                           this is the closest thing to genuine lead time
                           without a forecasting model.
  4. Isolation Forest   - multivariate: transaction_count, total_amount,
                           dr/cr imbalance, lag/diff/rolling-stat features,
                           week-over-week delta, calendar cyclical features.
                           Retrained on a periodic cadence on a trailing
                           window; flag threshold self-calibrated each
                           retrain from that window's own training scores
                           (so the flag rate tracks `expected_anomaly_rate`
                           instead of an arbitrary fixed cutoff).
  5. Local Outlier Factor (novelty mode) - same feature space as Isolation
                           Forest, retrained on the same cadence. LOF judges
                           local density rather than global isolation, so it
                           catches a different failure mode (a point that is
                           unremarkable globally but sits in a locally sparse
                           neighbourhood of its recent regime).

A point is a "Confirmed Anomaly" if >= vote_threshold of the 5 signals agree.
A point that trips the CUSUM pre-alarm band (building deviation, not yet a
confirmed breach) but isn't already a confirmed anomaly is "Building Stress"
-- the real-time early-signal tier.

Outputs (written to --out, default ./output):
  scored_series.csv       every hour with every signal + status
  confirmed_anomalies.csv rows where the ensemble confirms an anomaly
  building_stress.csv     rows in the CUSUM pre-alarm band only
  summary.json            machine-readable run summary
  dashboard.png           multi-panel diagnostic chart
  analysis_report.md      human-readable production analysis report
  run.log                 execution log

Use --simulate-stream to replay the scored series hour-by-hour and print
alerts as they would appear in a live monitor (demonstrates the same scoring
path running as a stream rather than a batch job).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = Path(
    r"C:\Users\reshm\OneDrive\Documents\work\timesfm\timesfm-forecasting\chaps_demo.csv"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass
class Config:
    data_path: Path = DEFAULT_DATA_PATH
    out_dir: Path = DEFAULT_OUT_DIR

    rolling_window_hours: int = 24 * 21     # 3-week trailing window for residual std / ML training
    seasonal_window_occ: int = 8            # trailing occurrences of same (hour, is_weekend) slot
    seasonal_min_occ: int = 4

    expected_anomaly_rate: float = 0.03     # ~3% of hours expected anomalous

    z_threshold: float = 3.0                # seasonal z-score

    ewma_lambda: float = 0.2                # EWMA smoothing weight
    ewma_l: float = 3.0                     # control-limit multiplier (sigma units)

    cusum_k: float = 0.5                    # slack, in sigma units
    cusum_h: float | None = None            # decision threshold; None = auto-calibrate
    cusum_warn_fraction: float = 0.6        # pre-alarm band = warn_fraction * H
    cusum_h_grid: tuple = (1.0, 20.0, 0.25)

    if_retrain_every: int = 24              # retrain Isolation Forest every N hours
    if_min_train: int = 24 * 10             # need >= 10 days of history to first fit
    if_n_estimators: int = 300

    lof_retrain_every: int = 24
    lof_min_train: int = 24 * 10
    lof_neighbors: int = 35

    vote_threshold: int = 3                 # out of 5 signals

    random_state: int = 42

    train_end: int = field(init=False, default=0)
    cusum_h_calibrated: float = field(init=False, default=0.0)

    def resolve(self, n_rows: int) -> None:
        self.train_end = max(int(n_rows * 0.4), self.if_min_train)


LOGGER = logging.getLogger("chaps_rt_anomaly")


def setup_logging(out_dir: Path, level: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-7s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(out_dir / "run.log", mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------
def clean_amount(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace("-", "0")
    return s.astype(float).abs()


def load_hourly_series(cfg: Config) -> pd.DataFrame:
    if not cfg.data_path.exists():
        raise FileNotFoundError(f"Input file not found: {cfg.data_path}")

    df = pd.read_csv(cfg.data_path)
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
    agg = agg.reset_index()

    if len(agg) < 24 * 30:
        raise ValueError(f"Only {len(agg)} hourly rows after cleaning; need at least 30 days.")

    LOGGER.info(
        "Loaded %d hourly rows: %s -> %s", len(agg), agg["Datetime"].min(), agg["Datetime"].max()
    )
    return agg


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["Datetime"].dt.hour
    df["dow"] = df["Datetime"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dr_cr_imbalance"] = (df["dr_amount"] - df["cr_amount"]) / (
        df["dr_amount"] + df["cr_amount"] + 1.0
    )
    df["tc_lag1"] = df["transaction_count"].shift(1)
    df["tc_diff1"] = df["transaction_count"].diff(1)
    df["tc_roll_mean_24"] = df["transaction_count"].shift(1).rolling(24).mean()
    df["tc_roll_std_24"] = df["transaction_count"].shift(1).rolling(24).std()
    df["tc_wow_diff"] = df["transaction_count"] - df["transaction_count"].shift(168)
    df["amt_roll_std_24"] = df["total_amount"].shift(1).rolling(24).std()
    return df


FEATURE_COLS = [
    "transaction_count", "total_amount", "dr_cr_imbalance",
    "tc_lag1", "tc_diff1", "tc_roll_mean_24", "tc_roll_std_24", "tc_wow_diff",
    "amt_roll_std_24",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
]


# ---------------------------------------------------------------------------
# Causal seasonal baseline (deseasonalized residual)
# ---------------------------------------------------------------------------
def seasonal_residual(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby(["hour", "is_weekend"])["transaction_count"]
    df["seasonal_baseline"] = grp.transform(
        lambda s: s.shift(1).rolling(cfg.seasonal_window_occ, min_periods=cfg.seasonal_min_occ).median()
    )
    df["residual"] = df["transaction_count"] - df["seasonal_baseline"]

    roll_std = df["residual"].shift(1).rolling(cfg.rolling_window_hours, min_periods=48).std()
    df["residual_roll_std"] = roll_std
    df["seasonal_z"] = df["residual"] / roll_std.replace(0, np.nan)
    df["z_flag"] = (df["seasonal_z"].abs() >= cfg.z_threshold).fillna(False)
    return df


# ---------------------------------------------------------------------------
# Signal: EWMA control chart on the seasonal residual
# ---------------------------------------------------------------------------
def ewma_signal(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    lam = cfg.ewma_lambda
    ewma = df["residual"].fillna(0.0).ewm(alpha=lam, adjust=False).mean()
    # asymptotic EWMA control-limit factor, scaled by the same trailing std
    # used for the seasonal z-score
    limit_factor = np.sqrt(lam / (2 - lam))
    ewma_limit = cfg.ewma_l * limit_factor * df["residual_roll_std"]

    df["ewma"] = ewma
    df["ewma_limit"] = ewma_limit
    df["ewma_flag"] = (ewma.abs() >= ewma_limit).fillna(False)
    return df


# ---------------------------------------------------------------------------
# Signal: CUSUM structural break (reset-on-signal + pre-alarm)
# ---------------------------------------------------------------------------
def simulate_cusum(dev: np.ndarray, K: float, H: float, warn_fraction: float):
    n = len(dev)
    cusum_up = np.zeros(n)
    cusum_dn = np.zeros(n)
    sb_flag = np.zeros(n, dtype=bool)
    sb_pre_alarm = np.zeros(n, dtype=bool)
    warn_level = warn_fraction * H

    for i in range(1, n):
        d = dev[i] if not np.isnan(dev[i]) else 0.0
        cusum_up[i] = max(0.0, cusum_up[i - 1] + d - K)
        cusum_dn[i] = max(0.0, cusum_dn[i - 1] - d - K)

        breached = (cusum_up[i] > H) or (cusum_dn[i] > H)
        sb_flag[i] = breached
        sb_pre_alarm[i] = (not breached) and (
            cusum_up[i] > warn_level or cusum_dn[i] > warn_level
        )
        if breached:
            cusum_up[i] = 0.0
            cusum_dn[i] = 0.0

    return cusum_up, cusum_dn, sb_flag, sb_pre_alarm


def calibrate_cusum_h(dev: np.ndarray, cfg: Config) -> float:
    train_dev = dev[: cfg.train_end]
    lo, hi, step = cfg.cusum_h_grid
    candidates = np.arange(lo, hi + 1e-9, step)
    best_h = candidates[-1]
    best_rate = None
    for h in candidates:
        _, _, flag, _ = simulate_cusum(train_dev, cfg.cusum_k, h, cfg.cusum_warn_fraction)
        rate = flag.mean()
        if rate <= cfg.expected_anomaly_rate:
            best_h = h
            best_rate = rate
            break
        best_rate = rate
    LOGGER.info(
        "CUSUM auto-calibration: K=%.2f -> H=%.2f (training flag rate %.1f%%, target %.1f%%)",
        cfg.cusum_k, best_h, best_rate * 100, cfg.expected_anomaly_rate * 100,
    )
    return float(best_h)


def cusum_structural_break(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    dev = df["seasonal_z"].to_numpy()
    H = cfg.cusum_h if cfg.cusum_h is not None else calibrate_cusum_h(dev, cfg)
    cusum_up, cusum_dn, sb_flag, sb_pre_alarm = simulate_cusum(dev, cfg.cusum_k, H, cfg.cusum_warn_fraction)

    df = df.copy()
    df["cusum_up"] = cusum_up
    df["cusum_dn"] = cusum_dn
    df["cusum_threshold"] = H
    df["cusum_warn_level"] = cfg.cusum_warn_fraction * H
    df["sb_flag"] = sb_flag
    df["sb_pre_alarm"] = sb_pre_alarm
    cfg.cusum_h_calibrated = H
    return df


# ---------------------------------------------------------------------------
# Signal: Isolation Forest (periodic retrain, self-calibrated threshold)
# ---------------------------------------------------------------------------
def isolation_forest_scores(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    feat = df[FEATURE_COLS]
    n = len(df)
    score = np.full(n, np.nan)
    threshold_used = np.full(n, np.nan)
    flag = np.zeros(n, dtype=bool)

    model = None
    scaler = None
    threshold = None

    for i in range(cfg.if_min_train, n):
        needs_retrain = model is None or (i - cfg.if_min_train) % cfg.if_retrain_every == 0
        if needs_retrain:
            win_start = max(0, i - cfg.rolling_window_hours)
            train_slice = feat.iloc[win_start:i].dropna()
            if len(train_slice) >= cfg.if_min_train:
                X_train = train_slice.values
                scaler = StandardScaler().fit(X_train)
                Xs = scaler.transform(X_train)
                model = IsolationForest(
                    n_estimators=cfg.if_n_estimators,
                    max_samples=min(256, len(Xs)),
                    contamination=cfg.expected_anomaly_rate,
                    random_state=cfg.random_state,
                    n_jobs=-1,
                ).fit(Xs)
                train_scores = -model.score_samples(Xs)
                threshold = float(np.quantile(train_scores, 1 - cfg.expected_anomaly_rate))
            else:
                model = None
                scaler = None
                threshold = None

        row = feat.iloc[[i]]
        if model is None or row.isna().any(axis=1).iloc[0]:
            continue
        x = scaler.transform(row.values)
        s = float(-model.score_samples(x)[0])
        score[i] = s
        threshold_used[i] = threshold
        flag[i] = s >= threshold

    df = df.copy()
    df["if_score"] = score
    df["if_threshold"] = threshold_used
    df["if_flag"] = flag
    return df


# ---------------------------------------------------------------------------
# Signal: Local Outlier Factor (novelty mode, periodic retrain)
# ---------------------------------------------------------------------------
def lof_scores(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    feat = df[FEATURE_COLS]
    n = len(df)
    score = np.full(n, np.nan)
    threshold_used = np.full(n, np.nan)
    flag = np.zeros(n, dtype=bool)

    model = None
    scaler = None
    threshold = None

    for i in range(cfg.lof_min_train, n):
        needs_retrain = model is None or (i - cfg.lof_min_train) % cfg.lof_retrain_every == 0
        if needs_retrain:
            win_start = max(0, i - cfg.rolling_window_hours)
            train_slice = feat.iloc[win_start:i].dropna()
            if len(train_slice) >= cfg.lof_min_train:
                X_train = train_slice.values
                scaler = StandardScaler().fit(X_train)
                Xs = scaler.transform(X_train)
                k = min(cfg.lof_neighbors, len(Xs) - 1)
                model = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Xs)
                train_scores = -model.decision_function(Xs)
                threshold = float(np.quantile(train_scores, 1 - cfg.expected_anomaly_rate))
            else:
                model = None
                scaler = None
                threshold = None

        row = feat.iloc[[i]]
        if model is None or row.isna().any(axis=1).iloc[0]:
            continue
        x = scaler.transform(row.values)
        s = float(-model.decision_function(x)[0])
        score[i] = s
        threshold_used[i] = threshold
        flag[i] = s >= threshold

    df = df.copy()
    df["lof_score"] = score
    df["lof_threshold"] = threshold_used
    df["lof_flag"] = flag
    return df


# ---------------------------------------------------------------------------
# Ensemble + status
# ---------------------------------------------------------------------------
def build_ensemble(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    flag_cols = ["z_flag", "ewma_flag", "sb_flag", "if_flag", "lof_flag"]
    for c in flag_cols:
        df[c] = df[c].fillna(False)
    df["votes"] = df[flag_cols].sum(axis=1).astype(int)
    df["is_anomaly"] = df["votes"] >= cfg.vote_threshold

    df["combined_score"] = (
        (df["seasonal_z"].abs() / (cfg.z_threshold * 2)).clip(0, 1).fillna(0) * 0.20
        + (df["ewma"].abs() / df["ewma_limit"].replace(0, np.nan)).clip(0, 1).fillna(0) * 0.15
        + df["sb_flag"].astype(float) * 0.20
        + df["if_score"].fillna(0).rank(pct=True) * 0.25
        + df["lof_score"].fillna(0).rank(pct=True) * 0.20
    )

    df["building_stress"] = df["sb_pre_alarm"] & ~df["is_anomaly"]

    def status(row) -> str:
        if row["is_anomaly"]:
            return "Confirmed Anomaly"
        if row["building_stress"]:
            return "Building Stress"
        if row["votes"] >= max(cfg.vote_threshold - 1, 1):
            return "Watch"
        return "Normal"

    df["status"] = df.apply(status, axis=1)
    df["cum_confirmed_anomalies"] = df["is_anomaly"].cumsum()
    df["cum_building_stress"] = df["building_stress"].cumsum()
    return df, flag_cols


# ---------------------------------------------------------------------------
# Live-stream simulation (demonstrates the same scoring path running hour by
# hour instead of as a single batch pass)
# ---------------------------------------------------------------------------
def simulate_stream(df: pd.DataFrame, delay_s: float = 0.0, tail: int | None = None) -> None:
    rows = df if tail is None else df.tail(tail)
    LOGGER.info("Starting live-stream simulation over %d hours ...", len(rows))
    for _, row in rows.iterrows():
        if row["status"] != "Normal":
            LOGGER.info(
                "[%s] %-18s votes=%d/5 tc=%.0f amt=%.0f combined_score=%.2f",
                row["Datetime"], row["status"], row["votes"],
                row["transaction_count"], row["total_amount"], row["combined_score"],
            )
        if delay_s:
            time.sleep(delay_s)
    LOGGER.info("Live-stream simulation complete.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_dashboard(df: pd.DataFrame, cfg: Config, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.style.use("dark_background")
    FG, BG, GRID = "#e6edf3", "#0d1117", "#21262d"
    RED, BLUE, GREEN, AMBER, PURPLE = "#f85149", "#58a6ff", "#3fb950", "#d29922", "#d2a8ff"

    fig = plt.figure(figsize=(16, 22), facecolor=BG)
    fig.suptitle("CHAPS Real-Time Anomaly Detection (transaction_count, hourly)",
                 color=FG, fontsize=14, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.6, wspace=0.28,
                            top=0.965, bottom=0.03, left=0.045, right=0.98)

    def style_ax(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=FG, fontsize=10, pad=6)
        ax.tick_params(colors=FG, labelsize=7)
        for s in ax.spines.values():
            s.set_edgecolor(GRID)

    dt = df["Datetime"]

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dt, df["transaction_count"], color=BLUE, lw=0.6, alpha=0.7, label="transaction_count")
    conf = df[df["is_anomaly"]]
    watch = df[(df["status"] == "Building Stress")]
    ax1.scatter(conf["Datetime"], conf["transaction_count"], color=RED, s=18, zorder=5,
                label=f"Confirmed anomaly (n={len(conf)})")
    ax1.scatter(watch["Datetime"], watch["transaction_count"], color=AMBER, s=10, zorder=4,
                marker="^", label=f"Building stress (n={len(watch)})")
    ax1.legend(fontsize=8, facecolor="#161b22", edgecolor=GRID, labelcolor=FG)
    style_ax(ax1, "Raw Series with Confirmed Anomalies & Building-Stress Flags")

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(dt, df["residual"], color=PURPLE, lw=0.5, alpha=0.8)
    ax2.axhline(0, color=GRID, lw=0.6)
    style_ax(ax2, "Seasonally-Adjusted Residual (actual - causal seasonal baseline)")

    ax3 = fig.add_subplot(gs[2, 0])
    colors_z = [RED if f else BLUE for f in df["z_flag"]]
    ax3.bar(dt, df["seasonal_z"].fillna(0), color=colors_z, width=0.03, alpha=0.85)
    ax3.axhline(cfg.z_threshold, color=RED, lw=1.0, linestyle="--")
    ax3.axhline(-cfg.z_threshold, color=RED, lw=1.0, linestyle="--")
    style_ax(ax3, "Seasonal Z-Score")

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(dt, df["ewma"], color=GREEN, lw=0.8, label="EWMA")
    ax4.plot(dt, df["ewma_limit"], color=RED, lw=0.8, linestyle="--", label="+limit")
    ax4.plot(dt, -df["ewma_limit"], color=RED, lw=0.8, linestyle="--")
    ax4.legend(fontsize=7, facecolor="#161b22", edgecolor=GRID, labelcolor=FG)
    style_ax(ax4, "EWMA Control Chart")

    ax5 = fig.add_subplot(gs[3, 0])
    ax5.plot(dt, df["cusum_up"], color=RED, lw=0.7, label="CUSUM+")
    ax5.plot(dt, df["cusum_dn"], color=BLUE, lw=0.7, label="CUSUM-")
    ax5.axhline(cfg.cusum_h_calibrated, color=RED, lw=1.0, linestyle="--", label="H")
    ax5.axhline(cfg.cusum_h_calibrated * cfg.cusum_warn_fraction, color=AMBER, lw=1.0, linestyle=":",
                label="pre-alarm")
    ax5.legend(fontsize=7, facecolor="#161b22", edgecolor=GRID, labelcolor=FG)
    style_ax(ax5, "CUSUM (resets on signal)")

    ax6 = fig.add_subplot(gs[3, 1])
    colors_if = [RED if f else PURPLE for f in df["if_flag"]]
    ax6.bar(dt, df["if_score"].fillna(0), color=colors_if, width=0.03, alpha=0.85)
    ax6.plot(dt, df["if_threshold"], color=AMBER, lw=0.8, linestyle="--")
    style_ax(ax6, "Isolation Forest Score (self-calibrated threshold)")

    ax7 = fig.add_subplot(gs[4, 0])
    colors_lof = [RED if f else "#39d0d8" for f in df["lof_flag"]]
    ax7.bar(dt, df["lof_score"].fillna(0), color=colors_lof, width=0.03, alpha=0.85)
    ax7.plot(dt, df["lof_threshold"], color=AMBER, lw=0.8, linestyle="--")
    style_ax(ax7, "Local Outlier Factor Score (self-calibrated threshold)")

    ax8 = fig.add_subplot(gs[4, 1])
    ax8.plot(dt, df["combined_score"], color=GREEN, lw=0.7)
    ax8.fill_between(dt, 0, df["combined_score"], color=GREEN, alpha=0.2)
    style_ax(ax8, "Combined Weighted Anomaly Score")

    ax9 = fig.add_subplot(gs[5, :])
    ax9.plot(dt, df["cum_confirmed_anomalies"], color=RED, lw=1.1, label="Cumulative confirmed anomalies")
    ax9.plot(dt, df["cum_building_stress"], color=AMBER, lw=1.1, label="Cumulative building-stress hours")
    ax9.legend(fontsize=8, facecolor="#161b22", edgecolor=GRID, labelcolor=FG)
    style_ax(ax9, "Cumulative Counts")

    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=130, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(df: pd.DataFrame, cfg: Config, flag_cols: list, out_path: Path) -> None:
    n = len(df)
    n_conf = int(df["is_anomaly"].sum())
    n_stress = int(df["building_stress"].sum())

    flag_counts = {
        "Seasonal z-score": int(df["z_flag"].sum()),
        "EWMA control chart": int(df["ewma_flag"].sum()),
        "CUSUM structural break": int(df["sb_flag"].sum()),
        "Isolation Forest": int(df["if_flag"].sum()),
        "Local Outlier Factor": int(df["lof_flag"].sum()),
    }

    last_row = df.iloc[-1]

    lines = []
    lines.append("# CHAPS Real-Time Anomaly Detection - Production Analysis Report\n")
    lines.append(f"**Data period:** {df['Datetime'].min()} -> {df['Datetime'].max()}  ")
    lines.append(f"**Hours analysed:** {n}  ")
    lines.append(f"**Current status (latest hour, {last_row['Datetime']}):** **{last_row['status']}**\n")

    lines.append("## 1. Executive Summary\n")
    lines.append(
        f"- {n_conf} confirmed anomalies out of {n} hours ({n_conf / n * 100:.2f}%), "
        f"vs. an assumed base rate of {cfg.expected_anomaly_rate * 100:.0f}%.\n"
        f"- {n_stress} hours sat in the CUSUM 'Building Stress' pre-alarm band "
        f"(cumulative deviation above {cfg.cusum_warn_fraction:.0%} of H but below full threshold) "
        f"without also being a confirmed anomaly.\n"
        f"- Ensemble requires >= {cfg.vote_threshold}/5 independent signals to agree before "
        f"confirming, which keeps false positives from any single noisy signal from driving alerts."
    )

    lines.append("\n## 2. Why Seasonal Adjustment Matters Here\n")
    lines.append(
        "CHAPS transaction_count is near-zero overnight and peaks mid-day, with a distinct "
        "weekday-vs-weekend profile. Feeding the raw series into a global z-score would flag "
        "every normal morning ramp-up as extreme. Instead, each hour's expected value is a "
        "causal rolling median of the last "
        f"{cfg.seasonal_window_occ} occurrences of that same (hour-of-day, weekend/weekday) "
        "slot -- using only slots that occurred *before* the current hour, so this is safe to "
        "run in production without look-ahead. The seasonal z-score, EWMA chart and CUSUM all "
        "operate on this deseasonalized residual, not the raw count."
    )

    lines.append("\n## 3. Signal-by-Signal Flag Rates\n")
    lines.append("| Signal | Flags | Rate |")
    lines.append("|---|---|---|")
    for k, v in flag_counts.items():
        lines.append(f"| {k} | {v} | {v / n * 100:.2f}% |")
    lines.append(
        f"| **Ensemble (>= {cfg.vote_threshold}/5 agree)** | **{n_conf}** | **{n_conf / n * 100:.2f}%** |"
    )
    lines.append(
        "\nIsolation Forest and Local Outlier Factor both look at the same multivariate feature "
        "space (transaction_count, total_amount, dr/cr imbalance, lag/diff/rolling stats, "
        "week-over-week delta, calendar cyclicals) but score anomalousness differently -- Isolation "
        "Forest by how few random splits it takes to isolate a point, LOF by local density relative "
        "to neighbours -- so they tend to disagree on borderline points and agree on the clearest "
        "ones, which is exactly the redundancy the ensemble vote is designed to exploit."
    )

    lines.append("\n## 4. Current Regime\n")
    lines.append(
        f"- CUSUM confirmation threshold H (auto-calibrated): {cfg.cusum_h_calibrated:.2f}\n"
        f"- Latest combined score: {last_row['combined_score']:.3f}\n"
        f"- Latest votes: {int(last_row['votes'])}/5 "
        f"({', '.join(c.replace('_flag','') for c in flag_cols if last_row[c])})"
    )

    lines.append("\n## 5. Production Recommendations\n")
    lines.append(
        "- Run this script (or the underlying scoring functions) on each new hourly CHAPS batch "
        "as it lands; alert on `status` transitioning to `Confirmed Anomaly` or `Building Stress`.\n"
        "- Re-validate `expected_anomaly_rate` periodically against realised anomaly frequency -- "
        "it drives the Isolation Forest / LOF thresholds and the CUSUM calibration.\n"
        "- Keep `run.log` and `scored_series.csv` for audit trail.\n"
        "- If a new payment corridor or seasonality shift changes the baseline, shrink "
        "`rolling_window_hours` temporarily so the seasonal baseline and ML training windows "
        "adapt faster to the new regime.\n"
        "- Use `--simulate-stream` to sanity-check how the same scoring path behaves as a live "
        "feed before wiring it into a scheduler."
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> tuple[Config, argparse.Namespace]:
    p = argparse.ArgumentParser(description="CHAPS real-time anomaly detection")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--rolling-window-hours", type=int, default=24 * 21)
    p.add_argument("--expected-anomaly-rate", type=float, default=0.03)
    p.add_argument("--z-threshold", type=float, default=3.0)
    p.add_argument("--vote-threshold", type=int, default=3)
    p.add_argument("--cusum-h", type=float, default=None)
    p.add_argument("--simulate-stream", action="store_true",
                    help="Replay the scored series hour-by-hour and print alerts as a live monitor would.")
    p.add_argument("--stream-tail", type=int, default=24 * 14,
                    help="Only replay the last N hours in --simulate-stream (default: last 14 days).")
    p.add_argument("--stream-delay", type=float, default=0.0,
                    help="Seconds to sleep between hours in --simulate-stream (default: 0, no delay).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    cfg = Config(
        data_path=args.data,
        out_dir=args.out,
        rolling_window_hours=args.rolling_window_hours,
        expected_anomaly_rate=args.expected_anomaly_rate,
        z_threshold=args.z_threshold,
        vote_threshold=args.vote_threshold,
        cusum_h=args.cusum_h,
    )
    return cfg, args


def main(argv=None) -> int:
    cfg, args = parse_args(argv)
    setup_logging(cfg.out_dir, args.log_level)

    try:
        df = load_hourly_series(cfg)
        cfg.resolve(len(df))
        LOGGER.info(
            "Config: rolling_window_hours=%d expected_anomaly_rate=%.3f z_threshold=%.2f "
            "vote_threshold=%d/5",
            cfg.rolling_window_hours, cfg.expected_anomaly_rate, cfg.z_threshold, cfg.vote_threshold,
        )

        df = add_calendar_features(df)
        df = add_ml_features(df)

        LOGGER.info("Signal 1/5: seasonal z-score (causal deseasonalized residual)")
        df = seasonal_residual(df, cfg)

        LOGGER.info("Signal 2/5: EWMA control chart")
        df = ewma_signal(df, cfg)

        LOGGER.info("Signal 3/5: CUSUM structural break (reset-on-signal + pre-alarm)")
        df = cusum_structural_break(df, cfg)

        LOGGER.info("Signal 4/5: Isolation Forest (retrain every %d hours)", cfg.if_retrain_every)
        df = isolation_forest_scores(df, cfg)

        LOGGER.info("Signal 5/5: Local Outlier Factor (novelty, retrain every %d hours)", cfg.lof_retrain_every)
        df = lof_scores(df, cfg)

        LOGGER.info("Building ensemble + status")
        df, flag_cols = build_ensemble(df, cfg)

        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.out_dir / "scored_series.csv", index=False)
        df[df["is_anomaly"]].to_csv(cfg.out_dir / "confirmed_anomalies.csv", index=False)
        df[df["building_stress"]].to_csv(cfg.out_dir / "building_stress.csv", index=False)

        summary = {
            "n_rows": int(len(df)),
            "date_range": [str(df["Datetime"].min()), str(df["Datetime"].max())],
            "config": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in cfg.__dict__.items()
            },
            "flag_counts": {c: int(df[c].sum()) for c in flag_cols},
            "n_confirmed_anomalies": int(df["is_anomaly"].sum()),
            "n_building_stress": int(df["building_stress"].sum()),
            "latest_status": df.iloc[-1]["status"],
        }
        (cfg.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

        LOGGER.info("Generating dashboard")
        make_dashboard(df, cfg, cfg.out_dir / "dashboard.png")

        LOGGER.info("Writing analysis report")
        write_report(df, cfg, flag_cols, cfg.out_dir / "analysis_report.md")

        LOGGER.info("Done. Outputs in %s", cfg.out_dir)

        if args.simulate_stream:
            simulate_stream(df, delay_s=args.stream_delay, tail=args.stream_tail)

        return 0
    except Exception:
        LOGGER.exception("Run failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

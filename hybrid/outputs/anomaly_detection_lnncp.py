"""
Anomaly detection (statistical + ML) and 2-day-ahead forecast for daily lnncp.

Statistical methods (on the raw series and on a rolling local window):
  1. Global Z-score            (|z| > 3)
  2. Modified Z-score / MAD    (robust to outliers, |mz| > 3.5)
  3. IQR (Tukey fences)        (outside Q1-1.5*IQR .. Q3+1.5*IQR)
  4. Rolling Z-score           (30-day centered window, |z| > 3) -- catches
     local anomalies that global stats miss because this series genuinely
     oscillates between a "small dip" and "large dip" regime every few days.

ML methods (unsupervised, features = value/lag/diff/rolling stats/day-of-week):
  5. Isolation Forest
  6. Local Outlier Factor

A point is flagged as a final ensemble anomaly if >= 2 of the 6 methods
agree. In practice, on this series the global/robust statistical methods
(z-score, modified z-score, IQR) find zero outliers -- every daily value
falls within its own 3-sigma / IQR-fence range because the series
naturally oscillates between a "small dip" and "large dip" regime rather
than producing rare extreme spikes. The signal comes from the rolling
z-score and, mainly, the ML methods (Isolation Forest, LOF), which judge
each day against its recent lag/diff/rolling-stat/day-of-week context
instead of the whole-history distribution.

Forecast (next 2 days):
  - SARIMAX(2,0,2)x(1,0,1,7) with day-of-week (sin/cos) exogenous regressors
  - Holt-Winters exponential smoothing (additive, 7-day seasonality)
  - Seasonal-naive (mean of the same weekday over the last 4 weeks)
  - Ensemble forecast = mean of the three
  Each forecast day is then checked against the historical IQR fences and
  modified Z-score to flag whether the predicted value would itself be
  anomalous relative to history.
"""

import json
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
    r"C:\Users\reshm\OneDrive\Documents\work\timesfm\hybrid\outputs\anomaly_detection_input.csv"
)
OUT_DIR = Path(
    r"C:\Users\reshm\OneDrive\Documents\work\timesfm\hybrid\outputs\anomaly_output"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "lnncp"
ROLL_WINDOW = 30
FORECAST_HORIZON = 2
CONTAMINATION = 0.03
VOTE_THRESHOLD = 2  # out of 6 methods -- see note below
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_series() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df["Period"] = pd.to_datetime(df["Period"], dayfirst=True)
    df = df.sort_values("Period").drop_duplicates("Period").set_index("Period")
    df.index.name = "date"

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_idx)
    df.index.name = "date"
    df[TARGET] = df[TARGET].interpolate(method="time")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dow = df.index.dayofweek
    df["dow"] = dow
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


# ---------------------------------------------------------------------------
# Statistical anomaly detection
# ---------------------------------------------------------------------------
def stat_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    y = df[TARGET]
    out = pd.DataFrame(index=df.index)

    mean, std = y.mean(), y.std()
    out["zscore"] = (y - mean) / std
    out["flag_zscore"] = out["zscore"].abs() > 3

    median = y.median()
    mad = (y - median).abs().median()
    mad_scale = 1.4826 * mad
    out["modified_zscore"] = (y - median) / mad_scale
    out["flag_modified_zscore"] = out["modified_zscore"].abs() > 3.5

    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out["flag_iqr"] = (y < lower) | (y > upper)

    roll_mean = y.rolling(ROLL_WINDOW, min_periods=10, center=True).mean()
    roll_std = y.rolling(ROLL_WINDOW, min_periods=10, center=True).std()
    out["rolling_zscore"] = (y - roll_mean) / roll_std
    out["flag_rolling_zscore"] = out["rolling_zscore"].abs() > 3

    bounds = {
        "global_mean": float(mean),
        "global_std": float(std),
        "median": float(median),
        "mad_scale": float(mad_scale),
        "iqr_lower": float(lower),
        "iqr_upper": float(upper),
    }
    return out, bounds


# ---------------------------------------------------------------------------
# ML anomaly detection
# ---------------------------------------------------------------------------
def ml_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    feat = df.copy()
    feat["lag1"] = feat[TARGET].shift(1)
    feat["diff1"] = feat[TARGET].diff(1)
    feat["roll_mean_7"] = feat[TARGET].shift(1).rolling(7).mean()
    feat["roll_std_7"] = feat[TARGET].shift(1).rolling(7).std()
    feature_cols = [TARGET, "lag1", "diff1", "roll_mean_7", "roll_std_7", "dow_sin", "dow_cos"]
    feat = feat.dropna(subset=feature_cols)

    X = StandardScaler().fit_transform(feat[feature_cols])

    iso = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=RANDOM_STATE)
    iso_pred = iso.fit_predict(X)
    iso_score = -iso.score_samples(X)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION)
    lof_pred = lof.fit_predict(X)
    lof_score = -lof.negative_outlier_factor_

    out = pd.DataFrame(index=feat.index)
    out["iso_forest_score"] = iso_score
    out["flag_iso_forest"] = iso_pred == -1
    out["lof_score"] = lof_score
    out["flag_lof"] = lof_pred == -1
    return out


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
def build_ensemble(stat_df: pd.DataFrame, ml_df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    combined = stat_df.join(ml_df, how="left")
    flag_cols = [c for c in combined.columns if c.startswith("flag_")]
    combined[flag_cols] = combined[flag_cols].fillna(False)
    combined["vote_count"] = combined[flag_cols].sum(axis=1).astype(int)
    combined["n_methods"] = len(flag_cols)
    combined["is_anomaly"] = combined["vote_count"] >= VOTE_THRESHOLD
    return combined, flag_cols


# ---------------------------------------------------------------------------
# Forecast (next 2 days)
# ---------------------------------------------------------------------------
def seasonal_naive_forecast(y: pd.Series, future_idx: pd.DatetimeIndex, weeks: int = 4) -> pd.Series:
    preds = []
    for d in future_idx:
        vals = [y.get(d - pd.Timedelta(days=7 * k)) for k in range(1, weeks + 1)]
        vals = [v for v in vals if v is not None and not pd.isna(v)]
        preds.append(float(np.mean(vals)) if vals else float(y.iloc[-7:].mean()))
    return pd.Series(preds, index=future_idx)


def forecast_next_days(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = df[TARGET].astype(float)
    exog = df[["dow_sin", "dow_cos"]]

    last_date = df.index.max()
    future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_dow = future_idx.dayofweek
    future_exog = pd.DataFrame(
        {"dow_sin": np.sin(2 * np.pi * future_dow / 7), "dow_cos": np.cos(2 * np.pi * future_dow / 7)},
        index=future_idx,
    )

    sarimax_fit = SARIMAX(
        y,
        exog=exog,
        order=(2, 0, 2),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    sarimax_fc = sarimax_fit.get_forecast(horizon, exog=future_exog)
    sarimax_mean = sarimax_fc.predicted_mean
    sarimax_ci = sarimax_fc.conf_int(alpha=0.05)

    hw_fit = ExponentialSmoothing(y, trend=None, seasonal="add", seasonal_periods=7).fit()
    hw_fc = hw_fit.forecast(horizon)

    naive_fc = seasonal_naive_forecast(y, future_idx)

    out = pd.DataFrame(index=future_idx)
    out["sarimax_forecast"] = sarimax_mean.values
    out["sarimax_lower95"] = sarimax_ci.iloc[:, 0].values
    out["sarimax_upper95"] = sarimax_ci.iloc[:, 1].values
    out["holtwinters_forecast"] = hw_fc.values
    out["seasonal_naive_forecast"] = naive_fc.values
    out["ensemble_forecast"] = out[
        ["sarimax_forecast", "holtwinters_forecast", "seasonal_naive_forecast"]
    ].mean(axis=1)
    out.index.name = "date"
    return out


def flag_forecast_anomalies(forecast_df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    df = forecast_df.copy()
    val = df["ensemble_forecast"]
    df["modified_zscore"] = (val - bounds["median"]) / bounds["mad_scale"]
    df["outside_iqr_fences"] = (val < bounds["iqr_lower"]) | (val > bounds["iqr_upper"])
    df["outside_modified_z"] = df["modified_zscore"].abs() > 3.5

    def risk(row):
        if row["outside_modified_z"] or row["outside_iqr_fences"]:
            return "Likely anomaly"
        if abs(row["modified_zscore"]) > 2.5:
            return "Watch"
        return "Normal"

    df["risk_level"] = df.apply(risk, axis=1)
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_full_series(df: pd.DataFrame, ensemble: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df.index, df[TARGET], color="#4e79a7", linewidth=0.9, label="lnncp")
    anoms = ensemble[ensemble["is_anomaly"]]
    ax.scatter(anoms.index, df.loc[anoms.index, TARGET], color="#e15759", s=28, zorder=5,
               label=f"Ensemble anomaly (n={len(anoms)})")
    ax.set_title("lnncp -- full history with ensemble-flagged anomalies")
    ax.set_xlabel("date")
    ax.set_ylabel("lnncp")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_recent_with_forecast(df: pd.DataFrame, ensemble: pd.DataFrame, forecast_df: pd.DataFrame,
                               out_path: Path, days: int = 90):
    start = df.index.max() - pd.Timedelta(days=days)
    hist = df.loc[start:]
    ens = ensemble.loc[start:]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hist.index, hist[TARGET], color="#4e79a7", linewidth=1.3, label="lnncp (history)")
    anoms = ens[ens["is_anomaly"]]
    ax.scatter(anoms.index, hist.loc[anoms.index, TARGET], color="#e15759", s=40, zorder=5,
               label="Ensemble anomaly")

    ax.plot(forecast_df.index, forecast_df["ensemble_forecast"], color="#59a14f", marker="o",
            linewidth=1.6, label="Ensemble forecast (next 2 days)")
    ax.fill_between(forecast_df.index, forecast_df["sarimax_lower95"], forecast_df["sarimax_upper95"],
                     color="#59a14f", alpha=0.2, label="SARIMAX 95% CI")
    for d, row in forecast_df.iterrows():
        if row["risk_level"] != "Normal":
            ax.scatter([d], [row["ensemble_forecast"]], color="#f28e2b", s=110, marker="*",
                       zorder=6, label=f"Forecast risk: {row['risk_level']}")

    ax.set_title(f"lnncp -- last {days} days + 2-day forecast")
    ax.set_xlabel("date")
    ax.set_ylabel("lnncp")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_method_agreement(ensemble: pd.DataFrame, flag_cols: list, out_path: Path):
    counts = ensemble[flag_cols].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values, color="#4e79a7")
    ax.set_title("Anomalies flagged per method")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading & preparing data ...")
    df = load_series()
    df = add_calendar_features(df)
    print(f"  {len(df)} daily rows, {df.index.min().date()} -> {df.index.max().date()}")

    print("\nRunning statistical anomaly detection (z-score, modified z-score/MAD, IQR, rolling z-score) ...")
    stat_df, bounds = stat_anomalies(df)

    print("Running ML anomaly detection (Isolation Forest, Local Outlier Factor) ...")
    ml_df = ml_anomalies(df)

    ensemble, flag_cols = build_ensemble(stat_df, ml_df)
    n_anom = int(ensemble["is_anomaly"].sum())
    print(f"\nMethod flag counts:\n{ensemble[flag_cols].sum().to_string()}")
    print(f"\nEnsemble anomalies (>= {VOTE_THRESHOLD}/{len(flag_cols)} methods agree): {n_anom} "
          f"of {len(ensemble)} days ({n_anom/len(ensemble)*100:.1f}%)")

    full = df.join(ensemble)
    full.to_csv(OUT_DIR / "lnncp_with_anomaly_scores.csv")
    full[full["is_anomaly"]].to_csv(OUT_DIR / "anomalies_detected.csv")

    print("\nForecasting next 2 days (SARIMAX + Holt-Winters + seasonal-naive ensemble) ...")
    forecast_df = forecast_next_days(df, FORECAST_HORIZON)
    forecast_df = flag_forecast_anomalies(forecast_df, bounds)
    forecast_df.to_csv(OUT_DIR / "forecast_next_2_days.csv")
    print(forecast_df.round(2).to_string())

    print("\nGenerating plots ...")
    plot_full_series(df, ensemble, OUT_DIR / "plot_full_series_anomalies.png")
    plot_recent_with_forecast(df, ensemble, forecast_df, OUT_DIR / "plot_recent_with_forecast.png")
    plot_method_agreement(ensemble, flag_cols, OUT_DIR / "plot_method_agreement.png")

    summary = {
        "n_days": int(len(df)),
        "date_range": [str(df.index.min().date()), str(df.index.max().date())],
        "historical_bounds": bounds,
        "methods": flag_cols,
        "vote_threshold": VOTE_THRESHOLD,
        "n_anomalies": n_anom,
        "anomaly_rate_pct": round(n_anom / len(ensemble) * 100, 2),
        "forecast": forecast_df.reset_index().assign(
            date=lambda d: d["date"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    print(f"\nAll outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

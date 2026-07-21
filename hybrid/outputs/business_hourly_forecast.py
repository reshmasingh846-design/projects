"""
Synthetic hourly bank-statement generator + business seasonality analysis +
next-day debit/credit forecast, per business line.

This mirrors the shape of a real intraday statement export (Effective
Timestamp, business unit, debit/credit amounts) but with fully synthetic
values -- no real figures are used. It is a simplified stand-in dataset for
prototyping the pipeline described below.

Pipeline:
  1. Generate hourly synthetic data for Apr-Jun 2026, one row per
     (hour, business), with separate Debit / Credit amounts. Each business
     has its own intraday shape (morning/afternoon/evening peaks) and
     weekly shape (weekday vs weekend).
  2. Morning / Afternoon / Evening / Night activity breakdown per business
     per side (debit, credit).
  3. Seasonality: hour-of-day profile, day-of-week profile, and an
     hour x day-of-week heatmap, per business.
  4. Next-day forecast: for each (business, side) series, SARIMAX with
     hour/day-of-week cyclical exogenous regressors + a same-hour-last-week
     seasonal-naive baseline, ensembled, forecasting all 24 hours of the
     next calendar day and summed to a daily total.

Businesses: Corporate Bank, FIC Trading, Corporate Cash Management
  - Corporate Bank: two peaks (morning settlement ~10:00, afternoon
    payment run ~15:00), quiet nights/weekends.
  - FIC Trading: three peaks aligned with Tokyo/London/NY session
    overlaps (~09:00, ~15:00, ~20:00), some residual weekend activity.
  - Corporate Cash Management: single sharp end-of-day cash-sweep peak
    (~17:30), otherwise low, essentially closed on weekends.
"""

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
OUT_DIR = Path(
    r"C:\Users\reshm\OneDrive\Documents\work\timesfm\hybrid\outputs\business_activity_output"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2026-04-01 00:00")
END = pd.Timestamp("2026-06-30 23:00")
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

BUSINESSES = ["Corporate Bank", "FIC Trading", "Corporate Cash Management"]
SIDES = ["Debit", "Credit"]

# (peak_hours, peak_widths, peak_weights, base_level, noise_sigma, weekend_factor,
#  jumbo_prob, jumbo_mult_range)
BUSINESS_PROFILE = {
    "Corporate Bank": {
        "Debit":  {"peaks": [10, 15], "widths": [1.6, 1.8], "weights": [1.0, 0.85],
                   "base": 5_000_000, "sigma": 0.30, "weekend_factor": 0.06,
                   "jumbo_prob": 0.015, "jumbo_range": (5, 10)},
        "Credit": {"peaks": [9, 14],  "widths": [1.6, 1.8], "weights": [1.0, 0.9],
                   "base": 5_200_000, "sigma": 0.30, "weekend_factor": 0.06,
                   "jumbo_prob": 0.015, "jumbo_range": (5, 10)},
    },
    "FIC Trading": {
        "Debit":  {"peaks": [9, 15, 20], "widths": [1.3, 1.5, 1.5], "weights": [0.9, 1.0, 0.75],
                   "base": 8_000_000, "sigma": 0.35, "weekend_factor": 0.15,
                   "jumbo_prob": 0.02, "jumbo_range": (4, 9)},
        "Credit": {"peaks": [9, 15, 20], "widths": [1.3, 1.5, 1.5], "weights": [0.9, 1.0, 0.8],
                   "base": 7_800_000, "sigma": 0.35, "weekend_factor": 0.15,
                   "jumbo_prob": 0.02, "jumbo_range": (4, 9)},
    },
    "Corporate Cash Management": {
        "Debit":  {"peaks": [17.5], "widths": [1.0], "weights": [1.0],
                   "base": 3_000_000, "sigma": 0.25, "weekend_factor": 0.02,
                   "jumbo_prob": 0.01, "jumbo_range": (3, 6)},
        "Credit": {"peaks": [17.5], "widths": [1.0], "weights": [1.0],
                   "base": 3_100_000, "sigma": 0.25, "weekend_factor": 0.02,
                   "jumbo_prob": 0.01, "jumbo_range": (3, 6)},
    },
}

SEGMENT_BOUNDS = {
    "Morning": (6, 12),    # 06:00-11:59
    "Afternoon": (12, 17), # 12:00-16:59
    "Evening": (17, 21),   # 17:00-20:59
    "Night": (21, 30),     # 21:00-05:59 (wraps past midnight, handled specially)
}


# ---------------------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------------------
def intraday_shape(hours: np.ndarray, peaks, widths, weights) -> np.ndarray:
    shape = np.zeros_like(hours, dtype=float)
    for peak, width, weight in zip(peaks, widths, weights):
        shape += weight * np.exp(-0.5 * ((hours - peak) / width) ** 2)
    return shape + 0.03  # small always-on floor


def generate_synthetic_data() -> pd.DataFrame:
    idx = pd.date_range(START, END, freq="h")
    hours = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()  # 0=Mon .. 6=Sun
    is_weekend = dow >= 5

    rows = []
    for business in BUSINESSES:
        for side in SIDES:
            cfg = BUSINESS_PROFILE[business][side]
            shape = intraday_shape(hours, cfg["peaks"], cfg["widths"], cfg["weights"])
            weekly_factor = np.where(is_weekend, cfg["weekend_factor"], 1.0)
            noise = rng.lognormal(mean=0.0, sigma=cfg["sigma"], size=len(idx))
            amount = cfg["base"] * shape * weekly_factor * noise

            jumbo_hit = rng.random(len(idx)) < cfg["jumbo_prob"]
            jumbo_mult = rng.uniform(*cfg["jumbo_range"], size=len(idx))
            amount = np.where(jumbo_hit, amount * jumbo_mult, amount)

            rows.append(pd.DataFrame({
                "Effective Timestamp": idx,
                "UBR L6 Name": business,
                "Side": side,
                "Amount (Local)": np.round(amount, 2),
            }))

    long_df = pd.concat(rows, ignore_index=True)
    wide = long_df.pivot_table(
        index=["Effective Timestamp", "UBR L6 Name"],
        columns="Side", values="Amount (Local)"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"Debit": "Debit Amount", "Credit": "Credit Amount"})
    wide["Net Amount"] = wide["Credit Amount"] - wide["Debit Amount"]
    wide = wide.sort_values(["UBR L6 Name", "Effective Timestamp"]).reset_index(drop=True)
    return wide


# ---------------------------------------------------------------------------
# 2. Morning / Afternoon / Evening / Night breakdown
# ---------------------------------------------------------------------------
def classify_segment(hour: int) -> str:
    if 6 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    return "Night"


def segment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["segment"] = d["Effective Timestamp"].dt.hour.map(classify_segment)
    summary = (
        d.groupby(["UBR L6 Name", "segment"])[["Debit Amount", "Credit Amount"]]
        .agg(["mean", "sum", "count"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    order = ["Morning", "Afternoon", "Evening", "Night"]
    summary = summary.reindex(
        pd.MultiIndex.from_product([BUSINESSES, order], names=["UBR L6 Name", "segment"])
    )
    return summary


# ---------------------------------------------------------------------------
# 3. Seasonality: hour-of-day, day-of-week, heatmap
# ---------------------------------------------------------------------------
def seasonality_profiles(df: pd.DataFrame):
    d = df.copy()
    d["hour"] = d["Effective Timestamp"].dt.hour
    d["dow"] = d["Effective Timestamp"].dt.dayofweek

    hourly = d.groupby(["UBR L6 Name", "hour"])[["Debit Amount", "Credit Amount"]].mean()
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    d["dow_name"] = d["dow"].map(dict(enumerate(dow_names)))
    weekly = d.groupby(["UBR L6 Name", "dow_name"])[["Debit Amount", "Credit Amount"]].mean()
    weekly = weekly.reindex(pd.MultiIndex.from_product([BUSINESSES, dow_names], names=["UBR L6 Name", "dow_name"]))

    return hourly, weekly, d


def plot_heatmaps(d: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(len(BUSINESSES), 2, figsize=(12, 3.6 * len(BUSINESSES)))
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, business in enumerate(BUSINESSES):
        for j, side in enumerate(["Debit Amount", "Credit Amount"]):
            ax = axes[i, j]
            sub = d[d["UBR L6 Name"] == business]
            pivot = sub.pivot_table(index="dow_name", columns="hour", values=side, aggfunc="mean")
            pivot = pivot.reindex(dow_names)
            im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
            ax.set_yticks(range(len(dow_names)))
            ax.set_yticklabels(dow_names, fontsize=8)
            ax.set_xticks(range(0, 24, 3))
            ax.set_xticklabels(range(0, 24, 3), fontsize=8)
            ax.set_title(f"{business} -- {side}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Hour x Day-of-week activity heatmap (mean amount)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_intraday_profiles(hourly: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"Corporate Bank": "#4e79a7", "FIC Trading": "#f28e2b", "Corporate Cash Management": "#59a14f"}
    for side, ax in zip(["Debit Amount", "Credit Amount"], axes):
        for business in BUSINESSES:
            s = hourly.loc[business, side]
            ax.plot(s.index, s.values, label=business, color=colors[business], marker="o", markersize=3)
        ax.set_title(side)
        ax.set_xlabel("hour of day")
        ax.set_ylabel("mean amount")
        ax.legend(fontsize=8)
    fig.suptitle("Intraday seasonality by business")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Next-day forecast (SARIMAX + seasonal-naive ensemble, per business/side)
# ---------------------------------------------------------------------------
def build_hourly_series(df: pd.DataFrame, business: str, side: str) -> pd.Series:
    col = f"{side} Amount"
    sub = df[df["UBR L6 Name"] == business].set_index("Effective Timestamp")[col]
    full_idx = pd.date_range(sub.index.min(), sub.index.max(), freq="h")
    return sub.reindex(full_idx).fillna(0.0)


def add_exog(idx: pd.DatetimeIndex) -> pd.DataFrame:
    hour = idx.hour
    dow = idx.dayofweek
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": (dow >= 5).astype(int),
    }, index=idx)


def forecast_next_day(y: pd.Series) -> pd.DataFrame:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    exog = add_exog(y.index)
    last_ts = y.index.max()
    future_idx = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=24, freq="h")
    future_exog = add_exog(future_idx)

    sarimax_fit = SARIMAX(
        y, exog=exog, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    sarimax_fc = np.clip(sarimax_fit.get_forecast(24, exog=future_exog).predicted_mean.to_numpy(), 0, None)

    seasonal_naive = np.array([
        y.get(ts - pd.Timedelta(days=7), np.nan) for ts in future_idx
    ])
    if np.isnan(seasonal_naive).any():
        same_hour_mean = y.groupby(y.index.hour).mean()
        seasonal_naive = np.where(
            np.isnan(seasonal_naive), same_hour_mean.reindex(future_idx.hour).to_numpy(), seasonal_naive
        )

    ensemble = (sarimax_fc + seasonal_naive) / 2.0
    return pd.DataFrame({
        "timestamp": future_idx,
        "sarimax_forecast": sarimax_fc,
        "seasonal_naive_forecast": seasonal_naive,
        "ensemble_forecast": ensemble,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating synthetic hourly business data (2026-04-01 -> 2026-06-30) ...")
    df = generate_synthetic_data()
    df.to_csv(OUT_DIR / "synthetic_hourly_business_data.csv", index=False)
    print(f"  {len(df)} rows, {df['Effective Timestamp'].min()} -> {df['Effective Timestamp'].max()}, "
          f"{df['UBR L6 Name'].nunique()} businesses")

    print("\nComputing morning/afternoon/evening/night segment breakdown ...")
    seg = segment_analysis(df)
    seg.to_csv(OUT_DIR / "segment_breakdown.csv")
    print(seg[["Debit Amount_mean", "Credit Amount_mean"]].round(0))

    print("\nComputing seasonality profiles (hour-of-day, day-of-week) ...")
    hourly, weekly, d_full = seasonality_profiles(df)
    hourly.to_csv(OUT_DIR / "hourly_seasonality_profile.csv")
    weekly.to_csv(OUT_DIR / "weekly_seasonality_profile.csv")
    plot_heatmaps(d_full, OUT_DIR / "plot_seasonality_heatmap.png")
    plot_intraday_profiles(hourly, OUT_DIR / "plot_intraday_profiles.png")

    print("\nForecasting next day (24h) debit & credit per business ...")
    forecast_rows = []
    daily_totals = []
    for business in BUSINESSES:
        for side in SIDES:
            print(f"  {business} / {side} ...")
            y = build_hourly_series(df, business, side)
            fc = forecast_next_day(y)
            fc["UBR L6 Name"] = business
            fc["Side"] = side
            forecast_rows.append(fc)
            daily_totals.append({
                "UBR L6 Name": business,
                "Side": side,
                "forecast_date": fc["timestamp"].dt.date.iloc[0],
                "sarimax_daily_total": fc["sarimax_forecast"].sum(),
                "seasonal_naive_daily_total": fc["seasonal_naive_forecast"].sum(),
                "ensemble_daily_total": fc["ensemble_forecast"].sum(),
            })

    forecast_df = pd.concat(forecast_rows, ignore_index=True)
    forecast_df.to_csv(OUT_DIR / "next_day_hourly_forecast.csv", index=False)

    daily_df = pd.DataFrame(daily_totals)
    daily_df.to_csv(OUT_DIR / "next_day_totals_by_business.csv", index=False)
    print("\nNext-day totals (ensemble):")
    print(daily_df[["UBR L6 Name", "Side", "forecast_date", "ensemble_daily_total"]]
          .round(0).to_string(index=False))

    print(f"\nAll outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()

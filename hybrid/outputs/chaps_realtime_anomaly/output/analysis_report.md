# CHAPS Real-Time Anomaly Detection - Production Analysis Report

**Data period:** 2025-01-02 00:00:00 -> 2025-03-31 23:00:00  
**Hours analysed:** 2136  
**Current status (latest hour, 2025-03-31 23:00:00):** **Watch**

## 1. Executive Summary

- 76 confirmed anomalies out of 2136 hours (3.56%), vs. an assumed base rate of 3%.
- 111 hours sat in the CUSUM 'Building Stress' pre-alarm band (cumulative deviation above 60% of H but below full threshold) without also being a confirmed anomaly.
- Ensemble requires >= 3/5 independent signals to agree before confirming, which keeps false positives from any single noisy signal from driving alerts.

## 2. Why Seasonal Adjustment Matters Here

CHAPS transaction_count is near-zero overnight and peaks mid-day, with a distinct weekday-vs-weekend profile. Feeding the raw series into a global z-score would flag every normal morning ramp-up as extreme. Instead, each hour's expected value is a causal rolling median of the last 8 occurrences of that same (hour-of-day, weekend/weekday) slot -- using only slots that occurred *before* the current hour, so this is safe to run in production without look-ahead. The seasonal z-score, EWMA chart and CUSUM all operate on this deseasonalized residual, not the raw count.

## 3. Signal-by-Signal Flag Rates

| Signal | Flags | Rate |
|---|---|---|
| Seasonal z-score | 69 | 3.23% |
| EWMA control chart | 200 | 9.36% |
| CUSUM structural break | 54 | 2.53% |
| Isolation Forest | 105 | 4.92% |
| Local Outlier Factor | 116 | 5.43% |
| **Ensemble (>= 3/5 agree)** | **76** | **3.56%** |

Isolation Forest and Local Outlier Factor both look at the same multivariate feature space (transaction_count, total_amount, dr/cr imbalance, lag/diff/rolling stats, week-over-week delta, calendar cyclicals) but score anomalousness differently -- Isolation Forest by how few random splits it takes to isolate a point, LOF by local density relative to neighbours -- so they tend to disagree on borderline points and agree on the clearest ones, which is exactly the redundancy the ensemble vote is designed to exploit.

## 4. Current Regime

- CUSUM confirmation threshold H (auto-calibrated): 6.00
- Latest combined score: 0.574
- Latest votes: 2/5 (ewma, lof)

## 5. Production Recommendations

- Run this script (or the underlying scoring functions) on each new hourly CHAPS batch as it lands; alert on `status` transitioning to `Confirmed Anomaly` or `Building Stress`.
- Re-validate `expected_anomaly_rate` periodically against realised anomaly frequency -- it drives the Isolation Forest / LOF thresholds and the CUSUM calibration.
- Keep `run.log` and `scored_series.csv` for audit trail.
- If a new payment corridor or seasonality shift changes the baseline, shrink `rolling_window_hours` temporarily so the seasonal baseline and ML training windows adapt faster to the new regime.
- Use `--simulate-stream` to sanity-check how the same scoring path behaves as a live feed before wiring it into a scheduler.

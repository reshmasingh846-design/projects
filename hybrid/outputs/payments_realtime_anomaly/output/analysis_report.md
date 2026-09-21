# Real-Time Payment Anomaly Detection - Analysis Report

Generated: 2026-09-21 19:09

## 1. What this replaces

The incumbent control is a static 95th/5th-percentile band on daily flow, investigated by hand after a breach. This system scores every minute of the settlement window in real time against a moving, seasonality-aware baseline, and names the causal transactions automatically.

## 2. Data

- Transactions: **1,085,673**
- Total value: **GBP 250.00bn**
- Period: 2026-01-19 06:00:00 -> 2026-09-18 18:00:00
- Sessions: 175 business days, 06:00-18:00
- History used to learn normal: 6 months (ends 2026-07-19)
- Minute rows scored: **126,175**

## 3. Detector flag rates

| Detector | Flagged minutes | Rate |
|---|---:|---:|
| Rolling z-score | 2,451 | 1.94% |
| Shewhart I-chart | 1,386 | 1.10% |
| EWMA (total flow) | 1,365 | 1.08% |
| EWMA (credit leg) | 6,082 | 4.82% |
| Isolation Forest | 435 | 0.34% |
| Ticket extreme (supplementary) | 141 | 0.11% |
| Static P95/P5 (incumbent) | 12,752 | 10.11% |

## 4. Ensemble status distribution

| Status | Minutes | Share |
|---|---:|---:|
| RED_CONFIRMED | 2,346 | 1.859% |
| RED_EARLY_WARNING | 34 | 0.027% |
| AMBER_BUILDING | 4,690 | 3.717% |
| GREEN | 119,105 | 94.397% |

## 5. Incident scorecard

| Incident | Detected | Mins to confirm | First AMBER | EWI lead time (min) | Peak EWI | Static line caught it? |
|---|:--:|---:|---|---:|---:|:--:|
| jumbo_single (24 Jul 07:56) | YES | 0 | - | - | 40 | yes |
| duplicate_burst (31 Jul 15:01) | YES | 0 | 15:02 | - | 93 | yes |
| level_shift_drift (07 Aug 12:14) | YES | 71 | 13:14 | 11 | 91 | yes |
| flow_stall (17 Aug 11:12) | YES | 2 | 11:12 | - | 93 | yes |
| cr_starvation (24 Aug 10:10) | YES | 12 | 10:13 | 9 | 62 | yes |
| late_surge (01 Sep 17:18) | YES | 16 | 16:33 | 61 | 89 | yes |
| jumbo_single (08 Sep 16:00) | YES | 0 | 15:30 | - | 86 | yes |
| level_shift_drift (16 Sep 07:43) | YES | 99 | 06:58 | 144 | 79 | yes |

- Ensemble detection rate: **8/8**
- Static P95/P5 line detection rate: **8/8**
- Median EWI lead time before confirmation: **36 minutes**
- Median time from incident onset to confirmation: **7 minutes**

### Alert load on quiet minutes

Detection rate alone flatters any detector -- one that fires constantly 'catches' everything. This is the alert volume outside every incident window, i.e. what an operator picks up on a normal day.

Runs of consecutive flagged minutes are collapsed into episodes -- one 40-minute drift is one ticket, not forty.

| Control | False-alarm minutes | Rate | Episodes | Episodes per session |
|---|---:|---:|---:|---:|
| This ensemble (confirmed) | 2,020 | 1.71% | 473 | 2.75 |
| EWI amber or worse | 5,614 | 4.74% | 630 | 3.66 |
| Incumbent static P95/P5 | 8,873 | 7.49% | 3,549 | 20.63 |

## 6. Alert triage (automatic attribution)

Cause labels assigned automatically at alert time:

| Likely cause | Alert minutes |
|---|---:|
| SINGLE_LARGE_PAYMENT | 1,095 |
| BROAD_ELEVATION | 647 |
| DR_CR_IMBALANCE | 385 |
| FLOW_STALL_or_OUTAGE | 203 |
| DUPLICATE_or_REPLAYED_FILE | 8 |
| VOLUME_BURST | 8 |

Ten largest alerts by excess over baseline:

| Time | Cause | Flow (GBP m) | Baseline (GBP m) | Excess (GBP m) | Top txn (GBP m) | % excess from top 5 | Votes |
|---|---|---:|---:|---:|---:|---:|:--:|
| 08 Sep 16:00 | SINGLE_LARGE_PAYMENT | 235.37 | 0.99 | 234.38 | 234.71 | 100% | 2/4 |
| 24 Jul 07:56 | SINGLE_LARGE_PAYMENT | 224.87 | 1.63 | 223.24 | 223.89 | 101% | 2/4 |
| 08 Apr 12:14 | SINGLE_LARGE_PAYMENT | 74.83 | 1.49 | 73.34 | 72.47 | 101% | 2/4 |
| 29 Apr 16:44 | SINGLE_LARGE_PAYMENT | 63.38 | 1.40 | 61.97 | 61.44 | 102% | 2/4 |
| 29 Jul 10:31 | SINGLE_LARGE_PAYMENT | 63.48 | 2.15 | 61.33 | 62.37 | 103% | 2/4 |
| 27 Feb 10:11 | SINGLE_LARGE_PAYMENT | 61.14 | 2.99 | 58.15 | 55.70 | 101% | 3/4 |
| 14 Sep 08:34 | SINGLE_LARGE_PAYMENT | 53.68 | 2.24 | 51.44 | 50.61 | 103% | 2/4 |
| 27 May 17:30 | SINGLE_LARGE_PAYMENT | 51.12 | 1.76 | 49.36 | 49.06 | 102% | 2/4 |
| 31 Jul 12:30 | SINGLE_LARGE_PAYMENT | 50.15 | 1.76 | 48.39 | 48.80 | 103% | 2/4 |
| 31 Jul 15:01 | DUPLICATE_or_REPLAYED_FILE | 46.27 | 1.29 | 44.98 | 4.04 | 31% | 3/4 |

## 7. Why each detector is in the ensemble

- **Rolling robust z-score** - the direct upgrade of the percentile line: same notion of 'too far from normal', but normal now moves with time-of-day and recent volatility. Median/MAD rather than mean/std so a genuine jumbo payment in the window does not inflate the yardstick and mask the next one.
- **Shewhart individuals chart** - sigma estimated from the mean moving range (MRbar / 1.128), which is insensitive to the outliers being hunted. Its Western Electric run rules (2-of-3 beyond 2 sigma, 4-of-5 beyond 1 sigma, 8 on one side) fire before a 3-sigma breach and feed the early-warning tier.
- **EWMA** - Shewhart is memoryless and therefore blind to a small sustained drift; EWMA accumulates it. This is the detector that catches the slow ramp hours before a static line would. Reset at each session open so overnight gaps never leak across.
- **Isolation Forest** - the only multivariate member: it sees amount, count, ticket size, DR/CR imbalance, burst ratio and volatility jointly, so it catches combinations that look unremarkable one dimension at a time. Threshold recalibrated from its own training scores at each retrain.

## 7b. How the detectors are combined (and why not a flat vote)

A plain 'k of 4 agree' rule was tried first and it failed in both directions, measurably. The four detectors do not observe the same class of event: z-score, Shewhart and Isolation Forest are **point** detectors that react to a single minute, while EWMA is a **sustained** detector that is deliberately slow. Under 2-of-4, a GBP 56m single payment scored z = 5.2 and was still rejected with one vote, because EWMA and Isolation Forest cannot physically react to a one-minute event; a 2-hour +85% drift was rejected for the mirror-image reason, since only EWMA can see drift at all.

So corroboration is required *within* a response group, and either group can confirm:

- **POINT** - 2 of the 3 point-capable detectors agree.
- **SUSTAINED** - EWMA on total flow stays outside its limit for 3 consecutive minutes.
- **SUSTAINED_CREDIT_LEG** - the same chart on the credit leg, held for 10 consecutive minutes.
- **TICKET** - a single payment exceeds the trailing q0.9995 ticket-size threshold.

Two further asymmetries were added because the data demanded them, not for symmetry's sake:

- **The low side must persist (3 minutes).** Payment arrivals are Poisson-sparse per minute, so a single near-empty minute is ordinary and recovers immediately. Confirming those instantly made them 39% of all alerts -- the largest single noise source in the system. On the high side one minute is still enough, because a large payment settling is real information; one empty minute is merely absence of evidence.
- **The credit leg is monitored separately.** Credits are about half of total value, so incoming liquidity can fall 85% while gross flow moves only ~40% -- inside normal variation. Watching only the aggregate makes a liquidity drought structurally invisible, and in testing the credit starvation incident was missed entirely until this chart was added; it is now caught with 178 of its 190 minutes confirmed.

The TICKET path is a supplementary control, not a fifth statistical method. It exists because one outsized payment is diluted by everything else settling in the same minute, so an aggregate control chart is a poor instrument for it -- whereas a threshold on the ticket itself answers the question directly and is the path that caught both single-payment incidents here, instantly.

Alert volume by path:

| Path | Alert minutes |
|---|---:|
| SUSTAINED_CREDIT_LEG | 1,198 |
| SUSTAINED | 875 |
| TICKET | 141 |
| POINT | 102 |
| POINT+SUSTAINED | 30 |

## 8. Operating notes

- POINT confirmation requires **2 of 3** point detectors; SUSTAINED requires **3 consecutive** EWMA breaches.
- EWI escalates at **50 (AMBER)** and **78 (RED)**.
- Isolation Forest retrains every **5 sessions** on a trailing **20-session** window.
- Every signal is computed point-in-time, so this same code path runs as a minute-by-minute stream (`--simulate-stream`) with no change to the maths.
- Tune `expected_anomaly_rate`, `z_threshold` and `vote_threshold` against your own alert-handling capacity before go-live; the values here are starting points.

## 9. Caveats

- **This is synthetic data.** The incidents were planted by the same code that scores them, so the detection rates above are a demonstration that the pipeline works end to end, not evidence of real-world accuracy. Every threshold needs recalibrating on your own history before it means anything.
- The statistical detectors run on a **log scale** because minute-level payment value is far too skewed for Gaussian control limits on raw amounts. Reported excesses are converted back to money.
- Tail assumptions dominate single-payment detection. In an earlier version of the generator, GBP 250m payments were routine, which made a GBP 56m 'anomaly' statistically unremarkable and correctly unflagged. Whether a large payment is an anomaly is a question about **your** tail, so fit `ticket_quantile` to real ticket sizes.
- Minutes in slots that normally carry fewer than `min_chartable_count` (2) payments are not charted: too sparse for a control chart, and charting them guarantees a false alarm every morning at open.
- Lead time is only claimed for incidents that **build**. A single outsized payment lands in one minute, so no detector can warn ahead of it, and any 'lead time' reported there would be a smoothing artefact.

# Real-Time Payment Anomaly Detection & Early Warning

Minute-level replacement for the static 95th/5th-percentile band on daily
payment flow. Scores every minute of the settlement window in real time,
and names the transactions that caused each alert automatically.

```bash
python payments_realtime_anomaly.py                   # generate + score
python payments_realtime_anomaly.py --simulate-stream # replay as a live feed
python payments_realtime_anomaly.py --months 6 --seed 7
```

Runs in ~60s. Needs `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

## What it does

Generates 8 months of synthetic minute-stamped payment transactions (DR/CR,
counterparty, channel, amount) across a 06:00-18:00 settlement window --
6 months of clean history to learn "normal", then a live period carrying
8 planted incidents. It then scores every minute walk-forward.

### The four requested detectors

| # | Detector | What it is for |
|---|---|---|
| 1 | Rolling robust z-score | Direct upgrade of the percentile line: the line now moves with time-of-day and recent volatility. Median/MAD so a jumbo payment doesn't inflate the yardstick. |
| 2 | Shewhart individuals chart | sigma from the mean moving range (MRbar/1.128), which is insensitive to the outliers being hunted. Western Electric run rules feed the early-warning tier. |
| 3 | EWMA control chart | Accumulates small sustained drift that Shewhart is structurally blind to. Reset at each session open. Run on total flow **and** on the credit leg. |
| 4 | Isolation Forest | The only multivariate member. Retrained on a cadence, threshold recalibrated from its own training scores each retrain. |

The incumbent static P95/P5 line is computed alongside as a benchmark.

### Results on this synthetic data

- **8 / 8** incidents detected.
- **2.75** false-alarm episodes per session, vs **20.63** for the static line.
- EWI lead time on the four gradual incidents: **144, 61, 11, 9** minutes
  before the ensemble confirmed.

## Three design decisions worth knowing

**Detectors run on a log scale.** Minute-level payment value is a sum of
lognormal tickets with a heavy tail. On raw amounts, Gaussian control limits
are meaningless -- EWMA alone flagged ~40% of all minutes. Reported excesses
are converted back to money.

**Combination routes on morphology, not a flat vote.** A "k of 4 agree" rule
fails in both directions: a GBP 56m single payment (z = 5.2) got one vote
because EWMA and Isolation Forest cannot react to a one-minute event, and a
2-hour drift got one vote because only EWMA can see drift. So corroboration
is required *within* a response group -- POINT (2 of 3 point detectors),
SUSTAINED (EWMA held N minutes), TICKET (one payment past the tail
threshold) -- and any path can confirm.

**The credit leg is monitored separately.** Credits are ~half of total value,
so incoming liquidity can collapse 85% while gross flow moves only ~40% --
inside normal variation. The credit-starvation incident was missed entirely
until this chart was added.

## Early Warning Indicator

A 0-100 composite of *sub-threshold* evidence: EWMA's approach to its limit,
live Western Electric run rules, z-score slope, Isolation Forest percentile,
volatility expansion, burst ratio, imbalance drift. Escalates GREEN -> AMBER
(50) -> RED (78). Thresholds were picked by sweeping against lead time: above
55 the slower drifts lose their warning entirely.

Lead time is only claimed for incidents that **build**. A single outsized
payment lands in one minute, so nothing can warn ahead of it.

## Automatic attribution

Every alert minute is joined back to the ledger: ranked contributing payments,
what share of the excess the top 5 explain, repeated (counterparty, amount)
pairs that signal a replayed file, and a triage label
(`SINGLE_LARGE_PAYMENT`, `DUPLICATE_or_REPLAYED_FILE`, `FLOW_STALL_or_OUTAGE`,
`VOLUME_BURST`, `DR_CR_IMBALANCE`, `BROAD_ELEVATION`). This is the manual
investigation step, done at alert time.

## Outputs (`./output/`)

| File | Contents |
|---|---|
| `scored_minutes.csv` | every minute, every signal, status |
| `alerts.csv` | confirmed anomalies with cause label + excess |
| `attribution.csv` | ranked causal transactions per alert |
| `early_warnings.csv` | EWI escalations |
| `incident_scorecard.csv` | per-incident detection + lead time |
| `summary.json` | machine-readable run summary |
| `dashboard.png` | 5-panel diagnostic chart |
| `analysis_report.md` | full written analysis |
| `synthetic_transactions.csv.gz` | the generated ledger (~1.1m rows) |

## Caveat

This is synthetic data, and the incidents are planted by the same code that
scores them. The numbers above show the pipeline works end to end; they are
not evidence of real-world accuracy. Every threshold -- especially
`ticket_quantile`, which depends entirely on your own tail -- needs
recalibrating on real history before it means anything.

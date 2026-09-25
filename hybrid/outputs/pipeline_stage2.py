"""Intraday CHAPS forecast - STAGE 2 (debit and credit only, no balance).

  lgbm   : features + gradient-boosting correction (scikit-learn) (Model A no Chronos, Model B with
           Chronos), quantiles 0.1/0.5/0.9, tuned on 3 expanding folds,
           compared with baseline and Chronos on VALIDATION.
  final  : large-payment add-on (+ optional scheduled payments), retrain the
           chosen model on TRAIN+VALIDATION, predict TEST once.
  report : one self-contained HTML report (actual vs forecast, method
           comparison, error analysis) from saved files only.

Run (after prep + chronos train/validation/test):
    python TRIALS/pipeline_stage2.py lgbm
    python TRIALS/pipeline_stage2.py final
    python TRIALS/pipeline_stage2.py report
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import intraday_forecast_trial as ift

import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance


class _Model:
    """Small wrapper so the rest of the code works like a LightGBM booster."""
    def __init__(self, est, n_iter):
        self.est, self.best_iteration = est, n_iter

    def predict(self, X):
        return self.est.predict(X)

    def current_iteration(self):
        return self.best_iteration

    def save_model(self, path):
        joblib.dump(self.est, str(path).replace(".txt", ".joblib"))


def _pinball(y, p, q):
    e = np.asarray(y) - p
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def _importance(m, X, y):
    """Permutation importance of the P50 model on up to 2,000 validation rows."""
    n = min(len(X), 2000)
    r = permutation_importance(m.est, X.iloc[:n], np.asarray(y)[:n], n_repeats=3,
                               random_state=42, scoring="neg_mean_absolute_error")
    return pd.Series(np.clip(r.importances_mean, 0, None), index=X.columns)

CFG = ift.CONFIG
COLORS = ift.COLORS
SERIES = ("debit", "credit")
HBINS = CFG["CHRONOS_HORIZONS_BINS"]          # [6, 12] -> 30 / 60 min
QS = (0.1, 0.5, 0.9)
DAY_BINS = 144                                # 06:00-17:55 in 5-min bins
OUT = CFG["OUTPUT_ROOT"]
MODEL_DIR = OUT / "models"
TABLE_P = OUT / "stage2_table.parquet"
BEST_P = OUT / "reports" / "best_config.json"
VAL_P = OUT / "validation" / "lgbm_val.parquet"
FINAL_P = OUT / "test" / "final_predictions.parquet"

CAL = ["dow", "is_month_end", "days_to_month_end", "is_quarter_end", "is_payday",
       "is_day_before_holiday", "is_day_after_holiday", "is_bridge_day",
       "days_to_next_holiday", "days_since_last_holiday",
       "event_before", "event_during", "event_after", "event_intensity"]
FEAT_A = ["minutes_since_open", "hour", "minute_of_hour", "dev_15m", "dev_30m",
          "dev_60m", "dev_today", "large_today", "log_base", "prev_day",
          "prev_week"] + CAL
FEAT_B = FEAT_A + ["chr_q10", "chr_q50", "chr_q90"]
GRID = [dict(learning_rate=lr, num_leaves=nl, min_data_in_leaf=md)
        for lr in (0.03, 0.1) for nl in (15, 31) for md in (20, 50)]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _csum(v):
    v = np.asarray(v, dtype="float64")
    nan = np.isnan(v)
    return (np.concatenate([[0.0], np.cumsum(np.where(nan, 0.0, v))]),
            np.concatenate([[0], np.cumsum(nan)]))


def _window(cs, cn, start, end):
    """Sum over bins [start, end); NaN if out of range or any bin is NaN."""
    start, end = np.asarray(start), np.asarray(end)
    out = np.full(len(start), np.nan)
    ok = (start >= 0) & (end <= len(cs) - 1) & (start <= end)
    s, e = start[ok], end[ok]
    tot = cs[e] - cs[s]
    tot[(cn[e] - cn[s]) > 0] = np.nan
    out[ok] = tot
    return out


def _lr(a, b):
    """Log ratio actual vs baseline (robust to zeros)."""
    return np.log1p(np.clip(a, 0, None)) - np.log1p(np.clip(b, 0, None))


def _to_amount(pred, base):
    return np.clip(np.expm1(pred + np.log1p(base)), 0, None)


def _cov(a, hi):
    m = ~(np.isnan(a) | np.isnan(hi))
    return float(np.mean(a[m] <= hi[m])) if m.any() else np.nan


def _qn(q):
    return f"q{int(round(q * 100)):02d}"


def _fit(X, y, params, q, n_iter=2000, Xv=None, yv=None, step=50, patience=3):
    """Quantile gradient boosting (scikit-learn). With (Xv, yv): early stopping
    on the validation pinball loss, adding `step` trees at a time."""
    est = HistGradientBoostingRegressor(
        loss="quantile", quantile=q, learning_rate=params["learning_rate"],
        max_leaf_nodes=params["num_leaves"], min_samples_leaf=params["min_data_in_leaf"],
        max_iter=n_iter if Xv is None else step, early_stopping=False,
        warm_start=Xv is not None, random_state=42)
    if Xv is None:
        est.fit(X, y)
        return _Model(est, n_iter)
    best, best_it, bad = np.inf, step, 0
    while est.max_iter <= n_iter:
        est.fit(X, y)                                  # warm start: adds trees
        loss = _pinball(yv, est.predict(Xv), q)
        if loss < best - 1e-9:
            best, best_it, bad = loss, est.max_iter, 0
        else:
            bad += 1
            if bad >= patience:
                break
        est.set_params(max_iter=est.max_iter + step)
    est.set_params(warm_start=False, max_iter=best_it)
    est.fit(X, y)                                      # refit at best size
    return _Model(est, best_it)


def _append_results(new):
    """Replace earlier stage-2 rows for the same splits, keep prep rows."""
    p = ift.table_path(CFG, "results_table.csv")
    new = new.assign(source="stage2")
    old = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if len(old) and "source" in old:
        old = old[~(old["source"].eq("stage2") & old["split"].isin(new["split"].unique()))]
    pd.concat([old, new], ignore_index=True).to_csv(p, index=False)


def _load_chronos():
    files = sorted(OUT.glob("*/chronos/*.parquet"))
    if not files:
        print("[stage2] no Chronos files found -> Model B skipped")
        return None
    c = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    c["issue_ts"] = pd.to_datetime(c["issue_ts"])
    return c.drop_duplicates("issue_ts").set_index("issue_ts")


# --------------------------------------------------------------------------
# feature table: one row per (issue point, series, horizon), past-only
# --------------------------------------------------------------------------
def build_table():
    t0 = time.time()
    df = ift._load_full_history(CFG)
    df, iss = ift._issue_points(df, CFG)                    # every 15 min
    split = df["split"].to_numpy()
    iss = iss[np.isin(split[iss], ["train", "validation", "test"])]
    day_start = np.arange(len(df)) - df.groupby("date").cumcount().to_numpy()
    chron = _load_chronos()

    hour, minute = df["hour"].to_numpy()[iss], df["minute"].to_numpy()[iss]
    meta = pd.DataFrame({"issue_ts": df["ts"].iloc[iss].values,
                         "split": split[iss],
                         "day_type": df["day_type"].to_numpy()[iss],
                         "hour": hour, "minute_of_hour": minute,
                         "minutes_since_open": hour * 60 + minute - 360})
    for c in CAL:
        meta[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()[iss]

    parts = []
    for s in SERIES:
        cl, cln = _csum(df[f"{s}_clean"])
        ca, can = _csum(df[f"{s}_amount"])
        cb, cbn = _csum(df[f"baseline_{s}"])
        cg, cgn = _csum(df[f"{s}_large"])
        common = {f"dev_{k*5}m": _lr(_window(cl, cln, iss - k, iss),
                                     _window(cb, cbn, iss - k, iss)) for k in (3, 6, 12)}
        common["dev_today"] = _lr(_window(cl, cln, day_start[iss], iss),
                                  _window(cb, cbn, day_start[iss], iss))
        common["large_today"] = np.log1p(_window(cg, cgn, day_start[iss], iss))
        for h in HBINS:
            base = _window(cb, cbn, iss, iss + h)
            t = meta.copy()
            t["series"], t["horizon_min"] = s, h * 5
            for k, v in common.items():
                t[k] = v
            t["log_base"] = np.log1p(base)
            t["prev_day"] = _lr(_window(cl, cln, iss - DAY_BINS, iss - DAY_BINS + h), base)
            t["prev_week"] = _lr(_window(cl, cln, iss - 5 * DAY_BINS,
                                         iss - 5 * DAY_BINS + h), base)
            t["baseline"] = base
            t["actual_clean"] = _window(cl, cln, iss, iss + h)
            t["actual"] = _window(ca, can, iss, iss + h)          # incl. large
            t["large_actual"] = _window(cg, cgn, iss, iss + h)
            t["y"] = _lr(t["actual_clean"].to_numpy(), base)
            if chron is not None:
                k = f"h{h*5:02d}"
                cj = chron.reindex(pd.to_datetime(t["issue_ts"]))[
                    [f"{s}_{_qn(q)}_{k}" for q in QS]].to_numpy(dtype="float64")
                for q, col in zip(QS, cj.T):
                    t[f"chr_{_qn(q)}"] = _lr(col, base)
                t["chronos_p50"] = cj[:, 1]
            else:
                for q in QS:
                    t[f"chr_{_qn(q)}"] = np.nan
                t["chronos_p50"] = np.nan
            parts.append(t)
    tab = pd.concat(parts, ignore_index=True)
    tab = tab[tab["baseline"].notna()].reset_index(drop=True)
    print(f"[stage2] feature table {len(tab):,} rows ({time.time()-t0:.1f}s)")
    return tab


# --------------------------------------------------------------------------
# LGBM stage
# --------------------------------------------------------------------------
def _tune(g):
    """3 expanding folds over train+validation dates, P50 Model A."""
    d = pd.to_datetime(g["issue_ts"]).dt.normalize()
    days = np.sort(d.unique())
    cuts = [days[int(len(days) * f)] for f in (0.55, 0.70, 0.85)]
    scores = []
    for params in GRID:
        w = []
        for c in cuts:
            a = g[d < c]
            b = g[(d >= c) & (d < c + np.timedelta64(42, "D"))]
            if len(a) < 200 or len(b) < 50:
                continue
            m = _fit(a[FEAT_A], a["y"], params, 0.5, n_iter=300)
            f = _to_amount(m.predict(b[FEAT_A]), b["baseline"].to_numpy())
            w.append(ift._wape(b["actual_clean"].to_numpy(float), f))
        scores.append(np.nanmean(w) if w else np.inf)
    return GRID[int(np.argmin(scores))], float(np.min(scores))


def run_lgbm():
    t0 = time.time()
    ift.ensure_dirs(CFG)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tab = build_table()
    tab.to_parquet(TABLE_P, index=False)

    best, val_out, res, imps = {"keys": {}}, [], [], {}
    for (s, h), g in tab.groupby(["series", "horizon_min"]):
        key = f"{s}_h{h}"
        g = g[g["y"].notna()]
        tr, va = g[g["split"] == "train"], g[g["split"] == "validation"]
        params, fold_wape = _tune(pd.concat([tr, va]))
        print(f"[lgbm] {key}: params={params} fold WAPE={fold_wape:.3f}")
        entry = {"params": params, "fold_wape": fold_wape, "models": {}}
        v = va[["issue_ts", "day_type", "hour", "series", "horizon_min", "baseline",
                "actual_clean", "actual", "chronos_p50"]].copy()
        has_chr = va["chr_q50"].notna().to_numpy()
        yv = va["y"].to_numpy()
        base = va["baseline"].to_numpy()

        for name, feats in (("A", FEAT_A), ("B", FEAT_B)):
            trm = tr if name == "A" else tr[tr["chr_q50"].notna()]
            vmask = np.ones(len(va), bool) if name == "A" else has_chr
            if len(trm) < 200 or vmask.sum() < 50:
                print(f"[lgbm] {key} model {name}: not enough rows, skipped")
                continue
            vam = va[vmask]
            preds, iters = {}, {}
            for q in QS:
                m = _fit(trm[feats], trm["y"], params, q, Xv=vam[feats], yv=vam["y"])
                iters[str(q)] = int(m.best_iteration or m.current_iteration())
                preds[q] = m.predict(va[feats])
                m.save_model(str(MODEL_DIR / f"val_{key}_{name}_{_qn(q)}.txt"))
                if q == 0.5:
                    imps[(key, name)] = _importance(m, vam[feats], vam["y"])
            ok = vmask & ~np.isnan(yv)
            c10 = float(np.quantile(yv[ok] - preds[0.1][ok], 0.1))   # calibrate
            c90 = float(np.quantile(yv[ok] - preds[0.9][ok], 0.9))   # to ~10/90%
            p50 = preds[0.5]
            p10 = np.minimum(preds[0.1] + c10, p50)
            p90 = np.maximum(preds[0.9] + c90, p50)
            for q, p in ((0.1, p10), (0.5, p50), (0.9, p90)):
                v[f"{name}_{_qn(q)}"] = np.where(vmask, _to_amount(p, base), np.nan)
            entry["models"][name] = {"iters": iters, "c10": c10, "c90": c90}

        # fair comparison on rows where every method exists
        cm = has_chr & v["actual_clean"].notna().to_numpy()
        a = v["actual_clean"].to_numpy(float)[cm]
        methods = {"baseline": v["baseline"], "chronos": v["chronos_p50"]}
        for name in entry["models"]:
            methods[f"lgbm_{name}"] = v[f"{name}_q50"]
        for mth, f in methods.items():
            f = f.to_numpy(float)[cm]
            row = {"method": mth, "series": s, "horizon_min": h, "split": "validation",
                   "day_type": "all", "wape": ift._wape(a, f), "n": int(cm.sum())}
            if mth.startswith("lgbm_"):
                row["coverage_p90"] = _cov(a, v[f"{mth[-1]}_q90"].to_numpy(float)[cm])
                entry["models"][mth[-1]]["val_wape"] = row["wape"]
            res.append(row)
        best["keys"][key] = entry
        val_out.append(v)

    wa = [e["models"]["A"]["val_wape"] for e in best["keys"].values() if "A" in e["models"]]
    wb = [e["models"]["B"]["val_wape"] for e in best["keys"].values() if "B" in e["models"]]
    best["chosen"] = "B" if len(wb) == len(wa) and np.mean(wb) < np.mean(wa) else "A"
    BEST_P.parent.mkdir(parents=True, exist_ok=True)
    BEST_P.write_text(json.dumps(best, indent=2, default=float))
    VAL_P.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(val_out, ignore_index=True).to_parquet(VAL_P, index=False)
    res = pd.DataFrame(res)
    _append_results(res)

    print("\n=== Validation WAPE (clean series) ===")
    print(res.pivot_table(index=["series", "horizon_min"], columns="method",
                          values="wape").round(3).to_string())
    print(f"[lgbm] chosen model: {best['chosen']}  "
          f"(A={np.mean(wa):.3f}, B={np.mean(wb) if wb else float('nan'):.3f})")

    # plots: feature importance (chosen model, averaged) + WAPE by method
    ch = best["chosen"]
    imp = pd.concat([v_ for (k, n), v_ in imps.items() if n == ch], axis=1).mean(axis=1)
    imp = (imp / imp.sum()).sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index, imp.values, color=COLORS["lightgbm"])
    ax.set_title(f"Gradient boosting (Model {ch}) feature importance, top 20")
    ax.set_xlabel("share of importance")
    fig.tight_layout(); fig.savefig(CFG["FIG_DIR"] / "step10_feature_importance.png"); plt.close(fig)

    piv = res.pivot_table(index=["series", "horizon_min"], columns="method", values="wape")
    fig, ax = plt.subplots(figsize=(9, 4))
    piv.plot.bar(ax=ax, color=[{"baseline": COLORS["baseline"], "chronos": COLORS["chronos"],
                                "lgbm_A": "#8fd18f", "lgbm_B": COLORS["lightgbm"]}.get(c, "k")
                               for c in piv.columns])
    ax.set_title("Validation WAPE by method (lower is better)"); ax.set_ylabel("WAPE")
    ax.set_xlabel("series, horizon (min)"); fig.tight_layout()
    fig.savefig(CFG["FIG_DIR"] / "step10_validation_wape.png"); plt.close(fig)
    print(f"[lgbm] done ({time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
# FINAL stage (test, run once)
# --------------------------------------------------------------------------
def _large_addon(tab):
    """Large-payment stats from TRAIN+VALIDATION windows (past for test)."""
    hist = tab[tab["split"].isin(["train", "validation"])].copy()
    hist["has"] = hist["large_actual"] > 0
    keys = ["series", "horizon_min", "dow", "hour", "is_month_end"]
    pos = hist[hist["has"]]
    st = hist.groupby(keys)["has"].mean().rename("prob").to_frame()
    st = st.join(pos.groupby(keys)["large_actual"].median().rename("med"))
    st = st.join(pos.groupby(keys)["large_actual"].quantile(0.95).rename("p95")).reset_index()
    gl = pos.groupby(["series", "horizon_min"])["large_actual"].agg(
        gmed="median", gp95=lambda x: x.quantile(0.95)).reset_index()
    return st, gl


def _scheduled(te):
    """Optional CSV cfg['SCHEDULED_PATH'] with columns ts, series, amount."""
    p = CFG.get("SCHEDULED_PATH")
    out = np.zeros(len(te))
    if p is None or not Path(p).exists():
        return out
    sc = pd.read_csv(p)
    sc["ts"] = (pd.to_datetime(sc["ts"]).dt.tz_localize(CFG["TIMEZONE"])
                .dt.tz_convert("UTC").dt.tz_localize(None))
    for s in SERIES:
        x = sc[sc["series"] == s].sort_values("ts")
        if x.empty:
            continue
        t = x["ts"].to_numpy(dtype="datetime64[ns]")
        c = np.concatenate([[0.0], np.cumsum(x["amount"].to_numpy(float))])
        m = (te["series"] == s).to_numpy()
        st_ = te.loc[m, "issue_ts"].to_numpy(dtype="datetime64[ns]")
        en = st_ + te.loc[m, "horizon_min"].to_numpy().astype("timedelta64[m]")
        out[m] = c[np.searchsorted(t, en)] - c[np.searchsorted(t, st_)]
    return out


def _fit_predict(fit, te, feats, params, mdl):
    preds = {}
    for q in QS:
        n = max(50, int(mdl["iters"][str(q)] * 1.1))
        preds[q] = _fit(fit[feats], fit["y"], params, q, n_iter=n).predict(te[feats])
    p50 = preds[0.5]
    return (np.minimum(preds[0.1] + mdl["c10"], p50), p50,
            np.maximum(preds[0.9] + mdl["c90"], p50))


def run_final():
    t0 = time.time()
    tab = pd.read_parquet(TABLE_P)
    best = json.loads(BEST_P.read_text())
    chosen = best["chosen"]
    st, gl = _large_addon(tab)
    test = tab[tab["split"] == "test"].copy()
    test = (test.merge(st, on=["series", "horizon_min", "dow", "hour", "is_month_end"], how="left")
                .merge(gl, on=["series", "horizon_min"], how="left"))
    test["prob"] = test["prob"].fillna(0.0)
    test["med"] = test["med"].fillna(test["gmed"]).fillna(0.0)
    test["p95"] = test["p95"].fillna(test["gp95"]).fillna(0.0)
    test["sched"] = _scheduled(test)

    outs = []
    for (s, h), te in test.groupby(["series", "horizon_min"]):
        key = f"{s}_h{h}"
        e = best["keys"][key]
        fit = tab[(tab["series"] == s) & (tab["horizon_min"] == h) &
                  tab["split"].isin(["train", "validation"]) & tab["y"].notna()]
        base = te["baseline"].to_numpy()
        # Model A always (also the fallback when Chronos is missing)
        pa = _fit_predict(fit, te, FEAT_A, e["params"], e["models"]["A"])
        p10, p50, p90 = pa
        if chosen == "B" and "B" in e["models"]:
            fb = fit[fit["chr_q50"].notna()]
            pb = _fit_predict(fb, te, FEAT_B, e["params"], e["models"]["B"])
            use_b = te["chr_q50"].notna().to_numpy()
            p10, p50, p90 = (np.where(use_b, b, a) for b, a in zip(pb, pa))
            print(f"[final] {key}: Model B on {use_b.mean():.0%} of rows, A fallback on rest")
        lg10, lg50, lg90 = (_to_amount(p, base) for p in (p10, p50, p90))
        sch, prob = te["sched"].to_numpy(), te["prob"].to_numpy()
        te = te.copy()
        te["lgbm_p50"] = lg50
        te["final_p10"] = lg10 + sch
        te["final_p50"] = lg50 + sch + prob * te["med"].to_numpy()
        # P90 only moves if a large payment is at least 10% likely in the window
        te["final_p90"] = lg90 + sch + np.where(prob >= 0.10, te["p95"].to_numpy(), 0.0)
        outs.append(te)

    pred = pd.concat(outs, ignore_index=True)[
        ["issue_ts", "day_type", "hour", "minute_of_hour", "dow", "is_day_after_holiday",
         "series", "horizon_min", "actual", "actual_clean", "large_actual", "baseline",
         "chronos_p50", "lgbm_p50", "final_p10", "final_p50", "final_p90", "prob"]]
    FINAL_P.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(FINAL_P, index=False)

    # test metrics vs ACTUAL TOTAL (clean + large), all methods
    rows = []
    groups = [("all", lambda d: pd.Series(True, index=d.index))]
    groups += [(dt, lambda d, dt=dt: d["day_type"] == dt) for dt in pred["day_type"].unique()]
    groups += [(f"min_{m:02d}", lambda d, m=m: d["minute_of_hour"] == m) for m in (0, 15, 30, 45)]
    groups += [("large_yes", lambda d: d["large_actual"] > 0),
               ("large_no", lambda d: d["large_actual"] <= 0)]
    for (s, h), g in pred.groupby(["series", "horizon_min"]):
        for label, fn in groups:
            d = g[fn(g)]
            if len(d) < 10:
                continue
            a = d["actual"].to_numpy(float)
            for mth, col in (("baseline", "baseline"), ("chronos", "chronos_p50"),
                             ("lgbm", "lgbm_p50"), ("final", "final_p50")):
                r = {"method": mth, "series": s, "horizon_min": h, "split": "test",
                     "day_type": label, "wape": ift._wape(a, d[col].to_numpy(float)),
                     "n": len(d)}
                if mth == "final":
                    r["coverage_p90"] = _cov(a, d["final_p90"].to_numpy(float))
                    r["coverage_p10"] = float(np.mean(a < d["final_p10"].to_numpy(float)))
                rows.append(r)
    res = pd.DataFrame(rows)
    _append_results(res)
    allr = res[res["day_type"] == "all"]
    print("\n=== TEST WAPE vs actual total (run once, do not tune after this) ===")
    print(allr.pivot_table(index=["series", "horizon_min"], columns="method",
                           values="wape").round(3).to_string())
    print(allr[allr["method"] == "final"][["series", "horizon_min", "coverage_p90"]]
          .round(3).to_string(index=False))
    print(f"[final] saved {FINAL_P} ({time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
# REPORT stage (saved files only): step by step, with charts and next steps
# --------------------------------------------------------------------------
def _img(fig, name):
    fig.tight_layout()
    fig.savefig(CFG["FIG_DIR"] / name, dpi=110)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _img_file(name):
    p = CFG["FIG_DIR"] / name
    if not p.exists():
        return "<p><i>(chart not available)</i></p>"
    return f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">'


def _reason(r):
    if r["large_actual"] > 0:
        return "large payment"
    if r["day_type"] != "normal":
        return r["day_type"].replace("_", " ")
    if r["actual"] > 2 * r["final_p50"]:
        return "unusual surge"
    if r["actual"] < 0.5 * r["final_p50"]:
        return "unusually quiet"
    return "normal variation"


def _wape_by(d, by, col):
    g = d.assign(ae=(d["actual"] - d[col]).abs(), aa=d["actual"].abs()).groupby(by)
    return g["ae"].sum() / g["aa"].sum()


def _pct(x):
    return f"{x:.1%}" if pd.notna(x) else "-"


def _verdict(imp, cov):
    if pd.isna(imp):
        return "-"
    if imp >= 0.15 and 0.85 <= cov <= 0.95:
        return "Good"
    if imp > 0.05:
        return "Acceptable"
    return "Not good"


def _box(title, items):
    return f'<div class="box"><b>{title}</b><ul>' + "".join(f"<li>{x}</li>" for x in items) + "</ul></div>"


def run_reports():
    t0 = time.time()
    tz, sp = CFG["TIMEZONE"], CFG["SPLITS"]
    pred = pd.read_parquet(FINAL_P)
    best = json.loads(BEST_P.read_text())
    res = pd.read_csv(ift.table_path(CFG, "results_table.csv"))
    hist = ift._load_full_history(CFG)
    hist["local"] = hist["ts"].dt.tz_convert(tz)
    hist["d"] = hist["local"].dt.date
    pred["local"] = pd.to_datetime(pred["issue_ts"]).dt.tz_localize("UTC").dt.tz_convert(tz)
    pred["date"] = pred["local"].dt.date
    h0 = int(pred["horizon_min"].min())
    p30 = pred[pred["horizon_min"] == h0]
    SC = 1e6                                               # plot in GBP m

    # ---------- headline metrics per series x horizon ----------
    rows = []
    for (s, h), g in pred.groupby(["series", "horizon_min"]):
        a = g["actual"].to_numpy(float)
        lg = g[g["large_actual"] > 0]
        rows.append({
            "series": s, "horizon": int(h),
            "baseline": ift._wape(a, g["baseline"].to_numpy(float)),
            "chronos": ift._wape(a, g["chronos_p50"].to_numpy(float)),
            "gbm": ift._wape(a, g["lgbm_p50"].to_numpy(float)),
            "final": ift._wape(a, g["final_p50"].to_numpy(float)),
            "cov90": _cov(a, g["final_p90"].to_numpy(float)),
            "bias": float((g["final_p50"] - g["actual"]).sum() / g["actual"].sum()),
            "large_caught": float(np.mean(lg["actual"] <= lg["final_p90"])) if len(lg) else np.nan,
            "n_large": len(lg)})
    M = pd.DataFrame(rows)
    M["improvement"] = (M["baseline"] - M["final"]) / M["baseline"]
    M["verdict"] = [_verdict(i, c) for i, c in zip(M["improvement"], M["cov90"])]
    lab = M["series"] + " " + M["horizon"].astype(str) + " min"

    # ---------- example days ----------
    def pick(mask):
        ds = sorted(p30.loc[mask, "date"].unique())
        return ds[len(ds) // 2] if ds else None

    dd = p30[p30["series"] == "debit"]
    d_norm = pick(p30["day_type"] == "normal")
    d_me = pick(p30["day_type"] == "month_end")
    d_ah = pick(p30["is_day_after_holiday"] == 1)
    d_worst = dd.assign(e=(dd["actual"] - dd["final_p50"]).abs()).groupby("date")["e"].sum().idxmax()
    d_large = dd.loc[dd["large_actual"].idxmax(), "date"] if (dd["large_actual"] > 0).any() else None

    ACT = ("actual", "actual", COLORS["actual"], "-")
    BASE = ("baseline", "baseline", COLORS["baseline"], "--")
    CHR = ("chronos_p50", "Chronos", COLORS["chronos"], "-")
    GBM = ("lgbm_p50", "gradient boosting", COLORS["lightgbm"], "-")
    FIN = ("final_p50", "final forecast", COLORS["final"], "-")

    def day_fig(d, title, lines, fname, band=True):
        if d is None:
            return "<p><i>(no such day in the test period)</i></p>"
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
        for ax, s in zip(axs, SERIES):
            g = p30[(p30["date"] == d) & (p30["series"] == s)].sort_values("local")
            x = g["local"].dt.tz_localize(None)
            if band:
                ax.fill_between(x, g["final_p10"] / SC, g["final_p90"] / SC,
                                color=COLORS["band"], alpha=0.6, label="likely range (P10-P90)")
            for col, lb, c, ls in lines:
                ax.plot(x, g[col] / SC, color=c, ls=ls, label=lb, lw=1.4 if col == "actual" else 1.1)
            ax.set_title(f"{title} {d} - {s}, next {h0} min")
            ax.set_ylabel("GBP m"); ax.set_xlabel("issue time")
            ax.legend(fontsize=7)
        return _img(fig, fname)

    # ---------- STEP 1: data ----------
    daily = hist.groupby("d")[["debit_amount", "credit_amount"]].sum()
    daily.index = pd.to_datetime(daily.index)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for name, colr in (("train", "#e8f0fe"), ("validation", "#fff4e0"), ("test", "#e9f7ef")):
        a_ = max(pd.Timestamp(sp[name]["start"]), daily.index.min())
        b_ = pd.Timestamp(sp[name]["end"]) if sp[name]["end"] else daily.index.max()
        ax.axvspan(a_, b_, color=colr, zorder=0, label=name)
    ax.plot(daily.index, daily["debit_amount"] / SC, color=COLORS["final"], lw=0.8, label="debit")
    ax.plot(daily.index, daily["credit_amount"] / SC, color=COLORS["chronos"], lw=0.8, label="credit")
    ax.set_title("Daily total payments and the train / validation / test periods")
    ax.set_ylabel("GBP m per day"); ax.legend(ncol=5, fontsize=8)
    img_data = _img(fig, "step15_01_data.png")
    data_txt = (f"{daily.index.min().date()} to {daily.index.max().date()}, {len(daily)} business days, "
                f"5-minute resolution 06:00-18:00. Average per day: debit "
                f"{daily['debit_amount'].mean()/SC:,.0f}m, credit {daily['credit_amount'].mean()/SC:,.0f}m.")

    # ---------- STEP 2: large payments ----------
    hist["month"] = hist["local"].dt.strftime("%Y-%m")
    mc = hist.groupby("month")[["debit_large_flag", "credit_large_flag"]].sum()
    fig, ax = plt.subplots(figsize=(12, 3.4))
    mc.plot.bar(ax=ax, color=[COLORS["final"], COLORS["chronos"]])
    ax.set_title("Number of 5-minute periods with a large one-off payment, per month")
    ax.set_ylabel("count"); ax.set_xlabel(""); ax.legend(["debit", "credit"])
    img_lp_month = _img(fig, "step15_02_large_by_month.png")
    share = {s: hist[f"{s}_large"].sum() / hist[f"{s}_amount"].sum() for s in SERIES}
    dbig = hist.loc[hist["debit_large"].idxmax(), "d"]
    hb = hist[hist["d"] == dbig]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    x = hb["local"].dt.tz_localize(None)
    ax.plot(x, hb["debit_amount"] / SC, color=COLORS["actual"], lw=0.8, label="total debit")
    ax.plot(x, hb["debit_clean"] / SC, color=COLORS["lightgbm"], lw=0.8, label="regular part (clean)")
    ax.set_title(f"Example {dbig}: the large payment is separated from the regular flow")
    ax.set_ylabel("GBP m per 5 min"); ax.legend(fontsize=8)
    img_lp_day = _img(fig, "step15_02_large_example.png")

    # ---------- STEP 3: baseline ----------
    hn = hist[hist["d"] == d_norm] if d_norm is not None else hist.iloc[:0]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))
    for ax, s in zip(axs, SERIES):
        x = hn["local"].dt.tz_localize(None)
        ax.plot(x, hn[f"{s}_clean"] / SC, color=COLORS["actual"], lw=0.8, label="actual (clean)")
        ax.plot(x, hn[f"baseline_{s}"] / SC, color=COLORS["baseline"], lw=1.4, label="baseline")
        ax.set_title(f"{d_norm} - {s}: actual vs baseline (5 min)"); ax.set_ylabel("GBP m")
        ax.legend(fontsize=8)
    img_base = _img(fig, "step15_03_baseline_day.png")
    pr = res[(res["method"] == "baseline") & (res["day_type"] == "all")]
    if "source" in pr:
        pr = pr[pr["source"].isna()]
    base_tab = (pr.pivot_table(index=["series", "horizon_min"], columns="split", values="wape")
                .map(_pct).to_html() if len(pr) else "")

    # ---------- STEP 4-6 tables ----------
    def mtab(cols, names):
        t = M[["series", "horizon"] + cols].copy()
        t.columns = ["Series", "Horizon (min)"] + names
        for c in names:
            t[c] = t[c].map(_pct)
        return t.to_html(index=False)

    chr_better = int((M["chronos"] < M["baseline"]).sum())
    gbm_better = int((M["gbm"] < M["chronos"]).sum())

    # ---------- STEP 8: improvement per step ----------
    fig, ax = plt.subplots(figsize=(9, 4))
    steps = ["Baseline", "+ Chronos", "+ Gradient boosting", "+ Large-payment add-on"]
    for i, r in M.iterrows():
        ax.plot(steps, [r["baseline"], r["chronos"], r["gbm"], r["final"]], marker="o",
                label=f"{r['series']} {r['horizon']} min")
    ax.set_title("Error (WAPE) after each step - lower is better"); ax.set_ylabel("WAPE")
    ax.legend(fontsize=8)
    img_steps = _img(fig, "step15_08_improvement_by_step.png")

    # ---------- STEP 9: error analysis ----------
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, s in zip(axs, SERIES):
        g = p30[p30["series"] == s]
        ax.plot(_wape_by(g, "hour", "baseline"), color=COLORS["baseline"], marker="o", label="baseline")
        ax.plot(_wape_by(g, "hour", "final_p50"), color=COLORS["final"], marker="o", label="final")
        ax.set_title(f"{s}: error by hour of day ({h0} min)"); ax.set_xlabel("hour"); ax.set_ylabel("WAPE")
        ax.legend()
    img_hour = _img(fig, "step15_09_wape_by_hour.png")

    dt_tab = pd.concat({s: _wape_by(p30[p30["series"] == s], "day_type", "final_p50")
                        for s in SERIES}, axis=1)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    dt_tab.plot.bar(ax=ax, color=[COLORS["final"], COLORS["chronos"]])
    ax.set_title(f"Error by day type ({h0} min)"); ax.set_ylabel("WAPE"); ax.tick_params(axis="x", rotation=0)
    img_dt = _img(fig, "step15_09_wape_by_daytype.png")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    me = pd.concat({s: _wape_by(p30[p30["series"] == s], "minute_of_hour", "final_p50")
                    for s in SERIES}, axis=1)
    me.plot.bar(ax=ax, color=[COLORS["final"], COLORS["chronos"]])
    ax.set_title("Error by minute of the hour when the forecast is issued")
    ax.set_xlabel("minutes past the hour"); ax.set_ylabel("WAPE"); ax.tick_params(axis="x", rotation=0)
    img_min = _img(fig, "step15_09_wape_by_minute.png")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    for s, c in zip(SERIES, (COLORS["final"], COLORS["chronos"])):
        g = p30[p30["series"] == s]
        b = g.assign(e=g["final_p50"] - g["actual"]).groupby("hour")
        ax.plot(b["e"].sum() / b["actual"].sum(), color=c, marker="o", label=s)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Bias by hour (above 0 = forecast too high)"); ax.set_xlabel("hour")
    ax.set_ylabel("bias / actual"); ax.legend()
    img_bias = _img(fig, "step15_09_bias_by_hour.png")

    fig, ax = plt.subplots(figsize=(10, 3.4))
    drift = []
    for s, c in zip(SERIES, (COLORS["final"], COLORS["chronos"])):
        g = p30[p30["series"] == s].assign(ae=lambda d: (d["actual"] - d["final_p50"]).abs())
        de = g.groupby("date")[["ae", "actual"]].sum()
        roll = de["ae"].rolling(7).sum() / de["actual"].rolling(7).sum()
        ax.plot(pd.to_datetime(roll.index), roll.values, color=c, label=s)
        if len(de) >= 40:
            f_, l_ = de.iloc[:20], de.iloc[-20:]
            drift.append((l_["ae"].sum() / l_["actual"].sum()) / (f_["ae"].sum() / f_["actual"].sum()))
    ax.set_title("Rolling 7-day error over the test period (is it getting worse?)")
    ax.set_ylabel("WAPE"); ax.legend()
    img_drift = _img(fig, "step15_09_drift.png")

    worst = pred.assign(err=(pred["actual"] - pred["final_p50"]).abs()).nlargest(20, "err")
    worst["reason"] = worst.apply(_reason, axis=1)
    worst_html = worst.assign(
        time=worst["local"].dt.strftime("%Y-%m-%d %H:%M"),
        actual=(worst["actual"] / SC).map("{:,.1f}m".format),
        forecast=(worst["final_p50"] / SC).map("{:,.1f}m".format),
        error=(worst["err"] / SC).map("{:,.1f}m".format))[
        ["time", "series", "horizon_min", "actual", "forecast", "error", "day_type", "reason"]
    ].to_html(index=False)

    # ---------- STEP 10: health checks and next steps ----------
    checks = []

    def check(name, ok, result, action):
        checks.append({"Check": name, "Result": result,
                       "Status": "✅ OK" if ok else "⚠️ Needs action",
                       "What to do if it needs action": action})

    imp = M["improvement"].mean()
    check("Beats the simple baseline", imp >= 0.05, f"average improvement {_pct(imp)}",
          "Check features for errors or leakage (helper prompt 'Leakage check'); retrain with "
          "more history; compare Model A only; add a simple 'same window last week' benchmark.")
    cg = ((M["baseline"] - M["chronos"]) / M["baseline"]).mean()
    check("Chronos adds value", cg > 0, f"Chronos vs baseline: {_pct(cg)} better",
          "Drop Chronos and use Model A (saves hours of compute), or give Chronos calendar "
          "covariates (holiday, month end, payday).")
    cov = M["cov90"].mean()
    check("Likely range (P90) is reliable", 0.85 <= cov <= 0.95, f"coverage {_pct(cov)} (target 90%)",
          "Below 85%: band too narrow - recalibrate on the latest 2 months and add known large "
          "payments. Above 95%: band too wide - recalibrate to narrow it.")
    ib = M["bias"].abs().idxmax()
    check("No systematic over/under-forecast", abs(M.loc[ib, "bias"]) <= 0.10,
          f"largest bias {_pct(M.loc[ib, 'bias'])} ({lab[ib]})",
          "Add a recent-level feature (last 4 weeks vs baseline) and retrain monthly so the "
          "forecast follows volume growth.")
    dmean = dt_tab.mean(axis=1).dropna()
    ratio = dmean.max() / dmean.get("normal", np.nan)
    check("Special days handled", not (ratio > 1.3),
          f"hardest: {dmean.idxmax().replace('_', ' ')} ({_pct(dmean.max())}) vs normal "
          f"({_pct(dmean.get('normal', np.nan))})",
          "Add a known calendar for that day type (e.g. month-end settlement schedule) or train "
          "a separate model for it.")
    dr = float(np.nanmax(drift)) if drift else np.nan
    check("Stable over time (no drift)", not (dr > 1.2),
          f"error in last 20 days = {dr:.2f} x first 20 days" if pd.notna(dr) else "not enough days",
          "Retrain on the latest data (monthly) or use a shorter, more recent training window.")
    lc = M["large_caught"].mean()
    check("Large payments inside the range", not (lc < 0.5), f"{_pct(lc)} of windows with a large payment",
          "Get a feed of scheduled / known large payments (SCHEDULED_PATH); try LARGE_M = 4.")
    w30 = M.loc[M["horizon"] == M["horizon"].min(), "final"].mean()
    w60 = M.loc[M["horizon"] == M["horizon"].max(), "final"].mean()
    check("30-minute forecast usable", not (w30 > 1.3 * w60), f"30 min {_pct(w30)} vs 60 min {_pct(w60)}",
          "Use the 60-minute forecast for decisions and treat the 30-minute one as indicative.")
    chk = pd.DataFrame(checks)
    n_bad = int(chk["Status"].str.contains("Needs").sum())
    overall = ("All checks passed: the forecast is ready for a monitored trial."
               if n_bad == 0 else
               f"{n_bad} of {len(chk)} checks need action - see the table below before using "
               "the forecast operationally.")

    roadmap = [
        "<b>Data:</b> more history (data starts Sep 2024), check data quality (gaps, duplicates), "
        "add a feed of scheduled / known large payments.",
        "<b>Features:</b> Chronos calendar covariates; a recent-level feature; extra special "
        "dates (quarter end, tax and settlement dates); split by payment type if available.",
        "<b>Models:</b> more tuning on walk-forward folds; separate models for morning / "
        "afternoon; an average of Chronos and gradient boosting; simple benchmarks for sanity.",
        "<b>Process:</b> retrain monthly; monitor daily WAPE and P90 coverage; alert when the "
        "rolling error rises by more than 20%.",
    ]

    # ---------- findings ----------
    findings = [
        f"The final forecast is on average <b>{_pct(imp)}</b> more accurate than the seasonal baseline.",
        f"Credit is {'easier' if M[M.series=='credit'].final.mean() < M[M.series=='debit'].final.mean() else 'harder'} "
        f"to forecast than debit (WAPE {_pct(M[M.series=='credit'].final.mean())} vs "
        f"{_pct(M[M.series=='debit'].final.mean())}).",
        f"60-minute forecasts are more accurate than 30-minute ones ({_pct(w60)} vs {_pct(w30)}), "
        "because single payments average out over longer windows.",
        f"Hardest day type: {dmean.idxmax().replace('_', ' ')}.",
    ]
    summ = pd.DataFrame({"Series": M["series"], "Horizon": M["horizon"].astype(str) + " min",
                         "Baseline error": M["baseline"].map(_pct),
                         "Final error": M["final"].map(_pct),
                         "Improvement": M["improvement"].map(_pct),
                         "P90 coverage (target 90%)": M["cov90"].map(_pct),
                         "Verdict": M["verdict"]}).to_html(index=False)

    style = ("body{font-family:Arial,sans-serif;max-width:1150px;margin:auto;padding:20px;color:#222;"
             "line-height:1.45}h1{color:#1f3b63}h2{color:#1f3b63;border-bottom:2px solid #ddd;"
             "padding-bottom:4px;margin-top:40px}h3{color:#33507a}table{border-collapse:collapse;"
             "margin:10px 0}td,th{border:1px solid #ccc;padding:5px 9px;text-align:right}"
             "th{background:#f1f3f6}img{max-width:100%;margin:8px 0}.box{background:#f6f8fb;"
             "padding:10px 18px;border-left:4px solid #1f3b63;margin:12px 0}.what{color:#555}"
             ".toc a{text-decoration:none}")
    toc = ["Summary", "How to read this report", "Step 1 - Data", "Step 2 - Large payments",
           "Step 3 - Seasonal baseline", "Step 4 - Chronos-2", "Step 5 - Gradient-boosting correction",
           "Step 6 - Large-payment add-on", "Step 7 - Final forecast vs actual",
           "Step 8 - What each step added", "Step 9 - Error analysis",
           "Step 10 - Health checks and next steps", "Limitations"]
    toc_html = "<ol class='toc'>" + "".join(f"<li><a href='#s{i}'>{t}</a></li>" for i, t in enumerate(toc)) + "</ol>"
    H = lambda i: f"<h2 id='s{i}'>{toc[i]}</h2>"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Intraday CHAPS forecast report</title><style>{style}</style></head><body>
<h1>Intraday CHAPS payment forecast - analysis report</h1>
<p>Forecasts of <b>debit</b> and <b>credit</b> payments for the next 30 and 60 minutes, updated every
15 minutes between 06:00 and 18:00. Results below are on the <b>test period</b>
({sp['test']['start']} onwards), which the models never saw during training or tuning.</p>
{toc_html}

{H(0)}{summ}
<div class="box"><b>Overall:</b> {overall}</div>
{_box("Key findings", findings)}

{H(1)}<ul>
<li><b>WAPE (error)</b> = total absolute forecast error / total actual amount. 20% means the
forecast is off by 20% of the real volume on average. Lower is better.</li>
<li><b>P50</b> is the central forecast. <b>P10-P90</b> is the likely range: the real amount should
fall below P90 about 90% of the time (<b>coverage</b>).</li>
<li><b>Horizon</b>: 30 min = total payments in the next 30 minutes after the forecast time.</li>
<li><b>Baseline</b>: the simple benchmark (average of the same time slot on the last 4 similar
weekdays). Every step must beat it.</li>
<li>Charts: <b>black</b> = actual, <b>grey dashed</b> = baseline, <b>blue</b> = Chronos,
<b>green</b> = gradient boosting, <b>red</b> = final forecast, <b>pink band</b> = likely range.</li></ul>

{H(2)}<p class="what"><b>What this step does:</b> loads the CHAPS minute data, fills missing
minutes with zero, adds calendar information (holidays, month end, payday, events) and sums
everything into 5-minute periods.</p><p><b>Result:</b> {data_txt}</p>{img_data}
<p><b>Split:</b> train {sp['train']['start']} - {sp['train']['end']} (data actually starts Sep 2024),
validation {sp['validation']['start']} - {sp['validation']['end']}, test {sp['test']['start']} onwards.
CHAPS is closed on bank holidays, so there is no data on those days.</p>

{H(3)}<p class="what"><b>What this step does:</b> finds one-off large payments (unusual for that
time slot <i>and</i> above the 99.5th percentile of train amounts) and separates them, so the
regular flow can be forecast cleanly. Large payments are added back in step 6.</p>
<p><b>Result:</b> large payments are {_pct(share['debit'])} of debit volume and
{_pct(share['credit'])} of credit volume.</p>{img_lp_month}{img_lp_day}

{H(4)}<p class="what"><b>What this step does:</b> the "normal" level for each weekday and 5-minute
slot = average of the last 4 comparable weeks, skipping holidays and events. It uses only
past data.</p>{img_base}<p><b>Baseline error on train and validation:</b></p>{base_tab}

{H(5)}<p class="what"><b>What this step does:</b> Chronos-2, a pre-trained AI time-series model,
reads the recent history and forecasts the next 60 minutes with a likely range.</p>
<p><b>Result:</b> Chronos beats the baseline in {chr_better} of {len(M)} series/horizon
combinations.</p>{mtab(["baseline", "chronos"], ["Baseline error", "Chronos error"])}
{day_fig(d_norm, "Normal day", [ACT, BASE, CHR], "step15_04_chronos_day.png", band=False)}

{H(6)}<p class="what"><b>What this step does:</b> a gradient-boosting model learns how far the real
flow will differ from the baseline, using how the day is going so far, the last 15-60 minutes,
yesterday and last week, calendar effects and (Model B) the Chronos forecast. Chosen model:
<b>{best['chosen']}</b> ({'with' if best['chosen'] == 'B' else 'without'} Chronos).</p>
<p><b>Result:</b> it beats Chronos in {gbm_better} of {len(M)} combinations.</p>
{mtab(["baseline", "chronos", "gbm"], ["Baseline error", "Chronos error", "Gradient boosting error"])}
<h3>Validation comparison (used to choose the model)</h3>{_img_file('step10_validation_wape.png')}
<h3>What drives the forecast</h3>{_img_file('step10_feature_importance.png')}
{day_fig(d_norm, "Normal day", [ACT, BASE, CHR, GBM], "step15_05_gbm_day.png", band=False)}

{H(7)}<p class="what"><b>What this step does:</b> adds the typical size of large payments for that
weekday, hour and month end (weighted by how likely one is), plus any known scheduled payments,
and widens the upper range when a large payment is likely.</p>
{mtab(["gbm", "final", "cov90", "large_caught"], ["Before add-on", "After add-on", "P90 coverage", "Large payments inside range"])}
{day_fig(d_large, "Day with a large payment", [ACT, GBM, FIN], "step15_06_large_day.png")}

{H(8)}<p class="what">The final forecast (red) with its likely range (pink) against what actually
happened (black), on typical and difficult days.</p>
{day_fig(d_norm, "Normal day", [ACT, BASE, FIN], "step15_07_day_normal.png")}
{day_fig(d_me, "Month end", [ACT, BASE, FIN], "step15_07_day_month_end.png")}
{day_fig(d_ah, "Day after a holiday", [ACT, BASE, FIN], "step15_07_day_after_holiday.png")}
{day_fig(d_worst, "Worst day", [ACT, BASE, FIN], "step15_07_day_worst.png")}

{H(9)}{img_steps}
<p>Each line should go down from left to right. If a step makes a line go up, that step is not
helping for that series and can be removed or improved.</p>

{H(10)}<h3>By hour of day</h3>{img_hour}<h3>By day type</h3>{img_dt}
<h3>By time of issue within the hour</h3>{img_min}<h3>Bias</h3>{img_bias}
<h3>Over time (drift)</h3>{img_drift}<h3>20 largest errors</h3>{worst_html}

{H(11)}<div class="box"><b>Overall:</b> {overall}</div>
{chk.to_html(index=False, escape=False)}
{_box("If the forecast is still not good enough - roadmap", roadmap)}

{H(12)}<ul>
<li>CHAPS is closed on bank holidays, so holiday effects are learned only from the days around them.</li>
<li>History starts in September 2024 (about 10 months of training data).</li>
<li>Single large payments are hard to time; the add-on covers their typical size, not their exact minute.</li>
<li>Chronos runs without calendar inputs.</li>
<li>P10/P90 are calibrated on the validation period; re-check coverage after each retrain.</li>
</ul></body></html>"""
    out = CFG["REPORTS_ROOT"] / "forecast_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n[report] overall: {overall}")
    print(chk[["Check", "Result", "Status"]].to_string(index=False))
    print(f"[report] written {out} ({time.time()-t0:.1f}s)")


STAGES = {"lgbm": run_lgbm, "final": run_final, "report": run_reports}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "all":
        for fn in STAGES.values():
            fn()
    elif stage in STAGES:
        STAGES[stage]()
    else:
        sys.exit(f"usage: python pipeline_stage2.py [{' | '.join(STAGES)} | all]")

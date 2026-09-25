"""Intraday CHAPS forecast - STAGE 2 (debit and credit only, no balance).

  lgbm   : features + LightGBM correction (Model A no Chronos, Model B with
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

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("lightgbm is missing: pip install lightgbm")

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


def _fit(X, y, params, q, n_iter=2000, Xv=None, yv=None):
    p = dict(objective="quantile", alpha=q, verbose=-1, seed=42,
             feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, **params)
    dtr = lgb.Dataset(X, y)
    if Xv is None:
        return lgb.train(p, dtr, n_iter)
    dv = lgb.Dataset(Xv, yv, reference=dtr)
    return lgb.train(p, dtr, n_iter, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


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
                    imps[(key, name)] = pd.Series(m.feature_importance("gain"), index=feats)
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
    ax.set_title(f"LightGBM (Model {ch}) feature importance, top 20")
    ax.set_xlabel("share of total gain")
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
# REPORT stage (saved files only)
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


def run_reports():
    t0 = time.time()
    pred = pd.read_parquet(FINAL_P)
    best = json.loads(BEST_P.read_text())
    tz = CFG["TIMEZONE"]
    pred["local"] = pd.to_datetime(pred["issue_ts"]).dt.tz_localize("UTC").dt.tz_convert(tz)
    pred["date"] = pred["local"].dt.date
    f = lambda x: f"{x:.1%}" if pd.notna(x) else "-"

    # 1. summary table
    summ = []
    for (s, h), g in pred.groupby(["series", "horizon_min"]):
        a = g["actual"].to_numpy(float)
        wb = ift._wape(a, g["baseline"].to_numpy(float))
        wf = ift._wape(a, g["final_p50"].to_numpy(float))
        summ.append({"Series": s, "Horizon": f"{h} min", "Baseline WAPE": f(wb),
                     "Final WAPE": f(wf), "Improvement": f((wb - wf) / wb),
                     "P90 coverage (target 90%)": f(_cov(a, g["final_p90"].to_numpy(float))),
                     "_imp": (wb - wf) / wb, "_cov": _cov(a, g["final_p90"].to_numpy(float))})
    summ = pd.DataFrame(summ)

    # 2. method comparison
    meth = {}
    for (s, h), g in pred.groupby(["series", "horizon_min"]):
        a = g["actual"].to_numpy(float)
        meth[f"{s} {h} min"] = {m: ift._wape(a, g[c].to_numpy(float)) for m, c in
                                (("Baseline", "baseline"), ("Chronos", "chronos_p50"),
                                 ("LightGBM", "lgbm_p50"), ("Final", "final_p50"))}
    meth = pd.DataFrame(meth).T
    fig, ax = plt.subplots(figsize=(9, 4))
    meth.plot.bar(ax=ax, color=[COLORS["baseline"], COLORS["chronos"],
                                COLORS["lightgbm"], COLORS["final"]])
    ax.set_title("Test WAPE by method (lower is better)"); ax.set_ylabel("WAPE")
    ax.set_xlabel(""); ax.tick_params(axis="x", rotation=0)
    img_meth = _img(fig, "step15_test_wape_by_method.png")

    # 3. actual vs forecast, 3 example days (30-min horizon)
    p30 = pred[pred["horizon_min"] == pred["horizon_min"].min()]
    picks = []
    for label, cond in (("Normal day", p30["day_type"] == "normal"),
                        ("Month end", p30["day_type"] == "month_end"),
                        ("Day after a holiday", p30["is_day_after_holiday"] == 1)):
        ds = sorted(p30.loc[cond, "date"].unique())
        if ds:
            picks.append((label, ds[len(ds) // 2]))
    img_days = []
    for label, d in picks:
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))
        for ax, s in zip(axs, SERIES):
            g = p30[(p30["date"] == d) & (p30["series"] == s)].sort_values("local")
            x = g["local"].dt.tz_localize(None)
            ax.fill_between(x, g["final_p10"], g["final_p90"], color=COLORS["band"],
                            alpha=0.6, label="P10-P90")
            ax.plot(x, g["actual"], color=COLORS["actual"], label="actual")
            ax.plot(x, g["baseline"], color=COLORS["baseline"], ls="--", label="baseline")
            ax.plot(x, g["final_p50"], color=COLORS["final"], label="final P50")
            ax.set_title(f"{label} {d} - {s}, next {int(g['horizon_min'].iloc[0])} min")
            ax.set_ylabel("GBP"); ax.legend(fontsize=7)
        img_days.append(_img(fig, f"step15_day_{label.split()[0].lower()}.png"))

    # scatter
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, s in zip(axs, SERIES):
        g = p30[p30["series"] == s]
        ax.scatter(g["final_p50"], g["actual"], s=4, alpha=0.3, color=COLORS["final"])
        lim = [max(1.0, float(np.nanmin(g[["actual", "final_p50"]].clip(lower=1).values))),
               float(np.nanmax(g[["actual", "final_p50"]].values))]
        ax.plot(lim, lim, color="k", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{s}: forecast vs actual (30 min)")
        ax.set_xlabel("final P50 [GBP]"); ax.set_ylabel("actual [GBP]")
    img_scatter = _img(fig, "step15_scatter.png")

    # 4. errors: by hour, day type, bias, drift
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, s in zip(axs, SERIES):
        g = p30[p30["series"] == s]
        ax.plot(_wape_by(g, "hour", "baseline"), color=COLORS["baseline"], label="baseline")
        ax.plot(_wape_by(g, "hour", "final_p50"), color=COLORS["final"], label="final")
        ax.set_title(f"{s}: WAPE by hour (30 min)"); ax.set_xlabel("hour"); ax.legend()
    img_hour = _img(fig, "step15_wape_by_hour.png")

    dt_tab = pd.concat({s: _wape_by(p30[p30["series"] == s], "day_type", "final_p50")
                        for s in SERIES}, axis=1)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    dt_tab.plot.bar(ax=ax, color=[COLORS["final"], COLORS["chronos"]])
    ax.set_title("Final WAPE by day type (30 min)"); ax.set_ylabel("WAPE")
    ax.tick_params(axis="x", rotation=0)
    img_dt = _img(fig, "step15_wape_by_daytype.png")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for s, c in zip(SERIES, (COLORS["final"], COLORS["chronos"])):
        g = p30[p30["series"] == s]
        b = g.assign(e=g["final_p50"] - g["actual"]).groupby("hour")
        ax.plot(b["e"].sum() / b["actual"].sum(), color=c, label=s)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Bias by hour (+ = over-forecast)"); ax.set_xlabel("hour")
    ax.set_ylabel("bias / actual"); ax.legend()
    img_bias = _img(fig, "step15_bias_by_hour.png")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    for s, c in zip(SERIES, (COLORS["final"], COLORS["chronos"])):
        g = p30[p30["series"] == s].assign(ae=lambda d: (d["actual"] - d["final_p50"]).abs())
        daily = g.groupby("date")[["ae", "actual"]].sum()
        roll = daily["ae"].rolling(7).sum() / daily["actual"].rolling(7).sum()
        ax.plot(pd.to_datetime(roll.index), roll.values, color=c, label=s)
    ax.set_title("Rolling 7-day WAPE over the test period (drift check)")
    ax.set_ylabel("WAPE"); ax.legend()
    img_drift = _img(fig, "step15_drift.png")

    # 5. large payments, worst windows
    lg = pred[pred["large_actual"] > 0]
    caught = float(np.mean(lg["actual"] <= lg["final_p90"])) if len(lg) else np.nan
    worst = pred.assign(err=(pred["actual"] - pred["final_p50"]).abs()).nlargest(20, "err")
    worst["reason"] = worst.apply(_reason, axis=1)
    worst_html = worst.assign(
        time=worst["local"].dt.strftime("%Y-%m-%d %H:%M"),
        actual=worst["actual"].map("{:,.0f}".format),
        forecast=worst["final_p50"].map("{:,.0f}".format),
        error=worst["err"].map("{:,.0f}".format))[
        ["time", "series", "horizon_min", "actual", "forecast", "error", "day_type", "reason"]
    ].to_html(index=False)

    # findings
    avg_imp = summ["_imp"].mean()
    dmean = dt_tab.mean(axis=1).dropna()
    covs = summ["_cov"]
    findings = [
        f"The final forecast is on average {avg_imp:.0%} more accurate than the seasonal baseline.",
        f"Hardest day type: {dmean.idxmax().replace('_', ' ')} (WAPE {dmean.max():.0%}); "
        f"easiest: {dmean.idxmin().replace('_', ' ')} ({dmean.min():.0%}).",
        f"Large payments inside the P90 range: {caught:.0%} of windows with a large payment."
        if pd.notna(caught) else "No large payments in the test period.",
    ]
    recs = [
        "Keep the 60-minute forecast as the main planning number; 30-minute windows are "
        "dominated by single payments.",
        "Widen the P90 band or add known scheduled payments (SCHEDULED_PATH) - coverage is "
        f"{covs.mean():.0%} vs the 90% target." if covs.mean() < 0.85 else
        "The P90 band is well calibrated; use it as the upper planning limit.",
        "Retrain monthly and watch the rolling WAPE chart for drift.",
    ]
    sp = CFG["SPLITS"]
    summ_html = summ.drop(columns=["_imp", "_cov"]).to_html(index=False)
    meth_html = meth.map(lambda x: f"{x:.1%}").to_html()
    style = ("body{font-family:Arial,sans-serif;max-width:1150px;margin:auto;padding:20px;"
             "color:#222}h1{color:#1f3b63}h2{color:#1f3b63;border-bottom:2px solid #ddd;"
             "padding-bottom:4px;margin-top:36px}table{border-collapse:collapse;margin:10px 0}"
             "td,th{border:1px solid #ccc;padding:5px 9px;text-align:right}th{background:#f1f3f6}"
             "img{max-width:100%;margin:8px 0}.box{background:#f6f8fb;padding:12px 18px;"
             "border-left:4px solid #1f3b63}")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Intraday CHAPS forecast report</title><style>{style}</style></head><body>
<h1>Intraday CHAPS payment forecast - test report</h1>
<p>Debit and credit forecasts for the next 30 and 60 minutes, updated every 15 minutes.
Test period {sp['test']['start']} onwards, evaluated once on data the models never saw.</p>

<h2>1. Summary</h2>{summ_html}
<div class="box"><b>Key findings</b><ul>{''.join(f'<li>{x}</li>' for x in findings)}</ul>
<b>Recommendations</b><ul>{''.join(f'<li>{x}</li>' for x in recs)}</ul></div>

<h2>2. What the pipeline does</h2><ol>
<li><b>Clean the data:</b> each minute is split into regular flow and one-off large payments
(unusual for that time slot <i>and</i> above the 99.5th percentile).</li>
<li><b>Seasonal baseline:</b> the normal level for each weekday and 5-minute slot, from the
last 4 comparable weeks (skipping holidays and events).</li>
<li><b>Chronos-2:</b> a pre-trained AI time-series model forecasts the next 60 minutes from
recent history.</li>
<li><b>LightGBM correction:</b> learns how far the real flow will differ from the baseline,
using how the day is going so far, recent trend, calendar effects and (Model B) Chronos.
Chosen model: <b>{best['chosen']}</b>.</li>
<li><b>Large-payment add-on:</b> adds the expected size of large payments for that weekday,
hour and month-end, plus any known scheduled payments.</li></ol>
<p>Data split: train {sp['train']['start']} to {sp['train']['end']} (data starts Sep 2024),
validation {sp['validation']['start']} to {sp['validation']['end']},
test {sp['test']['start']} onwards.</p>

<h2>3. Actual vs forecast</h2>
<p>Red line = final forecast, band = likely range (P10-P90), black = what actually happened.</p>
{''.join(img_days)}{img_scatter}

<h2>4. Method comparison (test)</h2>{meth_html}{img_meth}
<h3>Validation comparison and feature importance</h3>
{_img_file('step10_validation_wape.png')}{_img_file('step10_feature_importance.png')}

<h2>5. Error analysis</h2>{img_hour}{img_dt}{img_bias}{img_drift}
<h3>20 largest errors</h3>{worst_html}

<h2>6. Limitations and next steps</h2><ul>
<li>CHAPS is closed on bank holidays, so holiday effects are learned only from the days
around them.</li>
<li>History starts in September 2024 (about 10 months of training data); accuracy should
improve as more data arrives.</li>
<li>Single large payments are hard to predict in timing; the add-on covers their typical
size, not their exact minute.</li>
<li>Chronos runs without calendar inputs; adding them is a possible next step.</li>
<li>P10/P90 are calibrated on the validation period; re-check coverage after each retrain.</li>
</ul></body></html>"""
    out = CFG["REPORTS_ROOT"] / "forecast_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"[report] written {out} ({time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
# ANALYSIS stage (saved files only): deep-dive appended to the report
# --------------------------------------------------------------------------
AN_START, AN_END = "<!-- deep-analysis:start -->", "<!-- deep-analysis:end -->"
AN_LEVELS = ((0.1, "final_p10"), (0.5, "final_p50"), (0.9, "final_p90"))
AN_STYLE = ("body{font-family:Arial,sans-serif;max-width:1150px;margin:auto;padding:20px;"
            "color:#222}h1{color:#1f3b63}h2{color:#1f3b63;border-bottom:2px solid #ddd;"
            "padding-bottom:4px;margin-top:36px}h3{color:#31507d;margin-top:24px}"
            "table{border-collapse:collapse;margin:10px 0;font-size:13px}"
            "td,th{border:1px solid #ccc;padding:5px 9px;text-align:right}th{background:#f1f3f6}"
            "img{max-width:100%;margin:8px 0}.box{background:#f6f8fb;padding:12px 18px;"
            "border-left:4px solid #1f3b63}.note{color:#555;font-size:13px;margin:4px 0 14px}")


def _an_metrics(a, p):
    """WAPE, MAE, RMSE, bias, median APE and sMAPE on rows where both exist."""
    a, p = np.asarray(a, dtype="float64"), np.asarray(p, dtype="float64")
    m = ~(np.isnan(a) | np.isnan(p))
    a, p = a[m], p[m]
    if not len(a):
        return dict.fromkeys(("n", "wape", "mae", "rmse", "bias", "medape", "smape"), np.nan)
    e, den = p - a, np.abs(a).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(e) / np.where(a > 0, a, np.nan)
    return {"n": len(a), "wape": np.abs(e).sum() / den, "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": e.sum() / den,
            "medape": float(np.nanmedian(ape)),
            "smape": float(np.mean(2 * np.abs(e) / np.clip(np.abs(a) + np.abs(p), 1e-9, None)))}


def _an_pinball(a, p, q):
    """Quantile (pinball) loss - the proper score for a P10/P50/P90 forecast."""
    d = np.asarray(a, dtype="float64") - np.asarray(p, dtype="float64")
    return float(np.nanmean(np.maximum(q * d, (q - 1) * d)))


def _an_boot(d, ref, new, n=2000, seed=42):
    """Day-block paired bootstrap: P(`new` is no better than `ref` on WAPE)."""
    t = d.assign(e1=(d["actual"] - d[ref]).abs(), e2=(d["actual"] - d[new]).abs(),
                 aa=d["actual"].abs()).groupby("date")[["e1", "e2", "aa"]].sum()
    t = t[t["aa"] > 0]
    if len(t) < 5:
        return np.nan
    v = t.to_numpy(dtype="float64")
    s = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].sum(axis=1)
    return float(np.mean((s[:, 0] - s[:, 1]) <= 0))


def _an_lag1(g):
    """Lag-1 autocorrelation of the relative error: signal still on the table."""
    e = g.sort_values("issue_ts")["rel"].to_numpy(dtype="float64")
    e = e[~np.isnan(e)]
    return float(np.corrcoef(e[:-1], e[1:])[0, 1]) if len(e) > 10 else np.nan


def _an_group(d, by, labels):
    """Final vs baseline accuracy inside each group of `by`."""
    rows = []
    for k, g in d.groupby(by, observed=True):
        k = k if isinstance(k, tuple) else (k,)
        a = g["actual"].to_numpy(float)
        m, mb = _an_metrics(a, g["final_p50"]), _an_metrics(a, g["baseline"])
        rows.append({**dict(zip(labels, k)), "Windows": m["n"], "Value share": np.nansum(a),
                     "Baseline WAPE": mb["wape"], "Final WAPE": m["wape"],
                     "Skill": 1 - m["wape"] / mb["wape"], "Bias": m["bias"],
                     "P90 cov": _cov(a, g["final_p90"].to_numpy(float))})
    r = pd.DataFrame(rows)
    r["Value share"] = r["Value share"] / r["Value share"].sum()
    return r


def _an_html(df, pcts=(), nums=(), rnd=()):
    """Format selected columns and render a table."""
    out = df.copy()
    for c in pcts:
        out[c] = out[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    for c in nums:
        out[c] = out[c].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
    for c in rnd:
        out[c] = out[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    return out.to_html(index=False, escape=False)


def run_analysis():
    t0 = time.time()
    pred = pd.read_parquet(FINAL_P)
    best = json.loads(BEST_P.read_text())
    tz = CFG["TIMEZONE"]
    pred["local"] = pd.to_datetime(pred["issue_ts"]).dt.tz_localize("UTC").dt.tz_convert(tz)
    pred["date"] = pred["local"].dt.date
    pred["key"] = pred["series"] + " " + pred["horizon_min"].astype(int).astype(str) + " min"
    pred["err"] = pred["final_p50"] - pred["actual"]
    pred["ae"] = pred["err"].abs()
    pred["rel"] = pred["err"] / pred["actual"].clip(lower=1.0)
    pred["in_band"] = ((pred["actual"] >= pred["final_p10"]) &
                       (pred["actual"] <= pred["final_p90"]))
    pct = lambda x: f"{x:.1%}" if pd.notna(x) else "-"
    num = lambda x: f"{x:,.0f}" if pd.notna(x) else "-"
    short = pred[pred["horizon_min"] == pred["horizon_min"].min()]
    sec = []                                     # (title, html) in report order

    # A1. accuracy card -----------------------------------------------------
    acc = []
    for k, g in pred.groupby("key"):
        a = g["actual"].to_numpy(float)
        m, mb = _an_metrics(a, g["final_p50"]), _an_metrics(a, g["baseline"])
        w = float(np.nanmean(g["final_p90"] - g["final_p10"]))
        acc.append({"Target": k, "Windows": m["n"],
                    "Total actual (GBP m)": num(np.nansum(a) / 1e6),
                    "Mean window (GBP)": num(np.nanmean(a)),
                    "WAPE": pct(m["wape"]), "Median APE": pct(m["medape"]),
                    "sMAPE": pct(m["smape"]), "MAE (GBP)": num(m["mae"]),
                    "RMSE (GBP)": num(m["rmse"]), "Bias": pct(m["bias"]),
                    "Skill vs baseline": pct(1 - m["wape"] / mb["wape"]),
                    "P10-P90 hit (80%)": pct(float(g["in_band"].mean())),
                    "Band width / mean": pct(w / np.nanmean(a))})
    sec.append(("A1. Accuracy at a glance", pd.DataFrame(acc).to_html(index=False) +
                "<p class='note'>WAPE weights every pound equally, so it is driven by the "
                "busiest windows; median APE shows the typical window instead. Bias is the "
                "signed error as a share of turnover - positive means we over-forecast. "
                "RMSE well above MAE means the errors are dominated by a few big misses.</p>"))

    # A2. where the accuracy comes from ------------------------------------
    stage, sig = [], []
    for k, g in pred.groupby("key"):
        a = g["actual"].to_numpy(float)
        w = {lab: _an_metrics(a, g[c])["wape"] for lab, c in
             (("base", "baseline"), ("chr", "chronos_p50"),
              ("lgb", "lgbm_p50"), ("fin", "final_p50"))}
        stage.append({"Target": k, "Baseline WAPE": pct(w["base"]),
                      "Chronos alone": pct(w["chr"]),
                      "LightGBM correction": pct(w["lgb"]),
                      "Gain from LightGBM": pct(w["base"] - w["lgb"]),
                      "Gain from large-payment add-on": pct(w["lgb"] - w["fin"]),
                      "Final WAPE": pct(w["fin"]),
                      "Total skill": pct(1 - w["fin"] / w["base"])})
        sig.append({"Target": k, "Final vs baseline": pct(1 - w["fin"] / w["base"]),
                    "p": f"{_an_boot(g, 'baseline', 'final_p50'):.3f}",
                    "Final vs LightGBM": pct(1 - w["fin"] / w["lgb"]),
                    "p ": f"{_an_boot(g, 'lgbm_p50', 'final_p50'):.3f}",
                    "LightGBM vs Chronos": pct(1 - w["lgb"] / w["chr"]) if pd.notna(w["chr"]) else "-",
                    "p  ": f"{_an_boot(g, 'chronos_p50', 'lgbm_p50'):.3f}"})
    sec.append(("A2. Where the accuracy comes from",
                pd.DataFrame(stage).to_html(index=False) +
                "<h3>Is the gain real?</h3>" + pd.DataFrame(sig).to_html(index=False) +
                "<p class='note'>p is a day-block paired bootstrap (2,000 resamples of whole "
                "test days): the probability of seeing this improvement if the two methods were "
                "really equally good. Below 0.05 the gain is safe to quote; a large skill with a "
                "high p means one or two lucky days are carrying it.</p>"))

    # A3. error structure ---------------------------------------------------
    cols = dict(pcts=["Value share", "Baseline WAPE", "Final WAPE", "Skill", "Bias", "P90 cov"])
    g_hour = _an_group(pred, ["key", "hour"], ["Target", "Hour"])
    g_dow = _an_group(pred, ["key", "dow"], ["Target", "Weekday (0=Mon)"])
    g_dt = _an_group(pred, ["key", "day_type"], ["Target", "Day type"])
    g_min = _an_group(pred, ["key", "minute_of_hour"], ["Target", "Issued at minute"])
    dec = []
    for k, g in pred.groupby("key"):
        if len(g) < 30:
            continue
        q = pd.qcut(g["actual"].rank(method="first"), 10, labels=False) + 1
        for i, gg in g.groupby(q):
            a = gg["actual"].to_numpy(float)
            m = _an_metrics(a, gg["final_p50"])
            dec.append({"Target": k, "Volume decile": int(i), "Windows": m["n"],
                        "Mean actual (GBP)": float(np.nanmean(a)),
                        "Baseline WAPE": _an_metrics(a, gg["baseline"])["wape"],
                        "Final WAPE": m["wape"], "Bias": m["bias"],
                        "P90 cov": _cov(a, gg["final_p90"].to_numpy(float))})
    dec = pd.DataFrame(dec)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for k, g in dec.groupby("Target"):
        ax.plot(g["Volume decile"], g["Final WAPE"], marker="o", label=k)
    ax.set_title("Final WAPE by window-volume decile (1 = quietest, 10 = busiest)")
    ax.set_xlabel("decile of actual volume"); ax.set_ylabel("WAPE"); ax.legend(fontsize=8)
    img_dec = _img(fig, "step16_wape_by_volume_decile.png")
    sec.append(("A3. Where the errors sit",
                "<h3>By hour of day</h3>" + _an_html(g_hour, **cols) +
                "<h3>By weekday</h3>" + _an_html(g_dow, **cols) +
                "<h3>By day type</h3>" + _an_html(g_dt, **cols) +
                "<h3>By issue point inside the hour</h3>" + _an_html(g_min, **cols) +
                "<h3>By size of the window</h3>" +
                _an_html(dec, pcts=["Baseline WAPE", "Final WAPE", "Bias", "P90 cov"],
                         nums=["Mean actual (GBP)"]) + img_dec +
                "<p class='note'>Value share is how much of the test-period turnover each group "
                "carries: a bad WAPE on a group with a 1% value share costs far less than a small "
                "slip on a group carrying 20%. Quiet windows (low deciles) always look bad in "
                "percentage terms because the denominator is tiny - judge them on absolute "
                "pounds, and judge the busy deciles on WAPE.</p>"))

    # A4. error distribution and concentration ------------------------------
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    dist = pd.DataFrame({k: np.nanpercentile(g["rel"], qs) for k, g in pred.groupby("key")},
                        index=[f"P{q}" for q in qs])
    conc = []
    for k, g in pred.groupby("key"):
        v = np.sort(g["ae"].dropna().to_numpy(float))[::-1]
        tot = v.sum()
        row = {"Target": k, "Lag-1 error autocorrelation": _an_lag1(g)}
        for s in (0.01, 0.05, 0.10, 0.25):
            row[f"Worst {int(s * 100)}%"] = (v[:max(1, int(len(v) * s))].sum() / tot
                                             if tot > 0 else np.nan)
        conc.append(row)
    conc = pd.DataFrame(conc)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, s in zip(axs, SERIES):
        g = short[short["series"] == s]
        ax.hist(g["rel"].clip(-2, 2), bins=60, color=COLORS["final"])
        ax.axvline(0, color="k", lw=1)
        ax.set_title(f"{s}: relative error (forecast - actual) / actual")
        ax.set_xlabel("relative error"); ax.set_ylabel("windows")
    img_hist = _img(fig, "step16_error_histogram.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, g in pred.groupby("key"):
        v = np.sort(g["ae"].dropna().to_numpy(float))[::-1]
        ax.plot(np.arange(1, len(v) + 1) / len(v), np.cumsum(v) / v.sum(), label=k)
    ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls="--")
    ax.set_title("Concentration of absolute error")
    ax.set_xlabel("share of windows (worst first)"); ax.set_ylabel("share of total error")
    ax.legend(fontsize=8)
    img_lorenz = _img(fig, "step16_error_concentration.png")
    sec.append(("A4. Shape of the error distribution",
                "<h3>Percentiles of the relative error</h3>" +
                dist.map(lambda x: f"{x:.1%}").to_html() + img_hist +
                "<h3>Concentration and persistence</h3>" +
                _an_html(conc, pcts=[c for c in conc.columns if c.startswith("Worst")],
                         rnd=["Lag-1 error autocorrelation"]) + img_lorenz +
                "<p class='note'>A dashed diagonal would mean every window contributes equally. "
                "The further the curve bends to the top-left, the more the whole error is a "
                "handful of windows - those are an exception-handling problem, not a modelling "
                "problem. Lag-1 autocorrelation above about 0.2 means consecutive errors lean the "
                "same way, so a short-term correction on the last error would still add value.</p>"))

    # A5. calibration of the P10/P50/P90 band -------------------------------
    cal = []
    for k, g in pred.groupby("key"):
        a = g["actual"].to_numpy(float)
        pb = {q: _an_pinball(a, g[c].to_numpy(float), q) for q, c in AN_LEVELS}
        cal.append({"Target": k,
                    "P10 cov (10%)": float(np.mean(a < g["final_p10"].to_numpy(float))),
                    "P50 cov (50%)": float(np.mean(a < g["final_p50"].to_numpy(float))),
                    "P90 cov (90%)": _cov(a, g["final_p90"].to_numpy(float)),
                    "P10-P90 hit (80%)": float(g["in_band"].mean()),
                    "Mean band width (GBP)": float(np.nanmean(g["final_p90"] - g["final_p10"])),
                    "Band / mean actual": float(np.nanmean(g["final_p90"] - g["final_p10"])
                                                / np.nanmean(a)),
                    "Pinball P10": pb[0.1], "Pinball P50": pb[0.5], "Pinball P90": pb[0.9],
                    "Mean pinball": float(np.mean(list(pb.values())))})
    cal = pd.DataFrame(cal)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, s in zip(axs, SERIES):
        g = short[short["series"] == s]
        cov = g.assign(hit=g["actual"] <= g["final_p90"]).groupby("hour")["hit"].mean()
        ax.plot(cov.index, cov.values, marker="o", color=COLORS["final"], label="P90 coverage")
        ax.plot(cov.index, g.groupby("hour")["in_band"].mean().values, marker="s",
                color=COLORS["chronos"], label="P10-P90 hit rate")
        ax.axhline(0.9, color="k", lw=0.8, ls="--")
        ax.axhline(0.8, color="grey", lw=0.8, ls=":")
        ax.set_ylim(0, 1.05); ax.set_title(f"{s}: band coverage by hour")
        ax.set_xlabel("hour"); ax.legend(fontsize=7)
    img_cov = _img(fig, "step16_coverage_by_hour.png")
    sec.append(("A5. Is the uncertainty band honest?",
                _an_html(cal, pcts=["P10 cov (10%)", "P50 cov (50%)", "P90 cov (90%)",
                                    "P10-P90 hit (80%)", "Band / mean actual"],
                         nums=["Mean band width (GBP)", "Pinball P10", "Pinball P50",
                               "Pinball P90", "Mean pinball"]) + img_cov +
                "<p class='note'>Coverage is how often the actual really fell below that "
                "quantile; each one should land on its nominal level. Under-covering P90 means "
                "the band is too narrow and liquidity buffers built on it will be breached more "
                "often than advertised; over-covering means it is too wide to be useful. Pinball "
                "loss scores sharpness and coverage together - lower is better, and it is the "
                "number to watch when comparing two calibrations that both look on target.</p>"))

    # A6. large payments ----------------------------------------------------
    lgr = []
    for k, g in pred.groupby("key"):
        big, sml = g[g["large_actual"] > 0], g[g["large_actual"] <= 0]
        addon = (big["final_p50"] - big["lgbm_p50"]).sum()
        lgr.append({"Target": k, "Windows with a large payment": len(big),
                    "Share of windows": len(big) / len(g),
                    "Share of value": big["large_actual"].sum() / g["actual"].sum(),
                    "Share of total error": big["ae"].sum() / g["ae"].sum(),
                    "WAPE without large": _an_metrics(sml["actual"], sml["final_p50"])["wape"],
                    "WAPE with large": _an_metrics(big["actual"], big["final_p50"])["wape"],
                    "Inside P90": float(np.mean(big["actual"] <= big["final_p90"])) if len(big) else np.nan,
                    "Add-on / large value": addon / big["large_actual"].sum() if len(big) else np.nan})
    lgr = pd.DataFrame(lgr)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(lgr))
    ax.bar(x - 0.2, lgr["WAPE without large"], 0.4, color=COLORS["final"], label="no large payment")
    ax.bar(x + 0.2, lgr["WAPE with large"], 0.4, color=COLORS["chronos"], label="large payment in window")
    ax.set_xticks(x); ax.set_xticklabels(lgr["Target"]); ax.set_ylabel("WAPE")
    ax.set_title("Cost of large payments"); ax.legend(fontsize=8)
    img_lg = _img(fig, "step16_large_payments.png")
    sec.append(("A6. Large payments: the hard part",
                _an_html(lgr, pcts=["Share of windows", "Share of value", "Share of total error",
                                    "WAPE without large", "WAPE with large", "Inside P90",
                                    "Add-on / large value"]) + img_lg +
                "<p class='note'>Compare 'share of windows' with 'share of total error': that gap "
                "is the price of one-off payments. The add-on only adds the typical size for that "
                "weekday, hour and month-end position, so 'add-on / large value' well under 100% "
                "is expected - the band, not the P50, is what should absorb them. If 'inside P90' "
                "is low, feed known scheduled payments in through SCHEDULED_PATH rather than "
                "widening the band everywhere.</p>"))

    # A7. stability over the test period ------------------------------------
    dates = np.sort(pred["date"].unique())
    mid = dates[len(dates) // 2]
    stab = []
    for k, g in pred.groupby("key"):
        h1, h2 = g[g["date"] < mid], g[g["date"] >= mid]
        w1 = _an_metrics(h1["actual"], h1["final_p50"])["wape"]
        w2 = _an_metrics(h2["actual"], h2["final_p50"])["wape"]
        d = g.groupby("date")[["ae", "actual"]].sum()
        dw = (d["ae"] / d["actual"]).dropna()
        stab.append({"Target": k, "First half WAPE": w1, "Second half WAPE": w2,
                     "Change": w2 - w1, "Best day": dw.min(), "Median day": dw.median(),
                     "Worst day": dw.max(),
                     "Days above 2x median": float(np.mean(dw > 2 * dw.median()))})
    dly = pred.groupby("date")[["ae", "actual"]].sum()
    dly["WAPE"] = dly["ae"] / dly["actual"]
    wd = (dly.nlargest(10, "WAPE")
          .join(pred.groupby("date")["day_type"].first())
          .join(pred.groupby("date")["large_actual"].sum().rename("large payments (GBP)"))
          .reset_index())
    sec.append(("A7. Does accuracy hold across the test period?",
                _an_html(pd.DataFrame(stab),
                         pcts=["First half WAPE", "Second half WAPE", "Change", "Best day",
                               "Median day", "Worst day", "Days above 2x median"]) +
                f"<p class='note'>Split at {mid}. A second half materially worse than the first "
                "is drift - retrain before trusting the headline number.</p>"
                "<h3>10 worst days</h3>" +
                _an_html(wd[["date", "day_type", "WAPE", "actual", "large payments (GBP)"]],
                         pcts=["WAPE"], nums=["actual", "large payments (GBP)"]) +
                _img_file("step15_drift.png")))

    # A8. what was actually fitted ------------------------------------------
    cfg_rows = []
    for k, e in best["keys"].items():
        for name, m in e["models"].items():
            cfg_rows.append({"Target": k, "Model": name,
                             "learning_rate": e["params"]["learning_rate"],
                             "num_leaves": e["params"]["num_leaves"],
                             "min_data_in_leaf": e["params"]["min_data_in_leaf"],
                             "CV WAPE (tuning)": pct(e["fold_wape"]),
                             "Validation WAPE": pct(m.get("val_wape")),
                             "Trees (P50)": m["iters"].get("0.5"),
                             "P10 calibration shift": f"{m['c10']:+.3f}",
                             "P90 calibration shift": f"{m['c90']:+.3f}"})
    val_html = ""
    rp = ift.table_path(CFG, "results_table.csv")
    if rp.exists():
        r = pd.read_csv(rp)
        r = r[r["split"] == "validation"]
        if len(r):
            val_html = ("<h3>Validation, before the test run</h3>" +
                        r.pivot_table(index=["series", "horizon_min"], columns="method",
                                      values="wape").map(pct).to_html())
    sec.append(("A8. What was actually fitted",
                f"<p>Chosen model: <b>{best['chosen']}</b> "
                f"({'with' if best['chosen'] == 'B' else 'without'} Chronos as an input).</p>" +
                pd.DataFrame(cfg_rows).to_html(index=False) + val_html +
                "<p class='note'>Hyper-parameters come from 3 expanding-window folds on "
                "train+validation, so the validation WAPE beside them is close to honest but not "
                "untouched. The calibration shifts are the constants added to the raw P10/P90 to "
                "make validation coverage hit 10% and 90%; a large shift means the quantile "
                "regression itself was mis-calibrated and should be watched after each retrain. "
                "Test numbers above were produced once, after all of this was frozen.</p>"))

    # A9. read-out ----------------------------------------------------------
    a_all = pred["actual"].to_numpy(float)
    skill = 1 - _an_metrics(a_all, pred["final_p50"])["wape"] / _an_metrics(a_all, pred["baseline"])["wape"]
    bias = _an_metrics(a_all, pred["final_p50"])["bias"]
    cov90 = _cov(a_all, pred["final_p90"].to_numpy(float))
    top5 = conc["Worst 5%"].mean()
    ac = conc["Lag-1 error autocorrelation"].mean()
    hard = g_dt.groupby("Day type")["Final WAPE"].mean()
    hr = g_hour.groupby("Hour")["Final WAPE"].mean()
    drift = np.mean([s["Change"] for s in stab])
    err_big = lgr["Share of total error"].mean()
    read = [
        f"Overall the final forecast is {abs(skill):.0%} "
        f"{'more accurate than' if skill >= 0 else 'WORSE than'} the seasonal baseline, and it "
        f"beats the baseline significantly on "
        f"{sum(1 for s_ in sig if float(s_['p']) < 0.05)} of {len(sig)} targets (p&lt;0.05).",
        f"Bias is {bias:+.1%} of turnover - "
        + ("small enough to ignore." if abs(bias) < 0.02 else
           "large enough to correct with a constant multiplier at the next retrain."),
        f"P90 coverage is {cov90:.0%} against a 90% target"
        + ("; the band is usable as a planning limit as it stands." if abs(cov90 - 0.9) < 0.03
           else " - recalibrate before anyone sizes a buffer on it."),
        f"The worst 5% of windows carry {top5:.0%} of all error and windows containing a large "
        f"payment carry {err_big:.0%} of it"
        + ("; the error is concentrated in a few exceptional payments rather than a "
           "systematically wrong level, so handle it as exceptions." if top5 > 0.4 else
           "; the error is spread fairly evenly, so it is the general level that needs work, "
           "not the exceptions."),
        f"Hardest day type: {hard.idxmax().replace('_', ' ')} ({hard.max():.0%} WAPE); "
        f"hardest hour: {int(hr.idxmax())}:00 ({hr.max():.0%}); "
        f"easiest hour: {int(hr.idxmin())}:00 ({hr.min():.0%}).",
        f"Lag-1 error autocorrelation is {ac:.2f}"
        + ("; consecutive errors lean the same way, so a last-error correction term is worth "
           "trying." if abs(ac) > 0.2 else "; errors are close to independent, so there is little "
           "short-term signal left to harvest."),
        f"Second half of the test period vs first half: {drift:+.1%} WAPE"
        + ("; no drift worth acting on." if abs(drift) < 0.02 else
           "; treat this as drift and shorten the retrain cycle."),
    ]
    sec.append(("A9. Read-out",
                "<div class='box'><b>What the numbers say</b><ul>" +
                "".join(f"<li>{x}</li>" for x in read) + "</ul></div>"))

    # assemble --------------------------------------------------------------
    toc = "".join(f"<li><a href='#an{i}'>{t}</a></li>" for i, (t, _) in enumerate(sec, 1))
    body = "".join(f"<h2 id='an{i}'>{t}</h2>{h}" for i, (t, h) in enumerate(sec, 1))
    block = (f"{AN_START}\n<h1>Detailed analysis</h1>\n"
             f"<p>Everything below is computed from the saved test predictions - no model is "
             f"refitted and no tuning decision depends on it.</p><ol>{toc}</ol>{body}\n{AN_END}\n")

    out = CFG["REPORTS_ROOT"] / "forecast_analysis.html"
    out.write_text(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Intraday CHAPS forecast - detailed analysis</title><style>{AN_STYLE}</style></head>
<body>{block}</body></html>""", encoding="utf-8")
    print(f"[analysis] written {out}")

    rep = CFG["REPORTS_ROOT"] / "forecast_report.html"
    if rep.exists():                                  # splice into the main report
        h = rep.read_text(encoding="utf-8")
        if AN_START in h and AN_END in h:             # replace an earlier run
            h = h[:h.index(AN_START)] + h[h.index(AN_END) + len(AN_END):]
        h = h.replace("</body>", block + "</body>") if "</body>" in h else h + block
        rep.write_text(h, encoding="utf-8")
        print(f"[analysis] detailed section added to {rep}")
    else:
        print("[analysis] forecast_report.html not found - run the report stage to merge")
    print(f"[analysis] done ({time.time()-t0:.1f}s)")


STAGES = {"lgbm": run_lgbm, "final": run_final, "report": run_reports}
STAGES["analysis"] = run_analysis                     # detailed deep-dive

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "all":
        for fn in STAGES.values():
            fn()
    elif stage in STAGES:
        STAGES[stage]()
    else:
        sys.exit(f"usage: python pipeline_stage2.py [{' | '.join(STAGES)} | all]")

"""
Clean a raw intraday bank-statement export (Effective Timestamp,
Amount (Local), UBR L6 Name, TSO Identifier -- same schema as your real
statement export) into an hourly, per-business Debit/Credit table that's
ready to feed straight into business_hourly_forecast.py-style seasonality
analysis and forecasting.

Handles the data-quality issues typical of a raw Excel export:
  - Amount stored as text with thousand separators / accounting-style
    negatives, e.g. "(1,234.00)" -> -1234.00
  - Timestamps in day-first DD-MM-YYYY HH:MM format, some unparseable
  - Fully blank trailer rows (common at the bottom of Excel exports)
  - Blank / whitespace-only business names -> filled with "Unclassified"
  - Inconsistent whitespace in business names (e.g. "Corporate  Bank ")
  - Exact duplicate rows
  - Sub-hourly, irregular transaction timestamps -> bucketed to the hour
  - Missing hours in the resulting series (no transactions that hour) ->
    filled with 0 so downstream models see a complete grid
  - Extreme amounts are flagged (not dropped) for manual review, since a
    genuinely huge transfer isn't necessarily a data error

Nothing is silently discarded without being counted: every drop/fill is
tallied in cleaning_report.csv so you can see exactly what changed.

Usage:
    python clean_business_statement_data.py --input "StatementData_Bank of Japan_210720....xlsx"
    python clean_business_statement_data.py --input data.xlsx --sheet "JPY - analysis"
    python clean_business_statement_data.py --input data.csv --drop-tso-y
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_DIR_DEFAULT = Path(
    r"C:\Users\reshm\OneDrive\Documents\work\timesfm\hybrid\outputs\cleaned_business_data"
)

TIMESTAMP_COL = "Effective Timestamp"
AMOUNT_COL = "Amount (Local)"
BUSINESS_COL = "UBR L6 Name"
TSO_COL = "TSO Identifier"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean a raw intraday statement export into an hourly, per-business Debit/Credit table."
    )
    p.add_argument("--input", required=True, help="Path to the raw .xlsx/.xls/.csv export.")
    p.add_argument("--sheet", default=0, help="Sheet name or index for .xlsx input (default: first sheet).")
    p.add_argument("--output-dir", default=str(OUT_DIR_DEFAULT))
    p.add_argument(
        "--unclassified-label", default="Unclassified",
        help="Fill value used for blank/missing business names.",
    )
    p.add_argument(
        "--drop-tso-y", action="store_true",
        help="Drop rows where TSO Identifier == 'Y' (if you want those excluded from analysis).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw(path: Path, sheet) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {suf}")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_amount(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # accounting negatives: (1234.00) -> -1234.00
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)
    s = s.replace({"": np.nan, "-": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")


def clean(df_raw: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    report: dict = {"rows_in": len(df_raw)}
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = {TIMESTAMP_COL, AMOUNT_COL, BUSINESS_COL} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {sorted(missing_cols)}. Found columns: {list(df.columns)}"
        )

    # fully blank trailer rows (common at the bottom of Excel exports)
    df = df.dropna(how="all")
    report["rows_after_drop_blank_rows"] = len(df)

    # timestamp: day-first DD-MM-YYYY, coerce unparseable to NaT and drop
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], dayfirst=True, errors="coerce")
    report["rows_dropped_bad_timestamp"] = int(df[TIMESTAMP_COL].isna().sum())
    df = df.dropna(subset=[TIMESTAMP_COL])

    # amount: strip thousands separators / accounting negatives, coerce, drop unparseable
    df[AMOUNT_COL] = clean_amount(df[AMOUNT_COL])
    report["rows_dropped_bad_amount"] = int(df[AMOUNT_COL].isna().sum())
    df = df.dropna(subset=[AMOUNT_COL])

    # business name: trim, collapse whitespace, fill blanks
    df[BUSINESS_COL] = df[BUSINESS_COL].astype(str).str.strip()
    df[BUSINESS_COL] = df[BUSINESS_COL].replace({"": np.nan, "nan": np.nan, "None": np.nan})
    report["rows_blank_business_name_filled"] = int(df[BUSINESS_COL].isna().sum())
    df[BUSINESS_COL] = df[BUSINESS_COL].fillna(args.unclassified_label)
    df[BUSINESS_COL] = df[BUSINESS_COL].str.replace(r"\s+", " ", regex=True)

    # optional TSO filter
    if TSO_COL in df.columns:
        df[TSO_COL] = df[TSO_COL].astype(str).str.strip().str.upper()
        if args.drop_tso_y:
            before = len(df)
            df = df[df[TSO_COL] != "Y"]
            report["rows_dropped_tso_y"] = before - len(df)
    else:
        report["tso_column_missing"] = True

    # exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    report["rows_dropped_exact_duplicates"] = before - len(df)

    # split into debit/credit by sign convention (negative = debit, positive = credit)
    df["Debit Amount"] = np.where(df[AMOUNT_COL] < 0, -df[AMOUNT_COL], 0.0)
    df["Credit Amount"] = np.where(df[AMOUNT_COL] > 0, df[AMOUNT_COL], 0.0)

    # flag (don't drop) extreme amounts for manual review
    p99 = df[AMOUNT_COL].abs().quantile(0.99)
    df["flag_extreme_amount"] = df[AMOUNT_COL].abs() > p99 * 3
    report["rows_flagged_extreme_amount"] = int(df["flag_extreme_amount"].sum())

    report["rows_out_transaction_level"] = len(df)
    return df, report


# ---------------------------------------------------------------------------
# Hourly bucketing (matches business_hourly_forecast.py's schema)
# ---------------------------------------------------------------------------
def bucket_hourly(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Hour"] = d[TIMESTAMP_COL].dt.floor("h")
    agg = (
        d.groupby(["Hour", BUSINESS_COL])[["Debit Amount", "Credit Amount"]]
        .sum()
        .reset_index()
        .rename(columns={"Hour": TIMESTAMP_COL, BUSINESS_COL: "UBR L6 Name"})
    )
    return agg


def fill_hourly_grid(hourly: pd.DataFrame) -> pd.DataFrame:
    businesses = hourly["UBR L6 Name"].unique()
    full_idx = pd.date_range(hourly[TIMESTAMP_COL].min(), hourly[TIMESTAMP_COL].max(), freq="h")
    frames = []
    for b in businesses:
        sub = hourly[hourly["UBR L6 Name"] == b].set_index(TIMESTAMP_COL)[["Debit Amount", "Credit Amount"]]
        sub = sub.reindex(full_idx).fillna(0.0)
        sub["UBR L6 Name"] = b
        sub.index.name = "Effective Timestamp"
        frames.append(sub.reset_index())
    out = pd.concat(frames, ignore_index=True)
    out["Net Amount"] = out["Credit Amount"] - out["Debit Amount"]
    return out.sort_values(["UBR L6 Name", "Effective Timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw data from {input_path} ...")
    raw = load_raw(input_path, args.sheet)
    print(f"  {len(raw)} raw rows, columns: {list(raw.columns)}")

    print("\nCleaning ...")
    clean_df, report = clean(raw, args)
    for k, v in report.items():
        print(f"  {k}: {v}")
    clean_df.to_csv(out_dir / "transaction_level_clean.csv", index=False)

    print("\nBucketing to hourly, per business ...")
    hourly = bucket_hourly(clean_df)
    hourly_full = fill_hourly_grid(hourly)
    hourly_full.to_csv(out_dir / "hourly_business_data_clean.csv", index=False)
    print(f"  {len(hourly_full)} hourly rows, {hourly_full['UBR L6 Name'].nunique()} businesses, "
          f"{hourly_full['Effective Timestamp'].min()} -> {hourly_full['Effective Timestamp'].max()}")

    pd.DataFrame([report]).to_csv(out_dir / "cleaning_report.csv", index=False)

    print(f"\nDone. Outputs written to: {out_dir}")
    print("  transaction_level_clean.csv   -- cleaned, deduped transaction rows")
    print("  hourly_business_data_clean.csv -- hourly, per-business Debit/Credit/Net,")
    print("                                    same schema as business_hourly_forecast.py's")
    print("                                    synthetic_hourly_business_data.csv -- drop this")
    print("                                    straight into that pipeline's analysis/forecast steps.")
    print("  cleaning_report.csv           -- counts of everything dropped/filled/flagged")


if __name__ == "__main__":
    main()

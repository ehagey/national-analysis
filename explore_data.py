"""Data exploration script.

Loads the raw Sensor Tower exports and prints a comprehensive summary:
  - Row / column counts
  - Unique values for every categorical field
  - Date range and coverage
  - Unified Name vs App Name mapping (consolidation check)
  - Per-app time-series completeness (gaps, zeros, nulls)
  - Final Date × App pivots produced by utils/data.load()
  - Cross-file consistency between downloads.csv and dau.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "input_data"

SEP   = "─" * 72
SEP_S = "─" * 50


def hr(title: str = "", wide: bool = False) -> None:
    s = SEP if wide else SEP_S
    if title:
        print(f"\n{s}\n  {title}\n{s}")
    else:
        print(s)


def pct(n: int, total: int) -> str:
    return f"{n:>8,}  ({100*n/total:.1f}%)" if total else f"{n:>8,}"


def load_raw(fname: str, metric: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / fname, encoding="utf-16", sep="\t")
    df["Date"]  = pd.to_datetime(df["Date"],  errors="coerce")
    df[metric]  = pd.to_numeric(df[metric],   errors="coerce")
    return df


def pivot(df: pd.DataFrame, metric: str,
          country=None, platform=None) -> pd.DataFrame:
    PLATFORM_MAP = {"iOS": "App Store", "Android": "Google Play"}
    d = df.copy()
    if platform is not None and "Platform" in d.columns:
        d = d[d["Platform"] == PLATFORM_MAP.get(platform, platform)]
    if country == "US":
        d = d[d["Country / Region"] == "US"]
    elif country == "ex-US":
        d = d[d["Country / Region"] != "US"]
    return (d.groupby(["Date", "Unified Name"])[metric]
             .sum()
             .unstack())


# ── 1. Raw file overview ──────────────────────────────────────────────────────

for fname, metric in [("downloads.csv", "Downloads"), ("dau.csv", "DAU")]:
    hr(f"RAW FILE: {fname}  [{metric}]", wide=True)
    df = load_raw(fname, metric)

    print(f"\n  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    hr("Categorical fields — cardinality & unique values")
    cat_cols = ["Unified Name", "App Name", "Unified Publisher Name",
                "Publisher Name", "Country / Region", "Platform", "Device"]
    for col in cat_cols:
        if col not in df.columns:
            continue
        vals = sorted(df[col].dropna().unique())
        print(f"\n  {col}  ({len(vals)} unique)")
        for v in vals:
            cnt = (df[col] == v).sum()
            print(f"    {str(v):<44}  {pct(cnt, len(df))}")

    hr("Unified Name → App Name mapping  (consolidation check)")
    mapping = (df.groupby(["Unified Name", "App Name"])
                 .size()
                 .reset_index(name="rows")
                 .sort_values(["Unified Name", "rows"], ascending=[True, False]))
    for unified, grp in mapping.groupby("Unified Name"):
        tag = "  ← MULTIPLE VARIANTS" if len(grp) > 1 else ""
        print(f"\n  [{unified}]{tag}")
        for _, row in grp.iterrows():
            print(f"    App Name: {row['App Name']!r:<48}  rows: {row['rows']:,}")

    hr("Date range")
    print(f"  Min date : {df['Date'].min().date()}")
    print(f"  Max date : {df['Date'].max().date()}")
    print(f"  Null dates: {df['Date'].isna().sum():,}")

    hr(f"Metric distribution: {metric}")
    m = df[metric]
    print(f"  Null / NaN : {m.isna().sum():,}")
    print(f"  Zeros      : {(m == 0).sum():,}")
    print(f"  Negatives  : {(m < 0).sum():,}")
    for stat, val in m.dropna().describe(
            percentiles=[.01, .05, .25, .5, .75, .95, .99]).items():
        print(f"  {stat:<12}: {val:>15,.1f}")


# ── 2. Pivoted series (as seen by the pipeline) ───────────────────────────────

for fname, metric in [("downloads.csv", "Downloads"), ("dau.csv", "DAU")]:
    df_raw = load_raw(fname, metric)
    slices = [
        ("US / Combined",       "US",    None),
        ("US / iOS",            "US",    "iOS"),
        ("US / Android",        "US",    "Android"),
        ("Global / Combined",   None,    None),
        ("Global-ex-US",        "ex-US", None),
    ]
    hr(f"PIVOTED SERIES — {metric}  (Date × App as used by pipeline)", wide=True)

    for label, country, platform in slices:
        piv = pivot(df_raw, metric, country, platform)
        hr(f"Slice: {label}")
        print(f"  Shape: {piv.shape}  ({piv.shape[0]} dates × {piv.shape[1]} apps)")
        print(f"  Date range: {piv.index.min().date()} → {piv.index.max().date()}")
        print(f"  Apps: {list(piv.columns)}")
        print(f"\n  {'App':<38} {'NaN':>5}  {'Zeros':>5}  "
              f"{'Min':>12}  {'Mean':>12}  {'Max':>12}")
        print("  " + "─" * 90)
        for app in piv.columns:
            s     = piv[app]
            valid = s.dropna()
            print(f"  {app:<38} {s.isna().sum():>5}  {(s==0).sum():>5}  "
                  f"{valid.min():>12,.0f}  {valid.mean():>12,.0f}  "
                  f"{valid.max():>12,.0f}")

        dates = piv.dropna(how="all").index.sort_values()
        if len(dates) > 1:
            gaps = dates.to_series().diff().dropna()
            gaps = gaps[gaps > pd.Timedelta("1D")]
            print(f"\n  Date gaps: {len(gaps) if not gaps.empty else 'none'}")
            for d, g in gaps.items():
                print(f"    {d.date()}  ({g.days} days)")
        print()


# ── 3. Cross-file consistency ─────────────────────────────────────────────────

hr("CROSS-FILE CONSISTENCY  (apps and dates in both files)", wide=True)
dl_piv  = pivot(load_raw("downloads.csv", "Downloads"), "Downloads", "US")
dau_piv = pivot(load_raw("dau.csv",       "DAU"),       "DAU",       "US")

dl_apps,  dau_apps  = set(dl_piv.columns), set(dau_piv.columns)
dl_dates, dau_dates = set(dl_piv.index),   set(dau_piv.index)

print(f"\n  Downloads apps : {sorted(dl_apps)}")
print(f"  DAU apps       : {sorted(dau_apps)}")
print(f"\n  Only Downloads : {sorted(dl_apps - dau_apps) or 'none'}")
print(f"  Only DAU       : {sorted(dau_apps - dl_apps) or 'none'}")
print(f"\n  Downloads dates: {min(dl_dates).date()} → {max(dl_dates).date()}  "
      f"({len(dl_dates)} dates)")
print(f"  DAU dates      : {min(dau_dates).date()} → {max(dau_dates).date()}  "
      f"({len(dau_dates)} dates)")
only_dl  = sorted(dl_dates  - dau_dates)
only_dau = sorted(dau_dates - dl_dates)
if only_dl:
    print(f"  Dates only in Downloads: {[d.date() for d in only_dl]}")
if only_dau:
    print(f"  Dates only in DAU      : {[d.date() for d in only_dau]}")
if not only_dl and not only_dau:
    print("  Date sets are identical ✓")
print()

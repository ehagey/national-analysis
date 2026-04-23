"""Two-unit and multi-app spillover DiD pipeline (did_spill.py entry point).

Model A — Two-unit (Claude + ChatGPT only):
    log Y_it = α_i + γ·t + Σ φ_d·dow_d
             + δ_CL·(Post×Claude) + δ_GPT·(Post×ChatGPT) + ε_it

Model B — Multi-app (all apps):
    log Y_it = α_i + γ·t
             + λ_CL·(Claude×t) + λ_GPT·(ChatGPT×t)
             + Σ φ_d·dow_d
             + δ_CL·(Post×Claude) + δ_GPT·(Post×ChatGPT) + ε_it

HAC lags: dynamic Newey-West rule by default; explicit override for sensitivity.
"""

from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

from pipeline.estimation import hac_lags
from pipeline.types import SpilloverResult
from utils.config import (
    APP_CL, APP_GPT, ES_K, EVENT,
    MIN_PRE_DAYS, N_PERM, PLACEBO_CUTOFF,
    SHOCK_START, SONNET_REL, WINDOW_DEFAULT, WINDOWS,
)
from utils.data import load
from utils.plot_style import (
    EVENT_COLOR, HIST_COLOR, HIST_EDGE, app_color, app_label, style_ax,
)
from utils.reporting import VERSIONS, RNG_SEED, output_dir, rng, save_json
from utils.stats import stars

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

DATA_DIR  = Path(__file__).resolve().parent.parent / "input_data"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots" / "did_spill"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = output_dir("did_spill")

_WINDOW_START: dict[str, pd.Timestamp] = {lbl: ts for lbl, ts in WINDOWS}

HAC_LAGS_SENS = [5, 10, 12, 14]

CL_COLOR  = "#2563EB"
GPT_COLOR = "#DC2626"

FORMULA_A        = "log_Y ~ C(app) + t + C(dow) + DiD_CL + DiD_GPT"
FORMULA_A_SINGLE = "log_Y ~ C(app) + t + C(dow) + DiD_CL"  # for unit-placebo panels where DiD_GPT≡0
FORMULA_B      = "log_Y ~ C(app) + t + CLxt + GPTxt + C(dow) + DiD_CL + DiD_GPT"
FORMULA_B_QUAD = ("log_Y ~ C(app) + t + t2 + CLxt + GPTxt + CLxt2 + GPTxt2 "
                  "+ C(dow) + DiD_CL + DiD_GPT")


# ── Panel builders ─────────────────────────────────────────────────────────────

def build_panel_a(
    df: pd.DataFrame,
    window_start: pd.Timestamp | None = None,
    post_start: pd.Timestamp | None = None,
    app_cl: str = APP_CL,
    app_gpt: str = APP_GPT,
    drop_date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    date_end_exclusive: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    if window_start is None:
        window_start = WINDOW_DEFAULT
    if post_start is None:
        post_start = EVENT
    if app_cl not in df.columns or app_gpt not in df.columns:
        return None

    win  = df[df.index >= window_start][[app_cl, app_gpt]]
    rows = []
    for app in [app_cl, app_gpt]:
        col     = win[app]
        n_zeros = int((col == 0).sum())
        if n_zeros:
            print(f"  [warn] {app}: {n_zeros} zero(s) → NaN before log")
        tmp = pd.DataFrame({
            "log_Y": np.log(col.replace(0, np.nan)),
            "app":   app,
        }).reset_index(names="date")
        rows.append(tmp)

    panel = pd.concat(rows).dropna(subset=["log_Y"])

    if drop_date_ranges:
        for lo, hi in drop_date_ranges:
            panel = panel[~((panel["date"] >= lo) & (panel["date"] < hi))]
    if date_end_exclusive is not None:
        panel = panel[panel["date"] < date_end_exclusive]

    panel = panel.sort_values(["date", "app"]).reset_index(drop=True)
    panel["t"]       = (panel["date"] - window_start).dt.days.astype(float)
    panel["Post"]    = (panel["date"] >= post_start).astype(int)
    panel["isCL"]    = (panel["app"] == app_cl).astype(int)
    panel["isGPT"]   = (panel["app"] == app_gpt).astype(int)
    panel["DiD_CL"]  = panel["isCL"]  * panel["Post"]
    panel["DiD_GPT"] = panel["isGPT"] * panel["Post"]
    panel["dow"]     = panel["date"].dt.dayofweek.astype(str)
    return panel


def build_panel_b(
    df: pd.DataFrame,
    window_start: pd.Timestamp | None = None,
    post_start: pd.Timestamp | None = None,
    app_cl: str = APP_CL,
    app_gpt: str = APP_GPT,
    drop_date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    date_end_exclusive: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    if window_start is None:
        window_start = WINDOW_DEFAULT
    if post_start is None:
        post_start = EVENT

    apps = list(df.columns)
    if app_cl not in apps or app_gpt not in apps:
        return None

    win  = df[df.index >= window_start][apps]
    rows = []
    for app in apps:
        col     = win[app]
        n_zeros = int((col == 0).sum())
        if n_zeros:
            print(f"  [warn] {app}: {n_zeros} zero(s) → NaN before log")
        tmp = pd.DataFrame({
            "log_Y": np.log(col.replace(0, np.nan)),
            "app":   app,
        }).reset_index(names="date")
        rows.append(tmp)

    panel = pd.concat(rows).dropna(subset=["log_Y"])

    if drop_date_ranges:
        for lo, hi in drop_date_ranges:
            panel = panel[~((panel["date"] >= lo) & (panel["date"] < hi))]
    if date_end_exclusive is not None:
        panel = panel[panel["date"] < date_end_exclusive]

    panel = panel.sort_values(["date", "app"]).reset_index(drop=True)
    panel["t"]       = (panel["date"] - window_start).dt.days.astype(float)
    panel["Post"]    = (panel["date"] >= post_start).astype(int)
    panel["isCL"]    = (panel["app"] == app_cl).astype(int)
    panel["isGPT"]   = (panel["app"] == app_gpt).astype(int)
    panel["CLxt"]    = panel["isCL"]  * panel["t"]
    panel["GPTxt"]   = panel["isGPT"] * panel["t"]
    panel["t2"]      = panel["t"] ** 2
    panel["CLxt2"]   = panel["isCL"]  * panel["t2"]
    panel["GPTxt2"]  = panel["isGPT"] * panel["t2"]
    panel["DiD_CL"]  = panel["isCL"]  * panel["Post"]
    panel["DiD_GPT"] = panel["isGPT"] * panel["Post"]
    panel["dow"]     = panel["date"].dt.dayofweek.astype(str)
    return panel


# ── Estimation ────────────────────────────────────────────────────────────────

def _fit(formula: str, panel: pd.DataFrame, se_type: str = "HAC",
         lags: int | None = None) -> Any:
    if lags is None:
        lags = hac_lags(len(panel))
    if se_type == "HAC":
        return smf.ols(formula, data=panel).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags})
    return smf.ols(formula, data=panel).fit(cov_type="HC3")


def _extract(res: Any) -> tuple[tuple, tuple]:
    cl  = (float(res.params.get("DiD_CL",  np.nan)),
           float(res.bse.get("DiD_CL",     np.nan)),
           float(res.pvalues.get("DiD_CL", np.nan)))
    gpt = (float(res.params.get("DiD_GPT",  np.nan)),
           float(res.bse.get("DiD_GPT",     np.nan)),
           float(res.pvalues.get("DiD_GPT", np.nan)))
    return cl, gpt


def run_model(
    df: pd.DataFrame,
    model: str = "B",
    window_start: pd.Timestamp | None = None,
    post_start: pd.Timestamp | None = None,
    se_type: str = "HAC",
    hac_lags_override: int | None = None,
    trend: str = "linear",
    drop_date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    date_end_exclusive: pd.Timestamp | None = None,
    app_cl: str = APP_CL,
    app_gpt: str = APP_GPT,
) -> SpilloverResult | None:
    if window_start is None:
        window_start = WINDOW_DEFAULT
    if post_start is None:
        post_start = EVENT

    if model == "A":
        panel   = build_panel_a(df, window_start, post_start, app_cl, app_gpt,
                                drop_date_ranges, date_end_exclusive)
        formula = FORMULA_A
    else:
        panel   = build_panel_b(df, window_start, post_start, app_cl, app_gpt,
                                drop_date_ranges, date_end_exclusive)
        formula = FORMULA_B if trend == "linear" else FORMULA_B_QUAD

    if panel is None or panel["Post"].nunique() < 2:
        return None

    lags = hac_lags_override if hac_lags_override is not None else hac_lags(len(panel))
    try:
        res = _fit(formula, panel, se_type, lags)
    except Exception:
        return None

    cl, gpt   = _extract(res)
    n_units   = panel["app"].nunique()
    pct       = lambda c: float((np.exp(c) - 1) * 100)

    return SpilloverResult(
        coef_cl        = cl[0],
        pct_change_cl  = pct(cl[0]),
        pval_cl        = cl[2],
        se_cl          = cl[1],
        coef_gpt       = gpt[0],
        pct_change_gpt = pct(gpt[0]),
        pval_gpt       = gpt[2],
        se_gpt         = gpt[1],
        model          = model,
        hac_lags       = lags,
        n_pre          = int((panel["Post"] == 0).sum() // n_units),
        n_post         = int((panel["Post"] == 1).sum() // n_units),
        window_start   = window_start,
        post_start     = post_start,
    )


def _lag_cl(k: int) -> str:
    return f"cl_dm{abs(k)}" if k < 0 else f"cl_d{k}"


def _lag_gpt(k: int) -> str:
    return f"gpt_dm{abs(k)}" if k < 0 else f"gpt_d{k}"


def run_event_study(
    df: pd.DataFrame,
    window_start: pd.Timestamp | None = None,
) -> dict[int, tuple[float, float, float, float]] | None:
    panel = build_panel_b(df, window_start)
    if panel is None:
        return None

    panel = panel.copy()
    panel["k"] = (panel["date"] - EVENT).dt.days.clip(-ES_K, ES_K).astype(int)
    ks = [k for k in range(-ES_K, ES_K + 1) if k != -1]

    for k in ks:
        panel[_lag_cl(k)]  = ((panel["isCL"]  == 1) & (panel["k"] == k)).astype(int)
        panel[_lag_gpt(k)] = ((panel["isGPT"] == 1) & (panel["k"] == k)).astype(int)

    cl_terms  = " + ".join(_lag_cl(k)  for k in ks)
    gpt_terms = " + ".join(_lag_gpt(k) for k in ks)
    f   = f"log_Y ~ C(app) + t + CLxt + GPTxt + C(dow) + {cl_terms} + {gpt_terms}"
    lags = hac_lags(len(panel))
    try:
        res = _fit(f, panel, "HAC", lags)
    except Exception as e:
        print(f"  [warn] event study estimation failed: {e}")
        return None

    coefs: dict[int, tuple[float, float, float, float]] = {-1: (0.0, 0.0, 0.0, 0.0)}
    for k in ks:
        coefs[k] = (
            float(res.params.get(_lag_cl(k),  np.nan)),
            float(res.bse.get(_lag_cl(k),     np.nan)),
            float(res.params.get(_lag_gpt(k), np.nan)),
            float(res.bse.get(_lag_gpt(k),    np.nan)),
        )
    return coefs


# ── Print tables ──────────────────────────────────────────────────────────────

def _row_r(r: SpilloverResult) -> str:
    return (f"{r.coef_cl:+8.3f} {r.pct_change_cl:+8.1f}% "
            f"{r.pval_cl:.3f}{stars(r.pval_cl)}   "
            f"{r.coef_gpt:+8.3f} {r.pct_change_gpt:+8.1f}% "
            f"{r.pval_gpt:.3f}{stars(r.pval_gpt)}")


def print_diagnostics(dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    print(f"{'─'*70}\n  Residual Diagnostics\n{'─'*70}")
    for win_label, ws in WINDOWS:
        for model_name, build_fn, formula in [
            ("A", build_panel_a, FORMULA_A),
            ("B", build_panel_b, FORMULA_B),
        ]:
            for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
                panel = build_fn(df, ws)
                if panel is None:
                    continue
                lags  = hac_lags(len(panel))
                res   = _fit(formula, panel, "HAC", lags)
                resid = res.resid
                dw    = durbin_watson(resid)
                lb    = acorr_ljungbox(resid, lags=[10, 12], return_df=True)
                print(f"  {win_label} Model {model_name} {metric:<12} "
                      f"DW={dw:.2f}  LB(10)p={lb.loc[10,'lb_pvalue']:.3f}  "
                      f"LB(12)p={lb.loc[12,'lb_pvalue']:.3f}")
    print()


def print_main(df: pd.DataFrame, metric: str) -> None:
    print(f"{'─'*90}\n  Main Spillover Results — {metric}\n{'─'*90}")
    print(f"  {'Model':<8} {'Window':<6} {'SE':<8} "
          f"{'δ_CL':>8} {'Δ%_CL':>8} {'p_CL':>8}   "
          f"{'δ_GPT':>8} {'Δ%_GPT':>8} {'p_GPT':>8}")
    print("  " + "─" * 88)
    for win_label, ws in WINDOWS:
        for model in ["A", "B"]:
            for lags in HAC_LAGS_SENS:
                r = run_model(df, model, ws, hac_lags_override=lags)
                if r:
                    print(f"  {model:<8} {win_label:<6} {'HAC-'+str(lags):<8} {_row_r(r)}")
            r = run_model(df, model, ws, se_type="HC3")
            if r:
                print(f"  {model:<8} {win_label:<6} {'HC3':<8} {_row_r(r)}")
        print()


def print_quadratic_sensitivity(dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    print(f"{'─'*90}\n  Quadratic Trend Sensitivity — Model B (FEB1, HAC-5/-10)\n{'─'*90}")
    print(f"  {'Metric':<12} {'Trend':<12} {'SE':<8} "
          f"{'δ_CL':>8} {'Δ%_CL':>8} {'p_CL':>8}   "
          f"{'δ_GPT':>8} {'Δ%_GPT':>8} {'p_GPT':>8}")
    print("  " + "─" * 88)
    for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
        for trend_label, trend_type in [("Linear", "linear"), ("Quadratic", "quadratic")]:
            for lags in [5, 10]:
                r = run_model(df, "B", hac_lags_override=lags, trend=trend_type)
                if r:
                    print(f"  {metric:<12} {trend_label:<12} {'HAC-'+str(lags):<8} {_row_r(r)}")
        print()


def print_geo_platform(dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    scopes    = [("US", "US"), ("Global", None), ("Global-ex-US", "ex-US")]
    platforms = [("Combined", None), ("iOS", "iOS"), ("Android", "Android")]

    # Pre-load all slices once to avoid repeated disk reads inside the loop
    geo_data: dict = {}
    for scope_lbl, country in scopes:
        for plat_lbl, platform in platforms:
            geo_data[(scope_lbl, plat_lbl)] = {
                "dl":  load(DATA_DIR / "downloads.csv", "Downloads",
                            country=country, platform=platform),
                "dau": load(DATA_DIR / "dau.csv", "DAU",
                            country=country, platform=platform),
            }

    print(f"{'─'*90}\n  Geo × Platform Heterogeneity (Model B, FEB1, HAC)\n{'─'*90}")
    print(f"  {'Scope':<14} {'Platform':<10} "
          f"{'δ_CL_DL':>9} {'p':>7}  {'δ_GPT_DL':>9} {'p':>7}  "
          f"{'δ_CL_DAU':>9} {'p':>7}  {'δ_GPT_DAU':>9} {'p':>7}")
    for scope_lbl, country in scopes:
        for plat_lbl, platform in platforms:
            data  = geo_data[(scope_lbl, plat_lbl)]
            r_dl  = run_model(data["dl"],  "B")
            r_dau = run_model(data["dau"], "B")
            if r_dl is None or r_dau is None:
                print(f"  {scope_lbl:<14} {plat_lbl:<10} "
                      f"{'—':>9} {'—':>7}  {'—':>9} {'—':>7}  "
                      f"{'—':>9} {'—':>7}  {'—':>9} {'—':>7}")
                continue
            print(f"  {scope_lbl:<14} {plat_lbl:<10} "
                  f"{r_dl.coef_cl:+9.3f} {r_dl.pval_cl:.3f}{stars(r_dl.pval_cl)}  "
                  f"{r_dl.coef_gpt:+9.3f} {r_dl.pval_gpt:.3f}{stars(r_dl.pval_gpt)}  "
                  f"{r_dau.coef_cl:+9.3f} {r_dau.pval_cl:.3f}{stars(r_dau.pval_cl)}  "
                  f"{r_dau.coef_gpt:+9.3f} {r_dau.pval_gpt:.3f}{stars(r_dau.pval_gpt)}")
    print()


def print_event_window(dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    specs = [
        ("Point  (>= Feb 27)",   {}),
        ("Window (>= Feb 24)",   {"post_start": SHOCK_START}),
        ("Wash-in (excl 24-26)", {"drop_date_ranges": [(SHOCK_START, EVENT)]}),
    ]
    print(f"{'─'*90}\n  Event-Window Robustness (Model B, FEB1, HAC-5/-10)\n{'─'*90}")
    print(f"  {'Design':<26} {'SE':<8} "
          f"{'δ_CL':>8} {'Δ%_CL':>7} {'p_CL':>7}   "
          f"{'δ_GPT':>8} {'Δ%_GPT':>7} {'p_GPT':>7}")
    for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
        print(f"\n  {metric}")
        for label, kwargs in specs:
            for lags in [5, 10]:
                r = run_model(df, "B", hac_lags_override=lags, **kwargs)
                if r:
                    print(f"  {label:<26} HAC-{lags:<4} "
                          f"{r.coef_cl:+8.3f} {r.pct_change_cl:+7.1f}% "
                          f"{r.pval_cl:.3f}{stars(r.pval_cl)}   "
                          f"{r.coef_gpt:+8.3f} {r.pct_change_gpt:+7.1f}% "
                          f"{r.pval_gpt:.3f}{stars(r.pval_gpt)}")
    print()


def print_confounder_checks(dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    print(f"{'─'*90}\n  Confounder Checks (Model B, FEB1)\n{'─'*90}")
    print(f"  {'Design':<32} {'SE':<8} "
          f"{'δ_CL':>8} {'Δ%_CL':>7} {'p_CL':>7}   "
          f"{'δ_GPT':>8} {'Δ%_GPT':>7} {'p_GPT':>7}")
    specs: list[tuple[str, dict]] = [
        ("Falsif.: event=Feb17",
         {"post_start": SONNET_REL, "date_end_exclusive": SHOCK_START}),
        ("Excl. Feb17-23",
         {"drop_date_ranges": [(SONNET_REL, SHOCK_START)]}),
        ("Baseline",
         {}),
    ]
    for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
        print(f"\n  {metric}")
        for label, kwargs in specs:
            for lags in [5, 10]:
                r = run_model(df, "B", hac_lags_override=lags, **kwargs)
                if r and r.n_pre > 0:
                    print(f"  {label:<32} HAC-{lags:<4} "
                          f"{r.coef_cl:+8.3f} {r.pct_change_cl:+7.1f}% "
                          f"{r.pval_cl:.3f}{stars(r.pval_cl)}   "
                          f"{r.coef_gpt:+8.3f} {r.pct_change_gpt:+7.1f}% "
                          f"{r.pval_gpt:.3f}{stars(r.pval_gpt)}")
    print()


def print_event_study(dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    print(f"{'─'*90}\n  Event Study — Spillover Model B (FEB1, HAC)\n{'─'*90}")
    print(f"  {'k':>4}  {'δ_CL':>8} {'SE_CL':>7} {'δ_GPT':>8} {'SE_GPT':>7}  "
          f"{'δ_CL':>8} {'SE_CL':>7} {'δ_GPT':>8} {'SE_GPT':>7}")
    print(f"  {'':>4}  {'Downloads':^32}  {'DAU':^32}")
    print("  " + "─" * 86)

    coefs_dl  = run_event_study(dl)
    coefs_dau = run_event_study(dau)
    if coefs_dl is None or coefs_dau is None:
        print("  (insufficient data)")
        return

    for k in sorted(coefs_dl):
        c_cl_dl,  s_cl_dl,  c_gpt_dl,  s_gpt_dl  = coefs_dl[k]
        c_cl_dau, s_cl_dau, c_gpt_dau, s_gpt_dau = coefs_dau[k]
        omit = " ← omitted" if k == -1 else ""
        print(f"  {k:>4}  "
              f"{c_cl_dl:+8.3f} {s_cl_dl:7.3f} {c_gpt_dl:+8.3f} {s_gpt_dl:7.3f}  "
              f"{c_cl_dau:+8.3f} {s_cl_dau:7.3f} {c_gpt_dau:+8.3f} {s_gpt_dau:7.3f}"
              f"{omit}")
    print()


# ── Placebo & Permutation ─────────────────────────────────────────────────────

def placebo_time(
    df: pd.DataFrame,
    metric: str,
) -> tuple[list, float, float]:
    candidate_dates = pd.date_range(
        WINDOW_DEFAULT + timedelta(days=MIN_PRE_DAYS), PLACEBO_CUTOFF, freq="D"
    )
    results = []
    for fake in candidate_dates:
        r = run_model(df, "B", post_start=fake, hac_lags_override=5)
        if r:
            results.append((fake, r.coef_cl, r.coef_gpt))

    # Use dynamic HAC for the real reference to match the paper's reported spec;
    # HAC-5 is kept only in the placebo loop for comparability within the null.
    real = run_model(df, "B")
    if real is None:
        return results, float("nan"), float("nan")

    for name, real_coef, idx in [("CL", real.coef_cl, 1), ("GPT", real.coef_gpt, 2)]:
        coefs = [r[idx] for r in results]
        n     = len(coefs)
        rank  = sum(abs(c) >= abs(real_coef) for c in coefs)
        emp_p = rank / n if n > 0 else float("nan")
        print(f"  [{metric}] Time placebo {name}: "
              f"real δ={real_coef:+.3f}  emp-p={emp_p:.3f} ({rank}/{n})")
    return results, real.coef_cl, real.coef_gpt


def placebo_unit(
    df: pd.DataFrame,
    metric: str,
    apps: list[str],
) -> tuple[list, float]:
    controls = [a for a in apps if a not in [APP_CL, APP_GPT]]
    results  = []
    for app in controls:
        others = [a for a in controls if a != app]
        if not others or app not in df.columns:
            continue
        win = df[df.index >= WINDOW_DEFAULT][[app] + others].dropna()
        rows = []
        for a in [app] + others:
            if a not in win.columns:
                continue
            tmp = pd.DataFrame({
                "log_Y": np.log(win[a].replace(0, np.nan)),
                "app":   a,
            }).reset_index(names="date")
            rows.append(tmp)
        panel = pd.concat(rows).dropna(subset=["log_Y"])
        panel = panel.sort_values(["date", "app"]).reset_index(drop=True)
        panel["t"]       = (panel["date"] - WINDOW_DEFAULT).dt.days.astype(float)
        panel["Post"]    = (panel["date"] >= EVENT).astype(int)
        panel["isCL"]    = (panel["app"] == app).astype(int)
        panel["isGPT"]   = 0
        panel["DiD_CL"]  = panel["isCL"] * panel["Post"]
        panel["DiD_GPT"] = 0
        panel["dow"]     = panel["date"].dt.dayofweek.astype(str)
        try:
            lags = hac_lags(len(panel))
            # Use single-treatment formula: DiD_GPT is identically zero here
            # (zero-variance regressor), so including it in FORMULA_A would fail.
            res  = _fit(FORMULA_A_SINGLE, panel, "HAC", lags)
            coef = float(res.params.get("DiD_CL", np.nan))
            pval = float(res.pvalues.get("DiD_CL", np.nan))
            pct  = float((np.exp(coef) - 1) * 100)
            results.append((app.split()[0], coef, pval))
            print(f"  [{metric}] Unit placebo — {app.split()[0]:<12} "
                  f"δ={coef:+.3f}  Δ%={pct:+.1f}%  p={pval:.3f}{stars(pval)}")
        except Exception:
            continue

    real = run_model(df, "A", hac_lags_override=5)
    if real is None:
        return results, float("nan")
    cl_real = real.coef_cl
    coefs   = [r[1] for r in results]
    rank    = sum(abs(c) >= abs(cl_real) for c in coefs)
    emp_p   = (rank + 1) / (len(coefs) + 1)
    print(f"  [{metric}] Claude δ={cl_real:+.3f} | empirical p={emp_p:.3f}\n")
    return results, cl_real


def perm_inference(
    df: pd.DataFrame,
    metric: str,
    apps: list[str],
) -> tuple[list[float], list[float], list[float], list[float], float, float]:
    real = run_model(df, "B", hac_lags_override=5)
    if real is None:
        return [], [], [], [], float("nan"), float("nan")

    perm_dates = pd.date_range(
        WINDOW_DEFAULT + timedelta(days=MIN_PRE_DAYS), PLACEBO_CUTOFF, freq="D"
    ).tolist()

    n_cands = len(perm_dates)
    cl_date, gpt_date = [], []
    for idx in rng.choice(n_cands, size=min(N_PERM, n_cands), replace=False):
        r = run_model(df, "B", post_start=perm_dates[idx], hac_lags_override=5)
        if r:
            cl_date.append(r.coef_cl); gpt_date.append(r.coef_gpt)

    cl_treat, gpt_treat = [], []
    for _ in range(N_PERM):
        shuffled = [str(a) for a in rng.permutation(apps)]
        fake_cl, fake_gpt = shuffled[0], shuffled[1]
        # Build custom panel with shuffled treatment assignment
        win = df[df.index >= WINDOW_DEFAULT][[a for a in shuffled if a in df.columns]].dropna()
        rows = []
        for a in shuffled:
            if a not in win.columns:
                continue
            rows.append(pd.DataFrame({
                "log_Y": np.log(win[a].replace(0, np.nan)),
                "app":   a,
            }).reset_index(names="date"))
        if not rows:
            continue
        panel = pd.concat(rows).dropna(subset=["log_Y"])
        panel = panel.sort_values(["date", "app"]).reset_index(drop=True)
        panel["t"]       = (panel["date"] - WINDOW_DEFAULT).dt.days.astype(float)
        panel["Post"]    = (panel["date"] >= EVENT).astype(int)
        panel["isCL"]    = (panel["app"] == fake_cl).astype(int)
        panel["isGPT"]   = (panel["app"] == fake_gpt).astype(int)
        panel["CLxt"]    = panel["isCL"]  * panel["t"]
        panel["GPTxt"]   = panel["isGPT"] * panel["t"]
        panel["DiD_CL"]  = panel["isCL"]  * panel["Post"]
        panel["DiD_GPT"] = panel["isGPT"] * panel["Post"]
        panel["dow"]     = panel["date"].dt.dayofweek.astype(str)
        try:
            lags = hac_lags(len(panel))
            res  = _fit(FORMULA_B, panel, "HAC", lags)
            cl, gpt = _extract(res)
            cl_treat.append(cl[0]); gpt_treat.append(gpt[0])
        except Exception:
            continue

    def _p(coefs: list[float], ref: float) -> float:
        return float(np.mean(np.abs(coefs) >= abs(ref))) if coefs else float("nan")

    p_cl_date   = _p(cl_date,   real.coef_cl)
    p_gpt_date  = _p(gpt_date,  real.coef_gpt)
    p_cl_treat  = _p(cl_treat,  real.coef_cl)
    p_gpt_treat = _p(gpt_treat, real.coef_gpt)
    # n_cands is ~15; treatment permutations run the full N_PERM draws.
    n_date_actual = min(N_PERM, n_cands)
    print(f"  [{metric}] Permutation (date, N={n_date_actual})      — "
          f"Claude: p={p_cl_date:.3f}  ChatGPT: p={p_gpt_date:.3f}")
    print(f"  [{metric}] Permutation (treatment, N={N_PERM}) — "
          f"Claude: p={p_cl_treat:.3f}  ChatGPT: p={p_gpt_treat:.3f}  "
          f"(real δ_CL={real.coef_cl:+.3f}  δ_GPT={real.coef_gpt:+.3f})")
    return cl_date, gpt_date, cl_treat, gpt_treat, real.coef_cl, real.coef_gpt


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_raw_levels(dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Raw Daily Levels — US, All Apps", fontsize=13, fontweight="bold", y=1.01)
    for ax, (df, metric) in zip(axes, [(dl, "US Downloads"), (dau, "US DAU")]):
        for a in df.columns:
            ax.plot(df.index, df[a], color=app_color(a), lw=1.8, label=app_label(a))
        ax.axvline(EVENT, color=EVENT_COLOR, lw=1.2, ls="--", alpha=0.85, label="Feb 27")
        ax.set_title(metric); ax.set_ylabel(metric.split()[1])
        ax.legend(loc="upper left"); style_ax(ax)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    fname = PLOTS_DIR / "fig1_raw_levels.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


def plot_main_coefs(dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    SE_SPECS = [(f"HAC-{l}", "HAC", l) for l in HAC_LAGS_SENS] + [("HC3", "HC3", None)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Spillover DiD — δ_Claude and δ_ChatGPT across Specifications",
                 fontsize=13, fontweight="bold", y=1.01)
    for ax, (df, metric) in zip(axes, [(dl, "Downloads"), (dau, "DAU")]):
        x, xticks, xlabels = 0, [], []
        for win_label, ws in WINDOWS:
            for model in ["A", "B"]:
                group_start = x
                for se_label, se_type, lags in SE_SPECS:
                    r = run_model(df, model, ws,
                                  se_type=se_type,
                                  hac_lags_override=lags)
                    if r:
                        # 1.96 is asymptotic; qualitatively unchanged with t-critical
                        ax.errorbar(x - 0.15, r.coef_cl,  yerr=1.96 * r.se_cl,
                                    fmt="o", color=CL_COLOR,  ms=5, capsize=3, lw=1.3,
                                    label="δ_Claude"  if x == 0 else "")
                        ax.errorbar(x + 0.15, r.coef_gpt, yerr=1.96 * r.se_gpt,
                                    fmt="s", color=GPT_COLOR, ms=5, capsize=3, lw=1.3,
                                    label="δ_ChatGPT" if x == 0 else "")
                    x += 1
                mid = (group_start + x - 1) / 2
                xticks.append(mid); xlabels.append(f"{win_label}\nMod {model}")
                x += 0.5
        ax.axhline(0, color="#6B7280", lw=0.8, alpha=0.5)
        ax.set_xticks(xticks); ax.set_xticklabels(xlabels)
        ax.set_title(metric); ax.set_ylabel("δ (log pts)"); style_ax(ax)
        if metric == "Downloads":
            ax.legend()
    plt.tight_layout()
    fname = PLOTS_DIR / "fig2_main_coefs.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


def plot_event_study_fig(dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    coefs_dl  = run_event_study(dl)
    coefs_dau = run_event_study(dau)
    if coefs_dl is None or coefs_dau is None:
        print("Skipping event study plot (insufficient data)"); return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    fig.suptitle("Event Study — Spillover Model B (US, FEB1, HAC)",
                 fontsize=13, fontweight="bold", y=1.01)

    panel_configs = [
        (0, coefs_dl,  "Downloads", "δ_CL",  CL_COLOR,  0, 1),
        (0, coefs_dl,  "Downloads", "δ_GPT", GPT_COLOR, 2, 3),
        (1, coefs_dau, "DAU",       "δ_CL",  CL_COLOR,  0, 1),
        (1, coefs_dau, "DAU",       "δ_GPT", GPT_COLOR, 2, 3),
    ]
    for row, coefs, metric, coef_key, color, c_idx, s_idx in panel_configs:
        col = 0 if coef_key == "δ_CL" else 1
        ax  = axes[row, col]
        ks   = sorted(coefs)
        vals = [coefs[k][c_idx] for k in ks]
        # 1.96 is asymptotic; t-critical with small panels is ~2.0–2.1 (qualitatively unchanged)
        ci   = [1.96 * coefs[k][s_idx] for k in ks]

        ax.axhline(0, color="#6B7280", lw=0.8, alpha=0.5)
        ax.axvline(-0.5, color=EVENT_COLOR, lw=1.2, ls="--", alpha=0.75,
                   label="Feb 27 (event)")
        sonnet_k = (SONNET_REL - EVENT).days
        if -ES_K <= sonnet_k <= ES_K:
            ax.axvline(sonnet_k - 0.5, color="#D97706", lw=1.0, ls=":", alpha=0.8,
                       label="Feb 17 (Sonnet rel.)")
        ax.fill_between(ks, [v - e for v, e in zip(vals, ci)],
                        [v + e for v, e in zip(vals, ci)], alpha=0.12, color=color)
        ax.plot(ks, vals, "o-", color=color, ms=5, lw=1.8)
        ax.set_title(f"{'Claude' if col == 0 else 'ChatGPT'} — {metric}")
        ax.set_xlabel("Days relative to Feb 27"); ax.set_ylabel("δ_k (log pts)")
        ax.legend(); style_ax(ax)

    plt.tight_layout()
    fname = PLOTS_DIR / "fig3_event_study.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


def plot_placebos(placebo_store: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Spillover Placebo Tests (Model B, HAC)",
                 fontsize=13, fontweight="bold", y=1.01)
    for col, metric in enumerate(["Downloads", "DAU"]):
        store = placebo_store[metric]
        time_res, cl_real, gpt_real = store["time_res"], store["cl_real"], store["gpt_real"]
        unit_res, unit_cl           = store["unit_res"], store["unit_cl"]

        ax    = axes[0, col]
        coefs = [r[1] for r in time_res]
        n     = len(coefs)
        rank  = sum(abs(c) >= abs(cl_real) for c in coefs)
        emp_p = rank / n if n > 0 else float("nan")
        ax.hist(coefs, bins=15, color=HIST_COLOR, edgecolor=HIST_EDGE, lw=0.5, alpha=0.9)
        ax.axvline( cl_real,  color=CL_COLOR,  lw=2,
                    label=f"Claude δ = {cl_real:+.3f}   emp-p = {emp_p:.3f}")
        ax.axvline(-cl_real,  color=CL_COLOR,  lw=1.2, ls="--", alpha=0.45)
        ax.axvline( gpt_real, color=GPT_COLOR, lw=1.5, ls="-.",
                    label=f"ChatGPT δ = {gpt_real:+.3f}")
        ax.set_title(f"{metric} — Time placebo (N={n})")
        ax.set_xlabel("δ_CL (log pts)"); ax.set_ylabel("Count")
        ax.legend(); style_ax(ax)

        ax     = axes[1, col]
        labels = [r[0] for r in unit_res] + ["Claude"]
        vals   = [r[1] for r in unit_res] + [unit_cl]
        bar_colors = ["#CBD5E1"] * len(unit_res) + [CL_COLOR]
        ax.barh(labels, vals, color=bar_colors, edgecolor="white", height=0.55)
        ax.axvline(0, color="#6B7280", lw=0.8, alpha=0.5)
        ax.set_title(f"{metric} — Unit placebo")
        ax.set_xlabel("δ (log pts)"); style_ax(ax)

    plt.tight_layout()
    fname = PLOTS_DIR / "fig4_placebos.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


def plot_permutations(perm_store: dict) -> None:
    for perm_type, key_d, key_t, title_suffix in [
        ("Date",      "cl_date",  "gpt_date",  "Date"),
        ("Treatment", "cl_treat", "gpt_treat", "Treatment"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(f"Spillover Permutation Inference (N={N_PERM}, Model B — {title_suffix})",
                     fontsize=13, fontweight="bold", y=1.01)
        for col, metric in enumerate(["Downloads", "DAU"]):
            store = perm_store[metric]
            for row, (name, color, real_key) in enumerate([
                ("Claude",  CL_COLOR,  "cl_real"),
                ("ChatGPT", GPT_COLOR, "gpt_real"),
            ]):
                real  = store[real_key]
                coef_key = ("cl_" if name == "Claude" else "gpt_") + \
                           ("date" if perm_type == "Date" else "treat")
                coefs = store[coef_key]
                ax = axes[row, col]
                if not coefs:
                    ax.set_visible(False); continue
                p = float(np.mean(np.abs(coefs) >= abs(real)))
                ax.hist(coefs, bins=40, color=HIST_COLOR, edgecolor=HIST_EDGE, lw=0.5, alpha=0.9)
                ax.axvline( real, color=color, lw=2,
                            label=f"{name} δ = {real:+.3f}   p = {p:.3f}")
                ax.axvline(-real, color=color, lw=1.2, ls="--", alpha=0.45)
                ax.set_title(f"{metric} — {name} ({perm_type.lower()} perm)")
                ax.set_xlabel("δ (log pts)"); ax.set_ylabel("Count")
                ax.legend(); style_ax(ax)
        plt.tight_layout()
        fname = PLOTS_DIR / f"fig{'5' if perm_type == 'Date' else '6'}_permutations_{perm_type.lower()}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
        print(f"Saved: {fname}")


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_full_analysis() -> None:
    print("─" * 80)
    print(f"  Spillover DiD  |  {' · '.join(f'{k}={v}' for k, v in VERSIONS.items())}")
    print("─" * 80 + "\n")

    dl  = load(DATA_DIR / "downloads.csv", "Downloads", country="US")
    dau = load(DATA_DIR / "dau.csv",       "DAU",       country="US")
    apps = sorted(dl.columns.tolist())
    print(f"Apps: {apps}\n")

    print_diagnostics(dl, dau)
    for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
        print_main(df, metric)
    print_quadratic_sensitivity(dl, dau)
    print_geo_platform(dl, dau)
    print_event_window(dl, dau)
    print_confounder_checks(dl, dau)
    print_event_study(dl, dau)

    print(f"{'─'*72}\n  Placebo Tests\n{'─'*72}")
    placebo_store: dict = {}
    for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
        time_res, cl_real, gpt_real = placebo_time(df, metric)
        print()
        unit_res, unit_cl = placebo_unit(df, metric, apps)
        placebo_store[metric] = {
            "time_res": time_res, "cl_real": cl_real, "gpt_real": gpt_real,
            "unit_res": unit_res, "unit_cl": unit_cl,
        }

    print(f"{'─'*72}\n  Permutation Inference (N={N_PERM})\n{'─'*72}")
    global rng
    rng = np.random.default_rng(RNG_SEED)   # reset so results are order-independent
    perm_store: dict = {}
    for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
        cl_d, gpt_d, cl_t, gpt_t, cl_r, gpt_r = perm_inference(df, metric, apps)
        perm_store[metric] = {
            "cl_date": cl_d, "gpt_date": gpt_d,
            "cl_treat": cl_t, "gpt_treat": gpt_t,
            "cl_real": cl_r, "gpt_real": gpt_r,
        }
    print()

    plot_raw_levels(dl, dau)
    plot_main_coefs(dl, dau)
    plot_event_study_fig(dl, dau)
    plot_placebos(placebo_store)
    plot_permutations(perm_store)

    all_results: dict[str, Any] = {"versions": VERSIONS, "main": {}, "sensitivity": {}}
    for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
        all_results["main"][metric] = {}
        for model in ["A", "B"]:
            r = run_model(df, model)
            all_results["main"][metric][model] = r.to_dict() if r else None

    for win_label, ws in WINDOWS:
        for model in ["A", "B"]:
            for lags in HAC_LAGS_SENS:
                key = f"{win_label}_{model}_HAC{lags}"
                all_results["sensitivity"][key] = {}
                for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
                    r = run_model(df, model, ws, hac_lags_override=lags)
                    all_results["sensitivity"][key][metric] = r.to_dict() if r else None

    save_json(all_results, OUTPUT_DIR / "last_run.json")

"""Interrupted Time Series pipeline (its.py entry point).

Model (full):
    S_t = β0 + β1·t + β2·Post_t + β3·t_post + Σ φ_d·dow_d + ε_t

Model (parsimonious, drops β3):
    S_t = β0 + β1·t + β2·Post_t + Σ φ_d·dow_d + ε_t

Logit variant: log(S_t / (100 − S_t)) as outcome; marginal effect (pp) reported.

HAC default: Newey-West with HAC_LAGS_DEFAULT lags (conservative for daily series
with DOW controls). Sensitivity at 12 and 14.
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

from pipeline.types import ITSResult
from utils.config import (
    APP_CL, APP_GPT, EVENT, MIN_PRE_DAYS, N_PERM, PLACEBO_CUTOFF,
    SHOCK_START, SONNET_REL, WINDOW_DEFAULT, WINDOWS,
)
from utils.data import load
from utils.plot_style import EVENT_COLOR, HIST_COLOR, HIST_EDGE, style_ax
from utils.reporting import VERSIONS, RNG_SEED, output_dir, rng, save_json
from utils.stats import stars

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

DATA_DIR  = Path(__file__).resolve().parent.parent / "input_data"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots" / "its"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = output_dir("its")

_WINDOW_START: dict[str, pd.Timestamp] = {lbl: ts for lbl, ts in WINDOWS}

# HAC_LAGS_DEFAULT conservatively set to 10: daily time series can carry weekly
# autocorrelation even after DOW dummies; Andrews rule (~3) is too aggressive here.
HAC_LAGS_DEFAULT = 10
HAC_LAGS_SENS    = [10, 12, 14]

FORMULAS: dict[tuple[str, str], str] = {
    ("linear", "full"):   "S ~ t + Post + t_post + C(dow)",
    ("linear", "parsim"): "S ~ t + Post + C(dow)",
    ("logit",  "full"):   "logit_S ~ t + Post + t_post + C(dow)",
    ("logit",  "parsim"): "logit_S ~ t + Post + C(dow)",
}

ITS_SENSITIVITY_GRID: list[dict[str, Any]] = [
    {"window_label": wl, "outcome": out, "model": mod, "se": se, "lags": lags}
    for wl in ("FEB1", "JAN1")
    for out in ("linear", "logit")
    for mod in ("parsim", "full")
    for se, lags in [("HAC", 10), ("HAC", 12), ("HAC", 14), ("HC3", None)]
]


# ── Series builders ───────────────────────────────────────────────────────────

def build_share(
    df: pd.DataFrame,
    window_start: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    if window_start is None:
        window_start = WINDOW_DEFAULT
    if APP_CL not in df.columns or APP_GPT not in df.columns:
        return None

    sub   = df[[APP_CL, APP_GPT]].copy()
    sub   = sub[sub.index >= window_start].dropna()
    total = sub[APP_CL] + sub[APP_GPT]
    mask  = total > 0
    sub, total = sub[mask], total[mask]

    ts = pd.DataFrame(index=sub.index)
    ts["S"]       = (sub[APP_CL] / total) * 100
    s_clip        = ts["S"].clip(0.01, 99.99)
    ts["logit_S"] = np.log(s_clip / (100 - s_clip))
    ts["t"]       = (ts.index - window_start).days.astype(float)
    ts["t_post"]  = np.maximum(0, (ts.index - EVENT).days.astype(float))
    ts["Post"]    = (ts.index >= EVENT).astype(int)
    ts["dow"]     = ts.index.dayofweek.astype(str)
    return ts.reset_index(names="date")


def build_total_market(
    df: pd.DataFrame,
    window_start: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    if window_start is None:
        window_start = WINDOW_DEFAULT
    apps = [a for a in df.columns]
    if not apps:
        return None

    win   = df[df.index >= window_start][apps].dropna(how="all")
    total = win.sum(axis=1)
    total = total[total > 0]

    n_zeros = int((win == 0).sum().sum())
    if n_zeros:
        print(f"  [warn] total-market: {n_zeros} zero cell(s) included in sum")

    ts = pd.DataFrame(index=total.index)
    ts["log_total"] = np.log(total)
    ts["t"]         = (ts.index - window_start).days.astype(float)
    ts["Post"]      = (ts.index >= EVENT).astype(int)
    ts["dow"]       = ts.index.dayofweek.astype(str)
    return ts.reset_index(names="date")


# ── Estimation ────────────────────────────────────────────────────────────────

def _fit_hac(formula: str, data: pd.DataFrame, lags: int = HAC_LAGS_DEFAULT) -> Any:
    return smf.ols(formula, data=data).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})


def _fit_hc3(formula: str, data: pd.DataFrame) -> Any:
    return smf.ols(formula, data=data).fit(cov_type="HC3")


def _run_its_raw(
    df: pd.DataFrame,
    window_start: pd.Timestamp | None = None,
    outcome: str = "linear",
    model: str = "parsim",
    se_type: str = "HAC",
    lags: int = HAC_LAGS_DEFAULT,
) -> tuple[pd.DataFrame, Any] | None:
    ts = build_share(df, window_start)
    if ts is None:
        return None
    f   = FORMULAS[(outcome, model)]
    res = (_fit_hac(f, ts, lags) if se_type == "HAC" else _fit_hc3(f, ts))
    return ts, res


def run_its(
    df: pd.DataFrame,
    window_start: pd.Timestamp | None = None,
    outcome: str = "linear",
    model: str = "parsim",
    se_type: str = "HAC",
    lags: int = HAC_LAGS_DEFAULT,
) -> ITSResult | None:
    if window_start is None:
        window_start = WINDOW_DEFAULT
    raw = _run_its_raw(df, window_start, outcome, model, se_type, lags)
    if raw is None:
        return None
    ts, res = raw

    s_bar  = float(ts.loc[ts["Post"] == 0, "S"].mean())
    beta2  = float(res.params["Post"])
    se2    = float(res.bse["Post"])
    pval2  = float(res.pvalues["Post"])

    if outcome == "logit":
        me    = float(beta2 * s_bar * (100 - s_bar) / 100)
        me_se = float(se2   * s_bar * (100 - s_bar) / 100)
    else:
        me, me_se = beta2, se2

    beta3 = float(res.params.get("t_post", np.nan))
    pval3 = float(res.pvalues.get("t_post", np.nan))

    return ITSResult(
        beta2        = beta2,
        se2          = se2,
        pval2        = pval2,
        beta3        = beta3,
        pval3        = pval3,
        me           = me,
        me_se        = me_se,
        s_bar        = s_bar,
        outcome      = outcome,
        model        = model,
        hac_lags     = lags,
        window_start = window_start,
    )


# ── Print tables ──────────────────────────────────────────────────────────────

def print_diagnostics(dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    print(f"{'─'*60}\n  Residual Diagnostics\n{'─'*60}")
    for win_label, ws in WINDOWS:
        for outcome, model in [("linear", "full"), ("linear", "parsim")]:
            for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
                raw = _run_its_raw(df, ws, outcome, model)
                if raw is None:
                    continue
                ts, res = raw
                resid = res.resid
                dw    = durbin_watson(resid)
                lb    = acorr_ljungbox(resid, lags=[10, 12], return_df=True)
                print(f"  {win_label} {outcome:<7} {model:<7} {metric:<12} | "
                      f"DW={dw:.2f}  LB(10)p={lb.loc[10,'lb_pvalue']:.3f}  "
                      f"LB(12)p={lb.loc[12,'lb_pvalue']:.3f}")
    print()


def print_main_results(df: pd.DataFrame, metric: str) -> None:
    print(f"{'─'*80}\n  Main ITS Results — {metric}\n{'─'*80}")
    print(f"  {'Window':<6} {'Outcome':<8} {'Model':<8} "
          f"{'β2(ME)':>9} {'SE':>7} {'p':>7} {'β3':>9} {'p(β3)':>7} {'S̄%':>6}")
    print("  " + "─" * 78)
    for win_label, ws in WINDOWS:
        for outcome in ["linear", "logit"]:
            for model in ["parsim", "full"]:
                r = run_its(df, ws, outcome, model)
                if r is None:
                    continue
                b3_s  = f"{r.beta3:+.4f}" if not np.isnan(r.beta3) else "—"
                pv3_s = (f"{r.pval3:.3f}{stars(r.pval3)}" if not np.isnan(r.pval3) else "—")
                print(f"  {win_label:<6} {outcome:<8} {model:<8} "
                      f"{r.me:+9.4f}  {r.me_se:7.4f}  "
                      f"{r.pval2:.3f}{stars(r.pval2)}  "
                      f"{b3_s:>9}  {pv3_s:>9}  {r.s_bar:6.1f}")
    print()


def print_hac_sensitivity(df: pd.DataFrame, metric: str) -> None:
    print(f"{'─'*80}\n  HAC Bandwidth Sensitivity — {metric} (FEB1, linear, parsim)\n{'─'*80}")
    print(f"  {'Lags':<6} {'β2':>9} {'SE':>7} {'p':>8}")
    for lags in HAC_LAGS_SENS:
        r = run_its(df, outcome="linear", model="parsim", lags=lags)
        if r:
            print(f"  {lags:<6} {r.beta2:+9.4f}  {r.se2:7.4f}  "
                  f"{r.pval2:.3f}{stars(r.pval2)}")
    print()


def print_sensitivity_matrix(df: pd.DataFrame, metric: str) -> None:
    print(f"{'─'*90}\n  Sensitivity Matrix — {metric}\n{'─'*90}")
    print(f"  {'Window':<6} {'Outcome':<8} {'Model':<8} {'SE':<5} {'Lags':<5} "
          f"{'β2(ME)':>9} {'SE':>7} {'p':>8}")
    print("  " + "─" * 88)
    for spec in ITS_SENSITIVITY_GRID:
        ws   = _WINDOW_START[spec["window_label"]]
        lags = spec["lags"] if spec["lags"] is not None else HAC_LAGS_DEFAULT
        r    = run_its(df, ws, spec["outcome"], spec["model"],
                       se_type=spec["se"], lags=lags)
        if r is None:
            continue
        lags_s = str(spec["lags"]) if spec["lags"] is not None else "—"
        print(f"  {spec['window_label']:<6} {spec['outcome']:<8} {spec['model']:<8} "
              f"{spec['se']:<5} {lags_s:<5} "
              f"{r.me:+9.4f}  {r.me_se:7.4f}  {r.pval2:.3f}{stars(r.pval2)}")
    print()


def print_geo_platform() -> None:
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

    print(f"{'─'*72}\n  Geo × Platform Heterogeneity (FEB1, linear, parsim, HAC-10)\n{'─'*72}")
    print(f"  {'Scope':<14} {'Platform':<10} {'β2_DL':>9} {'p_DL':>8}  "
          f"{'β2_DAU':>9} {'p_DAU':>8}")
    for scope_lbl, country in scopes:
        for plat_lbl, platform in platforms:
            data  = geo_data[(scope_lbl, plat_lbl)]
            r_dl  = run_its(data["dl"])
            r_dau = run_its(data["dau"])
            dl_s  = (f"{r_dl.me:+9.4f}  {r_dl.pval2:.3f}{stars(r_dl.pval2)}"
                     if r_dl  else f"{'—':>9}  {'—':>8}")
            dau_s = (f"{r_dau.me:+9.4f}  {r_dau.pval2:.3f}{stars(r_dau.pval2)}"
                     if r_dau else f"{'—':>9}  {'—':>8}")
            print(f"  {scope_lbl:<14} {plat_lbl:<10} {dl_s}   {dau_s}")
    print()


def print_event_window(df: pd.DataFrame, metric: str) -> None:
    print(f"{'─'*72}\n  Event-Window Robustness — {metric}\n{'─'*72}")
    print(f"  {'Design':<26} {'β2(ME)':>9} {'SE':>7} {'p':>8}")

    ts_base = build_share(df)
    if ts_base is None:
        return

    f_p = FORMULAS[("linear", "parsim")]

    r = _fit_hac(f_p, ts_base)
    b, se, pv = r.params["Post"], r.bse["Post"], r.pvalues["Post"]
    print(f"  {'Point  (>= Feb 27)':<26} {b:+9.4f}  {se:7.4f}  {pv:.3f}{stars(pv)}")

    ts_w = ts_base.copy()
    ts_w["Post"]   = (ts_w["date"] >= SHOCK_START).astype(int)
    ts_w["t_post"] = np.maximum(0, (ts_w["date"] - SHOCK_START).dt.days.astype(float))
    r = _fit_hac(f_p, ts_w)
    b, se, pv = r.params["Post"], r.bse["Post"], r.pvalues["Post"]
    print(f"  {'Window (>= Feb 24)':<26} {b:+9.4f}  {se:7.4f}  {pv:.3f}{stars(pv)}")

    ts_wi = ts_base[
        ~((ts_base["date"] >= SHOCK_START) & (ts_base["date"] < EVENT))
    ].copy()
    ts_wi["Post"]   = (ts_wi["date"] >= EVENT).astype(int)
    ts_wi["t_post"] = np.maximum(0, (ts_wi["date"] - EVENT).dt.days.astype(float))
    r = _fit_hac(f_p, ts_wi)
    b, se, pv = r.params["Post"], r.bse["Post"], r.pvalues["Post"]
    print(f"  {'Wash-in (excl 24-26)':<26} {b:+9.4f}  {se:7.4f}  {pv:.3f}{stars(pv)}")
    print()


def print_confounder_checks(df: pd.DataFrame, metric: str) -> None:
    print(f"{'─'*72}\n  Confounder Checks — {metric}\n{'─'*72}")
    print(f"  {'Design':<32} {'β2(ME)':>9} {'SE':>7} {'p':>8}")

    ts_base = build_share(df)
    if ts_base is None:
        return
    f_p = FORMULAS[("linear", "parsim")]

    ts_f = ts_base[ts_base["date"] < SHOCK_START].copy()
    ts_f["Post"]   = (ts_f["date"] >= SONNET_REL).astype(int)
    ts_f["t_post"] = np.maximum(0, (ts_f["date"] - SONNET_REL).dt.days.astype(float))
    if ts_f["Post"].nunique() == 2:
        r = _fit_hac(f_p, ts_f)
        b, se, pv = r.params["Post"], r.bse["Post"], r.pvalues["Post"]
        print(f"  {'Falsif.: event=Feb17':<32} {b:+9.4f}  {se:7.4f}  {pv:.3f}{stars(pv)}")

    ts_e = ts_base[
        ~((ts_base["date"] >= SONNET_REL) & (ts_base["date"] < SHOCK_START))
    ].copy()
    r = _fit_hac(f_p, ts_e)
    b, se, pv = r.params["Post"], r.bse["Post"], r.pvalues["Post"]
    print(f"  {'Excl. Feb17-23':<32} {b:+9.4f}  {se:7.4f}  {pv:.3f}{stars(pv)}")

    r = _fit_hac(f_p, ts_base)
    b, se, pv = r.params["Post"], r.bse["Post"], r.pvalues["Post"]
    print(f"  {'Baseline (main spec)':<32} {b:+9.4f}  {se:7.4f}  {pv:.3f}{stars(pv)}")
    print()


def print_total_market_its(dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    print(f"{'─'*72}\n  Total AI Market Activity — ITS (FEB1, linear, HAC-10)\n{'─'*72}")
    print(f"  {'Metric':<12} {'Window':<6} {'β2':>9} {'SE':>7} {'Δ%':>8} {'p':>8}")
    print("  " + "─" * 70)
    for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
        for win_label, ws in WINDOWS:
            ts = build_total_market(df, ws)
            if ts is None:
                continue
            res = _fit_hac("log_total ~ t + Post + C(dow)", ts)
            b   = float(res.params["Post"])
            se  = float(res.bse["Post"])
            pv  = float(res.pvalues["Post"])
            pct = float((np.exp(b) - 1) * 100)
            print(f"  {metric:<12} {win_label:<6} {b:+9.4f}  {se:7.4f}  "
                  f"{pct:+8.1f}%  {pv:.3f}{stars(pv)}")
    print()


# ── Placebo & Permutation ─────────────────────────────────────────────────────

def placebo_time(
    df: pd.DataFrame,
    metric: str,
) -> tuple[list, float]:
    ts_base = build_share(df)
    if ts_base is None:
        return [], float("nan")

    candidate_dates = pd.date_range(
        WINDOW_DEFAULT + timedelta(days=MIN_PRE_DAYS), PLACEBO_CUTOFF, freq="D"
    )
    f_p = FORMULAS[("linear", "parsim")]
    results = []
    for fake in candidate_dates:
        ts = ts_base.copy()
        ts["Post"]   = (ts["date"] >= fake).astype(int)
        ts["t_post"] = np.maximum(0, (ts["date"] - fake).dt.days.astype(float))
        if ts["Post"].nunique() < 2:
            continue
        try:
            r = _fit_hac(f_p, ts)
            results.append((fake, float(r.params["Post"]), float(r.pvalues["Post"])))
        except Exception:
            continue

    # Real estimate from the same base series — ensures comparability
    ts_real = ts_base.copy()
    try:
        r_real_fit = _fit_hac(f_p, ts_real)
        real_b = float(r_real_fit.params["Post"])
    except Exception:
        return results, float("nan")

    coefs = [r[1] for r in results]
    n     = len(coefs)
    rank  = sum(abs(c) >= abs(real_b) for c in coefs)
    emp_p = (rank + 1) / (n + 1) if n > 0 else float("nan")
    print(f"  [{metric}] Time placebo: {n} fake dates | "
          f"real β2={real_b:+.4f} | empirical p={emp_p:.3f} ({rank}/{n})")
    return results, real_b


def placebo_unit(
    df: pd.DataFrame,
    metric: str,
) -> tuple[list, float]:
    apps  = sorted(df.columns.tolist())
    ctrls = [a for a in apps if a not in [APP_CL, APP_GPT]]
    f_p   = FORMULAS[("linear", "parsim")]
    results = []

    for ctrl in ctrls:
        if ctrl not in df.columns:
            continue
        sub   = df[[APP_GPT, ctrl]].dropna()
        sub   = sub[sub.index >= WINDOW_DEFAULT]
        total = sub[APP_GPT] + sub[ctrl]
        mask  = total > 0
        sub, total = sub[mask], total[mask]

        ts = pd.DataFrame(index=sub.index)
        ts["S"]      = (sub[APP_GPT] / total) * 100
        ts["t"]      = (ts.index - WINDOW_DEFAULT).days.astype(float)
        ts["t_post"] = np.maximum(0, (ts.index - EVENT).days.astype(float))
        ts["Post"]   = (ts.index >= EVENT).astype(int)
        ts["dow"]    = ts.index.dayofweek.astype(str)
        ts = ts.reset_index(names="date").dropna()
        if ts["Post"].nunique() < 2:
            continue
        try:
            r = _fit_hac(f_p, ts)
            b, pv = float(r.params["Post"]), float(r.pvalues["Post"])
            results.append((f"GPT/{ctrl.split()[0]}", b, pv))
            print(f"  [{metric}] Unit placebo — GPT vs {ctrl.split()[0]:<10} "
                  f"β2={b:+.4f}  p={pv:.3f}{stars(pv)}")
        except Exception:
            continue

    r_real = run_its(df, outcome="linear", model="parsim")
    if r_real is None:
        return results, float("nan")
    real_b = r_real.beta2
    coefs  = [r[1] for r in results]
    rank   = sum(abs(c) >= abs(real_b) for c in coefs)
    emp_p  = (rank + 1) / (len(coefs) + 1)
    print(f"  [{metric}] Claude share β2={real_b:+.4f} | empirical p={emp_p:.3f}\n")
    return results, real_b


def perm_inference(
    df: pd.DataFrame,
    metric: str,
) -> tuple[list[float], list[float], float]:
    f_p     = FORMULAS[("linear", "parsim")]
    ts_base = build_share(df)
    if ts_base is None:
        return [], [], float("nan")

    r_real = run_its(df, outcome="linear", model="parsim")
    if r_real is None:
        return [], [], float("nan")
    real_b = r_real.beta2

    perm_dates = pd.date_range(
        WINDOW_DEFAULT + timedelta(days=MIN_PRE_DAYS), PLACEBO_CUTOFF, freq="D"
    ).tolist()

    # Date permutation — without replacement, capped at unique candidate count
    n_cands    = len(perm_dates)
    date_coefs: list[float] = []
    for idx in rng.choice(n_cands, size=min(N_PERM, n_cands), replace=False):
        ts = ts_base.copy()
        fake = perm_dates[idx]
        ts["Post"]   = (ts["date"] >= fake).astype(int)
        ts["t_post"] = np.maximum(0, (ts["date"] - fake).dt.days.astype(float))
        if ts["Post"].nunique() < 2:
            continue
        try:
            r = _fit_hac(f_p, ts)
            date_coefs.append(float(r.params["Post"]))
        except Exception:
            continue

    # Treatment permutation: shuffle Post indicator across time to destroy
    # the association between the treatment break and outcome, producing a
    # symmetric null centred at zero (not a bimodal ±real_b distribution).
    treat_coefs: list[float] = []
    for _ in range(N_PERM):
        ts = ts_base.copy()
        ts["Post"] = rng.permutation(ts["Post"].values)
        post_dates = ts["date"][ts["Post"] == 1]
        if post_dates.empty or ts["Post"].nunique() < 2:
            continue
        ts["t_post"] = np.maximum(
            0, (ts["date"] - post_dates.min()).dt.days.astype(float)
        )
        try:
            r = _fit_hac(f_p, ts)
            treat_coefs.append(float(r.params["Post"]))
        except Exception:
            continue

    p_date  = float(np.mean(np.abs(date_coefs)  >= abs(real_b))) if date_coefs  else float("nan")
    p_treat = float(np.mean(np.abs(treat_coefs) >= abs(real_b))) if treat_coefs else float("nan")
    print(f"  [{metric}] Permutation — date: p={p_date:.3f}  "
          f"treatment: p={p_treat:.3f}  (real β2={real_b:+.4f})")
    return date_coefs, treat_coefs, real_b


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_share_raw() -> None:
    scopes    = [("US", "US"), ("Global", None), ("Global-ex-US", "ex-US")]
    platforms = [("Combined", None), ("iOS", "iOS"), ("Android", "Android")]

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=False)
    fig.suptitle("Raw Claude Share of (Claude + ChatGPT) — Geo × Platform",
                 fontsize=13, fontweight="bold", y=1.01)

    for ri, (scope_lbl, country) in enumerate(scopes):
        for ci, (plat_lbl, platform) in enumerate(platforms):
            ax = axes[ri, ci]
            for df_path, metric, color in [
                ("downloads.csv", "Downloads", "#2563EB"),
                ("dau.csv",       "DAU",       "#16A34A"),
            ]:
                df = load(DATA_DIR / df_path, metric, country=country, platform=platform)
                ts = build_share(df)
                if ts is None:
                    continue
                ax.plot(ts["date"], ts["S"], lw=1.5, color=color,
                        label=metric, alpha=0.85)
            ax.axvline(EVENT,      color=EVENT_COLOR, lw=1.0, ls="--", alpha=0.75)
            ax.axvline(SONNET_REL, color="#D97706",   lw=0.8, ls=":",  alpha=0.65)
            ax.set_title(f"{scope_lbl} / {plat_lbl}", fontsize=9)
            ax.set_ylabel("Share (%)", fontsize=8)
            style_ax(ax)
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            ax.tick_params(axis="y", labelsize=8)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    fname = PLOTS_DIR / "fig1_share_raw.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


def plot_pretrend_share(dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ITS Pre-Trend Check — Claude Share, US",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, (df, metric) in zip(axes, [(dl, "Downloads"), (dau, "DAU")]):
        ts = build_share(df)
        if ts is None:
            continue
        pre  = ts[ts["Post"] == 0]
        post = ts[ts["Post"] == 1]
        slope, intercept = np.polyfit(pre["t"].values, pre["S"].values, 1)
        trend = intercept + slope * ts["t"].values

        ax.plot(pre["date"],  pre["S"],  color="#64748B", lw=1.5, label="Pre")
        ax.plot(post["date"], post["S"], color="#2563EB", lw=1.8, label="Post")
        ax.plot(ts["date"], trend, ls="--", color="#94A3B8", lw=1.2,
                label=f"Pre-trend ({slope:+.4f}/d)")
        ax.axvline(EVENT,      color=EVENT_COLOR, lw=1.2, ls="--", alpha=0.75, label="Feb 27")
        ax.axvline(SONNET_REL, color="#D97706",   lw=1.0, ls=":",  alpha=0.75, label="Feb 17")
        ax.set_title(metric); ax.set_ylabel("Claude share (%)")
        ax.legend(); style_ax(ax)

    plt.tight_layout()
    fname = PLOTS_DIR / "fig2_pretrend.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


def plot_its_main(dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ITS — Claude Share of (Claude + ChatGPT), US",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, (df, metric) in zip(axes, [(dl, "Downloads"), (dau, "DAU")]):
        raw = _run_its_raw(df, outcome="linear", model="parsim")
        if raw is None:
            continue
        ts, res = raw
        r = run_its(df, outcome="linear", model="parsim")
        if r is None:
            continue

        post    = ts[ts["Post"] == 1]
        ts_cf   = ts.copy()
        ts_cf["Post"]   = 0
        ts_cf["t_post"] = 0.0
        cf_vals = res.predict(ts_cf)

        ax.plot(ts["date"], ts["S"], "o-", color="#2563EB",
                ms=3.5, lw=1.8, label="Actual share")
        ax.plot(ts["date"], cf_vals, "--", color="#94A3B8",
                lw=1.5, label="Counterfactual (pre-trend)")
        ax.fill_between(post["date"],
                        cf_vals[ts["Post"] == 1],
                        post["S"],
                        alpha=0.12, color="#2563EB", label="Treatment gap")
        ax.axvline(EVENT,      color=EVENT_COLOR, lw=1.2, ls="--", alpha=0.75, label="Feb 27")
        ax.axvline(SONNET_REL, color="#D97706",   lw=1.0, ls=":",  alpha=0.75, label="Feb 17 (Sonnet)")
        ax.set_title(f"{metric}   β₂ = {r.me:+.2f} pp   p = {r.pval2:.3f}")
        ax.set_xlabel("Date"); ax.set_ylabel("Claude share (%)")
        ax.legend(); style_ax(ax)

    plt.tight_layout()
    fname = PLOTS_DIR / "fig3_its_main.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


def plot_sensitivity(dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    pool = [(wl, ws, out, mod)
            for wl, ws in WINDOWS
            for out in ["linear", "logit"]
            for mod in ["parsim", "full"]]
    colors  = {"linear": "#2563EB", "logit": "#16A34A"}
    markers = {"parsim": "o", "full": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ITS Sensitivity — β₂ (Level Shift) across Specifications",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, (df, metric) in zip(axes, [(dl, "Downloads"), (dau, "DAU")]):
        for xi, (wl, ws, out, mod) in enumerate(pool):
            r = run_its(df, ws, out, mod)
            if r is None:
                continue
            # 1.96 is an asymptotic approximation; with n~200 a t-critical
            # would be ~2.0–2.1, leaving results qualitatively unchanged.
            ci = 1.96 * r.me_se
            ax.errorbar(xi, r.me, yerr=ci,
                        fmt=markers[mod], color=colors[out],
                        ms=6, capsize=3, lw=1.4,
                        label=f"{out}/{mod}" if xi < 4 else "")
        ax.axhline(0, color="#6B7280", lw=0.8, alpha=0.5)
        ax.set_xticks(range(len(pool)))
        ax.set_xticklabels([f"{wl}\n{out[:3]}/{mod[:3]}" for wl, _, out, mod in pool])
        ax.set_title(metric); ax.set_ylabel("β₂ / ME (pp)"); style_ax(ax)
        if metric == "Downloads":
            handles, lbls = ax.get_legend_handles_labels()
            ax.legend(handles[:4], lbls[:4])

    plt.tight_layout()
    fname = PLOTS_DIR / "fig4_sensitivity.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


def plot_placebos(placebo_store: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("ITS Placebo Tests — Claude Share (US)",
                 fontsize=13, fontweight="bold", y=1.01)

    for col, metric in enumerate(["Downloads", "DAU"]):
        store = placebo_store[metric]
        time_res, time_real = store["time_res"], store["time_real"]
        unit_res, unit_real = store["unit_res"], store["unit_real"]

        ax    = axes[0, col]
        coefs = [r[1] for r in time_res]
        n     = len(coefs)
        rank  = sum(abs(c) >= abs(time_real) for c in coefs)
        emp_p = (rank + 1) / (n + 1) if n > 0 else float("nan")
        ax.hist(coefs, bins=15, color=HIST_COLOR, edgecolor=HIST_EDGE, lw=0.5, alpha=0.9)
        ax.axvline( time_real, color=EVENT_COLOR, lw=2,
                    label=f"Real β₂ = {time_real:+.4f}   emp-p = {emp_p:.3f}")
        ax.axvline(-time_real, color=EVENT_COLOR, lw=1.2, ls="--", alpha=0.45)
        ax.set_title(f"{metric} — Time placebo (N={n})")
        ax.set_xlabel("β₂ (pp)"); ax.set_ylabel("Count")
        ax.legend(); style_ax(ax)

        ax     = axes[1, col]
        labels = [r[0] for r in unit_res] + ["Claude share"]
        vals   = [r[1] for r in unit_res] + [unit_real]
        bar_colors = ["#CBD5E1"] * len(unit_res) + ["#2563EB"]
        ax.barh(labels, vals, color=bar_colors, edgecolor="white", height=0.55)
        ax.axvline(0, color="#6B7280", lw=0.8, alpha=0.5)
        ax.set_title(f"{metric} — Unit placebo")
        ax.set_xlabel("β₂ (pp)"); style_ax(ax)

    plt.tight_layout()
    fname = PLOTS_DIR / "fig5_placebos.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


def plot_permutations(perm_store: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"ITS Permutation Inference — Claude Share (N={N_PERM})",
                 fontsize=13, fontweight="bold", y=1.01)

    for col, metric in enumerate(["Downloads", "DAU"]):
        store  = perm_store[metric]
        real_b = store["real_b"]
        for row, (coefs, title) in enumerate([
            (store["date_coefs"],  "Date permutation"),
            (store["treat_coefs"], "Treatment permutation"),
        ]):
            ax = axes[row, col]
            if not coefs or np.isnan(real_b):
                ax.set_visible(False); continue
            p = float(np.mean(np.abs(coefs) >= abs(real_b)))
            ax.hist(coefs, bins=40, color=HIST_COLOR, edgecolor=HIST_EDGE, lw=0.5, alpha=0.9)
            ax.axvline( real_b, color=EVENT_COLOR, lw=2,
                        label=f"Real β₂ = {real_b:+.4f}   p = {p:.3f}")
            ax.axvline(-real_b, color=EVENT_COLOR, lw=1.2, ls="--", alpha=0.45)
            ax.set_title(f"{metric} — {title}")
            ax.set_xlabel("β₂ (pp)"); ax.set_ylabel("Count")
            ax.legend(); style_ax(ax)

    plt.tight_layout()
    fname = PLOTS_DIR / "fig6_permutations.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {fname}")


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_full_analysis() -> None:
    print("─" * 80)
    print(f"  ITS  |  {' · '.join(f'{k}={v}' for k, v in VERSIONS.items())}")
    print("─" * 80 + "\n")

    dl  = load(DATA_DIR / "downloads.csv", "Downloads", country="US")
    dau = load(DATA_DIR / "dau.csv",       "DAU",       country="US")

    print_total_market_its(dl, dau)

    print_diagnostics(dl, dau)

    for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
        print_main_results(df, metric)
        print_hac_sensitivity(df, metric)
        print_sensitivity_matrix(df, metric)
        print_event_window(df, metric)
        print_confounder_checks(df, metric)

    print_geo_platform()

    print(f"{'─'*72}\n  Placebo Tests\n{'─'*72}")
    placebo_store: dict = {}
    for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
        time_res, time_real = placebo_time(df, metric)
        print()
        unit_res, unit_real = placebo_unit(df, metric)
        placebo_store[metric] = {
            "time_res": time_res, "time_real": time_real,
            "unit_res": unit_res, "unit_real": unit_real,
        }

    print(f"{'─'*72}\n  Permutation Inference (N={N_PERM})\n{'─'*72}")
    global rng
    rng = np.random.default_rng(RNG_SEED)   # reset so results are order-independent
    perm_store: dict = {}
    for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
        date_coefs, treat_coefs, real_b = perm_inference(df, metric)
        perm_store[metric] = {
            "date_coefs":  date_coefs,
            "treat_coefs": treat_coefs,
            "real_b":      real_b,
        }
    print()

    plot_share_raw()
    plot_pretrend_share(dl, dau)
    plot_its_main(dl, dau)
    plot_sensitivity(dl, dau)
    plot_placebos(placebo_store)
    plot_permutations(perm_store)

    all_results: dict[str, Any] = {"versions": VERSIONS, "main": {}, "sensitivity": {}}
    for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
        all_results["main"][metric] = {}
        for outcome in ("linear", "logit"):
            for model in ("parsim", "full"):
                r = run_its(df, outcome=outcome, model=model)
                all_results["main"][metric][f"{outcome}_{model}"] = (
                    r.to_dict() if r else None)

    for spec in ITS_SENSITIVITY_GRID:
        ws   = _WINDOW_START[spec["window_label"]]
        lags = spec["lags"] if spec["lags"] is not None else HAC_LAGS_DEFAULT
        key  = f"{spec['window_label']}_{spec['outcome']}_{spec['model']}_{spec['se']}"
        if spec["lags"] is not None:
            key += f"_{spec['lags']}"
        all_results["sensitivity"][key] = {}
        for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
            r = run_its(df, ws, spec["outcome"], spec["model"],
                        se_type=spec["se"], lags=lags)
            all_results["sensitivity"][key][metric] = r.to_dict() if r else None

    save_json(all_results, OUTPUT_DIR / "last_run.json")

"""App fixed-effects DiD analysis pipeline (did_app.py entry point).

Model:
    log Y_it = α_i + γ_i·t + Σ φ_d·dow_d + δ·(Treated_i × Post_t) + ε_it

α_i  : app fixed effects via C(app)
γ_i  : app-specific linear (or quadratic) trend via C(app):t
δ    : treatment effect (primary coefficient of interest)

HAC lags selected dynamically using Newey-West Andrews (1991) rule.
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

from pipeline.estimation import hac_lags
from pipeline.types import DiDResult
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
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots" / "did_app"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = output_dir("did_app")

_WINDOW_START: dict[str, pd.Timestamp] = {lbl: ts for lbl, ts in WINDOWS}

TREND_LABELS: dict[str, str] = {"linear": "Linear", "quadratic": "Quadratic"}
COLORS: dict[str, str]       = {"linear": "#2563EB", "quadratic": "#16A34A"}

SENSITIVITY_GRID: list[dict[str, Any]] = [
    {"window_label": wl, "trend": tt, "se": se}
    for wl in ("FEB1", "JAN1")
    for tt in ("linear", "quadratic")   # spline excluded: app-specific spline bases
                                        # are over-parameterized with 5 apps in the panel
    for se in ("HAC", "HC3")
]

FORMULAS: dict[str, str] = {
    "linear":    "log_Y ~ C(app) + C(app):t + C(dow) + DiD",
    "quadratic": "log_Y ~ C(app) + C(app):t + C(app):t2 + C(dow) + DiD",
}


# ── Panel ─────────────────────────────────────────────────────────────────────

def build_panel(
    df: pd.DataFrame,
    treated: str,
    controls: list[str],
    window_start: pd.Timestamp | None = None,
    post_start: pd.Timestamp | None = None,
    drop_dates: list[pd.Timestamp] | None = None,
    drop_date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    date_end_exclusive: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    if window_start is None:
        window_start = WINDOW_DEFAULT
    if post_start is None:
        post_start = EVENT

    ctrl = [a for a in controls if a in df.columns]
    if not ctrl or treated not in df.columns:
        return None

    win = df[df.index >= window_start]
    rows = []
    for app in [treated] + ctrl:
        col     = win[app]
        n_zeros = int((col == 0).sum())
        if n_zeros:
            print(f"  [warn] {app}: {n_zeros} zero(s) → NaN before log")
        tmp = pd.DataFrame({
            "log_Y":   np.log(col.replace(0, np.nan)),
            "app":     app,
            "treated": int(app == treated),
        }).reset_index(names="date")
        rows.append(tmp)

    panel = pd.concat(rows).dropna(subset=["log_Y"])

    if drop_dates:
        panel = panel[~panel["date"].isin(drop_dates)]
    if drop_date_ranges:
        for lo, hi in drop_date_ranges:
            panel = panel[~((panel["date"] >= lo) & (panel["date"] < hi))]
    if date_end_exclusive is not None:
        panel = panel[panel["date"] < date_end_exclusive]

    # Sort and reindex t from window_start (calendar time preserved after drops)
    panel = panel.sort_values(["date", "app"]).reset_index(drop=True)
    panel["t"]    = (panel["date"] - window_start).dt.days.astype(float)
    panel["t2"]   = panel["t"] ** 2
    panel["Post"] = (panel["date"] >= post_start).astype(int)
    panel["DiD"]  = panel["treated"] * panel["Post"]
    panel["dow"]  = panel["date"].dt.dayofweek.astype(str)
    return panel


# ── Estimation ────────────────────────────────────────────────────────────────

def _fit(panel: pd.DataFrame, formula: str, se_type: str = "HAC") -> Any:
    lags = hac_lags(len(panel))
    if se_type == "HAC":
        return smf.ols(formula, data=panel).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags})
    return smf.ols(formula, data=panel).fit(cov_type="HC3")


def run_did(
    df: pd.DataFrame,
    treated: str,
    controls: list[str],
    window_start: pd.Timestamp | None = None,
    post_start: pd.Timestamp | None = None,
    trend_type: str = "linear",
    se_type: str = "HAC",
    drop_dates: list[pd.Timestamp] | None = None,
    drop_date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    date_end_exclusive: pd.Timestamp | None = None,
) -> DiDResult | None:
    if window_start is None:
        window_start = WINDOW_DEFAULT
    if post_start is None:
        post_start = EVENT

    panel = build_panel(
        df, treated, controls, window_start, post_start,
        drop_dates, drop_date_ranges, date_end_exclusive,
    )
    if panel is None or panel["DiD"].nunique() < 2:
        return None

    try:
        res = _fit(panel, FORMULAS[trend_type], se_type)
    except Exception:
        return None

    n_units = panel["app"].nunique()
    coef    = float(res.params["DiD"])
    lags    = hac_lags(len(panel))

    return DiDResult(
        coef         = coef,
        pct_change   = float((np.exp(coef) - 1) * 100),
        pval         = float(res.pvalues["DiD"]),
        se           = float(res.bse["DiD"]),
        n_pre        = int((panel["Post"] == 0).sum() // n_units),
        n_post       = int((panel["Post"] == 1).sum() // n_units),
        hac_lags     = lags,
        trend        = trend_type,
        pool         = tuple(controls),
        window_start = window_start,
        post_start   = post_start,
    )


def did_coef(
    df: pd.DataFrame,
    treated: str,
    controls: list[str],
    event: pd.Timestamp,
    window_start: pd.Timestamp | None = None,
) -> DiDResult | None:
    return run_did(df, treated, controls,
                   window_start=window_start,
                   post_start=event)


def _lag_col(k: int) -> str:
    return f"dm{abs(k)}" if k < 0 else f"d{k}"


def run_event_study(
    df: pd.DataFrame,
    treated: str,
    controls: list[str],
    window_start: pd.Timestamp | None = None,
) -> dict[int, tuple[float, float]] | None:
    panel = build_panel(df, treated, controls, window_start)
    if panel is None:
        return None

    panel = panel.copy()
    panel["k"] = (panel["date"] - EVENT).dt.days.clip(-ES_K, ES_K).astype(int)
    ks = [k for k in range(-ES_K, ES_K + 1) if k != -1]
    for k in ks:
        panel[_lag_col(k)] = ((panel["treated"] == 1) & (panel["k"] == k)).astype(int)

    f   = ("log_Y ~ C(app) + C(app):t + C(dow) + "
           + " + ".join(_lag_col(k) for k in ks))
    try:
        res = _fit(panel, f)
    except Exception as e:
        print(f"  [warn] event study estimation failed: {e}")
        return None

    coefs: dict[int, tuple[float, float]] = {-1: (0.0, 0.0)}
    for k in ks:
        coefs[k] = (
            float(res.params.get(_lag_col(k), np.nan)),
            float(res.bse.get(_lag_col(k),    np.nan)),
        )
    return coefs


# ── Print tables ──────────────────────────────────────────────────────────────

def print_results(
    treated: str,
    apps: list[str],
    dl: pd.DataFrame,
    dau: pd.DataFrame,
) -> None:
    others  = [a for a in apps if a != treated]
    other   = APP_GPT if treated == APP_CL else APP_CL
    pool_ex = [a for a in others if a != other]

    specs = ([(a, [a]) for a in others]
             + [("Pool (all)", others),
                (f"Pool (excl {other.split()[0]})", pool_ex)])

    print(f"{'─'*72}\n  Treated: {treated}\n{'─'*72}")
    print(f"  {'Control':<34} {'δ':>7} {'Δ%':>8} {'p':>6}    {'δ':>7} {'Δ%':>8} {'p':>6}")
    print(f"  {'':>34} {'Downloads':^24}  {'DAU':^24}")
    for label, ctrls in specs:
        rd  = run_did(dl,  treated, ctrls)
        rda = run_did(dau, treated, ctrls)
        dl_s  = (f"{rd.coef:+.3f}  {rd.pct_change:+.1f}%  {rd.pval:.3f}{stars(rd.pval)}"
                 if rd  else "—")
        dau_s = (f"{rda.coef:+.3f}  {rda.pct_change:+.1f}%  {rda.pval:.3f}{stars(rda.pval)}"
                 if rda else "—")
        print(f"  {label:<34} {dl_s:<24}  {dau_s}")
    print()


def print_window_sensitivity(
    apps: list[str],
    dl: pd.DataFrame,
    dau: pd.DataFrame,
) -> None:
    pool_ex_cl  = [a for a in apps if a not in [APP_CL, APP_GPT]]
    pool_ex_gpt = pool_ex_cl

    print(f"{'─'*90}\n  Window × Trend Sensitivity (app-FE)\n{'─'*90}")
    for treated, pool in [(APP_CL, pool_ex_cl), (APP_GPT, pool_ex_gpt)]:
        print(f"\n  Treated: {treated}")
        hdr  = f"  {'Window':<8} {'Pre-days':>9}  {'Trend':<11}"
        hdr += f"  {'δ_DL':>8} {'Δ%_DL':>9} {'p_DL':>7}"
        hdr += f"  {'δ_DAU':>8} {'Δ%_DAU':>9} {'p_DAU':>7}"
        print(hdr)
        print("  " + "─" * 86)
        for win_label, ws in WINDOWS:
            pre_days = (EVENT - ws).days
            for tt in ("linear", "quadratic"):
                rd  = run_did(dl,  treated, pool, ws, trend_type=tt)
                rda = run_did(dau, treated, pool, ws, trend_type=tt)
                dl_s  = (f"{rd.coef:+.4f}  {rd.pct_change:+.1f}%  {rd.pval:.3f}{stars(rd.pval)}"
                         if rd  else "—")
                dau_s = (f"{rda.coef:+.4f}  {rda.pct_change:+.1f}%  {rda.pval:.3f}{stars(rda.pval)}"
                         if rda else "—")
                print(f"  {win_label:<8} {pre_days:>9}  "
                      f"{TREND_LABELS[tt]:<11}  {dl_s:<28}  {dau_s}")
        print()


def print_sensitivity_matrix(
    apps: list[str],
    dl: pd.DataFrame,
    dau: pd.DataFrame,
) -> None:
    pool_ex = [a for a in apps if a not in [APP_CL, APP_GPT]]

    print(f"{'─'*90}\n  Sensitivity Matrix: Window × Trend × SE (app-FE)\n{'─'*90}")
    print(f"  {'Treated':<10} {'Metric':<12} {'Window':<6} {'Trend':<11} "
          f"{'SE':<5} {'δ':>8} {'SE_δ':>7} {'Δ%':>9} {'p':>8}")
    for treated in [APP_CL, APP_GPT]:
        for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
            for spec in SENSITIVITY_GRID:
                ws = _WINDOW_START[spec["window_label"]]
                r  = run_did(df, treated, pool_ex,
                             window_start=ws,
                             trend_type=spec["trend"],
                             se_type=spec["se"])
                if r:
                    print(f"  {treated.split()[0]:<10} {metric:<12} "
                          f"{spec['window_label']:<6} {TREND_LABELS[spec['trend']]:<11} "
                          f"{spec['se']:<5} "
                          f"{r.coef:+.3f}  {r.se:.3f}  {r.pct_change:+.1f}%  "
                          f"{r.pval:.3f}{stars(r.pval)}")
        print()


def print_leave_one_out(
    apps: list[str],
    dl: pd.DataFrame,
    dau: pd.DataFrame,
) -> None:
    pool_full = [a for a in apps if a not in [APP_CL, APP_GPT]]

    print(f"{'─'*72}\n  Leave-One-Out Donor Robustness (Claude, FEB1, linear, HAC)\n{'─'*72}")
    print(f"  {'Pool':<32} {'δ_DL':>7} {'Δ%_DL':>8} {'p_DL':>8}    "
          f"{'δ_DAU':>7} {'Δ%_DAU':>8} {'p_DAU':>8}")
    print("  " + "─" * 70)

    rd  = run_did(dl,  APP_CL, pool_full)
    rda = run_did(dau, APP_CL, pool_full)
    dl_s  = (f"{rd.coef:+.3f}  {rd.pct_change:+.1f}%  {rd.pval:.3f}{stars(rd.pval)}"
             if rd  else "—")
    dau_s = (f"{rda.coef:+.3f}  {rda.pct_change:+.1f}%  {rda.pval:.3f}{stars(rda.pval)}"
             if rda else "—")
    print(f"  {'Full pool (excl. ChatGPT)':<32} {dl_s:<28} {dau_s}")

    for app in pool_full:
        loo_pool = [a for a in pool_full if a != app]
        label    = f"Excl. {app.split()[0]}"
        rd  = run_did(dl,  APP_CL, loo_pool)
        rda = run_did(dau, APP_CL, loo_pool)
        dl_s  = (f"{rd.coef:+.3f}  {rd.pct_change:+.1f}%  {rd.pval:.3f}{stars(rd.pval)}"
                 if rd  else "—")
        dau_s = (f"{rda.coef:+.3f}  {rda.pct_change:+.1f}%  {rda.pval:.3f}{stars(rda.pval)}"
                 if rda else "—")
        print(f"  {label:<32} {dl_s:<28} {dau_s}")
    print()


def print_geo_platform(apps: list[str]) -> None:
    pool_ex_cl  = [a for a in apps if a not in [APP_CL, APP_GPT]]
    pool_ex_gpt = pool_ex_cl

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

    print(f"{'─'*72}\n  Geo × Platform Heterogeneity (FEB1, linear, app-FE)\n{'─'*72}")
    for treated, pool in [(APP_CL, pool_ex_cl), (APP_GPT, pool_ex_gpt)]:
        print(f"  Treated: {treated}")
        print(f"  {'Scope':<14} {'Platform':<10} {'Δ%_DL':>9} {'p_DL':>8}  "
              f"{'Δ%_DAU':>9} {'p_DAU':>8}")
        for scope_lbl, country in scopes:
            for plat_lbl, platform in platforms:
                data      = geo_data[(scope_lbl, plat_lbl)]
                df_dl     = data["dl"]
                df_dau    = data["dau"]
                apps_here = sorted(set(df_dl.columns) & set(df_dau.columns))
                pool_here = [a for a in pool if a in apps_here]
                if treated not in apps_here or not pool_here:
                    print(f"  {scope_lbl:<14} {plat_lbl:<10} "
                          f"{'—':>9} {'—':>8}  {'—':>9} {'—':>8}")
                    continue
                rd  = run_did(df_dl,  treated, pool_here)
                rda = run_did(df_dau, treated, pool_here)
                dl_s  = (f"{rd.pct_change:+.1f}%  {rd.pval:.3f}{stars(rd.pval)}"
                         if rd  else "—")
                dau_s = (f"{rda.pct_change:+.1f}%  {rda.pval:.3f}{stars(rda.pval)}"
                         if rda else "—")
                print(f"  {scope_lbl:<14} {plat_lbl:<10} {dl_s:<18}  {dau_s}")
        print()


def print_event_window(
    apps: list[str],
    dl: pd.DataFrame,
    dau: pd.DataFrame,
) -> None:
    pool_ex = [a for a in apps if a not in [APP_CL, APP_GPT]]

    specs = [
        ("Point  (>= Feb 27)",    {}),
        ("Window (>= Feb 24)",    {"post_start": SHOCK_START}),
        ("Wash-in (excl 24-26)",  {"drop_date_ranges": [(SHOCK_START, EVENT)]}),
    ]

    print(f"{'─'*72}\n  Event-Window Robustness (app-FE)\n{'─'*72}")
    for treated in [APP_CL, APP_GPT]:
        print(f"\n  Treated: {treated}")
        print(f"  {'Design':<26} {'δ':>7} {'Δ%':>8} {'p':>8}    {'δ':>7} {'Δ%':>8} {'p':>8}")
        print(f"  {'':>26} {'Downloads':^26}  {'DAU':^26}")
        for label, kwargs in specs:
            rows_s = []
            for df in [dl, dau]:
                r = run_did(df, treated, pool_ex, **kwargs)
                rows_s.append(
                    f"{r.coef:+.3f}  {r.pct_change:+.1f}%  {r.pval:.3f}{stars(r.pval)}"
                    if r else "—"
                )
            print(f"  {label:<26} {rows_s[0]:<28} {rows_s[1]}")
    print()


def print_confounder_checks(
    apps: list[str],
    dl: pd.DataFrame,
    dau: pd.DataFrame,
) -> None:
    pool_ex = [a for a in apps if a not in [APP_CL, APP_GPT]]

    print(f"{'─'*72}\n  Confounder Checks (Claude, app-FE)\n{'─'*72}")
    print(f"  {'Design':<32} {'δ':>7} {'Δ%':>8} {'p':>8}    {'δ':>7} {'Δ%':>8} {'p':>8}")
    print(f"  {'':>32} {'Downloads':^26}  {'DAU':^26}")

    def _row(r: DiDResult | None) -> str:
        if r is None:
            return "—"
        return f"{r.coef:+.3f}  {r.pct_change:+.1f}%  {r.pval:.3f}{stars(r.pval)}"

    specs: list[tuple[str, Any]] = [
        (
            "Falsif.: event=Feb17, pre-Feb24",
            lambda d: run_did(d, APP_CL, pool_ex,
                              post_start=SONNET_REL,
                              date_end_exclusive=SHOCK_START),
        ),
        (
            "Main excl. Feb17-23",
            lambda d: run_did(d, APP_CL, pool_ex,
                              drop_date_ranges=[(SONNET_REL, SHOCK_START)]),
        ),
        (
            "Baseline (main spec)",
            lambda d: run_did(d, APP_CL, pool_ex),
        ),
    ]
    for label, pfn in specs:
        rows_s = [_row(pfn(dl)), _row(pfn(dau))]
        print(f"  {label:<32} {rows_s[0]:<28} {rows_s[1]}")
    print()


# ── Placebo & Permutation ─────────────────────────────────────────────────────

def placebo_time(
    df: pd.DataFrame,
    metric: str,
    treated: str,
    controls: list[str],
) -> tuple[list, DiDResult | None]:
    candidate_dates = pd.date_range(
        WINDOW_DEFAULT + timedelta(days=MIN_PRE_DAYS), PLACEBO_CUTOFF, freq="D"
    )
    results = []
    for fake in candidate_dates:
        r = did_coef(df, treated, controls, fake)
        if r:
            results.append((fake, r.coef, r.pval))

    real = did_coef(df, treated, controls, EVENT)
    if real is None:
        return results, None

    coefs = [r[1] for r in results]
    n     = len(coefs)
    rank  = sum(abs(c) >= abs(real.coef) for c in coefs)
    # +1 finite-sample correction (Davison & Hinkley; Phipson & Smyth 2010)
    emp_p = (rank + 1) / (n + 1) if n > 0 else np.nan
    label = treated.split()[0]
    print(f"  [{metric}|{label}] Time placebo: {n} fake dates | "
          f"real δ={real.coef:+.3f} | empirical p={emp_p:.3f} ({rank}/{n})")
    return results, real


def placebo_unit(
    df: pd.DataFrame,
    metric: str,
    treated: str,
    controls: list[str],
) -> tuple[list, DiDResult | None]:
    results = []
    for app in controls:
        other_ctrls = [a for a in controls if a != app]
        r = did_coef(df, app, other_ctrls, EVENT)
        if r:
            results.append((app.split()[0], r.coef, r.pval))
            print(f"  [{metric}|{treated.split()[0]}] Unit placebo — {app.split()[0]:<12} "
                  f"δ={r.coef:+.3f}  Δ%={r.pct_change:+.1f}%  p={r.pval:.3f}{stars(r.pval)}")

    real = did_coef(df, treated, controls, EVENT)
    if real is None:
        return results, None

    coefs = [r[1] for r in results]
    rank  = sum(abs(c) >= abs(real.coef) for c in coefs)
    emp_p = (rank + 1) / (len(coefs) + 1)
    print(f"  [{metric}|{treated.split()[0]}] {treated.split()[0]} δ={real.coef:+.3f} "
          f"| empirical p={emp_p:.3f}\n")
    return results, real


def perm_inference(
    df: pd.DataFrame,
    metric: str,
    treated: str,
    all_apps: list[str],
    controls: list[str],
) -> tuple[list[float], list[float], float]:
    real_r = did_coef(df, treated, controls, EVENT)
    if real_r is None:
        return [], [], float("nan")
    real = real_r.coef

    perm_dates = pd.date_range(
        WINDOW_DEFAULT + timedelta(days=MIN_PRE_DAYS), PLACEBO_CUTOFF, freq="D"
    ).tolist()

    # Sample without replacement, capped at the number of unique candidate dates
    n_cands    = len(perm_dates)
    date_coefs: list[float] = []
    for idx in rng.choice(n_cands, size=min(N_PERM, n_cands), replace=False):
        r = did_coef(df, treated, controls, perm_dates[idx])
        if r:
            date_coefs.append(r.coef)

    treat_coefs: list[float] = []
    for _ in range(N_PERM):
        fake_treated = str(rng.choice(all_apps))
        fake_ctrls   = [a for a in all_apps if a != fake_treated]
        r = did_coef(df, fake_treated, fake_ctrls, EVENT)
        if r:
            treat_coefs.append(r.coef)

    p_date  = float(np.mean(np.abs(date_coefs)  >= abs(real))) if date_coefs  else float("nan")
    p_treat = float(np.mean(np.abs(treat_coefs) >= abs(real))) if treat_coefs else float("nan")
    label   = treated.split()[0]
    print(f"  [{metric}|{label}] Permutation — date: p={p_date:.3f}  "
          f"treatment: p={p_treat:.3f}  (real δ={real:+.3f})")
    return date_coefs, treat_coefs, real


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_raw_levels(apps: list[str], dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Raw Daily Levels — US, All Apps", fontsize=13, fontweight="bold", y=1.01)
    for ax, (df, metric) in zip(axes, [(dl, "US Downloads"), (dau, "US DAU")]):
        for app in apps:
            if app not in df.columns:
                continue
            ax.plot(df.index, df[app], color=app_color(app), lw=1.8, label=app_label(app))
        ax.axvline(EVENT, color=EVENT_COLOR, lw=1.2, ls="--", alpha=0.85, label="Feb 27")
        ax.set_title(metric)
        ax.set_ylabel(metric.split()[1])
        ax.legend(loc="upper left")
        style_ax(ax)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    fname = PLOTS_DIR / "fig1_raw_levels.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_pretrend(apps: list[str], dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    pool_cl  = [a for a in apps if a not in [APP_CL,  APP_GPT]]
    pool_gpt = [a for a in apps if a not in [APP_GPT, APP_CL]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Pre-Trend Check: log(Treated) − mean(log(Controls)), US",
                 fontsize=13, fontweight="bold", y=1.01)

    specs = [
        (APP_CL,  pool_cl,  "Claude vs Pool (excl. ChatGPT)"),
        (APP_GPT, pool_gpt, "ChatGPT vs Pool (excl. Claude)"),
    ]
    for row, (treated, pool, title_base) in enumerate(specs):
        for col, (df, metric) in enumerate([(dl, "Downloads"), (dau, "DAU")]):
            ax   = axes[row, col]
            ctrl = [a for a in pool if a in df.columns]
            if treated not in df.columns or not ctrl:
                ax.set_visible(False); continue

            log_treated = np.log(df[treated].replace(0, np.nan))
            log_ctrl    = np.log(df[ctrl].replace(0, np.nan)).mean(axis=1)
            diff        = (log_treated - log_ctrl).dropna()
            diff        = diff[diff.index >= WINDOW_DEFAULT]

            pre  = diff[diff.index <  EVENT]
            post = diff[diff.index >= EVENT]

            if len(pre) > 1:
                t_pre  = (pre.index - WINDOW_DEFAULT).days.astype(float)
                slope  = np.polyfit(t_pre, pre.values, 1)
                t_all  = (diff.index - WINDOW_DEFAULT).days.astype(float)
                trend  = np.polyval(slope, t_all)
                ax.plot(diff.index, trend, ls="--", color="#94A3B8", lw=1.2,
                        label=f"Pre-trend ({slope[0]:+.4f}/d)")

            ax.plot(pre.index,  pre.values,  color="#64748B", lw=1.5, label="Pre")
            ax.plot(post.index, post.values, color=app_color(treated), lw=1.8, label="Post")
            ax.axvline(EVENT, color=EVENT_COLOR, lw=1.2, ls="--", alpha=0.75)
            ax.set_title(f"{title_base} — {metric}")
            ax.set_ylabel("log(treated) − mean(log(controls))")
            ax.legend()
            style_ax(ax)
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    fname = PLOTS_DIR / "fig2_pretrend.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_leave_one_out(apps: list[str], dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    pool_full = [a for a in apps if a not in [APP_CL, APP_GPT]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Leave-One-Out Donor Robustness — Claude (FEB1, linear, HAC)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, (df, metric) in zip(axes, [(dl, "Downloads"), (dau, "DAU")]):
        r_base = run_did(df, APP_CL, pool_full)
        if r_base is None:
            ax.set_visible(False); continue

        ax.axhline(r_base.coef, color="#CC4E00", lw=1.5, ls="--", alpha=0.75,
                   label=f"Full pool δ = {r_base.coef:+.3f}")
        # 1.96 is asymptotic; qualitatively unchanged with t-critical
        ax.fill_between([-0.5, len(pool_full) - 0.5],
                        r_base.coef - 1.96 * r_base.se,
                        r_base.coef + 1.96 * r_base.se,
                        alpha=0.08, color="#CC4E00")

        for xi, app in enumerate(pool_full):
            loo = [a for a in pool_full if a != app]
            r   = run_did(df, APP_CL, loo)
            if r is None:
                continue
            ax.errorbar(xi, r.coef, yerr=1.96 * r.se,
                        fmt="o", color="#2563EB", ms=7, capsize=4, lw=1.5,
                        label=f"Excl. {app.split()[0]}")

        ax.set_xticks(range(len(pool_full)))
        ax.set_xticklabels([f"Excl.\n{a.split()[0]}" for a in pool_full])
        ax.set_title(metric)
        ax.set_ylabel("δ (log pts)")
        ax.axhline(0, color="#6B7280", lw=0.7, alpha=0.4)
        style_ax(ax)
        ax.legend()

    plt.tight_layout()
    fname = PLOTS_DIR / "fig3_loo_donors.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_placebos(placebo_store: dict) -> None:
    for treated_app in [APP_CL, APP_GPT]:
        label = treated_app.split()[0]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(f"Placebo Tests — {label} (US, app-FE)",
                     fontsize=13, fontweight="bold", y=1.01)

        for col, metric in enumerate(["Downloads", "DAU"]):
            store = placebo_store[treated_app][metric]
            time_res, time_real = store["time_res"], store["time_real"]
            unit_res, unit_real = store["unit_res"], store["unit_real"]

            ax = axes[0, col]
            if time_real is None:
                ax.set_visible(False)
            else:
                coefs = [r[1] for r in time_res]
                n     = len(coefs)
                rank  = sum(abs(c) >= abs(time_real.coef) for c in coefs)
                emp_p = (rank + 1) / (n + 1) if n > 0 else np.nan
                ax.hist(coefs, bins=15, color=HIST_COLOR, edgecolor=HIST_EDGE, lw=0.5, alpha=0.9)
                ax.axvline( time_real.coef, color=EVENT_COLOR, lw=2,
                            label=f"Real δ = {time_real.coef:+.3f}   emp-p = {emp_p:.3f}")
                ax.axvline(-time_real.coef, color=EVENT_COLOR, lw=1.2, ls="--", alpha=0.45)
                ax.set_title(f"{metric} — Time placebo (N={n})")
                ax.set_xlabel("δ (log pts)")
                ax.set_ylabel("Count")
                ax.legend()
                style_ax(ax)

            ax = axes[1, col]
            if unit_real is None:
                ax.set_visible(False)
            else:
                labels_b   = [r[0] for r in unit_res] + [label]
                vals       = [r[1] for r in unit_res] + [unit_real.coef]
                bar_colors = ["#CBD5E1"] * len(unit_res) + [app_color(treated_app)]
                ax.barh(labels_b, vals, color=bar_colors, edgecolor="white", height=0.55)
                ax.axvline(0, color="#6B7280", lw=0.8, alpha=0.5)
                ax.set_title(f"{metric} — Unit placebo")
                ax.set_xlabel("δ (log pts)")
                style_ax(ax)

        plt.tight_layout()
        fname = PLOTS_DIR / f"fig4_placebo_{label.lower()}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")


def plot_permutations(perm_store: dict) -> None:
    for treated_app in [APP_CL, APP_GPT]:
        label = treated_app.split()[0]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(f"Permutation Inference — {label} (N={N_PERM}, app-FE)",
                     fontsize=13, fontweight="bold", y=1.01)

        for col, metric in enumerate(["Downloads", "DAU"]):
            date_coefs, treat_coefs, real = perm_store[treated_app][metric]
            for row, (coefs, title) in enumerate([
                (date_coefs,  "Date permutation"),
                (treat_coefs, "Treatment permutation"),
            ]):
                ax = axes[row, col]
                if not coefs or np.isnan(float(real)):
                    ax.set_visible(False); continue
                p = float(np.mean(np.abs(coefs) >= abs(real)))
                ax.hist(coefs, bins=40, color=HIST_COLOR, edgecolor=HIST_EDGE, lw=0.5, alpha=0.9)
                ax.axvline( real, color=EVENT_COLOR, lw=2,
                            label=f"Real δ = {real:+.3f}   p = {p:.3f}")
                ax.axvline(-real, color=EVENT_COLOR, lw=1.2, ls="--", alpha=0.45)
                ax.set_title(f"{metric} — {title}")
                ax.set_xlabel("δ (log pts)")
                ax.set_ylabel("Count")
                ax.legend()
                style_ax(ax)

        plt.tight_layout()
        fname = PLOTS_DIR / f"fig5_permutation_{label.lower()}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")


def plot_event_study(apps: list[str], dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    pool_ex = [a for a in apps if a not in [APP_CL, APP_GPT]]

    for treated_app in [APP_CL, APP_GPT]:
        label = treated_app.split()[0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
        fig.suptitle(f"Event Study — {label} (US, app-FE)",
                     fontsize=13, fontweight="bold", y=1.01)

        for ax, (df, metric) in zip(axes, [(dl, "Downloads"), (dau, "DAU")]):
            coefs = run_event_study(df, treated_app, pool_ex)
            if coefs is None:
                ax.set_visible(False); continue

            ks   = sorted(coefs)
            vals = [coefs[k][0] for k in ks]
            # 1.96 is asymptotic; t-critical with small panels is ~2.0–2.1 (qualitatively unchanged)
            ci   = [1.96 * coefs[k][1] for k in ks]

            ax.axhline(0, color="#6B7280", lw=0.8, alpha=0.5)
            ax.axvline(-0.5, color=EVENT_COLOR, lw=1.2, ls="--", alpha=0.75,
                       label="Feb 27 (event)")
            sonnet_k = (SONNET_REL - EVENT).days
            if -ES_K <= sonnet_k <= ES_K:
                ax.axvline(sonnet_k - 0.5, color="#D97706", lw=1.0, ls=":", alpha=0.8,
                           label="Feb 17 (Sonnet rel.)")
            ax.fill_between(ks,
                            [v - e for v, e in zip(vals, ci)],
                            [v + e for v, e in zip(vals, ci)],
                            alpha=0.12, color=app_color(treated_app))
            ax.plot(ks, vals, "o-", color=app_color(treated_app), ms=5, lw=1.8)
            ax.set_title(metric)
            ax.set_xlabel("Days relative to Feb 27")
            ax.set_ylabel("δ_k (log pts)")
            ax.legend()
            style_ax(ax)

        plt.tight_layout()
        fname = PLOTS_DIR / f"fig6_event_study_{label.lower()}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")


def plot_trend_sensitivity(apps: list[str], dl: pd.DataFrame, dau: pd.DataFrame) -> None:
    pool_ex = [a for a in apps if a not in [APP_CL, APP_GPT]]

    for treated_app in [APP_CL, APP_GPT]:
        label = treated_app.split()[0]
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle(f"Trend Sensitivity — {label} DiD (US, app-FE)",
                     fontsize=13, fontweight="bold", y=1.01)

        for ax, (df, metric) in zip(axes, [(dl, "Downloads"), (dau, "DAU")]):
            x_ticks, x_labels, x = [], [], 0
            for wi, (wl, ws) in enumerate(WINDOWS):
                for ti, tt in enumerate(("linear", "quadratic")):
                    r = run_did(df, treated_app, pool_ex, ws, trend_type=tt)
                    if r:
                        # 1.96 is asymptotic; qualitatively unchanged with t-critical
                        ax.errorbar(x, r.coef, yerr=1.96 * r.se, fmt="o",
                                    color=COLORS[tt], ms=6, capsize=3, lw=1.5,
                                    label=TREND_LABELS[tt] if wi == 0 else "")
                    x += 1
                x_ticks.append(x - 1.5)
                x_labels.append(wl)
                x += 0.8

            ax.axhline(0, color="#6B7280", lw=0.8, alpha=0.5)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_title(metric)
            ax.set_ylabel("δ (log pts)")
            style_ax(ax)
            if metric == "Downloads":
                handles, lbls = ax.get_legend_handles_labels()
                ax.legend(handles[:2], lbls[:2], title="Trend", title_fontsize=9)

        plt.tight_layout()
        fname = PLOTS_DIR / f"fig7_trend_sensitivity_{label.lower()}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_full_analysis() -> None:
    print("─" * 80)
    print(f"  App-FE DiD  |  {' · '.join(f'{k}={v}' for k, v in VERSIONS.items())}")
    print("─" * 80 + "\n")

    dl  = load(DATA_DIR / "downloads.csv", "Downloads", country="US")
    dau = load(DATA_DIR / "dau.csv",       "DAU",       country="US")

    apps     = sorted(dl.columns.tolist())
    controls = [a for a in apps if a not in [APP_CL, APP_GPT]]
    pool_ex  = controls
    print(f"Apps: {apps}\n")

    for treated in [APP_CL, APP_GPT]:
        print_results(treated, apps, dl, dau)
    print_window_sensitivity(apps, dl, dau)
    print_sensitivity_matrix(apps, dl, dau)
    print_leave_one_out(apps, dl, dau)
    print_geo_platform(apps)
    print_event_window(apps, dl, dau)
    print_confounder_checks(apps, dl, dau)

    print(f"{'─'*72}\n  Placebo Tests (app-FE)\n{'─'*72}")
    placebo_store: dict = {}
    for treated_app in [APP_CL, APP_GPT]:
        placebo_store[treated_app] = {}
        for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
            time_res, time_real = placebo_time(df, metric, treated_app, controls)
            print()
            unit_res, unit_real = placebo_unit(df, metric, treated_app, controls)
            placebo_store[treated_app][metric] = {
                "time_res": time_res, "time_real": time_real,
                "unit_res": unit_res, "unit_real": unit_real,
            }

    print(f"{'─'*72}\n  Permutation Inference (N={N_PERM}, app-FE)\n{'─'*72}")
    global rng
    rng = np.random.default_rng(RNG_SEED)   # reset so results are order-independent
    perm_store: dict = {}
    for treated_app in [APP_CL, APP_GPT]:
        perm_store[treated_app] = {}
        for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
            d_coefs, t_coefs, real_b = perm_inference(
                df, metric, treated_app, apps, controls)
            perm_store[treated_app][metric] = (d_coefs, t_coefs, real_b)
    print()

    plot_raw_levels(apps, dl, dau)
    plot_pretrend(apps, dl, dau)
    plot_leave_one_out(apps, dl, dau)
    plot_placebos(placebo_store)
    plot_permutations(perm_store)
    plot_event_study(apps, dl, dau)
    plot_trend_sensitivity(apps, dl, dau)

    all_results: dict[str, Any] = {"versions": VERSIONS, "main": {}, "sensitivity": {}}

    for treated in [APP_CL, APP_GPT]:
        t_key = treated.split()[0]
        all_results["main"][t_key] = {}
        for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
            r = run_did(df, treated, pool_ex)
            all_results["main"][t_key][metric] = r.to_dict() if r else None

    for spec in SENSITIVITY_GRID:
        ws  = _WINDOW_START[spec["window_label"]]
        key = f"{spec['window_label']}_{spec['trend']}_{spec['se']}"
        all_results["sensitivity"][key] = {}
        for treated in [APP_CL, APP_GPT]:
            t_key = treated.split()[0]
            all_results["sensitivity"][key][t_key] = {}
            for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
                r = run_did(df, treated, pool_ex,
                            window_start=ws,
                            trend_type=spec["trend"],
                            se_type=spec["se"])
                all_results["sensitivity"][key][t_key][metric] = r.to_dict() if r else None

    save_json(all_results, OUTPUT_DIR / "last_run.json")

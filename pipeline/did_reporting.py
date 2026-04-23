from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.estimation import (
    did_coef,
    fit_ols,
    formula,
    run_did,
    run_event_study,
)
from pipeline.panel import build_panel, validate_panel
from pipeline.types import DiDResult
from utils.config import (
    APP_CL,
    APP_GPT,
    EVENT,
    MIN_PRE_DAYS,
    N_PERM,
    PLACEBO_CUTOFF,
    SHOCK_START,
    SONNET_REL,
    WINDOW_DEFAULT,
    WINDOWS,
    ES_K,
)
from utils.data import load
from utils.plot_style import (
    EVENT_COLOR,
    HIST_COLOR,
    HIST_EDGE,
    app_color,
    app_label,
    style_ax,
)
from utils.reporting import VERSIONS, RNG_SEED, rng, output_dir, save_json
from utils.stats import pval_permutation, stars

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

DATA_DIR  = Path(__file__).resolve().parent.parent / "input_data"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots" / "did"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = output_dir("did")

_WINDOW_START = {lbl: ts for lbl, ts in WINDOWS}

SENSITIVITY_GRID: list[dict[str, Any]] = [
    {"window_label": wl, "trend": tt, "se": se, "pool": pool}
    for wl in ("FEB1", "JAN1")
    for tt in ("linear", "quadratic", "spline")
    for se in ("HC3", "HAC")
    for pool in ("log-sum", "avg-log")
]




def print_results(dl: pd.DataFrame, dau: pd.DataFrame, apps: list[str], treated: str) -> list:
    others  = [a for a in apps if a != treated]
    other   = APP_GPT if treated == APP_CL else APP_CL
    pool_ex = [a for a in others if a != other]

    specs = (
        [(a, [a]) for a in others]
        + [("Pool (all)",                      others)]
        + [(f"Pool (excl {other.split()[0]})", pool_ex)]
    )

    print(f"{'─'*80}\n  Treated: {treated}\n{'─'*80}")
    print(f"  {'Control':<36} {'δ_DL':>7} {'Δ%_DL':>8} {'p_DL':>7}   "
          f"{'δ_DAU':>7} {'Δ%_DAU':>8} {'p_DAU':>7}")

    rows = []
    for label, ctrls in specs:
        rd  = run_did(dl,  treated, ctrls)
        rda = run_did(dau, treated, ctrls)
        dl_s  = (f"{rd.coef:+.3f}  {rd.pct_change:+.1f}%  {rd.pval:.3f}{stars(rd.pval)}"
                 if rd  else "—")
        dau_s = (f"{rda.coef:+.3f}  {rda.pct_change:+.1f}%  {rda.pval:.3f}{stars(rda.pval)}"
                 if rda else "—")
        print(f"  {label:<36} {dl_s:<26} {dau_s}")
        rows.append({"label": label, "dl": rd, "dau": rda})
    print()
    return rows


def print_leave_one_out(dl: pd.DataFrame, dau: pd.DataFrame, apps: list[str], treated: str) -> list:
    other   = APP_GPT if treated == APP_CL else APP_CL
    pool_ex = [a for a in apps if a not in [treated, other]]

    print(f"{'─'*80}\n  Leave-One-Out Donors — {treated}\n{'─'*80}")
    print(f"  {'Pool':<40} {'δ_DL':>7} {'Δ%_DL':>8} {'p_DL':>7}   "
          f"{'δ_DAU':>7} {'Δ%_DAU':>8} {'p_DAU':>7}")

    specs = [("Full pool (excl. other main)", pool_ex)] + [
        (f"Excl. {a.split()[0]}", [x for x in pool_ex if x != a])
        for a in pool_ex
    ]

    rows = []
    for label, pool in specs:
        if not pool:
            continue
        rd  = run_did(dl,  treated, pool)
        rda = run_did(dau, treated, pool)
        dl_s  = (f"{rd.coef:+.3f}  {rd.pct_change:+.1f}%  {rd.pval:.3f}{stars(rd.pval)}"
                 if rd  else "—")
        dau_s = (f"{rda.coef:+.3f}  {rda.pct_change:+.1f}%  {rda.pval:.3f}{stars(rda.pval)}"
                 if rda else "—")
        print(f"  {label:<40} {dl_s:<26} {dau_s}")
        rows.append({"label": label, "dl": rd, "dau": rda})
    print()
    return rows


def print_window_sensitivity(dl: pd.DataFrame, dau: pd.DataFrame, apps: list[str]) -> dict:
    pool_ex_cl  = [a for a in apps if a not in [APP_CL,  APP_GPT]]
    pool_ex_gpt = [a for a in apps if a not in [APP_GPT, APP_CL]]

    TREND_TYPES  = ["linear", "quadratic", "spline"]
    TREND_LABELS = {
        "linear":    "Linear",
        "quadratic": "Quadratic",
        "spline":    "Nat. Spline",
    }

    print(f"{'─'*100}\n  Window × Trend Sensitivity\n{'─'*100}")
    results: dict[str, list] = {}

    for treated, pool in [(APP_CL, pool_ex_cl), (APP_GPT, pool_ex_gpt)]:
        print(f"\n  Treated: {treated}")
        print(f"  {'Window':<10} {'Pre-days':>9}  {'Trend':<13}"
              f"  {'δ_DL':>8} {'Δ%_DL':>9} {'p_DL':>8}"
              f"  {'δ_DAU':>8} {'Δ%_DAU':>9} {'p_DAU':>8}")
        print("  " + "─" * 98)

        results[treated] = []
        for win_label, ws in WINDOWS:
            pre_days = (EVENT - ws).days
            for tt in TREND_TYPES:
                rd  = run_did(dl,  treated, pool, window_start=ws, trend_type=tt)
                rda = run_did(dau, treated, pool, window_start=ws, trend_type=tt)
                dl_s  = (f"{rd.coef:+.4f}  {rd.pct_change:+.1f}%  {rd.pval:.3f}{stars(rd.pval)}"
                         if rd  else "—")
                dau_s = (f"{rda.coef:+.4f}  {rda.pct_change:+.1f}%  {rda.pval:.3f}{stars(rda.pval)}"
                         if rda else "—")
                print(f"  {win_label:<10} {pre_days:>9}  "
                      f"{TREND_LABELS[tt]:<13}  {dl_s:<30}  {dau_s}")
                results[treated].append({
                    "window": win_label, "trend": tt, "dl": rd, "dau": rda
                })
        print()
    return results


def print_sensitivity_matrix(dl: pd.DataFrame, dau: pd.DataFrame, apps: list[str]) -> list:
    pool_ex = [a for a in apps if a not in [APP_CL, APP_GPT]]
    TREND_LABELS = {
        "linear":    "Linear",
        "quadratic": "Quadratic",
        "spline":    "Nat. Spline",
    }

    print(f"{'─'*100}\n  Sensitivity Matrix: SE × Pool × Window × Trend\n{'─'*100}")
    print(f"  {'Treated':<10} {'Metric':<12} {'Window':<6} {'Trend':<13} "
          f"{'SEs':<5} {'Pool':<9} {'δ':>8} {'SE':>7} {'Δ%':>9} {'p':>8}")

    rows = []
    for treated in [APP_CL, APP_GPT]:
        for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
            for spec in SENSITIVITY_GRID:
                ws = _WINDOW_START[spec["window_label"]]
                r = run_did(
                    df, treated, pool_ex,
                    window_start=ws,
                    trend_type=spec["trend"],
                    se_type=spec["se"],
                    pool_method=spec["pool"],
                )
                if r:
                    print(
                        f"  {treated.split()[0]:<10} {metric:<12} "
                        f"{spec['window_label']:<6} {TREND_LABELS[spec['trend']]:<13} "
                        f"{spec['se']:<5} {spec['pool']:<9} "
                        f"{r.coef:+.3f}  {r.se:.3f}  {r.pct_change:+.1f}%  "
                        f"{r.pval:.3f}{stars(r.pval)}"
                    )
                    rows.append({
                        "treated": treated, "metric": metric,
                        "window": spec["window_label"], "trend": spec["trend"],
                        "se": spec["se"], "pool": spec["pool"],
                        "result": r,
                    })
        print()
    return rows


def print_geo_platform(dl: pd.DataFrame, dau: pd.DataFrame, apps: list[str]) -> list:
    pool_ex_cl  = [a for a in apps if a not in [APP_CL,  APP_GPT]]
    pool_ex_gpt = [a for a in apps if a not in [APP_GPT, APP_CL]]

    scopes    = [("US", "US"), ("Global", None), ("Global-ex-US", "ex-US")]
    platforms = [("Combined", None), ("iOS", "iOS"), ("Android", "Android")]

    geo_data: dict = {}
    for scope_lbl, country in scopes:
        for plat_lbl, platform in platforms:
            geo_data[(scope_lbl, plat_lbl)] = {
                "dl":  load(DATA_DIR / "downloads.csv", "Downloads",
                            country=country, platform=platform),
                "dau": load(DATA_DIR / "dau.csv", "DAU",
                            country=country, platform=platform),
            }

    print(f"{'─'*80}\n  Geo × Platform Heterogeneity (FEB1, linear)\n{'─'*80}")

    rows = []
    for treated, pool in [(APP_CL, pool_ex_cl), (APP_GPT, pool_ex_gpt)]:
        print(f"  Treated: {treated}")
        print(f"  {'Scope':<16} {'Platform':<10} {'Δ%_DL':>9} {'p_DL':>8}  "
              f"{'Δ%_DAU':>9} {'p_DAU':>8}")

        for scope_lbl, country in scopes:
            for plat_lbl, platform in platforms:
                data      = geo_data[(scope_lbl, plat_lbl)]
                df_dl     = data["dl"]
                df_dau    = data["dau"]
                apps_here = sorted(set(df_dl.columns) & set(df_dau.columns))
                pool_here = [a for a in pool if a in apps_here]

                if treated not in apps_here or not pool_here:
                    print(f"  {scope_lbl:<16} {plat_lbl:<10} {'—':>9} {'—':>8}  "
                          f"{'—':>9} {'—':>8}")
                    continue

                rd  = run_did(df_dl,  treated, pool_here, validate=False)
                rda = run_did(df_dau, treated, pool_here, validate=False)
                dl_s  = (f"{rd.pct_change:+.1f}%  {rd.pval:.3f}{stars(rd.pval)}"
                         if rd  else "—")
                dau_s = (f"{rda.pct_change:+.1f}%  {rda.pval:.3f}{stars(rda.pval)}"
                         if rda else "—")
                print(f"  {scope_lbl:<16} {plat_lbl:<10} {dl_s:<18}  {dau_s}")
                rows.append({
                    "treated": treated, "scope": scope_lbl, "platform": plat_lbl,
                    "dl": rd, "dau": rda,
                })
        print()
    return rows


def print_event_window(dl: pd.DataFrame, dau: pd.DataFrame, apps: list[str]) -> list:
    pool_ex = [a for a in apps if a not in [APP_CL, APP_GPT]]
    F = formula("linear", None)

    washin_drop = pd.date_range(SHOCK_START, EVENT - timedelta(days=1), freq="D").tolist()

    specs = [
        ("Point  (>= Feb 27)",
         lambda df: run_did(df, APP_CL, pool_ex)),
        ("Window (>= Feb 24)",
         lambda df: run_did(df, APP_CL, pool_ex, post_start=SHOCK_START)),
        ("Wash-in (excl Feb 24-26)",
         lambda df: run_did(df, APP_CL, pool_ex, drop_dates=washin_drop)),
    ]

    print(f"{'─'*80}\n  Event-Window Robustness (Claude, pool excl. ChatGPT)\n{'─'*80}")
    print(f"  {'Design':<30} {'δ_DL':>7} {'Δ%_DL':>8} {'p_DL':>8}   "
          f"{'δ_DAU':>7} {'Δ%_DAU':>8} {'p_DAU':>8}")

    rows = []
    for label, pfn in specs:
        row_data: dict[str, Any] = {"label": label}
        out = []
        for df in [dl, dau]:
            r = pfn(df)
            if r is None:
                out.append("—")
                row_data["dl" if df is dl else "dau"] = None
                continue
            out.append(
                f"{r.coef:+.3f}  {r.pct_change:+.1f}%  {r.pval:.3f}{stars(r.pval)}"
            )
            row_data["dl" if df is dl else "dau"] = r
        print(f"  {label:<30} {out[0]:<30} {out[1]}")
        rows.append(row_data)
    print()
    return rows


def print_confounder_checks(dl: pd.DataFrame, dau: pd.DataFrame, apps: list[str]) -> list:
    pool_ex = [a for a in apps if a not in [APP_CL, APP_GPT]]
    F = formula("linear", None)

    def sonnet_falsification(df):
        return build_panel(
            df, APP_CL, pool_ex, WINDOW_DEFAULT, SONNET_REL,
            date_end_exclusive=SHOCK_START,
        )

    def excl_sonnet_window(df):
        return build_panel(
            df, APP_CL, pool_ex, WINDOW_DEFAULT, EVENT,
            drop_date_ranges=[(SONNET_REL, SHOCK_START)],
        )

    specs = [
        ("Falsif.: event=Feb 17, pre-Feb 24", sonnet_falsification),
        ("Main excl. Feb 17-23",              excl_sonnet_window),
        ("Baseline (main spec)",
         lambda df: build_panel(df, APP_CL, pool_ex, WINDOW_DEFAULT, EVENT)),
    ]

    print(f"{'─'*80}\n  Confounder Checks (Claude, pool excl. ChatGPT)\n{'─'*80}")
    print(f"  {'Design':<38} {'δ_DL':>7} {'Δ%_DL':>8} {'p_DL':>8}   "
          f"{'δ_DAU':>7} {'Δ%_DAU':>8} {'p_DAU':>8}")

    rows = []
    for label, pfn in specs:
        out = []
        for df in [dl, dau]:
            p = pfn(df)
            if p is None or "DiD" not in p.columns or p["DiD"].nunique() < 2:
                out.append("—")
                continue
            validate_panel(p, APP_CL)
            res = fit_ols(p, F, "HAC")
            c   = float(res.params["DiD"])
            pv  = float(res.pvalues["DiD"])
            out.append(f"{c:+.3f}  {(np.exp(c)-1)*100:+.1f}%  {pv:.3f}{stars(pv)}")
            rows.append({"label": label, "metric": "dl" if df is dl else "dau",
                         "coef": c, "pval": pv})
        print(f"  {label:<38} {out[0]:<30} {out[1]}")
    print()
    return rows


def placebo_time(
    df: pd.DataFrame,
    metric: str,
    treated: str,
    pool: list[str],
) -> tuple:
    candidate_dates = pd.date_range(
        WINDOW_DEFAULT + timedelta(days=MIN_PRE_DAYS),
        PLACEBO_CUTOFF,
        freq="D",
    )
    results = []
    for fake in candidate_dates:
        r = did_coef(df, treated, pool, fake, validate=False)
        if r:
            results.append((fake, r.coef, r.pval))

    real = did_coef(df, treated, pool, EVENT, validate=False)
    label = treated.split()[0]
    if real is None:
        print(f"  [{metric}|{label}] Time placebo: skipped (no estimate)")
        return results, real
    coefs = [r[1] for r in results]
    emp_p = pval_permutation(real.coef, coefs)
    print(f"  [{metric}|{label}] Time placebo: {len(results)} fake dates  "
          f"real δ={real.coef:+.3f}  empirical p={emp_p:.3f} "
          f"({sum(abs(c) >= abs(real.coef) for c in coefs)}/{len(coefs)})")
    return results, real


def placebo_unit(
    df: pd.DataFrame,
    metric: str,
    treated: str,
    pool: list[str],
) -> tuple:
    results = []
    for app in pool:
        other_ctrls = [a for a in pool if a != app]
        r = did_coef(df, app, other_ctrls, EVENT, validate=False)
        if r:
            results.append((app.split()[0], r.coef, r.pval))
            print(f"  [{metric}|{treated.split()[0]}] Unit placebo — "
                  f"{app.split()[0]:<14} δ={r.coef:+.3f}  "
                  f"Δ%={r.pct_change:+.1f}%  p={r.pval:.3f}{stars(r.pval)}")

    real = did_coef(df, treated, pool, EVENT, validate=False)
    if real is None:
        print(f"  [{metric}|{treated.split()[0]}] Unit placebo: skipped\n")
        return results, real
    coefs = [r[1] for r in results]
    emp_p = pval_permutation(real.coef, coefs)
    print(f"  [{metric}|{treated.split()[0]}] {treated.split()[0]} "
          f"δ={real.coef:+.3f}  empirical p={emp_p:.3f}\n")
    return results, real


def perm_inference(
    df: pd.DataFrame,
    metric: str,
    treated: str,
    pool: list[str],
) -> tuple:
    real_val = did_coef(df, treated, pool, EVENT, validate=False)
    if real_val is None:
        return [], [], float("nan")
    real_coef = real_val.coef

    perm_dates = pd.date_range(
        WINDOW_DEFAULT + timedelta(days=MIN_PRE_DAYS),
        PLACEBO_CUTOFF,
        freq="D",
    ).tolist()

    all_apps = [treated] + pool

    n_dates = len(perm_dates)
    n_draws = min(N_PERM, n_dates)
    if n_draws < N_PERM:
        print(f"  NOTE: Date permutation capped at {n_dates} unique candidate "
              f"dates (N_PERM={N_PERM} exceeds available dates).")
    chosen_idx = rng.choice(n_dates, size=n_draws, replace=False)

    date_coefs = []
    for idx in chosen_idx:
        r = did_coef(df, treated, pool, perm_dates[idx], validate=False)
        if r:
            date_coefs.append(r.coef)

    treat_coefs = []
    candidate_treated = [treated] + list(pool)
    for _ in range(N_PERM):
        fake_treated = rng.choice(candidate_treated)
        fake_ctrls   = [a for a in all_apps if a != fake_treated]
        r = did_coef(df, fake_treated, fake_ctrls, EVENT, validate=False)
        if r:
            treat_coefs.append(r.coef)

    p_date  = pval_permutation(real_coef, date_coefs)
    p_treat = pval_permutation(real_coef, treat_coefs)
    label   = treated.split()[0]

    print(f"  [{metric}|{label}] Permutation — "
          f"date (N={n_draws}): p={p_date:.3f}  "
          f"treatment (N={N_PERM}): p={p_treat:.3f}  (real δ={real_coef:+.3f})")
    return date_coefs, treat_coefs, real_coef


def plot_raw_levels(dl: pd.DataFrame, dau: pd.DataFrame, apps: list[str]):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Raw Daily Levels — US, All Apps",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, (df, metric) in zip(axes, [(dl, "US Downloads"), (dau, "US DAU")]):
        for app in df.columns:
            ax.plot(df.index, df[app], color=app_color(app),
                    lw=1.8, label=app_label(app))
        ax.axvline(EVENT, color=EVENT_COLOR, lw=1.2, ls="--",
                   alpha=0.85, label="Feb 27")
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


def plot_pretrend(dl: pd.DataFrame, dau: pd.DataFrame, apps: list[str]):
    pool_cl  = [a for a in apps if a not in [APP_CL,  APP_GPT]]
    pool_gpt = [a for a in apps if a not in [APP_GPT, APP_CL]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Pre-Trend Check: log(Treated) − mean(log(Controls)), US",
        fontsize=13, fontweight="bold", y=1.01,
    )

    specs = [
        (APP_CL,  pool_cl,  "Claude vs Pool (excl. ChatGPT)"),
        (APP_GPT, pool_gpt, "ChatGPT vs Pool (excl. Claude)"),
    ]

    for row, (treated, pool, title_base) in enumerate(specs):
        for col, (df, metric) in enumerate([(dl, "Downloads"), (dau, "DAU")]):
            ax   = axes[row, col]
            ctrl = [a for a in pool if a in df.columns]
            if treated not in df.columns or not ctrl:
                ax.set_visible(False)
                continue

            log_treated = np.log(df[treated].replace(0, np.nan))
            log_ctrl    = np.log(df[ctrl].replace(0, np.nan)).mean(axis=1)
            diff        = (log_treated - log_ctrl).dropna()
            diff        = diff[diff.index >= WINDOW_DEFAULT]

            pre  = diff[diff.index <  EVENT]
            post = diff[diff.index >= EVENT]

            if len(pre) > 1:
                t_pre = (pre.index  - WINDOW_DEFAULT).days.astype(float)
                t_all = (diff.index - WINDOW_DEFAULT).days.astype(float)
                slope = np.polyfit(t_pre, pre.values, 1)
                trend = np.polyval(slope, t_all)
                ax.plot(diff.index, trend, ls="--", color="#94A3B8", lw=1.2,
                        label=f"Pre-trend ({slope[0]:+.4f}/d)")

            ax.plot(pre.index,  pre.values,  color="#64748B", lw=1.5, label="Pre")
            ax.plot(post.index, post.values,
                    color=app_color(treated), lw=1.8, label="Post")
            ax.axvline(EVENT, color=EVENT_COLOR, lw=1.2, ls="--", alpha=0.7)
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


def plot_placebos(placebo_store: dict):
    for treated_app in [APP_CL, APP_GPT]:
        label = treated_app.split()[0]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(
            f"Placebo Tests — {label} (US, excl. ChatGPT/Claude)",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for col, metric in enumerate(["Downloads", "DAU"]):
            store = placebo_store[treated_app][metric]
            time_res, time_real = store["time_res"], store["time_real"]
            unit_res, unit_real = store["unit_res"], store["unit_real"]

            if time_real is None:
                axes[0, col].set_visible(False)
                axes[1, col].set_visible(False)
                continue
            ax = axes[0, col]
            coefs = [r[1] for r in time_res]
            emp_p = pval_permutation(time_real.coef, coefs)
            ax.hist(coefs, bins=15, color=HIST_COLOR,
                    edgecolor=HIST_EDGE, lw=0.5, alpha=0.9)
            ax.axvline(time_real.coef, color=EVENT_COLOR, lw=2,
                       label=f"Real δ = {time_real.coef:+.3f}   emp-p = {emp_p:.3f}")
            ax.axvline(-time_real.coef, color=EVENT_COLOR,
                       lw=1.2, ls="--", alpha=0.45)
            ax.set_title(f"{metric} — Time placebo (N={len(coefs)})")
            ax.set_xlabel("δ (log pts)")
            ax.set_ylabel("Count")
            ax.legend()
            style_ax(ax)

            ax = axes[1, col]
            if unit_real is None:
                ax.set_visible(False)
                continue
            unit_coef = unit_real.coef
            bar_labels = [r[0] for r in unit_res] + [label]
            bar_vals   = [r[1] for r in unit_res] + [unit_coef]
            bar_colors = ["#CBD5E1"] * len(unit_res) + [app_color(treated_app)]
            ax.barh(bar_labels, bar_vals,
                    color=bar_colors, edgecolor="white", height=0.55)
            ax.axvline(0, color="#6B7280", lw=0.8, alpha=0.5)
            ax.set_title(f"{metric} — Unit placebo")
            ax.set_xlabel("δ (log pts)")
            style_ax(ax)

        plt.tight_layout()
        fname = PLOTS_DIR / f"fig3_placebo_{label.lower()}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")


def plot_permutations(perm_results: dict):
    for treated_app in [APP_CL, APP_GPT]:
        label = treated_app.split()[0]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(
            f"Permutation Inference — {label} (N={N_PERM})",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for col, metric in enumerate(["Downloads", "DAU"]):
            date_coefs, treat_coefs, real = perm_results[treated_app][metric]
            for row, (coefs, title) in enumerate([
                (date_coefs,  "Date permutation"),
                (treat_coefs, "Treatment permutation"),
            ]):
                ax = axes[row, col]
                p  = pval_permutation(real, coefs)
                ax.hist(coefs, bins=40, color=HIST_COLOR,
                        edgecolor=HIST_EDGE, lw=0.5, alpha=0.9)
                ax.axvline( real, color=EVENT_COLOR, lw=2,
                            label=f"Real δ = {real:+.3f}   p = {p:.3f}")
                ax.axvline(-real, color=EVENT_COLOR,
                           lw=1.2, ls="--", alpha=0.45)
                ax.set_title(f"{metric} — {title}")
                ax.set_xlabel("δ (log pts)")
                ax.set_ylabel("Count")
                ax.legend()
                style_ax(ax)

        plt.tight_layout()
        fname = PLOTS_DIR / f"fig4_permutation_{label.lower()}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")


def plot_event_study(dl: pd.DataFrame, dau: pd.DataFrame, apps: list[str]):
    pool_ex = [a for a in apps if a not in [APP_CL, APP_GPT]]

    for treated_app in [APP_CL, APP_GPT]:
        label = treated_app.split()[0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
        fig.suptitle(
            f"Event Study — {label} (US, pool excl. ChatGPT/Claude)",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for ax, (df, metric) in zip(axes, [(dl, "Downloads"), (dau, "DAU")]):
            coefs = run_event_study(df, treated_app, pool_ex)
            if coefs is None:
                ax.set_visible(False)
                continue

            ks   = sorted(coefs)
            vals = [coefs[k][0] for k in ks]
            # 1.96 is asymptotic; t-critical with small panels is ~2.0–2.1 (qualitatively unchanged)
            ci   = [1.96 * coefs[k][1] for k in ks]

            ax.axhline(0, color="#6B7280", lw=0.8, alpha=0.5)
            ax.axvline(-0.5, color=EVENT_COLOR, lw=1.2, ls="--",
                       alpha=0.75, label="Feb 27 (event)")

            sonnet_k = (SONNET_REL - EVENT).days
            if -ES_K <= sonnet_k <= ES_K:
                ax.axvline(sonnet_k - 0.5, color="#D97706", lw=1.0, ls=":",
                           alpha=0.8, label="Feb 17 (Sonnet rel.)")

            ax.fill_between(
                ks,
                [v - e for v, e in zip(vals, ci)],
                [v + e for v, e in zip(vals, ci)],
                alpha=0.12, color="#2563EB",
            )
            ax.plot(ks, vals, "o-", color="#2563EB", ms=5, lw=1.8)
            ax.set_title(metric)
            ax.set_xlabel("Days relative to Feb 27")
            ax.set_ylabel("δ_k (log pts)")
            ax.legend()
            style_ax(ax)

        plt.tight_layout()
        fname = PLOTS_DIR / f"fig5_event_study_{label.lower()}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")


def plot_trend_sensitivity(dl: pd.DataFrame, dau: pd.DataFrame, apps: list[str]):
    pool_ex      = [a for a in apps if a not in [APP_CL, APP_GPT]]
    TREND_TYPES  = ["linear", "quadratic", "spline"]
    TREND_LABELS = ["Linear", "Quadratic", "Nat. Spline"]
    WIN_LABELS   = [w[0] for w in WINDOWS]
    WIN_STARTS   = [w[1] for w in WINDOWS]
    COLORS       = ["#2563EB", "#16A34A", "#D97706"]

    for treated_app in [APP_CL, APP_GPT]:
        label = treated_app.split()[0]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Trend Sensitivity — {label} DiD (US, pool excl. ChatGPT/Claude)",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for ax, (df, metric) in zip(axes, [(dl, "Downloads"), (dau, "DAU")]):
            x_ticks, x_labels, x = [], [], 0
            for wi, (wl, ws) in enumerate(zip(WIN_LABELS, WIN_STARTS)):
                for ti, tt in enumerate(TREND_TYPES):
                    r = run_did(df, treated_app, pool_ex, window_start=ws, trend_type=tt)
                    if r:
                        # 1.96 is asymptotic; qualitatively unchanged with t-critical
                        ax.errorbar(
                            x, r.coef, yerr=1.96 * r.se,
                            fmt="o", color=COLORS[ti], ms=6,
                            capsize=3, lw=1.5,
                            label=TREND_LABELS[ti] if wi == 0 else "",
                        )
                    x += 1
                x_ticks.append(x - len(TREND_TYPES) / 2 - 0.5)
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
                ax.legend(handles[:3], lbls[:3], title="Trend", title_fontsize=9)

        plt.tight_layout()
        fname = PLOTS_DIR / f"fig6_trend_sensitivity_{label.lower()}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")


def run_full_analysis() -> dict[str, Any]:
    dl  = load(DATA_DIR / "downloads.csv", "Downloads", country="US")
    dau = load(DATA_DIR / "dau.csv",       "DAU",       country="US")
    apps = sorted(set(dl.columns.tolist()) | set(dau.columns.tolist()))
    print(f"Apps detected: {apps}\n")

    all_results: dict[str, Any] = {
        "versions": VERSIONS,
        "rng_seed": RNG_SEED,
        "sensitivity_grid": SENSITIVITY_GRID,
        "rows": [],
    }

    print_results(dl, dau, apps, APP_CL)
    print_results(dl, dau, apps, APP_GPT)

    print_leave_one_out(dl, dau, apps, APP_CL)
    print_leave_one_out(dl, dau, apps, APP_GPT)

    print_window_sensitivity(dl, dau, apps)
    sens_rows = print_sensitivity_matrix(dl, dau, apps)
    for item in sens_rows:
        r = item.get("result")
        if isinstance(r, DiDResult):
            all_results["rows"].append({**{k: v for k, v in item.items() if k != "result"}, "result": r.to_dict()})

    print_geo_platform(dl, dau, apps)
    print_event_window(dl, dau, apps)
    print_confounder_checks(dl, dau, apps)

    print(f"{'─'*80}\n  Placebo Tests\n{'─'*80}")
    placebo_store: dict = {}
    for treated_app in [APP_CL, APP_GPT]:
        pool_app = [a for a in apps if a not in [APP_CL, APP_GPT]]
        placebo_store[treated_app] = {}
        for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
            time_res, time_real = placebo_time(df, metric, treated_app, pool_app)
            print()
            unit_res, unit_real = placebo_unit(df, metric, treated_app, pool_app)
            placebo_store[treated_app][metric] = {
                "time_res":  time_res,
                "time_real": time_real,
                "unit_res":  unit_res,
                "unit_real": unit_real,
            }

    print(f"{'─'*80}\n  Permutation Inference (N={N_PERM})\n{'─'*80}")
    global rng
    rng = np.random.default_rng(RNG_SEED)   # reset so results are order-independent
    perm_results: dict = {}
    for treated_app in [APP_CL, APP_GPT]:
        pool_app = [a for a in apps if a not in [APP_CL, APP_GPT]]
        perm_results[treated_app] = {}
        for df, metric in [(dl, "Downloads"), (dau, "DAU")]:
            perm_results[treated_app][metric] = perm_inference(
                df, metric, treated_app, pool_app
            )
    print()

    plot_raw_levels(dl, dau, apps)
    plot_pretrend(dl, dau, apps)
    plot_placebos(placebo_store)
    plot_permutations(perm_results)
    plot_event_study(dl, dau, apps)
    plot_trend_sensitivity(dl, dau, apps)

    save_json(all_results, OUTPUT_DIR / "last_run.json")

    return all_results

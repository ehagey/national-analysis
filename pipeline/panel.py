from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from utils.config import MIN_PRE_DAYS


def build_panel(
    df: pd.DataFrame,
    treated: str,
    controls: list[str],
    window_start: pd.Timestamp,
    post_start: pd.Timestamp,
    drop_dates: Sequence[pd.Timestamp] | None = None,
    drop_date_ranges: Sequence[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    date_end_exclusive: pd.Timestamp | None = None,
    pool_method: Literal["avg-log", "log-sum"] = "avg-log",
) -> pd.DataFrame | None:
    ctrl = [a for a in controls if a in df.columns]
    if not ctrl or treated not in df.columns:
        return None

    win = df[df.index >= window_start]

    n_zeros_t = int((win[treated] == 0).sum())
    if n_zeros_t:
        print(f"  WARNING [build_panel] {treated}: {n_zeros_t} zero-value "
              f"day(s) replaced with NaN before log-transform.")

    t_df = pd.DataFrame({
        "log_Y":   np.log(win[treated].replace(0, np.nan)),
        "treated": 1,
    }).reset_index(names="date")

    if pool_method == "log-sum":
        ctrl_series = np.log(win[ctrl].replace(0, np.nan).sum(axis=1))
    else:
        n_zeros_c = int((win[ctrl] == 0).sum().sum())
        if n_zeros_c:
            print(f"  WARNING [build_panel] controls {ctrl}: {n_zeros_c} "
                  f"zero-value cell(s) replaced with NaN before log-transform.")
        ctrl_series = np.log(win[ctrl].replace(0, np.nan)).mean(axis=1)

    c_df = pd.DataFrame({
        "log_Y":   ctrl_series,
        "treated": 0,
    }).reset_index(names="date")

    panel = pd.concat([t_df, c_df]).dropna(subset=["log_Y"])

    if drop_dates:
        panel = panel[~panel["date"].isin(drop_dates)].copy()
    for lo, hi in drop_date_ranges or []:
        panel = panel[~((panel["date"] >= lo) & (panel["date"] < hi))].copy()
    if date_end_exclusive is not None:
        panel = panel[panel["date"] < date_end_exclusive].copy()

    panel = panel.sort_values(["date", "treated"]).reset_index(drop=True)
    panel["t"]    = (panel["date"] - window_start).dt.days.astype(float)
    panel["t2"]   = panel["t"] ** 2
    panel["Post"] = (panel["date"] >= post_start).astype(int)
    panel["DiD"]  = panel["treated"] * panel["Post"]
    panel["TxT"]  = panel["treated"] * panel["t"]
    panel["dow"]  = panel["date"].dt.dayofweek.astype(str)
    return panel


def validate_panel(panel: pd.DataFrame, treated: str) -> None:
    t1 = panel.loc[panel["treated"] == 1].sort_values("date")
    assert t1["date"].is_monotonic_increasing, f"{treated}: treated rows not sorted by date"
    dd = t1["date"].diff().dt.days.dropna()
    assert (dd >= 0).all(), f"{treated}: non-increasing dates in treated series"
    n_pre  = int((t1["Post"] == 0).sum())
    n_post = int((t1["Post"] == 1).sum())
    assert n_pre >= MIN_PRE_DAYS, f"{treated}: only {n_pre} pre-event days"
    assert n_post >= 3, f"{treated}: only {n_post} post-event days"
    zero_share = float(panel["log_Y"].isna().mean())
    assert zero_share < 0.05, f"{treated}: {zero_share:.0%} of log_Y is NaN — check zeros"

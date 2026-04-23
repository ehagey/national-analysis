from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix

from pipeline.panel import build_panel, validate_panel
from pipeline.types import DiDResult
from utils.config import EVENT, WINDOW_DEFAULT, ES_K


def hac_lags(n_obs: int) -> int:
    # Newey–West Andrews (1991) rule: floor(4 * (n/100)^(2/9))
    return max(1, int(np.floor(4 * (n_obs / 100) ** (2 / 9))))


def add_spline_cols(panel: pd.DataFrame, n_knots: int = 3) -> tuple[pd.DataFrame, list[str]]:
    lo, hi = float(panel["t"].min()), float(panel["t"].max())
    pre_t  = panel.loc[panel["Post"] == 0, "t"]
    knots  = np.quantile(pre_t, np.linspace(0.25, 0.75, n_knots))
    knot_str = ", ".join(str(k) for k in knots)
    basis = dmatrix(
        f"cr(t, knots=[{knot_str}], lower_bound={lo}, upper_bound={hi}) - 1",
        data=panel, return_type="dataframe",
    )
    post = panel.loc[panel["Post"] == 1, "t"]
    if len(post):
        assert float(post.max()) <= hi + 1e-9, "post-event t exceeds spline upper bound"
        assert float(post.min()) >= lo - 1e-9, "post-event t below spline lower bound"

    basis.columns = [f"sp{i+1}" for i in range(basis.shape[1])]
    for col in basis.columns:
        panel[col]        = basis[col].values
        panel[f"Tx{col}"] = panel["treated"] * panel[col]
    return panel, list(basis.columns)


def formula(trend_type: str, spline_cols: list[str] | None) -> str:
    if trend_type == "linear":
        return "log_Y ~ treated + t + TxT + C(dow) + DiD"
    if trend_type == "quadratic":
        return "log_Y ~ treated + t + t2 + TxT + C(dow) + DiD"
    if trend_type == "spline":
        sp_terms   = " + ".join(spline_cols)
        txsp_terms = " + ".join(f"Tx{c}" for c in spline_cols)
        return f"log_Y ~ treated + {sp_terms} + {txsp_terms} + C(dow) + DiD"
    raise ValueError(f"Unknown trend_type: {trend_type}")


def _fit(panel: pd.DataFrame, f: str, se_type: str, maxlags: int):
    if se_type == "HAC":
        return smf.ols(f, data=panel).fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags}
        )
    return smf.ols(f, data=panel).fit(cov_type="HC3")


def fit_ols(panel: pd.DataFrame, f: str, se_type: str = "HAC"):
    return _fit(panel, f, se_type, hac_lags(len(panel)))


def _did_result_from_fit(
    res,
    panel: pd.DataFrame,
    trend: str,
    pool: list[str],
    window_start: pd.Timestamp,
    post_start: pd.Timestamp,
    maxlags: int,
) -> DiDResult:
    coef = float(res.params["DiD"])
    se   = float(res.bse["DiD"])
    pv   = float(res.pvalues["DiD"])
    t1   = panel.loc[panel["treated"] == 1]
    n_pre  = int((t1["Post"] == 0).sum())
    n_post = int((t1["Post"] == 1).sum())
    return DiDResult(
        coef=coef,
        pct_change=(np.exp(coef) - 1) * 100,
        pval=pv,
        se=se,
        n_pre=n_pre,
        n_post=n_post,
        hac_lags=maxlags,
        trend=trend,
        pool=tuple(pool),
        window_start=window_start,
        post_start=post_start,
    )


def run_did(
    df: pd.DataFrame,
    treated: str,
    controls: list[str],
    window_start: pd.Timestamp | None = None,
    post_start: pd.Timestamp | None = None,
    drop_dates: list[pd.Timestamp] | None = None,
    drop_date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    date_end_exclusive: pd.Timestamp | None = None,
    trend_type: str = "linear",
    se_type: str = "HAC",
    pool_method: str = "avg-log",
    validate: bool = True,
) -> DiDResult | None:
    from utils.config import WINDOW_DEFAULT

    ws = window_start if window_start is not None else WINDOW_DEFAULT
    ps = post_start if post_start is not None else EVENT

    panel = build_panel(
        df, treated, controls, ws, ps,
        drop_dates=drop_dates,
        drop_date_ranges=drop_date_ranges,
        date_end_exclusive=date_end_exclusive,
        pool_method=pool_method,
    )
    if panel is None:
        return None
    if validate:
        validate_panel(panel, treated)

    n_obs = len(panel)
    maxlags = hac_lags(n_obs)

    spline_cols: list[str] | None = None
    if trend_type == "spline":
        panel, spline_cols = add_spline_cols(panel)

    f   = formula(trend_type, spline_cols)
    res = _fit(panel, f, se_type, maxlags)
    return _did_result_from_fit(
        res, panel, trend_type, controls, ws, ps, maxlags,
    )


def did_coef(
    df: pd.DataFrame,
    treated: str,
    controls: list[str],
    event: pd.Timestamp,
    window_start: pd.Timestamp | None = None,
    validate: bool = True,
) -> DiDResult | None:
    return run_did(
        df, treated, controls,
        window_start=window_start,
        post_start=event,
        trend_type="linear",
        se_type="HAC",
        validate=validate,
    )


def lag_col(k: int) -> str:
    return f"dm{abs(k)}" if k < 0 else f"d{k}"


def run_event_study(
    df: pd.DataFrame,
    treated: str,
    controls: list[str],
    window_start: pd.Timestamp | None = None,
    validate: bool = True,
) -> dict[int, tuple[float, float]] | None:
    from utils.config import WINDOW_DEFAULT

    ws = window_start if window_start is not None else WINDOW_DEFAULT
    panel = build_panel(df, treated, controls, ws, EVENT)
    if panel is None:
        return None
    if validate:
        validate_panel(panel, treated)

    maxlags = hac_lags(len(panel))

    panel["k"] = (panel["date"] - EVENT).dt.days.clip(-ES_K, ES_K).astype(int)
    ks = [k for k in range(-ES_K, ES_K + 1) if k != -1]

    reserved = set(panel.columns)
    for k in ks:
        name = lag_col(k)
        assert name not in reserved, (
            f"lag_col({k})='{name}' collides with existing column."
        )
        panel[name] = ((panel["treated"] == 1) & (panel["k"] == k)).astype(int)

    f = ("log_Y ~ treated + t + TxT + C(dow) + "
         + " + ".join(lag_col(k) for k in ks))
    res = smf.ols(f, data=panel).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )

    coefs: dict[int, tuple[float, float]] = {-1: (0.0, 0.0)}
    for k in ks:
        coefs[k] = (
            float(res.params.get(lag_col(k), np.nan)),
            float(res.bse.get(lag_col(k), np.nan)),
        )
    return coefs

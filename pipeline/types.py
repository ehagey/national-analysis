from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


def _nan_to_none(x: float) -> float | None:
    try:
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return x  # type: ignore[return-value]


@dataclass(frozen=True)
class DiDResult:
    coef: float
    pct_change: float
    pval: float
    se: float
    n_pre: int
    n_post: int
    hac_lags: int
    trend: str
    pool: tuple[str, ...]
    window_start: pd.Timestamp
    post_start: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "coef":         self.coef,
            "pct_change":   self.pct_change,
            "pval":         self.pval,
            "se":           self.se,
            "n_pre":        self.n_pre,
            "n_post":       self.n_post,
            "hac_lags":     self.hac_lags,
            "trend":        self.trend,
            "pool":         list(self.pool),
            "window_start": self.window_start.isoformat(),
            "post_start":   self.post_start.isoformat(),
        }


@dataclass(frozen=True)
class SpilloverResult:
    """Two-treatment spillover DiD: δ_Claude and δ_ChatGPT."""
    coef_cl:        float
    pct_change_cl:  float
    pval_cl:        float
    se_cl:          float
    coef_gpt:       float
    pct_change_gpt: float
    pval_gpt:       float
    se_gpt:         float
    model:          str            # "A" | "B"
    hac_lags:       int
    n_pre:          int
    n_post:         int
    window_start:   pd.Timestamp
    post_start:     pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "coef_cl":        self.coef_cl,
            "pct_change_cl":  self.pct_change_cl,
            "pval_cl":        self.pval_cl,
            "se_cl":          self.se_cl,
            "coef_gpt":       self.coef_gpt,
            "pct_change_gpt": self.pct_change_gpt,
            "pval_gpt":       self.pval_gpt,
            "se_gpt":         self.se_gpt,
            "model":          self.model,
            "hac_lags":       self.hac_lags,
            "n_pre":          self.n_pre,
            "n_post":         self.n_post,
            "window_start":   self.window_start.isoformat(),
            "post_start":     self.post_start.isoformat(),
        }


@dataclass(frozen=True)
class ITSResult:
    """Single-series ITS result for Claude's share of (Claude + ChatGPT)."""
    beta2:        float          # level shift at event
    se2:          float
    pval2:        float
    beta3:        float          # slope change post-event (NaN for parsim)
    pval3:        float          # (NaN for parsim)
    me:           float          # marginal effect in pp (=beta2 for linear)
    me_se:        float
    s_bar:        float          # pre-period mean share (%)
    outcome:      str            # "linear" | "logit"
    model:        str            # "parsim" | "full"
    hac_lags:     int
    window_start: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "beta2":        self.beta2,
            "se2":          self.se2,
            "pval2":        self.pval2,
            "beta3":        _nan_to_none(self.beta3),
            "pval3":        _nan_to_none(self.pval3),
            "me":           self.me,
            "me_se":        self.me_se,
            "s_bar":        self.s_bar,
            "outcome":      self.outcome,
            "model":        self.model,
            "hac_lags":     self.hac_lags,
            "window_start": self.window_start.isoformat(),
        }

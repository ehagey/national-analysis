"""Shared constants used across all DiD and ITS scripts."""

from datetime import timedelta
import pandas as pd

EVENT          = pd.Timestamp("2026-02-27")
SHOCK_START    = pd.Timestamp("2026-02-24")   # first day of the information window
SONNET_REL     = pd.Timestamp("2026-02-17")   # Claude Sonnet 3.7 release date
PLACEBO_CUTOFF = SHOCK_START - timedelta(days=4)   # last valid fake event date

WINDOWS = [
    ("FEB1", pd.Timestamp("2026-02-01")),
    ("JAN1", pd.Timestamp("2026-01-01")),
]
WINDOW_DEFAULT = WINDOWS[0][1]   # Feb 1 baseline window

APP_CL  = "Claude by Anthropic"
APP_GPT = "ChatGPT"

HAC_LAGS     = 5      # Newey-West bandwidth for main specifications
ES_K         = 10     # event-study window: ±ES_K days
MIN_PRE_DAYS = 5      # minimum pre-event days required for a valid placebo date
N_PERM       = 1_000  # permutation draws

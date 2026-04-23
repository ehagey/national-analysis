# AI Market Impact of the DoD Deal

Estimates the causal effect of the February 27, 2026 Department of Defense
AI contract announcement on Claude (Anthropic) and ChatGPT (OpenAI) using
Sensor Tower daily downloads and DAU data. Four complementary empirical
strategies are implemented.

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Data

Place two Sensor Tower UTF-16 TSV exports in `input_data/`:

```
input_data/
  downloads.csv   — daily downloads by app / country / platform
  dau.csv         — daily active users by app / country / platform
```

Required columns: `Unified Name`, `App Name`, `Date`, `Country / Region`,
`Platform`, and the metric column (`Downloads` or `DAU`).

The five apps expected (as they appear in the `Unified Name` column):

| Unified Name | Role |
|---|---|
| `Claude by Anthropic` | Treated |
| `ChatGPT` | Treated |
| `DeepSeek - Your AI Assistant` | Control |
| `Google Gemini` | Control |
| `Perplexity - AI Search & Chat` | Control |

---

## Usage

```bash
python did.py        # Pooled-control DiD
python did_app.py    # App fixed-effects DiD
python did_spill.py  # Spillover DiD (joint Claude + ChatGPT model)
python its.py        # Interrupted time series on Claude's market share
python explore_data.py  # Data diagnostics (run this first to verify inputs)
```

Results print to stdout. Figures are saved to `plots/<script>/`.

---

## Methods

| Script | Model | Estimand |
|---|---|---|
| `did.py` | Pooled-control DiD | ATT on log Downloads / DAU, Claude and ChatGPT |
| `did_app.py` | App fixed-effects DiD with app-specific trends | Same, no pooled control |
| `did_spill.py` | Joint two-unit and multi-app DiD (Models A & B) | Simultaneous ATT; tests for spillover |
| `its.py` | Interrupted time series on Claude's share of (Claude + ChatGPT) | Level shift β₂ in pp |

All specs include day-of-week fixed effects and HAC standard errors (Newey-West).
Robustness checks: estimation window, trend type, SE type, geo/platform slice,
event-window definition, and confounder falsification tests.

---

## Configuration

Key parameters in `utils/config.py`:

| Setting | Value | Description |
|---|---|---|
| `EVENT` | 2026-02-27 | DoD contract announcement date |
| `SHOCK_START` | 2026-02-24 | Start of information window |
| `SONNET_REL` | 2026-02-17 | Claude Sonnet 3.7 release (used in falsification) |
| `WINDOWS` | FEB1, JAN1 | Estimation windows |
| `N_PERM` | 1 000 | Permutation draws |
| `ES_K` | ±10 days | Event-study window |

---

## Project Structure

```
did.py               Pooled-control DiD entry point
did_app.py           App fixed-effects DiD entry point
did_spill.py         Spillover DiD entry point
its.py               Interrupted time series entry point
explore_data.py      Data diagnostic script
pipeline/            Estimation, panel building, and reporting modules
utils/
  config.py          Shared constants
  data.py            Sensor Tower CSV loader
  plot_style.py      Shared matplotlib style and app colors
  reporting.py       RNG seed, output directory, JSON serialization
  stats.py           Permutation p-value and significance stars
input_data/          Place input CSVs here (not tracked in git)
plots/               Output figures (generated on run, not tracked)
output/              JSON result summaries (generated on run, not tracked)
requirements.txt
```

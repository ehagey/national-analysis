"""Shared matplotlib style and color utilities."""

import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.frameon":    False,
    "lines.linewidth":   1.8,
    "patch.linewidth":   0.5,
})

HIST_COLOR  = "#CBD5E1"
HIST_EDGE   = "#94A3B8"
EVENT_COLOR = "#991B1B"

APP_COLORS = {
    "ChatGPT":             "#16A34A",
    "Claude by Anthropic": "#CC4E00",
    "DeepSeek":            "#94A3B8",   # matches "DeepSeek - Your AI Assistant" via startswith
    "Google Gemini":       "#2563EB",   # full Unified Name; "Gemini" alone would not match
    "Perplexity":          "#9333EA",   # matches "Perplexity - AI Search & Chat" via startswith
}


def app_color(name: str) -> str:
    for key, col in APP_COLORS.items():
        if name.startswith(key):
            return col
    return "#888888"


def app_label(name: str) -> str:
    return (name.replace("by Anthropic", "")
                .replace("- You", "- You...")
                .replace("- S", "- S...")
                .strip())


def style_ax(ax) -> None:
    """Remove top/right spines and apply a light horizontal grid."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6)

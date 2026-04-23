"""Shared statistical utilities."""

import numpy as np


def stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def pval_permutation(real_coef: float, null_coefs: list) -> float:
    """Two-sided permutation p-value with +1 correction (Phipson & Smyth 2010)."""
    n    = len(null_coefs)
    rank = sum(abs(c) >= abs(real_coef) for c in null_coefs)
    return (rank + 1) / (n + 1)

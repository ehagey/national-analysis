"""Data loading utilities for Sensor Tower exports."""

import pandas as pd


def load(path, metric: str, country=None, platform=None) -> pd.DataFrame:
    """
    Load a Sensor Tower export (UTF-16 TSV) and return a Date × App pivot.

    Parameters
    ----------
    path     : file path (str or Path)
    metric   : column name in the file (e.g. "Downloads", "DAU")
    country  : "US" | "ex-US" | None (all countries combined)
    platform : "iOS" | "Android" | None (both platforms combined)
    """
    df = pd.read_csv(path, encoding="utf-16", sep="\t")
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df[metric]  = pd.to_numeric(df[metric],  errors="coerce")

    PLATFORM_MAP = {"iOS": "App Store", "Android": "Google Play"}
    if platform is not None and "Platform" in df.columns:
        df = df[df["Platform"] == PLATFORM_MAP.get(platform, platform)]
    if country == "US":
        df = df[df["Country / Region"] == "US"]
    elif country == "ex-US":
        df = df[df["Country / Region"] != "US"]

    return (df.groupby(["Date", "Unified Name"])[metric]
              .sum()
              .unstack())

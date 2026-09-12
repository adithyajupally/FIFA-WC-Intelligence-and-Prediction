"""Simple goalscorer analytics helpers for goalscorers.csv.

This module is intentionally small because the original goalscorer notebook
was not available in the uploaded project files. Adapt column names only if
your actual goalscorers.csv uses different names.
"""

import pandas as pd


def load_goalscorers(path):
    return pd.read_csv(path)


def prepare_goalscorers(df):
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "year" not in df.columns and "date" in df.columns:
        df["year"] = df["date"].dt.year

    if "scorer" in df.columns:
        df["scorer"] = df["scorer"].astype("string").str.strip()

    return df


def top_scorers(df, n=20):
    if "scorer" not in df.columns:
        raise ValueError("Expected a 'scorer' column.")

    return (
        df["scorer"]
        .value_counts()
        .head(n)
        .rename_axis("scorer")
        .reset_index(name="goals")
    )

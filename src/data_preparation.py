"""Core data preparation functions for the FIFA international-results dataset."""

import pandas as pd
import numpy as np


def load_results(path):
    """Load the raw international match results."""
    return pd.read_csv(path, parse_dates=["date"])


def prepare_matches(df):
    """Clean results and create the foundational features used by later analysis."""
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])

    numeric_cols = ["home_score", "away_score"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    df["total_goals"] = df["home_score"] + df["away_score"]
    df["goal_difference"] = df["home_score"] - df["away_score"]
    df["goal_difference_abs"] = df["goal_difference"].abs()

    df["match_result"] = np.select(
        [
            df["home_score"] > df["away_score"],
            df["home_score"] < df["away_score"],
        ],
        [
            "Home Team Win",
            "Away Team Win",
        ],
        default="Draw",
    )

    df["home_win"] = (df["match_result"] == "Home Team Win").astype(int)

    df["year"] = df["date"].dt.year
    df["decade"] = (df["year"] // 10) * 10

    df["home_points"] = np.select(
        [
            df["match_result"] == "Home Team Win",
            df["match_result"] == "Draw",
        ],
        [3, 1],
        default=0,
    )

    df["away_points"] = np.select(
        [
            df["match_result"] == "Away Team Win",
            df["match_result"] == "Draw",
        ],
        [3, 1],
        default=0,
    )

    df["high_scoring_match"] = np.where(
        df["total_goals"] >= 4, "Yes", "No"
    )

    df["goal_margin_category"] = np.select(
        [
            df["goal_difference_abs"] == 0,
            df["goal_difference_abs"].between(1, 2),
            df["goal_difference_abs"].between(3, 4),
        ],
        [
            "Draw",
            "Close",
            "Moderate",
        ],
        default="Heavy",
    )

    if "neutral" in df.columns:
        df["is_neutral"] = (
            df["neutral"]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )
    elif "is_neutral" not in df.columns:
        df["is_neutral"] = 0

    return df


def save_prepared_matches(df, path):
    """Save the prepared match dataset."""
    df.to_csv(path, index=False)

"""Historical football analytics functions used by notebook 03."""

import pandas as pd


def goal_scoring_by_decade(df):
    return (
        df.groupby("decade")
        .agg(
            matches=("total_goals", "size"),
            avg_goals=("total_goals", "mean"),
        )
        .reset_index()
    )


def competitiveness_by_decade(df):
    return (
        df.groupby("decade")["goal_difference_abs"]
        .mean()
        .reset_index(name="avg_goal_difference")
    )


def result_distribution_by_decade(df):
    counts = (
        df.groupby(["decade", "match_result"])
        .size()
        .reset_index(name="matches")
    )
    return counts


def high_scoring_trend(df):
    return (
        df.groupby("decade")["high_scoring_match"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="high_scoring_pct")
    )


def tournament_summary(df, min_matches=20):
    counts = df["tournament"].value_counts()
    valid = counts[counts >= min_matches].index

    return (
        df[df["tournament"].isin(valid)]
        .groupby("tournament")
        .agg(
            matches=("total_goals", "size"),
            avg_goals=("total_goals", "mean"),
        )
        .reset_index()
        .sort_values("matches", ascending=False)
    )


def neutral_venue_trend(df):
    return (
        df.groupby("decade")["is_neutral"]
        .mean()
        .mul(100)
        .reset_index(name="neutral_pct")
    )

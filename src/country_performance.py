"""Country, home-advantage, neutral-venue and ranking calculations."""

import pandas as pd


def build_country_records(df):
    """Convert each match into home and away country perspectives."""
    home = pd.DataFrame({
        "country": df["home_team"],
        "goals_scored": df["home_score"],
        "goals_conceded": df["away_score"],
        "result": df["match_result"].map({
            "Home Team Win": "Win",
            "Draw": "Draw",
            "Away Team Win": "Loss",
        }),
    })

    away = pd.DataFrame({
        "country": df["away_team"],
        "goals_scored": df["away_score"],
        "goals_conceded": df["home_score"],
        "result": df["match_result"].map({
            "Home Team Win": "Loss",
            "Draw": "Draw",
            "Away Team Win": "Win",
        }),
    })

    return pd.concat([home, away], ignore_index=True)


def country_stats(df, min_matches=30):
    """Return overall country performance with a minimum-match filter."""
    records = build_country_records(df)

    stats = (
        records.groupby("country")
        .agg(
            matches_played=("country", "size"),
            wins=("result", lambda x: (x == "Win").sum()),
            draws=("result", lambda x: (x == "Draw").sum()),
            losses=("result", lambda x: (x == "Loss").sum()),
            goals_scored=("goals_scored", "sum"),
            goals_conceded=("goals_conceded", "sum"),
        )
        .reset_index()
    )

    stats = stats[stats["matches_played"] >= min_matches].copy()

    stats["goal_difference"] = (
        stats["goals_scored"] - stats["goals_conceded"]
    )
    stats["win_pct"] = (
        stats["wins"] / stats["matches_played"] * 100
    ).round(2)
    stats["goals_per_match"] = (
        stats["goals_scored"] / stats["matches_played"]
    ).round(2)
    stats["conceded_per_match"] = (
        stats["goals_conceded"] / stats["matches_played"]
    ).round(2)
    stats["gd_per_match"] = (
        stats["goal_difference"] / stats["matches_played"]
    ).round(2)

    return stats


def custom_ranking(country_stats_df, gd_weight=0.1):
    """Create the project's simple analytical ranking, not an official FIFA ranking."""
    result = country_stats_df.copy()
    result["ranking_score"] = (
        result["wins"] * 3
        + result["draws"]
        + result["goal_difference"] * gd_weight
    )
    return result.sort_values("ranking_score", ascending=False)


def outcome_percentages(df):
    pct = df["match_result"].value_counts(normalize=True).mul(100)

    return {
        "Home Win %": pct.get("Home Team Win", 0),
        "Draw %": pct.get("Draw", 0),
        "Away Win %": pct.get("Away Team Win", 0),
    }


def neutral_vs_non_neutral(df):
    neutral = df[df["is_neutral"] == 1]
    non_neutral = df[df["is_neutral"] == 0]

    return {
        "neutral": outcome_percentages(neutral),
        "non_neutral": outcome_percentages(non_neutral),
    }


def home_performance(df, min_matches=30):
    data = df[df["is_neutral"] == 0].copy()

    result = (
        data.groupby("home_team")
        .agg(
            home_matches=("match_result", "size"),
            home_wins=("match_result", lambda x: (x == "Home Team Win").sum()),
            home_goals=("home_score", "sum"),
            home_conceded=("away_score", "sum"),
        )
        .reset_index()
        .rename(columns={"home_team": "country"})
    )

    result = result[result["home_matches"] >= min_matches].copy()
    result["home_win_pct"] = (
        result["home_wins"] / result["home_matches"] * 100
    ).round(2)
    result["home_gd_per_match"] = (
        (result["home_goals"] - result["home_conceded"])
        / result["home_matches"]
    ).round(2)

    return result


def away_performance(df, min_matches=30):
    data = df[df["is_neutral"] == 0].copy()

    result = (
        data.groupby("away_team")
        .agg(
            away_matches=("match_result", "size"),
            away_wins=("match_result", lambda x: (x == "Away Team Win").sum()),
            away_goals=("away_score", "sum"),
            away_conceded=("home_score", "sum"),
        )
        .reset_index()
        .rename(columns={"away_team": "country"})
    )

    result = result[result["away_matches"] >= min_matches].copy()
    result["away_win_pct"] = (
        result["away_wins"] / result["away_matches"] * 100
    ).round(2)
    result["away_gd_per_match"] = (
        (result["away_goals"] - result["away_conceded"])
        / result["away_matches"]
    ).round(2)

    return result

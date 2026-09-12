"""Leakage-safe historical features and prediction helpers for the football ML model."""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


FEATURE_COLUMNS = [
    "home_team_win_rate",
    "away_team_win_rate",
    "home_team_avg_goals_scored",
    "away_team_avg_goals_scored",
    "home_team_avg_goals_conceded",
    "away_team_avg_goals_conceded",
    "home_team_last_5_win_rate",
    "away_team_last_5_win_rate",
    "home_team_last_5_avg_goals",
    "away_team_last_5_avg_goals",
    "head_to_head_home_win_rate",
    "is_neutral",
    "tournament",
]


def _win_rate(history):
    if not history:
        return 0.5
    return sum(x["win"] for x in history) / len(history)


def _average(history, key):
    if not history:
        return 0.0
    return float(np.mean([x[key] for x in history]))


def create_historical_features(df):
    """Create leakage-safe features for every historical match."""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "home_team", "away_team", "match_result"])
    data = data.sort_values("date").reset_index(drop=True)

    team_history = {}
    h2h_history = {}
    rows = []

    for _, match in data.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        home_hist = team_history.get(home, [])
        away_hist = team_history.get(away, [])
        home_last5 = home_hist[-5:]
        away_last5 = away_hist[-5:]

        pair = (home, away)
        reverse_pair = (away, home)
        previous = h2h_history.get(pair, [])
        reverse = h2h_history.get(reverse_pair, [])

        # Keep the same H2H definition used when the model is trained.
        h2h_results = list(previous)
        h2h_results += [
            {"home_win": 0 if item["home_win"] == 1 else 1}
            for item in reverse
        ]
        h2h_rate = (
            np.mean([item["home_win"] for item in h2h_results])
            if h2h_results else 0.5
        )

        rows.append({
            "home_team_win_rate": _win_rate(home_hist),
            "away_team_win_rate": _win_rate(away_hist),
            "home_team_avg_goals_scored": _average(home_hist, "goals_scored"),
            "away_team_avg_goals_scored": _average(away_hist, "goals_scored"),
            "home_team_avg_goals_conceded": _average(home_hist, "goals_conceded"),
            "away_team_avg_goals_conceded": _average(away_hist, "goals_conceded"),
            "home_team_last_5_win_rate": _win_rate(home_last5),
            "away_team_last_5_win_rate": _win_rate(away_last5),
            "home_team_last_5_avg_goals": _average(home_last5, "goals_scored"),
            "away_team_last_5_avg_goals": _average(away_last5, "goals_scored"),
            "head_to_head_home_win_rate": h2h_rate,
            "is_neutral": match["is_neutral"],
            "tournament": match["tournament"],
        })

        if match["match_result"] == "Home Team Win":
            home_win, away_win = 1, 0
        elif match["match_result"] == "Away Team Win":
            home_win, away_win = 0, 1
        else:
            home_win, away_win = 0, 0

        team_history.setdefault(home, []).append({
            "win": home_win,
            "goals_scored": match["home_score"],
            "goals_conceded": match["away_score"],
        })
        team_history.setdefault(away, []).append({
            "win": away_win,
            "goals_scored": match["away_score"],
            "goals_conceded": match["home_score"],
        })

        h2h_history.setdefault(pair, []).append({
            "home_win": 1 if match["match_result"] == "Home Team Win" else 0
        })

    features = pd.DataFrame(rows)

    return pd.concat(
        [
            data[["date", "home_team", "away_team", "match_result"]].reset_index(drop=True),
            features,
        ],
        axis=1,
    )


def _build_current_prediction_features(df, home_team, away_team, neutral, tournament):
    """Build the exact feature row needed for a future match prediction.

    Only matches already present in df are used as historical information.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "home_team", "away_team", "match_result"])
    data = data.sort_values("date")

    team_history = {}
    h2h_history = {}

    for _, match in data.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        pair = (home, away)

        if match["match_result"] == "Home Team Win":
            home_win, away_win = 1, 0
        elif match["match_result"] == "Away Team Win":
            home_win, away_win = 0, 1
        else:
            home_win, away_win = 0, 0

        team_history.setdefault(home, []).append({
            "win": home_win,
            "goals_scored": match["home_score"],
            "goals_conceded": match["away_score"],
        })
        team_history.setdefault(away, []).append({
            "win": away_win,
            "goals_scored": match["away_score"],
            "goals_conceded": match["home_score"],
        })
        h2h_history.setdefault(pair, []).append({
            "home_win": 1 if match["match_result"] == "Home Team Win" else 0
        })

    home_hist = team_history.get(home_team, [])
    away_hist = team_history.get(away_team, [])
    previous = h2h_history.get((home_team, away_team), [])
    reverse = h2h_history.get((away_team, home_team), [])

    h2h_results = list(previous)
    h2h_results += [
        {"home_win": 0 if item["home_win"] == 1 else 1}
        for item in reverse
    ]
    h2h_rate = (
        np.mean([item["home_win"] for item in h2h_results])
        if h2h_results else 0.5
    )

    return pd.DataFrame([{
        "home_team_win_rate": _win_rate(home_hist),
        "away_team_win_rate": _win_rate(away_hist),
        "home_team_avg_goals_scored": _average(home_hist, "goals_scored"),
        "away_team_avg_goals_scored": _average(away_hist, "goals_scored"),
        "home_team_avg_goals_conceded": _average(home_hist, "goals_conceded"),
        "away_team_avg_goals_conceded": _average(away_hist, "goals_conceded"),
        "home_team_last_5_win_rate": _win_rate(home_hist[-5:]),
        "away_team_last_5_win_rate": _win_rate(away_hist[-5:]),
        "home_team_last_5_avg_goals": _average(home_hist[-5:], "goals_scored"),
        "away_team_last_5_avg_goals": _average(away_hist[-5:], "goals_scored"),
        "head_to_head_home_win_rate": h2h_rate,
        "is_neutral": int(bool(neutral)),
        "tournament": tournament,
    }], columns=FEATURE_COLUMNS)


def predict_match(model, history_df, home_team, away_team, neutral=False, tournament="Friendly"):
    """Predict a future match from two team names and historical match data.

    Returns the predicted result and class probabilities.
    """
    if not home_team or not away_team:
        raise ValueError("home_team and away_team are required")
    if home_team == away_team:
        raise ValueError("home_team and away_team must be different")

    features = _build_current_prediction_features(
        history_df,
        home_team,
        away_team,
        neutral,
        tournament,
    )

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    probability_dict = {
        str(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_result": str(prediction),
        "probabilities": probability_dict,
    }


def chronological_split(model_df, train_fraction=0.80):
    """Split older matches into training and newer matches into testing."""
    split = int(len(model_df) * train_fraction)
    X = model_df[FEATURE_COLUMNS]
    y = model_df["match_result"]
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def build_models():
    """Build the three simple ML models plus the baseline."""
    categorical = ["tournament"]
    numeric = [c for c in FEATURE_COLUMNS if c not in categorical]

    def pipeline(model):
        preprocessor = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ])
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    return {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": pipeline(LogisticRegression(max_iter=1000)),
        "Decision Tree": pipeline(DecisionTreeClassifier(max_depth=6, random_state=42)),
        "Random Forest": pipeline(RandomForestClassifier(
            n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
        )),
    }


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Train and compare models using accuracy and macro F1."""
    results = []
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Macro F1": f1_score(y_test, prediction, average="macro"),
        })
        trained[name] = model

    results = pd.DataFrame(results).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    return results, trained


def save_model(model, path):
    """Save the complete preprocessing + model pipeline."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """Load a saved model pipeline."""
    return joblib.load(path)

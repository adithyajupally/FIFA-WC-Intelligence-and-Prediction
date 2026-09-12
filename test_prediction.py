"""
Local test for the Football Intelligence match prediction system.

Run this file from the ROOT of your Git repository:

    python test_prediction.py

Expected project structure:

    Football-Intelligence-Analytics/
    ├── data/
    │   └── matches_prepared.csv
    ├── models/
    │   └── match_prediction_model.pkl
    ├── src/
    │   └── match_prediction.py
    └── test_prediction.py   <-- this file

This script checks that:
1. The prepared data can be loaded.
2. The saved model can be loaded.
3. The application-facing predict_match() function works.
"""

from pathlib import Path
import sys
import pandas as pd
import joblib

# Make sure Python can find the src/ folder.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from match_prediction import predict_match


# ---------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------

DATA_PATH = ROOT / "data" / "matches_prepared.csv"
MODEL_PATH = ROOT / "models" / "match_prediction_model.pkl"


# ---------------------------------------------------------
# 2. Check required files
# ---------------------------------------------------------

print("=" * 60)
print("FOOTBALL INTELLIGENCE - LOCAL PREDICTION TEST")
print("=" * 60)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find prepared dataset:\n{DATA_PATH}\n\n"
        "Make sure matches_prepared.csv is inside data/."
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Could not find trained model:\n{MODEL_PATH}\n\n"
        "Make sure match_prediction_model.pkl is inside models/."
    )

print("\n[1/4] Required files found.")


# ---------------------------------------------------------
# 3. Load data and model
# ---------------------------------------------------------

history_df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

print("[2/4] Dataset and trained model loaded.")
print(f"      Historical matches: {len(history_df):,}")


# ---------------------------------------------------------
# 4. Make a sample prediction
# ---------------------------------------------------------

print("\n[3/4] Making sample prediction...")

result = predict_match(
    model=model,
    history_df=history_df,
    home_team="Brazil",
    away_team="Germany",
    neutral=False,
    tournament="Friendly",
)


# ---------------------------------------------------------
# 5. Display prediction
# ---------------------------------------------------------

print("\n[4/4] PREDICTION")
print("-" * 60)

print(f"Home Team:       {result['home_team']}")
print(f"Away Team:       {result['away_team']}")
print(f"Neutral Venue:   {result['neutral']}")
print(f"Tournament:      {result['tournament']}")
print()
print(f"Predicted Result: {result['predicted_result']}")
print()

print("Probabilities:")

for outcome, probability in result["probabilities"].items():
    print(f"  {outcome:<20} {probability:.2%}")

print("-" * 60)
print("SUCCESS: Prediction function works.")
print("=" * 60)

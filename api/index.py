from pathlib import Path
import sys

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Make the project root importable when this file is run directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.match_prediction import predict_match


app = FastAPI(
    title="Football Intelligence Prediction API",
    description="Predict international football match outcomes using the trained ML model.",
    version="1.0.0",
)


# Load these once when the API starts.
MODEL_PATH = ROOT / "models" / "match_prediction_model.pkl"
DATA_PATH = ROOT / "data" / "matches_prepared.csv"

try:
    model = joblib.load(MODEL_PATH)
    history_df = pd.read_csv(DATA_PATH)
except Exception as e:
    model = None
    history_df = None
    startup_error = str(e)


class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    neutral: bool = False
    tournament: str = "Friendly"


@app.get("/")
def root():
    return {
        "message": "Football Intelligence Prediction API",
        "status": "running",
        "endpoint": "/predict",
    }


@app.get("/health")
def health():
    if model is None or history_df is None:
        return {
            "status": "error",
            "message": startup_error,
        }

    return {
        "status": "ok",
        "model_loaded": True,
        "data_loaded": True,
    }


@app.post("/predict")
def predict(request: MatchRequest):
    if model is None or history_df is None:
        raise HTTPException(
            status_code=500,
            detail="Model or historical data could not be loaded.",
        )

    if request.home_team.strip() == "" or request.away_team.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Home team and away team are required.",
        )

    if request.home_team.strip().lower() == request.away_team.strip().lower():
        raise HTTPException(
            status_code=400,
            detail="Home team and away team must be different.",
        )

    try:
        result = predict_match(
            model=model,
            history_df=history_df,
            home_team=request.home_team.strip(),
            away_team=request.away_team.strip(),
            neutral=request.neutral,
            tournament=request.tournament.strip(),
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

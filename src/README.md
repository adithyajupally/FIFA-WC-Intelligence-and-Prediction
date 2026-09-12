# src

This folder contains the reusable Python code behind the Football Intelligence Analytics System.

## Modules

- `data_preparation.py` — cleans the international match data and creates the Phase 2 features.
- `historical_analytics.py` — reusable calculations from Phase 3.
- `country_performance.py` — country performance, home advantage, neutral venue and ranking calculations from Phase 4.
- `goalscorer_analytics.py` — small helpers for the goalscorer data.
- `match_prediction.py` — leakage-safe historical features, chronological split, model training/evaluation and model saving for Phase 6.

## Main prediction flow

```text
matches_prepared.csv
        ↓
create_historical_features()
        ↓
chronological_split()
        ↓
build_models()
        ↓
evaluate_models()
        ↓
save_model()
        ↓
prediction API
        ↓
Vercel frontend
```

## Suggested project structure

```text
football-intelligence/
├── data/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── historical_analytics.py
│   ├── country_performance.py
│   ├── goalscorer_analytics.py
│   ├── match_prediction.py
│   └── README.md
├── models/
├── app/
└── requirements.txt
```

The notebooks remain useful as the project's analysis/storytelling layer. The `src` code is the reusable implementation layer.

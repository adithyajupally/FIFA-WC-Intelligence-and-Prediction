"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const teams = [
  "Argentina", "Australia", "Belgium", "Brazil", "Canada", "Croatia",
  "England", "France", "Germany", "Italy", "Japan", "Mexico",
  "Netherlands", "Portugal", "Spain", "United States", "Uruguay"
];

const tournaments = [
  "Friendly", "World Cup", "UEFA Euro", "Copa America",
  "African Cup of Nations", "AFC Asian Cup"
];

export default function Home() {
  const [homeTeam, setHomeTeam] = useState("Brazil");
  const [awayTeam, setAwayTeam] = useState("Germany");
  const [neutral, setNeutral] = useState(false);
  const [tournament, setTournament] = useState("Friendly");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function predict() {
    setError("");
    setResult(null);

    if (homeTeam === awayTeam) {
      setError("Please select two different teams.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          home_team: homeTeam,
          away_team: awayTeam,
          neutral,
          tournament
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setResult(data);
    } catch (err) {
      setError(
        `${err.message} Make sure the Python API is running and NEXT_PUBLIC_API_URL is correct.`
      );
    } finally {
      setLoading(false);
    }
  }

  const probabilities = result?.probabilities || {};
  const homeProb = Math.round((probabilities["Home Team Win"] || 0) * 100);
  const drawProb = Math.round((probabilities["Draw"] || 0) * 100);
  const awayProb = Math.round((probabilities["Away Team Win"] || 0) * 100);

  return (
    <main className="page">
      <section className="hero">
        <div className="eyebrow">FOOTBALL INTELLIGENCE ANALYTICS</div>
        <h1>Match Predictor</h1>
        <p>
          Enter two international teams and let the historical ML model estimate
          the match outcome.
        </p>
      </section>

      <section className="card">
        <div className="match-grid">
          <div>
            <label>Home Team</label>
            <select value={homeTeam} onChange={(e) => setHomeTeam(e.target.value)}>
              {teams.map((team) => <option key={team}>{team}</option>)}
            </select>
          </div>

          <div className="vs">VS</div>

          <div>
            <label>Away Team</label>
            <select value={awayTeam} onChange={(e) => setAwayTeam(e.target.value)}>
              {teams.map((team) => <option key={team}>{team}</option>)}
            </select>
          </div>
        </div>

        <div className="options">
          <div>
            <label>Tournament</label>
            <select value={tournament} onChange={(e) => setTournament(e.target.value)}>
              {tournaments.map((item) => <option key={item}>{item}</option>)}
            </select>
          </div>

          <label className="checkbox">
            <input
              type="checkbox"
              checked={neutral}
              onChange={(e) => setNeutral(e.target.checked)}
            />
            Neutral venue
          </label>
        </div>

        <button onClick={predict} disabled={loading}>
          {loading ? "Predicting..." : "Predict Match"}
        </button>

        {error && <div className="error">{error}</div>}
      </section>

      {result && (
        <section className="result card">
          <div className="result-heading">
            <span>MODEL PREDICTION</span>
            <h2>{result.predicted_result}</h2>
            <p>{homeTeam} vs {awayTeam}</p>
          </div>

          <div className="probabilities">
            <Probability label={`${homeTeam} Win`} value={homeProb} />
            <Probability label="Draw" value={drawProb} />
            <Probability label={`${awayTeam} Win`} value={awayProb} />
          </div>

          <p className="note">
            Probabilities are statistical estimates based on historical data,
            not guaranteed outcomes.
          </p>
        </section>
      )}
    </main>
  );
}

function Probability({ label, value }) {
  return (
    <div className="probability">
      <div className="prob-top">
        <span>{label}</span>
        <strong>{value}%</strong>
      </div>
      <div className="bar">
        <div className="fill" style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

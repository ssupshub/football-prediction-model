import { useState, useEffect, useCallback } from 'react';
import './index.css';

const API_BASE = import.meta.env.VITE_API_BASE_URL
  ? import.meta.env.VITE_API_BASE_URL.replace(/\/$/, '')
  : '/api';

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  const [teams, setTeams]                 = useState([]);
  const [homeTeam, setHomeTeam]           = useState('');
  const [awayTeam, setAwayTeam]           = useState('');
  const [loading, setLoading]             = useState(false);
  const [fetchingTeams, setFetchingTeams] = useState(true);
  const [error, setError]                 = useState(null);
  const [prediction, setPrediction]       = useState(null);

  useEffect(() => {
    setFetchingTeams(true);
    fetch(`${API_BASE}/teams`)
      .then(r => { if (!r.ok) throw new Error(`Server error ${r.status}`); return r.json(); })
      .then(data => {
        const list = data.teams ?? [];
        setTeams(list);
        if (list.length >= 2) { setHomeTeam(list[0]); setAwayTeam(list[1]); }
      })
      .catch(() => setError('Could not connect to the backend. Make sure it is running on port 8000.'))
      .finally(() => setFetchingTeams(false));
  }, []);

  const handlePredict = useCallback(async (e) => {
    e.preventDefault();
    setError(null);
    setPrediction(null);
    if (homeTeam === awayTeam) { setError('Home and Away teams cannot be the same.'); return; }
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam }),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail ?? `Error ${res.status}`);
      }
      setPrediction(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [homeTeam, awayTeam]);

  return (
    <div className="app-container">
      <h1 className="title">AI Match Predictor</h1>
      <p className="subtitle">
        Trained on 15,960 matches &middot; 6 leagues &middot; 7 seasons &middot; 120 teams
      </p>

      <div className="glass-card">
        <form className="prediction-form" onSubmit={handlePredict}>
          <div className="team-selectors">
            <TeamSelect
              label="Home Team"
              value={homeTeam}
              teams={teams}
              loading={fetchingTeams}
              onChange={setHomeTeam}
            />
            <div className="vs-badge">VS</div>
            <TeamSelect
              label="Away Team"
              value={awayTeam}
              teams={teams}
              loading={fetchingTeams}
              onChange={setAwayTeam}
            />
          </div>

          {error && <div className="error-message" role="alert">{error}</div>}

          <button
            type="submit"
            className={`submit-btn ${loading ? 'loading' : ''}`}
            disabled={loading || fetchingTeams || teams.length === 0}
          >
            {loading ? '' : 'Predict Match Outcome'}
          </button>
        </form>

        {prediction && (
          <Results
            prediction={prediction}
            homeTeam={homeTeam}
            awayTeam={awayTeam}
          />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function TeamSelect({ label, value, teams, loading, onChange }) {
  return (
    <div className="input-group">
      <label>{label}</label>
      {loading
        ? <div className="select-placeholder">Loading teams&hellip;</div>
        : (
          <select value={value} onChange={e => onChange(e.target.value)} required>
            {teams.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        )
      }
    </div>
  );
}

function Results({ prediction, homeTeam, awayTeam }) {
  const colorMap = {
    'Home Win': 'var(--home-color)',
    'Away Win': 'var(--away-color)',
    'Draw':     'var(--draw-color)',
  };

  return (
    <div className="results-container">
      <div className="prediction-winner">
        Predicted Outcome:&nbsp;
        <span className="winner-highlight" style={{ color: colorMap[prediction.prediction] }}>
          {prediction.prediction}
        </span>
      </div>

      <div className="elo-row">
        <EloChip label={homeTeam} elo={prediction.home_elo} color="var(--home-color)" />
        <span className="elo-sep">ELO Rating</span>
        <EloChip label={awayTeam} elo={prediction.away_elo} color="var(--away-color)" />
      </div>

      <div className="bars-container">
        <ProgressBar
          label={`Home Win (${homeTeam})`}
          value={prediction.home_win_probability}
          type="home"
        />
        <ProgressBar
          label="Draw"
          value={prediction.draw_probability}
          type="draw"
        />
        <ProgressBar
          label={`Away Win (${awayTeam})`}
          value={prediction.away_win_probability}
          type="away"
        />
      </div>
    </div>
  );
}

function EloChip({ label, elo, color }) {
  return (
    <div className="elo-chip">
      <span className="elo-team" style={{ color }}>{label}</span>
      <span className="elo-value">{elo}</span>
    </div>
  );
}

function ProgressBar({ label, value, type }) {
  const pct = (Math.min(Math.max(value, 0), 1) * 100).toFixed(1);
  return (
    <div className={`bar-wrapper ${type}-bar`}>
      <div className="bar-labels">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div
        className="progress-bg"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin="0"
        aria-valuemax="100"
      >
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default App;

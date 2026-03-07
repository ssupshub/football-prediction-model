import { useState, useEffect, useCallback, useRef } from "react";
import "./index.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL
  ? import.meta.env.VITE_API_BASE_URL.replace(/\/$/, "")
  : "/api";

// ---------------------------------------------------------------------------
// Animated number counter
// ---------------------------------------------------------------------------
function AnimatedNumber({ value, decimals = 1, suffix = "" }) {
  const [display, setDisplay] = useState(0);
  const rafRef = useRef(null);

  useEffect(() => {
    const start = 0;
    const end = value;
    const duration = 900;
    const startTime = performance.now();

    const tick = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(start + (end - start) * eased);
      if (progress < 1) rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [value]);

  return <>{display.toFixed(decimals)}{suffix}</>;
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
export default function App() {
  const [teams, setTeams]                 = useState([]);
  const [homeTeam, setHomeTeam]           = useState("");
  const [awayTeam, setAwayTeam]           = useState("");
  const [loading, setLoading]             = useState(false);
  const [fetchingTeams, setFetchingTeams] = useState(true);
  const [error, setError]                 = useState(null);
  const [prediction, setPrediction]       = useState(null);
  const [revealed, setRevealed]           = useState(false);

  useEffect(() => {
    let cancelled = false;
    setFetchingTeams(true);
    fetch(`${API_BASE}/teams`)
      .then((r) => {
        if (!r.ok) throw new Error(`Server responded with ${r.status}`);
        return r.json();
      })
      .then((data) => {
        if (cancelled) return;
        const list = Array.isArray(data.teams) ? data.teams : [];
        setTeams(list);
        if (list.length >= 2) {
          setHomeTeam(list[0]);
          setAwayTeam(list[1]);
        }
      })
      .catch((err) => {
        if (cancelled) return;
        setError(`Backend unreachable: ${err.message}`);
      })
      .finally(() => { if (!cancelled) setFetchingTeams(false); });
    return () => { cancelled = true; };
  }, []);

  const handlePredict = useCallback(async () => {
    setError(null);
    setPrediction(null);
    setRevealed(false);

    if (homeTeam === awayTeam) {
      setError("Select two different teams.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ home_team: homeTeam, away_team: awayTeam }),
      });
      if (!res.ok) {
        let detail = `Error ${res.status}`;
        try { detail = (await res.json()).detail ?? detail; } catch {}
        throw new Error(detail);
      }
      const data = await res.json();
      if (typeof data.home_win_probability !== "number") throw new Error("Unexpected response format.");
      setPrediction(data);
      setTimeout(() => setRevealed(true), 50);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [homeTeam, awayTeam]);

  return (
    <div className="app">
      {/* Background grid */}
      <div className="bg-grid" aria-hidden="true" />

      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo-lockup">
            <span className="logo-icon" aria-hidden="true">⚽</span>
            <div>
              <div className="logo-text">MATCH ORACLE</div>
              <div className="logo-sub">AI · v4 · 200 teams · 10 leagues</div>
            </div>
          </div>
          <div className="header-badge">ML POWERED</div>
        </div>
      </header>

      {/* Hero strip */}
      <section className="hero">
        <div className="hero-inner">
          <h1 className="hero-title">
            <span className="hero-title-line">WHO</span>
            <span className="hero-title-line accent">WINS?</span>
          </h1>
          <p className="hero-desc">
            Trained on <strong>38,000 matches</strong> · 27 features · ELO + xG + form + H2H
          </p>
        </div>
      </section>

      {/* Main card */}
      <main className="main">
        <div className="card">

          {/* Team picker */}
          <div className="picker">
            <TeamColumn
              side="HOME"
              value={homeTeam}
              teams={teams}
              loading={fetchingTeams}
              onChange={setHomeTeam}
              accent="var(--home)"
            />

            <div className="versus-block" aria-hidden="true">
              <div className="versus-line" />
              <div className="versus-text">VS</div>
              <div className="versus-line" />
            </div>

            <TeamColumn
              side="AWAY"
              value={awayTeam}
              teams={teams}
              loading={fetchingTeams}
              onChange={setAwayTeam}
              accent="var(--away)"
            />
          </div>

          {/* Error */}
          {error && (
            <div className="error-strip" role="alert">
              <span className="error-icon">!</span>
              {error}
            </div>
          )}

          {/* CTA */}
          <button
            className={`cta-btn ${loading ? "cta-loading" : ""}`}
            disabled={loading || fetchingTeams || teams.length === 0}
            onClick={handlePredict}
            type="button"
          >
            {loading ? (
              <span className="cta-spinner" aria-label="Predicting…" />
            ) : (
              <>
                <span className="cta-label">PREDICT OUTCOME</span>
                <span className="cta-arrow" aria-hidden="true">→</span>
              </>
            )}
          </button>
        </div>

        {/* Results */}
        {prediction && (
          <ResultsPanel
            prediction={prediction}
            homeTeam={homeTeam}
            awayTeam={awayTeam}
            revealed={revealed}
          />
        )}

        {/* Stats strip */}
        <div className="stats-strip">
          {[
            { label: "MATCHES TRAINED", value: "38,000" },
            { label: "LEAGUES",         value: "10" },
            { label: "SEASONS",         value: "10" },
            { label: "FEATURES",        value: "27" },
          ].map(({ label, value }) => (
            <div className="stat-item" key={label}>
              <div className="stat-value">{value}</div>
              <div className="stat-label">{label}</div>
            </div>
          ))}
        </div>
      </main>

      <footer className="footer">
        Predictions are probabilistic · Not financial advice · For entertainment
      </footer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// TeamColumn
// ---------------------------------------------------------------------------
function TeamColumn({ side, value, teams, loading, onChange, accent }) {
  return (
    <div className="team-col">
      <div className="team-side-label" style={{ color: accent }}>{side}</div>
      {loading ? (
        <div className="team-skeleton">Loading…</div>
      ) : (
        <div className="select-wrap">
          <select
            value={value}
            onChange={(e) => onChange(e.target.value)}
            style={{ "--accent": accent }}
            className="team-select"
          >
            {teams.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
          <span className="select-chevron" aria-hidden="true" style={{ color: accent }}>▾</span>
        </div>
      )}
      {value && !loading && (
        <div className="team-name-display" style={{ "--accent": accent }}>
          {value}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ResultsPanel
// ---------------------------------------------------------------------------
function ResultsPanel({ prediction, homeTeam, awayTeam, revealed }) {
  const outcomeColor = {
    "Home Win": "var(--home)",
    "Away Win": "var(--away)",
    Draw:       "var(--draw)",
  };

  const bars = [
    { label: homeTeam,   value: prediction.home_win_probability, type: "home", id: "hw" },
    { label: "Draw",     value: prediction.draw_probability,     type: "draw", id: "dr" },
    { label: awayTeam,   value: prediction.away_win_probability, type: "away", id: "aw" },
  ];

  const winner = bars.reduce((a, b) => (b.value > a.value ? b : a));

  return (
    <div className={`results ${revealed ? "results--visible" : ""}`}>

      {/* Verdict banner */}
      <div className="verdict-banner">
        <div className="verdict-label">PREDICTED RESULT</div>
        <div
          className="verdict-outcome"
          style={{ color: outcomeColor[prediction.prediction] ?? "var(--text)" }}
        >
          {prediction.prediction.toUpperCase()}
        </div>
        <div className="verdict-team">
          {prediction.prediction === "Draw"
            ? "No clear favourite"
            : prediction.prediction === "Home Win"
              ? homeTeam
              : awayTeam}
        </div>
      </div>

      {/* ELO comparison */}
      <div className="elo-track">
        <EloBlock team={homeTeam} elo={prediction.home_elo} color="var(--home)" revealed={revealed} />
        <div className="elo-divider">
          <div className="elo-divider-label">ELO</div>
        </div>
        <EloBlock team={awayTeam} elo={prediction.away_elo} color="var(--away)" revealed={revealed} align="right" />
      </div>

      {/* Probability bars */}
      <div className="prob-grid">
        {bars.map((bar) => (
          <ProbBar
            key={bar.id}
            label={bar.label}
            value={bar.value}
            type={bar.type}
            isWinner={bar.id === winner.id}
            revealed={revealed}
          />
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// EloBlock
// ---------------------------------------------------------------------------
function EloBlock({ team, elo, color, revealed, align = "left" }) {
  return (
    <div className={`elo-block elo-block--${align}`}>
      <div className="elo-team-name" style={{ color }}>{team}</div>
      <div className="elo-score" style={{ color }}>
        {revealed ? <AnimatedNumber value={elo} decimals={0} /> : "—"}
      </div>
      <div className="elo-score-label">ELO RATING</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ProbBar
// ---------------------------------------------------------------------------
function ProbBar({ label, value, type, isWinner, revealed }) {
  const safe = typeof value === "number" && isFinite(value) ? value : 0;
  const pct  = (Math.min(Math.max(safe, 0), 1) * 100);

  return (
    <div className={`prob-row ${isWinner ? "prob-row--winner" : ""}`}>
      <div className="prob-meta">
        <span className="prob-label">{label}</span>
        <span className="prob-pct">
          {revealed ? <AnimatedNumber value={pct} decimals={1} suffix="%" /> : "—"}
        </span>
      </div>
      <div className="prob-track" role="progressbar" aria-valuenow={pct.toFixed(1)} aria-valuemin="0" aria-valuemax="100">
        <div
          className={`prob-fill prob-fill--${type}`}
          style={{ width: revealed ? `${pct}%` : "0%" }}
        />
        {isWinner && <div className="prob-winner-pip" />}
      </div>
    </div>
  );
}

// ─── HomePage ─────────────────────────────────────────────────────────────────
// The prediction page. Contains: TeamColumn, ResultsPanel, DetailedReport,
// EloBlock, ProbBar — all emoji-free.

import { useState, useCallback } from "react";
import AnimatedNumber from "../components/AnimatedNumber.jsx";

// ─── HomePage ─────────────────────────────────────────────────────────────────
export default function HomePage({ teams, fetchingTeams, goTo, apiBase }) {
  const [homeTeam, setHomeTeam]     = useState(teams[0] ?? "");
  const [awayTeam, setAwayTeam]     = useState(teams[1] ?? "");
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [revealed, setRevealed]     = useState(false);
  const [activeTab, setActiveTab]   = useState("prediction");

  // Sync selects when teams list arrives after mount
  useState(() => {
    if (!homeTeam && teams.length >= 1) setHomeTeam(teams[0]);
    if (!awayTeam && teams.length >= 2) setAwayTeam(teams[1]);
  });

  const handlePredict = useCallback(async () => {
    setError(null); setPrediction(null); setRevealed(false);
    if (homeTeam === awayTeam) { setError("Select two different teams."); return; }
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/predict`, {
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
      setActiveTab("prediction");
      setTimeout(() => setRevealed(true), 50);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [homeTeam, awayTeam, apiBase]);

  return (
    <>
      {/* Hero */}
      <section className="hero">
        <div className="hero-inner">
          <div className="hero-eyebrow">AI Football Prediction Engine</div>
          <h1 className="hero-title">
            <span className="hero-title-line">WHO</span>
            <span className="hero-title-line accent">WINS?</span>
          </h1>
          <p className="hero-desc">
            Trained on <strong>38,000 matches</strong> · 27 features · ELO + xG + form + H2H
          </p>
        </div>
      </section>

      <main className="main">

        {/* Predictor card */}
        <div className="card">
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

          {error && (
            <div className="error-strip" role="alert">
              <span className="error-icon">!</span>
              {error}
            </div>
          )}

          <button
            className={`cta-btn ${loading ? "cta-loading" : ""}`}
            disabled={loading || fetchingTeams || teams.length === 0}
            onClick={handlePredict}
            type="button"
          >
            {loading
              ? <span className="cta-spinner" aria-label="Predicting" />
              : <><span className="cta-label">PREDICT OUTCOME</span><span className="cta-arrow">→</span></>
            }
          </button>
        </div>

        {/* Result tabs — appear after prediction */}
        {prediction && (
          <>
            <div className="tab-bar">
              {[
                { id: "prediction", label: "Prediction" },
                { id: "report",     label: "Detailed Report" },
              ].map(t => (
                <button
                  key={t.id}
                  className={`tab-btn ${activeTab === t.id ? "tab-btn--active" : ""}`}
                  onClick={() => setActiveTab(t.id)}
                  type="button"
                >
                  {t.label}
                </button>
              ))}
            </div>

            {activeTab === "prediction" && (
              <ResultsPanel
                prediction={prediction}
                homeTeam={homeTeam}
                awayTeam={awayTeam}
                revealed={revealed}
              />
            )}

            {activeTab === "report" && (
              <DetailedReport
                prediction={prediction}
                homeTeam={homeTeam}
                awayTeam={awayTeam}
                revealed={revealed}
              />
            )}
          </>
        )}

        {/* Stats strip */}
        <div className="stats-strip">
          {[
            { label: "MATCHES TRAINED", value: "38,000" },
            { label: "LEAGUES",         value: "10"     },
            { label: "SEASONS",         value: "10"     },
            { label: "FEATURES",        value: "27"     },
          ].map(({ label, value }) => (
            <div className="stat-item" key={label}>
              <div className="stat-value">{value}</div>
              <div className="stat-label">{label}</div>
            </div>
          ))}
        </div>

        {/* Page-nav cards */}
        <div className="page-nav-cards">
          <button className="page-nav-card" onClick={() => goTo("leagues")} type="button">
            <div className="page-nav-card-body">
              <div className="page-nav-card-title">10 Leagues Covered</div>
              <div className="page-nav-card-sub">Premier League, La Liga, Bundesliga &amp; 7 more</div>
            </div>
            <div className="page-nav-card-arrow">→</div>
          </button>
          <button className="page-nav-card" onClick={() => goTo("howit")} type="button">
            <div className="page-nav-card-body">
              <div className="page-nav-card-title">How It Works</div>
              <div className="page-nav-card-sub">Data pipeline, 27 features, model training</div>
            </div>
            <div className="page-nav-card-arrow">→</div>
          </button>
        </div>

      </main>

      <footer className="footer">
        Predictions are probabilistic · Not financial advice · For entertainment
      </footer>
    </>
  );
}

// ─── TeamColumn ───────────────────────────────────────────────────────────────
function TeamColumn({ side, value, teams, loading, onChange, accent }) {
  return (
    <div className="team-col">
      <div className="team-side-label" style={{ color: accent }}>{side}</div>
      {loading ? (
        <div className="team-skeleton">Loading teams…</div>
      ) : (
        <div className="select-wrap">
          <select
            value={value}
            onChange={e => onChange(e.target.value)}
            style={{ "--accent": accent }}
            className="team-select"
          >
            {teams.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
          <span className="select-chevron" aria-hidden="true" style={{ color: accent }}>▾</span>
        </div>
      )}
      {value && !loading && (
        <div className="team-name-display" style={{ "--accent": accent }}>{value}</div>
      )}
    </div>
  );
}

// ─── ResultsPanel ─────────────────────────────────────────────────────────────
function ResultsPanel({ prediction, homeTeam, awayTeam, revealed }) {
  const outcomeColor = {
    "Home Win": "var(--home)",
    "Away Win": "var(--away)",
    Draw:       "var(--draw)",
  };
  const bars = [
    { label: homeTeam, value: prediction.home_win_probability, type: "home", id: "hw" },
    { label: "Draw",   value: prediction.draw_probability,     type: "draw", id: "dr" },
    { label: awayTeam, value: prediction.away_win_probability, type: "away", id: "aw" },
  ];
  const winner = bars.reduce((a, b) => b.value > a.value ? b : a);

  return (
    <div className={`results ${revealed ? "results--visible" : ""}`}>

      <div className="verdict-banner">
        <div className="verdict-label">PREDICTED RESULT</div>
        <div className="verdict-outcome" style={{ color: outcomeColor[prediction.prediction] ?? "var(--text)" }}>
          {prediction.prediction.toUpperCase()}
        </div>
        <div className="verdict-team">
          {prediction.prediction === "Draw"
            ? "No clear favourite"
            : prediction.prediction === "Home Win" ? homeTeam : awayTeam}
        </div>
      </div>

      <div className="elo-track">
        <EloBlock team={homeTeam} elo={prediction.home_elo} color="var(--home)" revealed={revealed} align="left" />
        <div className="elo-divider"><div className="elo-divider-label">ELO</div></div>
        <EloBlock team={awayTeam} elo={prediction.away_elo} color="var(--away)" revealed={revealed} align="right" />
      </div>

      <div className="prob-grid">
        {bars.map(bar => (
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

// ─── EloBlock ─────────────────────────────────────────────────────────────────
function EloBlock({ team, elo, color, revealed, align }) {
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

// ─── ProbBar ──────────────────────────────────────────────────────────────────
function ProbBar({ label, value, type, isWinner, revealed }) {
  const safe = typeof value === "number" && isFinite(value) ? value : 0;
  const pct  = Math.min(Math.max(safe, 0), 1) * 100;
  return (
    <div className={`prob-row ${isWinner ? "prob-row--winner" : ""}`}>
      <div className="prob-meta">
        <span className="prob-label">{label}</span>
        <span className="prob-pct">
          {revealed ? <AnimatedNumber value={pct} decimals={1} suffix="%" /> : "—"}
        </span>
      </div>
      <div
        className="prob-track"
        role="progressbar"
        aria-valuenow={pct.toFixed(1)}
        aria-valuemin="0"
        aria-valuemax="100"
      >
        <div className={`prob-fill prob-fill--${type}`} style={{ width: revealed ? `${pct}%` : "0%" }} />
        {isWinner && <div className="prob-winner-pip" />}
      </div>
    </div>
  );
}

// ─── DetailedReport ───────────────────────────────────────────────────────────
function DetailedReport({ prediction, homeTeam, awayTeam, revealed }) {
  const eloDiff  = prediction.home_elo - prediction.away_elo;
  const eloEdge  = Math.abs(eloDiff);
  const eloFavor = eloDiff > 0 ? homeTeam : awayTeam;
  const eloColor = eloDiff > 0 ? "var(--home)" : "var(--away)";

  const homeWin = (prediction.home_win_probability * 100).toFixed(1);
  const draw    = (prediction.draw_probability * 100).toFixed(1);
  const awayWin = (prediction.away_win_probability * 100).toFixed(1);

  const confidence      = Math.max(prediction.home_win_probability, prediction.draw_probability, prediction.away_win_probability);
  const confidenceLabel = confidence > 0.6 ? "HIGH" : confidence > 0.45 ? "MEDIUM" : "LOW";
  const confidenceColor = confidence > 0.6 ? "var(--home)" : confidence > 0.45 ? "#f0c040" : "var(--away)";

  const toOdds = p => p > 0 ? (1 / p).toFixed(2) : "—";

  const verdict     = prediction.prediction;
  const summaryText = verdict === "Draw"
    ? `This looks like a balanced contest. Neither side has a decisive edge — draw probability at ${draw}% reflects closely matched form and ELO.`
    : `${verdict === "Home Win" ? homeTeam : awayTeam} are favoured with ${verdict === "Home Win" ? homeWin : awayWin}% win probability. ${
        eloEdge > 50
          ? `An ELO advantage of ${eloEdge.toFixed(0)} points suggests a meaningful quality gap.`
          : `The ELO gap is narrow (${eloEdge.toFixed(0)} pts), keeping this fixture competitive.`
      }`;

  return (
    <div className={`results results--report ${revealed ? "results--visible" : ""}`}>

      <div className="report-section report-header-section">
        <div className="report-section-title">MATCH SUMMARY</div>
        <p className="report-summary-text">{summaryText}</p>
        <div className="report-confidence">
          <span className="report-confidence-label">MODEL CONFIDENCE</span>
          <span className="report-confidence-value" style={{ color: confidenceColor }}>{confidenceLabel}</span>
          <span className="report-confidence-pct"   style={{ color: confidenceColor }}>{(confidence * 100).toFixed(1)}%</span>
        </div>
      </div>

      <div className="report-section">
        <div className="report-section-title">HEAD-TO-HEAD SNAPSHOT</div>
        <div className="h2h-grid">
          {[
            { label: homeTeam, val: `${homeWin}%`, sub: "Win probability",  color: "var(--home)" },
            { label: "DRAW",   val: `${draw}%`,    sub: "Draw probability", color: "var(--draw)" },
            { label: awayTeam, val: `${awayWin}%`, sub: "Win probability",  color: "var(--away)" },
          ].map(c => (
            <div className="h2h-cell" key={c.label}>
              <div className="h2h-label" style={{ color: c.color }}>{c.label}</div>
              <div className="h2h-big"   style={{ color: c.color }}>{c.val}</div>
              <div className="h2h-sub">{c.sub}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="report-section">
        <div className="report-section-title">ELO STRENGTH ANALYSIS</div>
        <div className="elo-analysis">
          {[
            { team: homeTeam, elo: prediction.home_elo, color: "var(--home)" },
            { team: awayTeam, elo: prediction.away_elo, color: "var(--away)" },
          ].map(t => (
            <div className="elo-analysis-row" key={t.team}>
              <span className="elo-analysis-label" style={{ color: t.color }}>{t.team}</span>
              <div className="elo-analysis-bar-wrap">
                <div
                  className="elo-analysis-bar"
                  style={{ width: `${Math.min(100, (t.elo / 2000) * 100)}%`, background: t.color }}
                />
              </div>
              <span className="elo-analysis-val">
                {revealed ? <AnimatedNumber value={t.elo} decimals={0} /> : "—"}
              </span>
            </div>
          ))}
        </div>
        <div className="elo-edge-callout">
          <span>ELO edge:</span>
          <span style={{ color: eloColor, fontWeight: 700 }}>{eloFavor} +{eloEdge.toFixed(0)} pts</span>
        </div>
      </div>

      <div className="report-section">
        <div className="report-section-title">IMPLIED DECIMAL ODDS</div>
        <div className="odds-grid">
          {[
            { label: homeTeam, odds: toOdds(prediction.home_win_probability), color: "var(--home)" },
            { label: "Draw",   odds: toOdds(prediction.draw_probability),     color: "var(--draw)" },
            { label: awayTeam, odds: toOdds(prediction.away_win_probability), color: "var(--away)" },
          ].map(o => (
            <div className="odds-card" key={o.label}>
              <div className="odds-team"  style={{ color: o.color }}>{o.label}</div>
              <div className="odds-value" style={{ color: o.color }}>{o.odds}x</div>
              <div className="odds-sub">decimal odds</div>
            </div>
          ))}
        </div>
        <p className="odds-disclaimer">
          Derived from model probabilities only. Not affiliated with any bookmaker.
        </p>
      </div>

    </div>
  );
}

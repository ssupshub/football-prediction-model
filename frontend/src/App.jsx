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
    const end = value, duration = 900, startTime = performance.now();
    const tick = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(end * eased);
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
  const [teams, setTeams]               = useState([]);
  const [homeTeam, setHomeTeam]         = useState("");
  const [awayTeam, setAwayTeam]         = useState("");
  const [loading, setLoading]           = useState(false);
  const [fetchingTeams, setFetchingTeams] = useState(true);
  const [error, setError]               = useState(null);
  const [prediction, setPrediction]     = useState(null);
  const [revealed, setRevealed]         = useState(false);
  const [activeTab, setActiveTab]       = useState("prediction"); // prediction | report | howit

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/teams`)
      .then(r => { if (!r.ok) throw new Error(`Server responded with ${r.status}`); return r.json(); })
      .then(data => {
        if (cancelled) return;
        const list = Array.isArray(data.teams) ? data.teams : [];
        setTeams(list);
        if (list.length >= 2) { setHomeTeam(list[0]); setAwayTeam(list[1]); }
      })
      .catch(err => { if (!cancelled) setError(`Backend unreachable: ${err.message}`); })
      .finally(() => { if (!cancelled) setFetchingTeams(false); });
    return () => { cancelled = true; };
  }, []);

  const handlePredict = useCallback(async () => {
    setError(null); setPrediction(null); setRevealed(false);
    if (homeTeam === awayTeam) { setError("Select two different teams."); return; }
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam }),
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
  }, [homeTeam, awayTeam]);

  return (
    <div className="app">
      <div className="bg-grid" aria-hidden="true" />

      {/* ── HEADER (centred) ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo-lockup">
            <span className="logo-icon" aria-hidden="true">⚽</span>
            <div>
              <div className="logo-text">MATCH ORACLE</div>
              <div className="logo-sub">AI · v4 · 200 teams · 10 leagues</div>
            </div>
          </div>
          <nav className="header-nav">
            <a href="#how-it-works" className="nav-link" onClick={e => { e.preventDefault(); setActiveTab("howit"); }}>How it works</a>
            <a href="#report" className="nav-link" onClick={e => { e.preventDefault(); if(prediction) setActiveTab("report"); }}>
              Analysis {prediction && <span className="nav-dot" />}
            </a>
          </nav>
          <div className="header-badge">ML POWERED</div>
        </div>
      </header>

      {/* ── HERO (centred) ── */}
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

      {/* ── MAIN ── */}
      <main className="main">

        {/* Predictor card */}
        <div className="card">
          <div className="picker">
            <TeamColumn side="HOME" value={homeTeam} teams={teams} loading={fetchingTeams} onChange={setHomeTeam} accent="var(--home)" />
            <div className="versus-block" aria-hidden="true">
              <div className="versus-line" /><div className="versus-text">VS</div><div className="versus-line" />
            </div>
            <TeamColumn side="AWAY" value={awayTeam} teams={teams} loading={fetchingTeams} onChange={setAwayTeam} accent="var(--away)" />
          </div>

          {error && (
            <div className="error-strip" role="alert">
              <span className="error-icon">!</span>{error}
            </div>
          )}

          <button
            className={`cta-btn ${loading ? "cta-loading" : ""}`}
            disabled={loading || fetchingTeams || teams.length === 0}
            onClick={handlePredict} type="button"
          >
            {loading
              ? <span className="cta-spinner" aria-label="Predicting…" />
              : <><span className="cta-label">PREDICT OUTCOME</span><span className="cta-arrow">→</span></>
            }
          </button>
        </div>

        {/* Tab bar — only show when prediction exists */}
        {prediction && (
          <div className="tab-bar">
            {[
              { id: "prediction", label: "⚡ Prediction" },
              { id: "report",     label: "📊 Detailed Report" },
              { id: "howit",      label: "🧠 How It Works" },
            ].map(t => (
              <button
                key={t.id}
                className={`tab-btn ${activeTab === t.id ? "tab-btn--active" : ""}`}
                onClick={() => setActiveTab(t.id)}
                type="button"
              >{t.label}</button>
            ))}
          </div>
        )}

        {/* ── TAB: Prediction ── */}
        {prediction && activeTab === "prediction" && (
          <ResultsPanel prediction={prediction} homeTeam={homeTeam} awayTeam={awayTeam} revealed={revealed} />
        )}

        {/* ── TAB: Detailed Report ── */}
        {prediction && activeTab === "report" && (
          <DetailedReport prediction={prediction} homeTeam={homeTeam} awayTeam={awayTeam} revealed={revealed} />
        )}

        {/* ── TAB: How It Works (always available) ── */}
        {activeTab === "howit" && (
          <HowItWorks />
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
          <select value={value} onChange={e => onChange(e.target.value)} style={{ "--accent": accent }} className="team-select">
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

// ---------------------------------------------------------------------------
// ResultsPanel
// ---------------------------------------------------------------------------
function ResultsPanel({ prediction, homeTeam, awayTeam, revealed }) {
  const outcomeColor = { "Home Win": "var(--home)", "Away Win": "var(--away)", Draw: "var(--draw)" };
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
          {prediction.prediction === "Draw" ? "No clear favourite"
            : prediction.prediction === "Home Win" ? homeTeam : awayTeam}
        </div>
      </div>
      <div className="elo-track">
        <EloBlock team={homeTeam} elo={prediction.home_elo} color="var(--home)" revealed={revealed} />
        <div className="elo-divider"><div className="elo-divider-label">ELO</div></div>
        <EloBlock team={awayTeam} elo={prediction.away_elo} color="var(--away)" revealed={revealed} align="right" />
      </div>
      <div className="prob-grid">
        {bars.map(bar => (
          <ProbBar key={bar.id} label={bar.label} value={bar.value} type={bar.type} isWinner={bar.id === winner.id} revealed={revealed} />
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
      <div className="elo-score" style={{ color }}>{revealed ? <AnimatedNumber value={elo} decimals={0} /> : "—"}</div>
      <div className="elo-score-label">ELO RATING</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ProbBar
// ---------------------------------------------------------------------------
function ProbBar({ label, value, type, isWinner, revealed }) {
  const safe = typeof value === "number" && isFinite(value) ? value : 0;
  const pct  = Math.min(Math.max(safe, 0), 1) * 100;
  return (
    <div className={`prob-row ${isWinner ? "prob-row--winner" : ""}`}>
      <div className="prob-meta">
        <span className="prob-label">{label}</span>
        <span className="prob-pct">{revealed ? <AnimatedNumber value={pct} decimals={1} suffix="%" /> : "—"}</span>
      </div>
      <div className="prob-track" role="progressbar" aria-valuenow={pct.toFixed(1)} aria-valuemin="0" aria-valuemax="100">
        <div className={`prob-fill prob-fill--${type}`} style={{ width: revealed ? `${pct}%` : "0%" }} />
        {isWinner && <div className="prob-winner-pip" />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// DetailedReport — NEW feature tab
// ---------------------------------------------------------------------------
function DetailedReport({ prediction, homeTeam, awayTeam, revealed }) {
  const eloDiff  = prediction.home_elo - prediction.away_elo;
  const eloEdge  = Math.abs(eloDiff);
  const eloFavor = eloDiff > 0 ? homeTeam : awayTeam;
  const eloColor = eloDiff > 0 ? "var(--home)" : "var(--away)";

  const homeWin  = (prediction.home_win_probability * 100).toFixed(1);
  const draw     = (prediction.draw_probability * 100).toFixed(1);
  const awayWin  = (prediction.away_win_probability * 100).toFixed(1);

  const confidence = Math.max(
    prediction.home_win_probability,
    prediction.draw_probability,
    prediction.away_win_probability
  );
  const confidenceLabel = confidence > 0.6 ? "HIGH" : confidence > 0.45 ? "MEDIUM" : "LOW";
  const confidenceColor = confidence > 0.6 ? "var(--home)" : confidence > 0.45 ? "#f0c040" : "var(--away)";

  // Implied odds (bookmaker-style)
  const toOdds = p => p > 0 ? (1 / p).toFixed(2) : "—";

  const verdict = prediction.prediction;
  const summaryText = verdict === "Draw"
    ? `This looks like a balanced contest. The model gives neither side a decisive edge, with both teams near parity in ELO. Draw probability at ${draw}% reflects closely matched form.`
    : `${verdict === "Home Win" ? homeTeam : awayTeam} are favoured with ${verdict === "Home Win" ? homeWin : awayWin}% win probability. ${
        eloEdge > 50
          ? `Their ELO advantage of ${eloEdge.toFixed(0)} points suggests a meaningful quality gap.`
          : `The ELO gap is narrow (${eloEdge.toFixed(0)} pts), so this remains competitive.`
      }`;

  return (
    <div className={`results results--report ${revealed ? "results--visible" : ""}`}>

      {/* Summary */}
      <div className="report-section report-header-section">
        <div className="report-section-title">MATCH SUMMARY</div>
        <p className="report-summary-text">{summaryText}</p>
        <div className="report-confidence">
          <span className="report-confidence-label">MODEL CONFIDENCE</span>
          <span className="report-confidence-value" style={{ color: confidenceColor }}>{confidenceLabel}</span>
          <span className="report-confidence-pct" style={{ color: confidenceColor }}>
            {(confidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Head to Head */}
      <div className="report-section">
        <div className="report-section-title">HEAD-TO-HEAD SNAPSHOT</div>
        <div className="h2h-grid">
          <div className="h2h-cell h2h-cell--home">
            <div className="h2h-label" style={{ color: "var(--home)" }}>{homeTeam}</div>
            <div className="h2h-big" style={{ color: "var(--home)" }}>{homeWin}%</div>
            <div className="h2h-sub">Win probability</div>
          </div>
          <div className="h2h-cell h2h-cell--draw">
            <div className="h2h-label" style={{ color: "var(--draw)" }}>DRAW</div>
            <div className="h2h-big" style={{ color: "var(--draw)" }}>{draw}%</div>
            <div className="h2h-sub">Draw probability</div>
          </div>
          <div className="h2h-cell h2h-cell--away">
            <div className="h2h-label" style={{ color: "var(--away)" }}>{awayTeam}</div>
            <div className="h2h-big" style={{ color: "var(--away)" }}>{awayWin}%</div>
            <div className="h2h-sub">Win probability</div>
          </div>
        </div>
      </div>

      {/* ELO Analysis */}
      <div className="report-section">
        <div className="report-section-title">ELO STRENGTH ANALYSIS</div>
        <div className="elo-analysis">
          <div className="elo-analysis-row">
            <span className="elo-analysis-label" style={{ color: "var(--home)" }}>{homeTeam}</span>
            <div className="elo-analysis-bar-wrap">
              <div className="elo-analysis-bar" style={{
                width: `${Math.min(100, (prediction.home_elo / 2000) * 100)}%`,
                background: "var(--home)"
              }} />
            </div>
            <span className="elo-analysis-val">{revealed ? <AnimatedNumber value={prediction.home_elo} decimals={0} /> : "—"}</span>
          </div>
          <div className="elo-analysis-row">
            <span className="elo-analysis-label" style={{ color: "var(--away)" }}>{awayTeam}</span>
            <div className="elo-analysis-bar-wrap">
              <div className="elo-analysis-bar" style={{
                width: `${Math.min(100, (prediction.away_elo / 2000) * 100)}%`,
                background: "var(--away)"
              }} />
            </div>
            <span className="elo-analysis-val">{revealed ? <AnimatedNumber value={prediction.away_elo} decimals={0} /> : "—"}</span>
          </div>
        </div>
        <div className="elo-edge-callout">
          <span>ELO edge:</span>
          <span style={{ color: eloColor, fontWeight: 700 }}>{eloFavor} +{eloEdge.toFixed(0)} pts</span>
        </div>
      </div>

      {/* Implied Odds */}
      <div className="report-section">
        <div className="report-section-title">IMPLIED DECIMAL ODDS</div>
        <div className="odds-grid">
          {[
            { label: homeTeam,  odds: toOdds(prediction.home_win_probability), color: "var(--home)" },
            { label: "Draw",    odds: toOdds(prediction.draw_probability),     color: "var(--draw)" },
            { label: awayTeam,  odds: toOdds(prediction.away_win_probability), color: "var(--away)" },
          ].map(o => (
            <div className="odds-card" key={o.label}>
              <div className="odds-team" style={{ color: o.color }}>{o.label}</div>
              <div className="odds-value" style={{ color: o.color }}>{o.odds}x</div>
              <div className="odds-sub">decimal odds</div>
            </div>
          ))}
        </div>
        <p className="odds-disclaimer">Derived from model probabilities only. Not affiliated with any bookmaker.</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// HowItWorks — NEW info tab
// ---------------------------------------------------------------------------
function HowItWorks() {
  const steps = [
    {
      num: "01", title: "Data Generation",
      desc: "38,000 synthetic matches generated across 10 leagues and 10 seasons using Poisson-based goal simulation, team attack/defence profiles, home advantage, and form tracking.",
      icon: "⚙️",
    },
    {
      num: "02", title: "Feature Engineering",
      desc: "27 features extracted per match: ELO ratings, rolling xG, shots on target, possession, corners, yellow cards, goal difference, head-to-head win rate, and league physicality index.",
      icon: "🔬",
    },
    {
      num: "03", title: "Model Training",
      desc: "Three models compete — Logistic Regression, Random Forest (400 trees), and XGBoost (tuned via RandomizedSearchCV). Best by weighted F1 is selected and probability-calibrated with isotonic regression.",
      icon: "🤖",
    },
    {
      num: "04", title: "ELO Rating System",
      desc: "Every team maintains an ELO rating (k=32) updated after each match. Ratings start at 1500 and converge to reflect true strength over 10 seasons of data.",
      icon: "📈",
    },
    {
      num: "05", title: "Prediction",
      desc: "At inference, the model reads current ELO ratings, rolling averages from the last 10 matches, and H2H history to output Home Win / Draw / Away Win probabilities.",
      icon: "⚡",
    },
  ];

  const features = [
    { cat: "ELO", items: "Home ELO, Away ELO, ELO Diff" },
    { cat: "Form", items: "Last 5 match points (home & away), Form diff" },
    { cat: "Goals", items: "Avg scored & conceded (last 10)" },
    { cat: "xG", items: "Avg xG for & against (last 10)" },
    { cat: "Shots", items: "Avg shots on target (home & away)" },
    { cat: "Possession", items: "Avg possession % (home & away)" },
    { cat: "Set Pieces", items: "Avg corners (home & away)" },
    { cat: "Discipline", items: "Avg yellow cards (home & away)" },
    { cat: "H2H", items: "Home win rate from last 10 meetings" },
    { cat: "League", items: "League encoding + physicality index" },
  ];

  return (
    <div className="how-it-works results--visible" style={{ opacity: 1, transform: "none" }}>
      <div className="how-header">
        <div className="report-section-title">HOW THE MODEL WORKS</div>
        <p className="how-intro">Match Oracle uses machine learning trained on synthetic match data that mirrors real football dynamics.</p>
      </div>

      <div className="steps-list">
        {steps.map(s => (
          <div className="step-item" key={s.num}>
            <div className="step-icon">{s.icon}</div>
            <div className="step-body">
              <div className="step-num">{s.num}</div>
              <div className="step-title">{s.title}</div>
              <div className="step-desc">{s.desc}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="features-table-section">
        <div className="report-section-title" style={{ marginBottom: "1rem" }}>ALL 27 FEATURES</div>
        <div className="features-table">
          {features.map(f => (
            <div className="feature-row" key={f.cat}>
              <div className="feature-cat">{f.cat}</div>
              <div className="feature-items">{f.items}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="leagues-section">
        <div className="report-section-title" style={{ marginBottom: "1rem" }}>10 LEAGUES COVERED</div>
        <div className="leagues-grid">
          {[
            { name: "Premier League", flag: "🏴󠁧󠁢󠁥󠁮󠁧󠁿" },
            { name: "La Liga", flag: "🇪🇸" },
            { name: "Bundesliga", flag: "🇩🇪" },
            { name: "Serie A", flag: "🇮🇹" },
            { name: "Ligue 1", flag: "🇫🇷" },
            { name: "Eredivisie", flag: "🇳🇱" },
            { name: "Scottish Premiership", flag: "🏴󠁧󠁢󠁳󠁣󠁴󠁿" },
            { name: "Primeira Liga", flag: "🇵🇹" },
            { name: "Super Lig", flag: "🇹🇷" },
            { name: "Championship", flag: "🏴󠁧󠁢󠁥󠁮󠁧󠁿" },
          ].map(l => (
            <div className="league-chip" key={l.name}>
              <span className="league-flag">{l.flag}</span>
              <span className="league-name">{l.name}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

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
// Pages: "home" | "leagues" | "howit"
// ---------------------------------------------------------------------------
export default function App() {
  const [page, setPage]                 = useState("home"); // home | leagues | howit
  const [teams, setTeams]               = useState([]);
  const [homeTeam, setHomeTeam]         = useState("");
  const [awayTeam, setAwayTeam]         = useState("");
  const [loading, setLoading]           = useState(false);
  const [fetchingTeams, setFetchingTeams] = useState(true);
  const [error, setError]               = useState(null);
  const [prediction, setPrediction]     = useState(null);
  const [revealed, setRevealed]         = useState(false);
  const [activeTab, setActiveTab]       = useState("prediction");
  const topRef = useRef(null);

  // Navigate to a page and scroll to top
  const goTo = useCallback((p) => {
    setPage(p);
    setTimeout(() => topRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 40);
  }, []);

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

  const switchTab = useCallback((tab) => {
    setActiveTab(tab);
  }, []);

  // ─── Shared header ────────────────────────────────────────────────────────
  const Header = () => (
    <header className="header">
      <div className="header-inner">
        {/* Left nav */}
        <nav className="header-nav">
          <button
            className={`nav-link ${page === "leagues" ? "nav-link--active" : ""}`}
            onClick={() => goTo("leagues")} type="button"
          >Leagues</button>
          <button
            className={`nav-link ${page === "howit" ? "nav-link--active" : ""}`}
            onClick={() => goTo("howit")} type="button"
          >How It Works</button>
        </nav>

        {/* Centre logo — always navigates home */}
        <button className="logo-lockup logo-btn" onClick={() => goTo("home")} type="button">
          <span className="logo-icon" aria-hidden="true">⚽</span>
          <div>
            <div className="logo-text">MATCH ORACLE</div>
            <div className="logo-sub">AI · v4 · 200 teams · 10 leagues</div>
          </div>
        </button>

        {/* Right badge */}
        <div className="header-right">
          {page !== "home" && (
            <button className="nav-link nav-back" onClick={() => goTo("home")} type="button">
              ← Predict
            </button>
          )}
          <div className="header-badge">ML POWERED</div>
        </div>
      </div>
    </header>
  );

  // ─── PAGE: HOME ───────────────────────────────────────────────────────────
  if (page === "home") return (
    <div className="app">
      <div className="bg-grid" aria-hidden="true" />
      <div ref={topRef} />
      <Header />

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
            <TeamColumn side="HOME" value={homeTeam} teams={teams} loading={fetchingTeams} onChange={setHomeTeam} accent="var(--home)" />
            <div className="versus-block" aria-hidden="true">
              <div className="versus-line" /><div className="versus-text">VS</div><div className="versus-line" />
            </div>
            <TeamColumn side="AWAY" value={awayTeam} teams={teams} loading={fetchingTeams} onChange={setAwayTeam} accent="var(--away)" />
          </div>
          {error && <div className="error-strip" role="alert"><span className="error-icon">!</span>{error}</div>}
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

        {/* Prediction tabs — only after a prediction */}
        {prediction && (
          <>
            <div className="tab-bar">
              {[
                { id: "prediction", label: "⚡ Prediction" },
                { id: "report",     label: "📊 Detailed Report" },
              ].map(t => (
                <button
                  key={t.id}
                  className={`tab-btn ${activeTab === t.id ? "tab-btn--active" : ""}`}
                  onClick={() => switchTab(t.id)} type="button"
                >{t.label}</button>
              ))}
            </div>
            {activeTab === "prediction" && (
              <ResultsPanel prediction={prediction} homeTeam={homeTeam} awayTeam={awayTeam} revealed={revealed} />
            )}
            {activeTab === "report" && (
              <DetailedReport prediction={prediction} homeTeam={homeTeam} awayTeam={awayTeam} revealed={revealed} />
            )}
          </>
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

        {/* Quick-nav cards to other pages */}
        <div className="page-nav-cards">
          <button className="page-nav-card" onClick={() => goTo("leagues")} type="button">
            <div className="page-nav-card-icon">🌍</div>
            <div className="page-nav-card-body">
              <div className="page-nav-card-title">10 Leagues Covered</div>
              <div className="page-nav-card-sub">Premier League, La Liga, Bundesliga & 7 more</div>
            </div>
            <div className="page-nav-card-arrow">→</div>
          </button>
          <button className="page-nav-card" onClick={() => goTo("howit")} type="button">
            <div className="page-nav-card-icon">🧠</div>
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
    </div>
  );

  // ─── PAGE: LEAGUES ────────────────────────────────────────────────────────
  if (page === "leagues") return (
    <div className="app">
      <div className="bg-grid" aria-hidden="true" />
      <div ref={topRef} />
      <Header />

      <section className="page-hero">
        <div className="page-hero-inner">
          <div className="hero-eyebrow">DATA COVERAGE</div>
          <h1 className="page-hero-title">10 LEAGUES<br /><span className="accent">COVERED</span></h1>
          <p className="hero-desc">
            Every prediction draws from 10 seasons of match data — 38,000 matches, 200 teams across Europe.
          </p>
        </div>
      </section>

      <main className="main">
        <LeaguesCovered />
      </main>

      <footer className="footer">
        Predictions are probabilistic · Not financial advice · For entertainment
      </footer>
    </div>
  );

  // ─── PAGE: HOW IT WORKS ───────────────────────────────────────────────────
  if (page === "howit") return (
    <div className="app">
      <div className="bg-grid" aria-hidden="true" />
      <div ref={topRef} />
      <Header />

      <section className="page-hero">
        <div className="page-hero-inner">
          <div className="hero-eyebrow">UNDER THE HOOD</div>
          <h1 className="page-hero-title">HOW IT<br /><span className="accent">WORKS</span></h1>
          <p className="hero-desc">
            From raw match simulation to calibrated probabilities — here's the full pipeline.
          </p>
        </div>
      </section>

      <main className="main">
        <HowItWorks />
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
// DetailedReport
// ---------------------------------------------------------------------------
function DetailedReport({ prediction, homeTeam, awayTeam, revealed }) {
  const eloDiff  = prediction.home_elo - prediction.away_elo;
  const eloEdge  = Math.abs(eloDiff);
  const eloFavor = eloDiff > 0 ? homeTeam : awayTeam;
  const eloColor = eloDiff > 0 ? "var(--home)" : "var(--away)";
  const homeWin  = (prediction.home_win_probability * 100).toFixed(1);
  const draw     = (prediction.draw_probability * 100).toFixed(1);
  const awayWin  = (prediction.away_win_probability * 100).toFixed(1);
  const confidence = Math.max(prediction.home_win_probability, prediction.draw_probability, prediction.away_win_probability);
  const confidenceLabel = confidence > 0.6 ? "HIGH" : confidence > 0.45 ? "MEDIUM" : "LOW";
  const confidenceColor = confidence > 0.6 ? "var(--home)" : confidence > 0.45 ? "#f0c040" : "var(--away)";
  const toOdds = p => p > 0 ? (1 / p).toFixed(2) : "—";
  const verdict = prediction.prediction;
  const summaryText = verdict === "Draw"
    ? `This looks like a balanced contest. The model gives neither side a decisive edge, with both teams near parity in ELO. Draw probability at ${draw}% reflects closely matched form.`
    : `${verdict === "Home Win" ? homeTeam : awayTeam} are favoured with ${verdict === "Home Win" ? homeWin : awayWin}% win probability. ${eloEdge > 50 ? `Their ELO advantage of ${eloEdge.toFixed(0)} points suggests a meaningful quality gap.` : `The ELO gap is narrow (${eloEdge.toFixed(0)} pts), so this remains competitive.`}`;

  return (
    <div className={`results results--report ${revealed ? "results--visible" : ""}`}>
      <div className="report-section report-header-section">
        <div className="report-section-title">MATCH SUMMARY</div>
        <p className="report-summary-text">{summaryText}</p>
        <div className="report-confidence">
          <span className="report-confidence-label">MODEL CONFIDENCE</span>
          <span className="report-confidence-value" style={{ color: confidenceColor }}>{confidenceLabel}</span>
          <span className="report-confidence-pct" style={{ color: confidenceColor }}>{(confidence * 100).toFixed(1)}%</span>
        </div>
      </div>
      <div className="report-section">
        <div className="report-section-title">HEAD-TO-HEAD SNAPSHOT</div>
        <div className="h2h-grid">
          <div className="h2h-cell"><div className="h2h-label" style={{ color: "var(--home)" }}>{homeTeam}</div><div className="h2h-big" style={{ color: "var(--home)" }}>{homeWin}%</div><div className="h2h-sub">Win probability</div></div>
          <div className="h2h-cell"><div className="h2h-label" style={{ color: "var(--draw)" }}>DRAW</div><div className="h2h-big" style={{ color: "var(--draw)" }}>{draw}%</div><div className="h2h-sub">Draw probability</div></div>
          <div className="h2h-cell"><div className="h2h-label" style={{ color: "var(--away)" }}>{awayTeam}</div><div className="h2h-big" style={{ color: "var(--away)" }}>{awayWin}%</div><div className="h2h-sub">Win probability</div></div>
        </div>
      </div>
      <div className="report-section">
        <div className="report-section-title">ELO STRENGTH ANALYSIS</div>
        <div className="elo-analysis">
          {[{ team: homeTeam, elo: prediction.home_elo, color: "var(--home)" }, { team: awayTeam, elo: prediction.away_elo, color: "var(--away)" }].map(t => (
            <div className="elo-analysis-row" key={t.team}>
              <span className="elo-analysis-label" style={{ color: t.color }}>{t.team}</span>
              <div className="elo-analysis-bar-wrap"><div className="elo-analysis-bar" style={{ width: `${Math.min(100,(t.elo/2000)*100)}%`, background: t.color }} /></div>
              <span className="elo-analysis-val">{revealed ? <AnimatedNumber value={t.elo} decimals={0} /> : "—"}</span>
            </div>
          ))}
        </div>
        <div className="elo-edge-callout"><span>ELO edge:</span><span style={{ color: eloColor, fontWeight: 700 }}>{eloFavor} +{eloEdge.toFixed(0)} pts</span></div>
      </div>
      <div className="report-section">
        <div className="report-section-title">IMPLIED DECIMAL ODDS</div>
        <div className="odds-grid">
          {[{ label: homeTeam, odds: toOdds(prediction.home_win_probability), color: "var(--home)" }, { label: "Draw", odds: toOdds(prediction.draw_probability), color: "var(--draw)" }, { label: awayTeam, odds: toOdds(prediction.away_win_probability), color: "var(--away)" }].map(o => (
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
// LeaguesCovered — full page content
// ---------------------------------------------------------------------------
function LeaguesCovered() {
  const leagues = [
    { name: "Premier League",       flag: "🏴󠁧󠁢󠁥󠁮󠁧󠁿", country: "England",     teams: 20, avg: "2.7", phys: "Medium" },
    { name: "La Liga",              flag: "🇪🇸", country: "Spain",       teams: 20, avg: "2.6", phys: "Low" },
    { name: "Bundesliga",           flag: "🇩🇪", country: "Germany",     teams: 20, avg: "3.0", phys: "Medium" },
    { name: "Serie A",              flag: "🇮🇹", country: "Italy",       teams: 20, avg: "2.7", phys: "Medium" },
    { name: "Ligue 1",              flag: "🇫🇷", country: "France",      teams: 20, avg: "2.7", phys: "Medium" },
    { name: "Eredivisie",           flag: "🇳🇱", country: "Netherlands", teams: 20, avg: "3.1", phys: "Low" },
    { name: "Scottish Premiership", flag: "🏴󠁧󠁢󠁳󠁣󠁴󠁿", country: "Scotland",   teams: 20, avg: "2.9", phys: "High" },
    { name: "Primeira Liga",        flag: "🇵🇹", country: "Portugal",    teams: 20, avg: "2.6", phys: "Medium" },
    { name: "Super Lig",            flag: "🇹🇷", country: "Turkey",      teams: 20, avg: "2.8", phys: "Very High" },
    { name: "Championship",         flag: "🏴󠁧󠁢󠁥󠁮󠁧󠁿", country: "England",     teams: 20, avg: "2.5", phys: "Very High" },
  ];

  const physColor = { "Low": "#7c8fa6", "Medium": "#f0c040", "High": "#ff9040", "Very High": "#ff3c5a" };

  return (
    <div className="page-content-block">
      {/* Summary bar */}
      <div className="leagues-summary-bar">
        {[{ val: "10", lbl: "Leagues" }, { val: "200", lbl: "Teams" }, { val: "38,000", lbl: "Matches" }, { val: "10", lbl: "Seasons" }].map(s => (
          <div className="leagues-summary-item" key={s.lbl}>
            <div className="leagues-summary-val">{s.val}</div>
            <div className="leagues-summary-lbl">{s.lbl}</div>
          </div>
        ))}
      </div>

      {/* League cards */}
      <div className="leagues-cards-grid">
        {leagues.map((l, i) => (
          <div className="league-card" key={l.name}>
            <div className="league-card-num">0{i + 1}</div>
            <div className="league-card-flag">{l.flag}</div>
            <div className="league-card-body">
              <div className="league-card-name">{l.name}</div>
              <div className="league-card-country">{l.country}</div>
            </div>
            <div className="league-card-stats">
              <div className="league-card-stat">
                <span className="league-card-stat-val">{l.avg}</span>
                <span className="league-card-stat-lbl">avg goals</span>
              </div>
              <div className="league-card-divider" />
              <div className="league-card-stat">
                <span className="league-card-stat-val" style={{ color: physColor[l.phys] }}>{l.phys}</span>
                <span className="league-card-stat-lbl">physicality</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Footer note */}
      <div className="page-note">
        Each league contributes 3,800 matches (20 teams × 380 fixtures × 10 seasons). League physicality is a model feature encoding foul and card rates per competition.
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// HowItWorks — full page content
// ---------------------------------------------------------------------------
function HowItWorks() {
  const steps = [
    { num: "01", title: "Data Generation",    icon: "⚙️", desc: "38,000 synthetic matches generated across 10 leagues and 10 seasons using Poisson-based goal simulation, team attack/defence profiles, home advantage, and form tracking." },
    { num: "02", title: "Feature Engineering", icon: "🔬", desc: "27 features extracted per match: ELO ratings, rolling xG, shots on target, possession, corners, yellow cards, goal difference, head-to-head win rate, and league physicality index." },
    { num: "03", title: "Model Training",      icon: "🤖", desc: "Three models compete — Logistic Regression, Random Forest (400 trees), and XGBoost (tuned via RandomizedSearchCV with 30 iterations). Best by weighted F1 is selected and probability-calibrated." },
    { num: "04", title: "ELO Rating System",   icon: "📈", desc: "Every team maintains an ELO rating (k=32) updated after each match. Ratings start at 1500 and converge to reflect true strength over 10 seasons of data." },
    { num: "05", title: "Prediction",          icon: "⚡", desc: "At inference, the model reads current ELO ratings, rolling averages from the last 10 matches, and H2H history to output Home Win / Draw / Away Win probabilities." },
  ];

  const features = [
    { cat: "ELO",          items: "Home ELO, Away ELO, ELO Diff" },
    { cat: "Form",         items: "Last 5 match points (home & away), Form diff" },
    { cat: "Goals",        items: "Avg scored & conceded (rolling last 10)" },
    { cat: "xG",           items: "Avg xG for & against (last 10)" },
    { cat: "Shots",        items: "Avg shots on target (home & away)" },
    { cat: "Possession",   items: "Avg possession % (home & away)" },
    { cat: "Set Pieces",   items: "Avg corners (home & away)" },
    { cat: "Discipline",   items: "Avg yellow cards (home & away)" },
    { cat: "H2H",          items: "Home win rate from last 10 meetings" },
    { cat: "League",       items: "League encoding + physicality index" },
  ];

  return (
    <div className="page-content-block">
      {/* Pipeline steps */}
      <div className="hiw-block">
        <div className="hiw-block-title">THE PIPELINE</div>
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
      </div>

      {/* Features table */}
      <div className="hiw-block">
        <div className="hiw-block-title">ALL 27 FEATURES</div>
        <div className="features-table">
          {features.map(f => (
            <div className="feature-row" key={f.cat}>
              <div className="feature-cat">{f.cat}</div>
              <div className="feature-items">{f.items}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Model comparison */}
      <div className="hiw-block">
        <div className="hiw-block-title">MODELS EVALUATED</div>
        <div className="models-grid">
          {[
            { name: "Logistic Regression", detail: "StandardScaler pipeline · C=0.5 · 5-fold CV", badge: "Baseline" },
            { name: "Random Forest",        detail: "400 estimators · max depth 12 · min samples leaf 3", badge: "Ensemble" },
            { name: "XGBoost",              detail: "RandomizedSearchCV · 30 iterations · 3-fold CV", badge: "Best ★" },
          ].map(m => (
            <div className="model-card" key={m.name}>
              <div className="model-badge">{m.badge}</div>
              <div className="model-name">{m.name}</div>
              <div className="model-detail">{m.detail}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="page-note">
        The winning model is wrapped in isotonic calibration fitted on a dedicated hold-out split, so the output probabilities are well-calibrated and not just raw scores.
      </div>
    </div>
  );
}

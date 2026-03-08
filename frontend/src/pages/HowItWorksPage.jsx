// ─── HowItWorksPage ───────────────────────────────────────────────────────────
// Edit the data constants below to customise all content without touching JSX.

// ── Data ─────────────────────────────────────────────────────────────────────

const PIPELINE_STEPS = [
  {
    num:   "01",
    title: "Data Generation",
    tag:   "Foundation",
    desc:  "38,000 synthetic matches generated across 10 leagues and 10 seasons using Poisson-based goal simulation with per-team attack / defence profiles, per-league home advantage, and a form tracker. A momentum signal carries win-rate history across seasons.",
  },
  {
    num:   "02",
    title: "Feature Engineering",
    tag:   "Signal extraction",
    desc:  "27 features extracted chronologically per match. Rolling stats use only past data to prevent leakage: ELO ratings, xG, shots on target, possession, corners, yellow cards, 10-match goal difference, head-to-head win rate, and league physicality index.",
  },
  {
    num:   "03",
    title: "Model Selection",
    tag:   "Training",
    desc:  "Three candidate models are trained and evaluated on a stratified test split. Logistic Regression sets the baseline; Random Forest adds non-linear power; XGBoost is hyperparameter-tuned via RandomizedSearchCV with 30 iterations and 3-fold cross-validation.",
  },
  {
    num:   "04",
    title: "Probability Calibration",
    tag:   "Reliability",
    desc:  "The best model by weighted F1 is wrapped in isotonic calibration on a dedicated hold-out split. Raw scores are often over-confident — calibration maps them to probabilities that reflect true outcome frequencies.",
  },
  {
    num:   "05",
    title: "ELO Ratings",
    tag:   "Team strength",
    desc:  "Every team maintains a live ELO rating (k = 32, starting at 1500) updated after each match. Ratings converge over 10 seasons of data and are the strongest single predictor in the feature set.",
  },
  {
    num:   "06",
    title: "Inference",
    tag:   "Prediction",
    desc:  "At prediction time the API reads current ELO ratings, rolling averages from each team's last 10 league matches, and head-to-head history. The calibrated model returns three probabilities: Home Win, Draw, Away Win.",
  },
];

const FEATURES = [
  { cat: "ELO",        count: 3,  items: "Home ELO · Away ELO · ELO Diff" },
  { cat: "Form",       count: 3,  items: "Last 5 match points (home & away) · Form diff" },
  { cat: "Goals",      count: 4,  items: "Avg scored & conceded — rolling last 10 (home & away)" },
  { cat: "xG",         count: 4,  items: "Avg expected goals for & against — last 10 (home & away)" },
  { cat: "Shots",      count: 2,  items: "Avg shots on target — last 10 (home & away)" },
  { cat: "Possession", count: 2,  items: "Avg possession % — last 10 (home & away)" },
  { cat: "Set Pieces", count: 2,  items: "Avg corners — last 10 (home & away)" },
  { cat: "Discipline", count: 2,  items: "Avg yellow cards — last 10 (home & away)" },
  { cat: "H2H",        count: 1,  items: "Home win rate from last 10 head-to-head meetings" },
  { cat: "League",     count: 2,  items: "League encoding · Physicality index (foul / card rate)" },
];

const MODELS = [
  {
    rank:   "01",
    badge:  "Baseline",
    name:   "Logistic Regression",
    detail: "StandardScaler pipeline · C = 0.5 · max_iter = 2000 · 5-fold cross-validation",
    best:   false,
  },
  {
    rank:   "02",
    badge:  "Ensemble",
    name:   "Random Forest",
    detail: "400 estimators · max depth 12 · min samples leaf 3 · random_state = 42",
    best:   false,
  },
  {
    rank:   "03",
    badge:  "Selected",
    name:   "XGBoost",
    detail: "RandomizedSearchCV · 30 iterations · 3-fold CV · isotonic calibration on hold-out",
    best:   true,
  },
];

const ACCURACY_STATS = [
  { val: "~0.52", label: "Weighted F1",      note: "Across all 3 outcomes" },
  { val: "38k",   label: "Training matches", note: "10 leagues × 10 seasons" },
  { val: "15%",   label: "Test split",       note: "Stratified hold-out" },
  { val: "15%",   label: "Calibration",      note: "Isotonic regression" },
];

// ── Component ─────────────────────────────────────────────────────────────────
export default function HowItWorksPage() {
  return (
    <>
      {/* Page hero */}
      <section className="page-hero">
        <div className="page-hero-inner">
          <div className="hero-eyebrow">Under the hood</div>
          <h1 className="page-hero-title">
            HOW IT<br />
            <span className="accent">WORKS</span>
          </h1>
          <p className="hero-desc">
            From raw match simulation to calibrated win probabilities — the complete pipeline.
          </p>
        </div>
      </section>

      <main className="main">
        <div className="page-content-block">

          {/* ── 1. Pipeline steps ── */}
          <section className="hiw-section">
            <div className="hiw-section-header">
              <div className="hiw-section-label">Step by step</div>
              <h2 className="hiw-section-title">The Pipeline</h2>
            </div>

            <div className="pipeline-grid">
              {PIPELINE_STEPS.map((s) => (
                <div className="pipeline-card" key={s.num}>
                  <div className="pipeline-card-top">
                    <span className="pipeline-num">{s.num}</span>
                    <span className="pipeline-tag">{s.tag}</span>
                  </div>
                  <div className="pipeline-card-title">{s.title}</div>
                  <p className="pipeline-card-desc">{s.desc}</p>
                </div>
              ))}
            </div>
          </section>

          {/* ── 2. Features ── */}
          <section className="hiw-section">
            <div className="hiw-section-header">
              <div className="hiw-section-label">Input to the model</div>
              <h2 className="hiw-section-title">All 27 Features</h2>
            </div>

            <div className="features-panel">
              {FEATURES.map((f) => (
                <div className="feature-row" key={f.cat}>
                  <div className="feature-left">
                    <span className="feature-cat">{f.cat}</span>
                    <span className="feature-count">{f.count}</span>
                  </div>
                  <div className="feature-items">{f.items}</div>
                </div>
              ))}
              <div className="feature-row feature-row--total">
                <div className="feature-left">
                  <span className="feature-cat feature-cat--total">Total</span>
                  <span className="feature-count feature-count--total">27</span>
                </div>
                <div className="feature-items feature-items--total">features per match prediction</div>
              </div>
            </div>
          </section>

          {/* ── 3. Models ── */}
          <section className="hiw-section">
            <div className="hiw-section-header">
              <div className="hiw-section-label">Trained and compared</div>
              <h2 className="hiw-section-title">Models Evaluated</h2>
            </div>

            <div className="models-panel">
              {MODELS.map((m) => (
                <div className={`model-row ${m.best ? "model-row--best" : ""}`} key={m.name}>
                  <div className="model-row-rank">{m.rank}</div>
                  <div className="model-row-body">
                    <div className="model-row-name">{m.name}</div>
                    <div className="model-row-detail">{m.detail}</div>
                  </div>
                  <div className={`model-row-badge ${m.best ? "model-row-badge--best" : ""}`}>
                    {m.badge}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* ── 4. Accuracy ── */}
          <section className="hiw-section">
            <div className="hiw-section-header">
              <div className="hiw-section-label">Performance</div>
              <h2 className="hiw-section-title">Expected Accuracy</h2>
            </div>

            <div className="accuracy-strip">
              {ACCURACY_STATS.map((a) => (
                <div className="accuracy-item" key={a.label}>
                  <div className="accuracy-val">{a.val}</div>
                  <div className="accuracy-label">{a.label}</div>
                  <div className="accuracy-note">{a.note}</div>
                </div>
              ))}
            </div>
          </section>

          {/* Disclaimer */}
          <p className="page-note">
            Football is inherently unpredictable — even the best models struggle past ~55% accuracy on unseen
            fixtures. These predictions reflect historical patterns and statistical tendencies, not certainties.
          </p>

        </div>
      </main>

      <footer className="footer">
        Predictions are probabilistic · Not financial advice · For entertainment
      </footer>
    </>
  );
}

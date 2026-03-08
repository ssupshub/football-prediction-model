// ─── Header ──────────────────────────────────────────────────────────────────
// Props: page (string), goTo (fn)

export default function Header({ page, goTo }) {
  return (
    <header className="header">
      <div className="header-inner">

        {/* Left — page links */}
        <nav className="header-nav" aria-label="Main navigation">
          <button
            className={`nav-link ${page === "leagues" ? "nav-link--active" : ""}`}
            onClick={() => goTo("leagues")}
            type="button"
          >
            Leagues
          </button>
          <button
            className={`nav-link ${page === "howit" ? "nav-link--active" : ""}`}
            onClick={() => goTo("howit")}
            type="button"
          >
            How It Works
          </button>
        </nav>

        {/* Centre — logo, always goes home */}
        <button
          className="logo-lockup logo-btn"
          onClick={() => goTo("home")}
          type="button"
          aria-label="Go to home"
        >
          <span className="logo-icon" aria-hidden="true">⚽</span>
          <div>
            <div className="logo-text">MATCH ORACLE</div>
            <div className="logo-sub">AI · v4 · 200 teams · 10 leagues</div>
          </div>
        </button>

        {/* Right — back link on inner pages + badge */}
        <div className="header-right">
          {page !== "home" && (
            <button
              className="nav-link nav-back"
              onClick={() => goTo("home")}
              type="button"
            >
              ← Predict
            </button>
          )}
          <div className="header-badge">ML POWERED</div>
        </div>

      </div>
    </header>
  );
}

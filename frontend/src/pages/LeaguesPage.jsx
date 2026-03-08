// ─── LeaguesPage ─────────────────────────────────────────────────────────────
// Edit LEAGUES array below to add, remove or update league data.
// physColor maps physicality labels to accent colours.

const LEAGUES = [
  { name: "Premier League",       flag: "🏴󠁧󠁢󠁥󠁮󠁧󠁿", country: "England",     avg: "2.7", phys: "Medium"    },
  { name: "La Liga",              flag: "🇪🇸",       country: "Spain",       avg: "2.6", phys: "Low"       },
  { name: "Bundesliga",           flag: "🇩🇪",       country: "Germany",     avg: "3.0", phys: "Medium"    },
  { name: "Serie A",              flag: "🇮🇹",       country: "Italy",       avg: "2.7", phys: "Medium"    },
  { name: "Ligue 1",              flag: "🇫🇷",       country: "France",      avg: "2.7", phys: "Medium"    },
  { name: "Eredivisie",           flag: "🇳🇱",       country: "Netherlands", avg: "3.1", phys: "Low"       },
  { name: "Scottish Premiership", flag: "🏴󠁧󠁢󠁳󠁣󠁴󠁿",   country: "Scotland",   avg: "2.9", phys: "High"      },
  { name: "Primeira Liga",        flag: "🇵🇹",       country: "Portugal",    avg: "2.6", phys: "Medium"    },
  { name: "Super Lig",            flag: "🇹🇷",       country: "Turkey",      avg: "2.8", phys: "Very High" },
  { name: "Championship",         flag: "🏴󠁧󠁢󠁥󠁮󠁧󠁿",   country: "England",     avg: "2.5", phys: "Very High" },
];

const PHYS_COLOR = {
  "Low":       "#7c8fa6",
  "Medium":    "#f0c040",
  "High":      "#ff9040",
  "Very High": "#ff3c5a",
};

export default function LeaguesPage() {
  return (
    <>
      {/* Page hero */}
      <section className="page-hero">
        <div className="page-hero-inner">
          <div className="hero-eyebrow">DATA COVERAGE</div>
          <h1 className="page-hero-title">
            10 LEAGUES<br />
            <span className="accent">COVERED</span>
          </h1>
          <p className="hero-desc">
            Every prediction draws from 10 seasons of match data — 38,000 matches,
            200 teams across Europe.
          </p>
        </div>
      </section>

      <main className="main">
        <div className="page-content-block">

          {/* Summary stats bar */}
          <div className="leagues-summary-bar">
            {[
              { val: "10",     lbl: "Leagues"  },
              { val: "200",    lbl: "Teams"    },
              { val: "38,000", lbl: "Matches"  },
              { val: "10",     lbl: "Seasons"  },
            ].map(s => (
              <div className="leagues-summary-item" key={s.lbl}>
                <div className="leagues-summary-val">{s.val}</div>
                <div className="leagues-summary-lbl">{s.lbl}</div>
              </div>
            ))}
          </div>

          {/* League cards grid */}
          <div className="leagues-cards-grid">
            {LEAGUES.map((l, i) => (
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
                    <span
                      className="league-card-stat-val"
                      style={{ color: PHYS_COLOR[l.phys] }}
                    >
                      {l.phys}
                    </span>
                    <span className="league-card-stat-lbl">physicality</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Footer note */}
          <div className="page-note">
            Each league contributes 3,800 matches (20 teams × 380 fixtures × 10 seasons).
            League physicality is a model feature encoding the foul and card rate of each competition
            — Championship scores highest (0.85), Eredivisie lowest (0.45).
          </div>

        </div>
      </main>

      <footer className="footer">
        Predictions are probabilistic · Not financial advice · For entertainment
      </footer>
    </>
  );
}

import { useState, useCallback, useEffect, useRef } from "react";
import "./index.css";

import Header         from "./components/Header.jsx";
import HomePage       from "./pages/HomePage.jsx";
import LeaguesPage    from "./pages/LeaguesPage.jsx";
import HowItWorksPage from "./pages/HowItWorksPage.jsx";

const API_BASE = import.meta.env.VITE_API_BASE_URL
  ? import.meta.env.VITE_API_BASE_URL.replace(/\/$/, "")
  : "/api";

// ─── App — router + data layer ───────────────────────────────────────────────
export default function App() {
  const [page, setPage]                   = useState("home");
  const [teams, setTeams]                 = useState([]);
  const [fetchingTeams, setFetchingTeams] = useState(true);
  const topRef = useRef(null);

  const goTo = useCallback((p) => {
    setPage(p);
    setTimeout(() => topRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 40);
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/teams`)
      .then(r => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then(data => { if (!cancelled) setTeams(Array.isArray(data.teams) ? data.teams : []); })
      .catch(() => {})
      .finally(() => { if (!cancelled) setFetchingTeams(false); });
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="app">
      <div className="bg-grid" aria-hidden="true" />
      <div ref={topRef} />
      <Header page={page} goTo={goTo} />

      {page === "home"    && <HomePage       teams={teams} fetchingTeams={fetchingTeams} goTo={goTo} apiBase={API_BASE} />}
      {page === "leagues" && <LeaguesPage    />}
      {page === "howit"   && <HowItWorksPage />}
    </div>
  );
}

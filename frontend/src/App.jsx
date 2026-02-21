import { useState, useEffect } from 'react';
import './index.css';

function App() {
  const [teams, setTeams] = useState([]);
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    // Fetch available teams
    fetch('http://localhost:8000/teams')
      .then(res => res.json())
      .then(data => {
        if(data.teams) {
            setTeams(data.teams);
            if(data.teams.length > 1) {
                setHomeTeam(data.teams[0]);
                setAwayTeam(data.teams[1]);
            }
        }
      })
      .catch(err => {
        console.error("Error fetching teams:", err);
        setError("Could not connect to the backend server. Make sure it is running.");
      });
  }, []);

  const handlePredict = async (e) => {
    e.preventDefault();
    if (homeTeam === awayTeam) {
      setError("Home and Away teams cannot be the same.");
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to fetch prediction');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1 className="title">AI Match Predictor</h1>
      
      <div className="glass-card">
        <form className="prediction-form" onSubmit={handlePredict}>
          <div className="team-selectors">
            <div className="input-group">
              <label>Home Team</label>
              <select 
                value={homeTeam} 
                onChange={(e) => setHomeTeam(e.target.value)}
                required
              >
                {teams.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
            
            <div className="vs-badge">VS</div>
            
            <div className="input-group">
              <label>Away Team</label>
              <select 
                value={awayTeam} 
                onChange={(e) => setAwayTeam(e.target.value)}
                required
              >
                {teams.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
          </div>

          {error && <div className="error-message">{error}</div>}

          <button 
            type="submit" 
            className={`submit-btn ${loading ? 'loading' : ''}`}
            disabled={loading || teams.length === 0}
          >
            {loading ? '' : 'Predict Match Outcome'}
          </button>
        </form>

        {prediction && (
          <div className="results-container">
            <div className="prediction-winner">
              Predicted Winner: <span className="winner-highlight">{prediction.prediction}</span>
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
        )}
      </div>
    </div>
  );
}

function ProgressBar({ label, value, type }) {
  const percentage = (value * 100).toFixed(1);
  
  return (
    <div className={`bar-wrapper ${type}-bar`}>
      <div className="bar-labels">
        <span>{label}</span>
        <span>{percentage}%</span>
      </div>
      <div className="progress-bg">
        <div 
          className="progress-fill" 
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  );
}

export default App;

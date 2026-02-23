# AI Football Match Predictor v2

An ML-powered football match outcome predictor with a FastAPI backend and React frontend. Trained on **15,960 synthetic matches** simulated from realistic team profiles across 6 major leagues and 7 seasons.

---

## Features

- Predicts **Home Win / Draw / Away Win** probabilities for any fixture
- Displays **ELO ratings** for both teams
- 120 teams across 6 leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie)
- 26 engineered features: ELO, xG, form, shots on target, possession, corners, cards, H2H history
- Three competing models evaluated per run; best is selected automatically and probability-calibrated

---

## Project Structure

```
football-predictor/
├── backend/
│   ├── data_generator.py     # Generates 15,960 synthetic matches across 6 leagues & 7 seasons
│   ├── model_training.py     # Feature engineering + trains LR / RF / XGBoost, saves best model
│   ├── main.py               # FastAPI app — /teams, /predict, /health endpoints
│   ├── utils.py              # ELO rating calculation helper
│   ├── requirements.txt      # Python dependencies
│   ├── Dockerfile            # Container: generates data, trains model, starts server
│   └── run_all.bat           # Windows one-click local setup
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # React UI: team selectors, ELO chips, probability bars
│   │   ├── index.css         # Glassmorphism dark-theme styles
│   │   └── main.jsx          # React entry point
│   ├── public/
│   │   └── vite.svg
│   ├── index.html
│   ├── vite.config.js        # Dev proxy: /api → localhost:8000
│   ├── package.json
│   ├── eslint.config.js
│   └── .env.example          # Environment variable reference (copy → .env)
├── .gitignore
└── README.md
```

---

## Local Development

### Prerequisites

- Python 3.10+
- Node.js 18+

### 1. Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate training data (produces football_data.csv — 15,960 matches)
python data_generator.py

# Train models (produces football_model.pkl and current_state.pkl)
python model_training.py

# Start API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API is now available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

### 2. Frontend

Open a new terminal:

```bash
cd frontend

# Install dependencies
npm install

# Start dev server (proxies /api → localhost:8000 automatically)
npm run dev
```

App is now available at `http://localhost:5173`

### Windows Shortcut

```bat
cd backend
run_all.bat
```

---

## Environment Variables

Copy `.env.example` to `.env` in the relevant folder and update values as needed.

### Backend (`backend/.env`)

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `football_model.pkl` | Path to trained model file |
| `STATE_PATH` | `current_state.pkl` | Path to ELO + stats state file |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins (comma-separated in production) |
| `PORT` | `8000` | Server port |

### Frontend (`frontend/.env`)

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | *(empty)* | Backend URL. Leave empty in dev (Vite proxy handles it). Set to your Render URL in production. |

> **Important:** All Vite frontend variables must be prefixed with `VITE_` to be accessible in the browser.

---

## API Reference

### `GET /health`
Returns model readiness and number of teams loaded.

```json
{ "ready": true, "teams_loaded": 120 }
```

### `GET /teams`
Returns an alphabetical list of all 120 available teams.

```json
{ "teams": ["AC Milan", "Ajax", "Almeria", "..."] }
```

### `POST /predict`
Predicts match outcome probabilities and returns ELO ratings.

**Request:**
```json
{
  "home_team": "Arsenal",
  "away_team": "Liverpool"
}
```

**Response:**
```json
{
  "home_win_probability": 0.47,
  "draw_probability": 0.21,
  "away_win_probability": 0.32,
  "prediction": "Home Win",
  "home_elo": 1534.2,
  "away_elo": 1521.8
}
```

**Error responses:**

| Status | Reason |
|---|---|
| `404` | Team name not found |
| `422` | Home and Away teams are the same |
| `503` | Model not loaded (run `model_training.py` first) |

---

## Deployment

### Backend → Render (Docker)

1. Push repository to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set **Root Directory** to `backend`
5. Set **Environment** to `Docker`
6. Under **Environment Variables**, add:
   - `ALLOWED_ORIGINS` = `https://your-app.vercel.app`
7. Click **Deploy** (first build takes ~10 minutes — it generates data and trains the model)
8. Copy your service URL: `https://your-backend.onrender.com`

> **Note:** Render free-tier services sleep after 15 minutes of inactivity. The first request after sleep may take ~30 seconds.

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) → **Add New Project**
2. Import your GitHub repository
3. Set **Root Directory** to `frontend`
4. Framework will be auto-detected as **Vite**
5. Under **Environment Variables**, add:
   - `VITE_API_BASE_URL` = `https://your-backend.onrender.com`
6. Click **Deploy** (~1 minute)

**Fix page refresh 404 (optional):** Create `frontend/public/vercel.json`:
```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/" }]
}
```

---

## Model Details

### Data Generation

| Property | Value |
|---|---|
| Leagues | Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie |
| Seasons | 2018–19 through 2024–25 (7 seasons) |
| Teams | 120 |
| Matches | 15,960 |
| Format | Double round-robin per league per season |

Each match is simulated using:
- Separate **attack / defence ratings** per team (not a combined score)
- **xG (expected goals)** via Poisson sampling
- **Home fortress** multiplier per team
- **Form factor** from last 6 results (range 0.80–1.20)
- **Season fatigue** — performance slightly declines late in season
- **Head-to-head** history across all previous seasons

### Features (26 total)

| Category | Features |
|---|---|
| ELO | `Home_ELO`, `Away_ELO`, `ELO_Diff` |
| Form | `Home_Form5`, `Away_Form5`, `Form_Diff` |
| Goals | `Home_Avg_Scored`, `Home_Avg_Conceded`, `Away_Avg_Scored`, `Away_Avg_Conceded` |
| Goal Diff | `Home_GD_Last10`, `Away_GD_Last10` |
| xG | `Home_Avg_xG`, `Away_Avg_xG`, `Home_Avg_xGA`, `Away_Avg_xGA` |
| Shots | `Home_Avg_SoT`, `Away_Avg_SoT` |
| Possession | `Home_Avg_Poss`, `Away_Avg_Poss` |
| Corners | `Home_Avg_Corners`, `Away_Avg_Corners` |
| Discipline | `Home_Avg_Yellow`, `Away_Avg_Yellow` |
| H2H | `H2H_HomeWinRate` |
| League | `League_Enc` |

### Models Evaluated

| Model | Notes |
|---|---|
| Logistic Regression | StandardScaler pipeline, `C=0.5`, 5-fold CV |
| Random Forest | 300 estimators, max depth 12 |
| XGBoost | RandomizedSearchCV (20 iterations, 3-fold CV) |

The best model by weighted F1 score is selected and wrapped in **isotonic probability calibration** (`CalibratedClassifierCV`) for reliable confidence scores.

### Performance (v1 → v2)

| Metric | v1 | v2 |
|---|---|---|
| Training matches | 2,000 | **15,960** |
| Features | 8 | **26** |
| F1 Score (weighted) | 0.39 | **0.49** (+26%) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, uvicorn |
| ML | scikit-learn, XGBoost, pandas, numpy, scipy |
| Frontend | React 19, Vite 8, plain CSS |
| Containerisation | Docker |
| Hosting (backend) | Render |
| Hosting (frontend) | Vercel |

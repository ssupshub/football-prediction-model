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

## Bug Fixes (v2.1)

### `backend/main.py`
- **[Critical]** `H2H_HomeWinRate` was hardcoded to `0.45` in `build_features()` — predictions now use real head-to-head history loaded from `current_state.pkl`
- **[Critical]** Added `_get_classes()` helper for robust `classes_` extraction from `CalibratedClassifierCV` across all sklearn versions (handles both `.estimator` and `.base_estimator` attribute names)
- **[Minor]** Narrowed `except` block in `/predict` endpoint so only `predict_proba` failures are caught, with a descriptive error message

### `backend/model_training.py`
- **[Critical]** Fixed calibration data leakage: calibrator now fits on a dedicated `X_cal`/`y_cal` split, not on the same `X_test`/`y_test` used to compute evaluation metrics
- **[Critical]** H2H records are now tracked during feature engineering, persisted in `current_state.pkl`, and loaded at inference time
- **[Minor]** Fixed return signature of `engineer_features()` to include `h2h_serializable`

### `backend/data_generator.py`
- **[Important]** Replaced global `np.random` calls with an isolated `np.random.RandomState` instance (`np_rng`) threaded through all simulation functions — removes side-effects on global random state and improves reproducibility
- **[Minor]** Removed `random.seed()` / `np.random.seed()` global mutations from `generate_historical_data()`

### `backend/run_all.bat`
- **[Minor]** Added `if errorlevel 1` checks after every step so failures are reported and the script stops instead of silently continuing

### `frontend/src/App.jsx`
- **[Minor]** Error details are now captured and displayed in the teams-fetch `.catch()` handler (previously discarded)
- **[Minor]** `ProgressBar` now guards against `null`/`undefined`/`NaN` probability values with an `isFinite()` check before calling `.toFixed()`

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

---

## Model Details

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

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, uvicorn |
| ML | scikit-learn, XGBoost, pandas, numpy, scipy |
| Frontend | React 19, Vite 8, plain CSS |
| Containerisation | Docker |
| Hosting (backend) | Render |
| Hosting (frontend) | Vercel |

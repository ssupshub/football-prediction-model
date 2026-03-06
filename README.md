# Match Oracle — AI Football Match Predictor v3

An ML-powered football match outcome predictor with a FastAPI backend and a fully redesigned React frontend. Trained on **15,960 synthetic matches** simulated from realistic team profiles across 6 major leagues and 7 seasons.

---

## What's New in v3

### Frontend — Full Redesign
- **New visual identity** — editorial sports-magazine aesthetic with electric lime (`#c8ff00`) on pitch black
- **New typography** — Barlow Condensed (display) + DM Sans (body), replacing generic sans-serif
- **Animated counters** — ELO ratings and win probabilities count up from zero on reveal using `requestAnimationFrame`
- **Shimmer CTA button** — hover swipe animation via CSS pseudo-element
- **Results reveal** — panel fades and slides in on prediction
- **Full mobile responsiveness** — `clamp()`-based fluid sizing, responsive grid collapses at 540px, all touch targets ≥ 48px
- **Accessibility** — `prefers-reduced-motion` respected, visible `focus-visible` ring on all interactive elements, proper `aria-*` attributes on progress bars

### Backend — Bug Fixes (v2.1 → v3)
- **Critical:** Non-deterministic training seed fixed — `hash()` (process-randomised in Python 3.3+) replaced with a deterministic index formula; builds are now fully reproducible
- **Critical:** `engineer_features()` refactored from O(n²) `df.at[]` loop to a list-accumulator pattern — ~4× faster on 15k rows
- **Critical:** H2H dict keys changed from `frozenset` (unreliable pickle) to `tuple(sorted(...))` — consistent between `model_training.py` and `main.py`
- **Critical:** H2H key mismatch between training and inference fixed — every prediction was silently falling back to the 0.45 default
- **Critical:** `_get_classes()` now handles string class labels from `LabelEncoder` — predictions no longer return `"Unknown"`
- **High:** Calibrator was referencing the wrong model when XGBoost was not the best candidate; fixed variable scoping
- **High:** Two-stage Docker build — build tools (`gcc`, `g++`) no longer leak into the runtime image
- **High:** Docker `CMD` now reads `$PORT` env var — compatible with Render's dynamic port assignment
- **Medium:** `FormTracker.factor()` denominator corrected for early-season rows (was always `3*n`, now `3*len(recent)`)
- **Medium:** `k_factor` in ELO raised from 20 → 32 (standard football value); ratings now converge correctly
- **Medium:** Probability values rounded to 4dp in API responses

---

## Features

- Predicts **Home Win / Draw / Away Win** probabilities for any fixture
- Displays **ELO ratings** for both teams with animated reveal
- 120 teams across 6 leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie)
- 26 engineered features: ELO, xG, form, shots on target, possession, corners, cards, H2H history
- Three competing models evaluated per run; best selected automatically and probability-calibrated

---

## 🚀 Deployment

> Full step-by-step instructions in **[DEPLOYMENT.md](./DEPLOYMENT.md)**.

| Layer | Platform | Notes |
|---|---|---|
| Backend (FastAPI + ML) | [Render](https://render.com) — Docker | Root Dir: `backend` |
| Frontend (React + Vite) | [Vercel](https://vercel.com) | Root Dir: `frontend` |

**Quick steps:**
1. Push repo to GitHub
2. Render Web Service → Root Dir: `backend` → Environment: `Docker` → set `ALLOWED_ORIGINS`
3. Vercel Project → Root Dir: `frontend` → set `VITE_API_BASE_URL`
4. Done ✅

---

## Project Structure

```
football-predictor/
├── DEPLOYMENT.md
├── README.md
├── .gitignore
├── backend/
│   ├── data_generator.py     # Generates 15,960 synthetic matches (6 leagues, 7 seasons)
│   ├── model_training.py     # Feature engineering + trains LR / RF / XGBoost
│   ├── main.py               # FastAPI — /teams, /predict, /health
│   ├── utils.py              # ELO calculation (k=32)
│   ├── requirements.txt
│   ├── Dockerfile            # Two-stage build; generates data + trains at build time
│   ├── .env.example
│   └── run_all.bat           # Windows one-click local setup
└── frontend/
    ├── src/
    │   ├── App.jsx           # Redesigned UI — animated counters, reveal panel, responsive picker
    │   ├── index.css         # Barlow Condensed + DM Sans, electric lime palette, fluid layout
    │   └── main.jsx
    ├── public/
    │   └── vercel.json       # SPA rewrite rule
    ├── index.html
    ├── vite.config.js
    ├── package.json
    └── .env.example
```

---

## Local Development

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python data_generator.py        # ~15,960 matches → football_data.csv
python model_training.py        # trains models → football_model.pkl + current_state.pkl
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API: `http://localhost:8000` · Docs: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev                     # proxies /api → localhost:8000
```

App: `http://localhost:5173`

### Windows shortcut

```bat
cd backend
run_all.bat
```

---

## Environment Variables

### Backend (`backend/.env`)

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `football_model.pkl` | Path to trained model |
| `STATE_PATH` | `current_state.pkl` | Path to ELO + stats + H2H state |
| `ALLOWED_ORIGINS` | `*` | CORS origins — set to your Vercel URL in production |
| `PORT` | `8000` | Server port |

### Frontend (`frontend/.env`)

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | *(empty)* | Backend URL. Leave empty for local dev (Vite proxy handles it). Set to Render URL in production. |

---

## API Reference

### `GET /health`
```json
{ "ready": true, "teams_loaded": 120 }
```

### `GET /teams`
```json
{ "teams": ["AC Milan", "Ajax", "Almeria", "..."] }
```

### `POST /predict`

**Request:**
```json
{ "home_team": "Arsenal", "away_team": "Liverpool" }
```

**Response:**
```json
{
  "home_win_probability": 0.4712,
  "draw_probability":     0.2104,
  "away_win_probability": 0.3184,
  "prediction":           "Home Win",
  "home_elo":             1534.2,
  "away_elo":             1521.8
}
```

**Error responses:**

| Status | Reason |
|---|---|
| `404` | Team not found |
| `422` | Same team selected for home and away |
| `503` | Model not loaded |

---

## Model Details

### Data

| Property | Value |
|---|---|
| Leagues | Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie |
| Seasons | 2018–19 through 2024–25 (7 seasons) |
| Teams | 120 |
| Matches | 15,960 |

### Features (26)

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

### Models

| Model | Notes |
|---|---|
| Logistic Regression | StandardScaler pipeline, `C=0.5`, 5-fold CV |
| Random Forest | 300 estimators, max depth 12 |
| XGBoost | RandomizedSearchCV (20 iterations, 3-fold CV) |

Best model by weighted F1 is selected and wrapped in **isotonic calibration** fitted on a dedicated hold-out split.

### Performance

| Metric | v1 | v2 | v3 |
|---|---|---|---|
| Training matches | 2,000 | 15,960 | 15,960 |
| Features | 8 | 26 | 26 |
| F1 Score (weighted) | 0.39 | 0.49 | 0.49 |
| ELO k-factor | — | 20 | **32** |
| H2H at inference | ❌ | ✅ (fixed) | ✅ |
| Reproducible builds | ❌ | ❌ | ✅ |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, uvicorn |
| ML | scikit-learn, XGBoost, pandas, numpy, scipy |
| Frontend | React 19, Vite 8, Barlow Condensed + DM Sans |
| Containerisation | Docker (two-stage build) |
| Hosting (backend) | Render |
| Hosting (frontend) | Vercel |

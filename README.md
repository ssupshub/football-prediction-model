# AI Football Match Predictor

An end-to-end Machine Learning football outcome prediction system featuring:

- **Models**: Logistic Regression, Random Forest, XGBoost (best F1 score wins)
- **Features**: ELO ratings, rolling form, average goals scored/conceded
- **Simulation**: Poisson distribution goal model (standalone helper)
- **Backend**: Python 3.10 + FastAPI
- **Frontend**: React 19 + Vite with glassmorphism UI

---

## Project Structure

```
.
├── backend/
│   ├── data_generator.py   # Generates synthetic historical data
│   ├── model_training.py   # Trains & evaluates all models; saves best
│   ├── main.py             # FastAPI application
│   ├── utils.py            # ELO calculation helper
│   ├── requirements.txt    # All Python dependencies (incl. ML libs)
│   └── Dockerfile
└── frontend/
    ├── src/
    │   ├── App.jsx         # Main React component
    │   └── index.css       # Glassmorphism styles
    ├── vite.config.js      # Dev proxy → localhost:8000
    ├── .env.example        # Copy to .env for production URL
    └── package.json
```

---

## Local Setup & Development

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate

pip install -r requirements.txt  # Installs FastAPI, sklearn, xgboost, pandas …
python data_generator.py         # Creates football_data.csv
python model_training.py         # Trains models, saves .pkl files
uvicorn main:app --reload        # API live at http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev       # App live at http://localhost:5173
```

The Vite dev server proxies `/api/*` → `http://localhost:8000/*` automatically —
no CORS issues and no hardcoded URLs needed during development.

---

## Deployment

### Backend (Render / Docker)

1. Push code to GitHub.
2. Render Dashboard → **New → Web Service → Docker**.
3. Connect repo; Render uses the `Dockerfile` automatically.
4. The image builds, generates data, trains models, then starts Uvicorn on port 8000.

> **Optional env vars you can set in Render:**
> - `ALLOWED_ORIGINS` — comma-separated list of allowed frontend origins (default `*`)

### Frontend (Vercel)

1. Vercel Dashboard → **Add New → Project → Import repo**.
2. Set **Root Directory** to `frontend`.
3. Add environment variable:
   ```
   VITE_API_BASE_URL=https://your-backend.onrender.com
   ```
4. Deploy — Vercel detects Vite automatically.

---

## API Reference

| Method | Endpoint  | Description                     |
|--------|-----------|----------------------------------|
| GET    | `/`       | Health check                     |
| GET    | `/health` | Model readiness + team count     |
| GET    | `/teams`  | List all available teams         |
| POST   | `/predict`| Predict match outcome            |

### POST `/predict`

```json
// Request
{ "home_team": "Arsenal", "away_team": "Liverpool" }

// Response
{
  "home_win_probability": 0.42,
  "draw_probability": 0.24,
  "away_win_probability": 0.34,
  "prediction": "Home Win"
}
```

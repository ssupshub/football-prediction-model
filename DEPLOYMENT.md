# 🚀 Deployment Guide — AI Football Match Predictor

> **Backend → Render** | **Frontend → Vercel**

This guide walks you through deploying the full-stack Football Match Predictor from zero to a live, publicly accessible URL. Every step explains **what** you're doing, **why** it matters, and **how** to do it.

---

## 📋 Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 — Push Your Code to GitHub](#3-step-1--push-your-code-to-github)
4. [Step 2 — Deploy the Backend on Render](#4-step-2--deploy-the-backend-on-render)
5. [Step 3 — Deploy the Frontend on Vercel](#5-step-3--deploy-the-frontend-on-vercel)
6. [Step 4 — Connect Frontend to Backend](#6-step-4--connect-frontend-to-backend)
7. [Step 5 — Verify the Live App](#7-step-5--verify-the-live-app)
8. [Environment Variables Reference](#8-environment-variables-reference)
9. [Troubleshooting](#9-troubleshooting)
10. [How It All Fits Together](#10-how-it-all-fits-together)

---

## 1. Overview & Architecture

### What are we deploying?

| Layer | Technology | Host |
|---|---|---|
| **Backend API** | Python · FastAPI · ML model (scikit-learn / XGBoost) | [Render](https://render.com) |
| **Frontend UI** | React 19 · Vite | [Vercel](https://vercel.com) |

### Why Render for the backend?

- **Docker support** — our backend uses a `Dockerfile` that generates training data and trains the model at build time. Render's Docker environment handles this perfectly.
- **Free tier** — Render's free web service is sufficient for demo/portfolio use.
- **Persistent build artifacts** — the trained `.pkl` model files survive inside the container between requests.
- **Zero server management** — no need to configure Nginx, gunicorn workers, or SSL certificates manually.

### Why Vercel for the frontend?

- **Vite/React native support** — Vercel auto-detects Vite projects and configures the build with zero config.
- **Global CDN** — static assets are served from edge nodes worldwide, making the UI fast for all users.
- **Free tier** — unlimited personal projects on the free Hobby plan.
- **Preview deployments** — every Git push to a branch gets its own preview URL automatically.

### How they communicate

```
User's browser
      │
      ▼
┌─────────────────┐         HTTPS POST /predict
│  Vercel (React) │  ──────────────────────────►  ┌──────────────────────┐
│  Static Files   │                                │  Render (FastAPI)    │
│  + CDN          │  ◄──────────────────────────  │  ML Model + ELO Data │
└─────────────────┘         JSON Response          └──────────────────────┘
```

The React frontend calls the FastAPI backend over HTTPS. The backend runs inside a Docker container on Render, with the trained model already baked in at build time.

---

## 2. Prerequisites

Before starting, make sure you have:

- [ ] A **GitHub account** — [github.com](https://github.com)
- [ ] A **Render account** — [render.com](https://render.com) (sign up free, no credit card needed)
- [ ] A **Vercel account** — [vercel.com](https://vercel.com) (sign up free with GitHub)
- [ ] **Git** installed locally — [git-scm.com](https://git-scm.com)
- [ ] Your project code on your local machine (the `football-predictor/` folder)

> **Why GitHub?** Both Render and Vercel integrate directly with GitHub. When you push code, they automatically re-deploy — no manual uploads needed.

---

## 3. Step 1 — Push Your Code to GitHub

### What & Why

Both Render and Vercel pull your code directly from a GitHub repository. You need to push your project there first.

### How

**3.1 — Create a new GitHub repository**

1. Go to [github.com/new](https://github.com/new)
2. Name it `football-predictor` (or anything you like)
3. Set visibility to **Public** or **Private** (both work)
4. **Do NOT** initialise with a README, `.gitignore`, or licence (you already have them)
5. Click **Create repository**

**3.2 — Initialise Git and push**

Open a terminal in your `football-predictor/` folder:

```bash
# Initialise a git repository
git init

# Stage all files
git add .

# Create the first commit
git commit -m "Initial commit — AI Football Match Predictor"

# Point to your new GitHub repo (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/football-predictor.git

# Push to GitHub
git push -u origin main
```

> **Why commit before pushing?** Git requires at least one commit before it can push. The `-u origin main` flag sets the default remote branch so future `git push` commands need no arguments.

**3.3 — Verify**

Refresh your GitHub repository page. You should see all your files listed there.

---

## 4. Step 2 — Deploy the Backend on Render

### What & Why

Render will pull your repository, build the Docker image (which runs `data_generator.py` and `model_training.py` to produce the trained model), and start the FastAPI server. This means every fresh deployment automatically retrains the model from scratch — no pre-built `.pkl` files need to be committed to Git.

### How

**4.1 — Create a new Web Service**

1. Log in to [render.com](https://render.com)
2. Click **New +** in the top-right corner
3. Select **Web Service**

**4.2 — Connect your GitHub repository**

1. Click **Connect account** next to GitHub (if not already connected)
2. Authorise Render to access your repositories
3. Search for `football-predictor` and click **Connect**

**4.3 — Configure the service**

Fill in the settings form:

| Field | Value | Why |
|---|---|---|
| **Name** | `football-predictor-api` | Becomes part of your URL |
| **Region** | Choose nearest to you | Lower latency |
| **Branch** | `main` | Deploys from your main branch |
| **Root Directory** | `backend` | Tells Render where the `Dockerfile` lives |
| **Environment** | `Docker` | Uses your `Dockerfile` instead of buildpacks |
| **Instance Type** | `Free` | Sufficient for demo purposes |

> **Why Root Directory = `backend`?** Your repository root contains both `backend/` and `frontend/` folders. Setting Root Directory to `backend` means Render only looks inside that folder for the `Dockerfile` and source files.

> **Why Docker environment?** The `Dockerfile` does heavy work — it installs GCC, runs `data_generator.py` (~15,960 match simulation), and trains three ML models. Docker ensures this runs in a controlled, reproducible environment.

**4.4 — Add Environment Variables**

Scroll to the **Environment Variables** section and add:

| Key | Value | Why |
|---|---|---|
| `ALLOWED_ORIGINS` | `https://YOUR-APP.vercel.app` | Restricts CORS to your frontend only (fill this in after Vercel deployment; use `*` for now) |

> **What is CORS?** Cross-Origin Resource Sharing. By default, browsers block API calls from one domain to another. Setting `ALLOWED_ORIGINS` to your Vercel URL tells the FastAPI backend to allow requests from your frontend. Using `*` allows all origins — acceptable during setup but should be tightened for production.

**4.5 — Deploy**

Click **Create Web Service**.

Render will now:
1. Clone your repository
2. Build the Docker image (`docker build`)
3. Run `python data_generator.py` → generates `football_data.csv`
4. Run `python model_training.py` → trains models, saves `football_model.pkl` and `current_state.pkl`
5. Start the server with `uvicorn main:app --host 0.0.0.0 --port 8000`

> ⏱️ **The first build takes 8–15 minutes** because it installs Python packages, generates 15,960 match records, and trains three ML models including an XGBoost hyperparameter search. Subsequent deploys are faster because Docker layer caching skips unchanged steps.

**4.6 — Confirm it's running**

Once the build log shows `==> Your service is live 🎉`, click the URL at the top of the page (e.g. `https://football-predictor-api.onrender.com`).

You should see:

```json
{"status": "ok", "message": "Football Match Predictor API v2"}
```

Also test the health endpoint:

```
https://football-predictor-api.onrender.com/health
```

Expected response:

```json
{"ready": true, "teams_loaded": 120}
```

And the teams list:

```
https://football-predictor-api.onrender.com/teams
```

> 📝 **Copy your Render URL** — you will need it in Step 4.

---

## 5. Step 3 — Deploy the Frontend on Vercel

### What & Why

Vercel builds the React/Vite app into static HTML, CSS, and JavaScript files, then serves them from a global CDN. The app is then accessible at a `vercel.app` subdomain instantly.

### How

**5.1 — Import your project**

1. Log in to [vercel.com](https://vercel.com)
2. Click **Add New…** → **Project**
3. Under **Import Git Repository**, find `football-predictor` and click **Import**

**5.2 — Configure the project**

Vercel will detect it as a monorepo (backend + frontend folders). Configure:

| Field | Value | Why |
|---|---|---|
| **Root Directory** | `frontend` | Tells Vercel where `package.json` and `vite.config.js` live |
| **Framework Preset** | `Vite` (auto-detected) | Uses Vite's build command and output directory |
| **Build Command** | `npm run build` (auto-filled) | Runs `vite build` to produce the `dist/` folder |
| **Output Directory** | `dist` (auto-filled) | Where Vite puts the compiled static files |

> **Why Root Directory = `frontend`?** Without this, Vercel would look for `package.json` in the repository root and fail to find it.

**5.3 — Add Environment Variables**

Still on the configuration screen, expand **Environment Variables** and add:

| Key | Value | Why |
|---|---|---|
| `VITE_API_BASE_URL` | `https://football-predictor-api.onrender.com` | Tells the React app where to send API requests. Replace with your actual Render URL. |

> **Why the `VITE_` prefix?** Vite only exposes environment variables prefixed with `VITE_` to the browser bundle at build time. Variables without this prefix are ignored for security — you don't want server secrets accidentally shipped to the browser.

> **Why is this needed in production but not locally?** Locally, `vite.config.js` has a dev proxy that rewrites `/api/*` to `http://localhost:8000/*`. This proxy only works during local development (`npm run dev`). In production, the built static files are served from Vercel's CDN with no proxy — the frontend needs the full backend URL to make direct HTTPS calls.

**5.4 — Deploy**

Click **Deploy**.

Vercel will:
1. Install Node.js dependencies (`npm install`)
2. Run `npm run build` (Vite compiles React → static files in `dist/`)
3. Upload the `dist/` folder to Vercel's global CDN
4. Assign you a URL like `https://football-predictor-xyz.vercel.app`

> ⏱️ **This takes about 1–2 minutes** — much faster than the backend because it's just compiling JavaScript.

**5.5 — Fix page-refresh 404 (important)**

By default, Vercel serves a 404 for any URL that isn't a file (e.g. if the user refreshes on a sub-route). For a single-page app, all routes should fall back to `index.html`.

Create the file `frontend/public/vercel.json`:

```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/" }]
}
```

Then commit and push:

```bash
git add frontend/public/vercel.json
git commit -m "Add Vercel SPA rewrite rule"
git push
```

Vercel will automatically re-deploy within seconds.

> **Why is this needed?** React Router (or any client-side router) handles navigation in JavaScript. When a user lands directly on `/predict` or refreshes, the browser asks Vercel's server for that path. Without the rewrite rule, the server returns 404 because `/predict` isn't a real file. The rewrite tells Vercel to always serve `index.html`, and React takes over routing from there.

---

## 6. Step 4 — Connect Frontend to Backend

### What & Why

Now that both services are live, you need to make sure the frontend points to the correct backend URL, and the backend accepts requests from the frontend domain.

### How

**6.1 — Update CORS on the backend (Render)**

1. Go to your Render service dashboard
2. Click **Environment** in the left sidebar
3. Update the `ALLOWED_ORIGINS` variable:

| Key | Value |
|---|---|
| `ALLOWED_ORIGINS` | `https://football-predictor-xyz.vercel.app` |

Replace `football-predictor-xyz.vercel.app` with your actual Vercel URL.

4. Click **Save Changes** — Render will automatically redeploy

> **Why update this now?** During setup you used `*` as a placeholder. Restricting CORS to your exact Vercel domain prevents other websites from abusing your API endpoint.

**6.2 — Confirm the environment variable on Vercel**

1. Go to your Vercel project dashboard
2. Click **Settings** → **Environment Variables**
3. Confirm `VITE_API_BASE_URL` is set to your Render URL (e.g. `https://football-predictor-api.onrender.com`)
4. If you need to change it, edit and then go to **Deployments** → click the three dots on the latest deployment → **Redeploy**

---

## 7. Step 5 — Verify the Live App

### What & Why

Before sharing the link, run a quick end-to-end check to confirm every piece works together.

### How

**7.1 — Test the backend directly**

Open these URLs in your browser (replace with your Render URL):

```
✅ https://football-predictor-api.onrender.com/
   → {"status":"ok","message":"Football Match Predictor API v2"}

✅ https://football-predictor-api.onrender.com/health
   → {"ready":true,"teams_loaded":120}

✅ https://football-predictor-api.onrender.com/teams
   → {"teams":["AC Milan","Ajax","Almeria",...]}
```

**7.2 — Test a prediction via the API docs**

1. Go to `https://football-predictor-api.onrender.com/docs`
2. Expand `POST /predict`
3. Click **Try it out**
4. Enter:
   ```json
   {
     "home_team": "Arsenal",
     "away_team": "Liverpool"
   }
   ```
5. Click **Execute**
6. You should get a `200 OK` response with probabilities and ELO ratings

**7.3 — Test the full frontend**

1. Open your Vercel URL (e.g. `https://football-predictor-xyz.vercel.app`)
2. The team dropdowns should populate with 120 teams
3. Select a home and away team
4. Click **Predict Match Outcome**
5. Probability bars and ELO ratings should appear

> ⚠️ **First request after inactivity may be slow (20–30 seconds).** Render's free tier spins down services after 15 minutes of no traffic. The first request "wakes" the container. Subsequent requests are fast. This is a free-tier limitation — paid Render plans stay always-on.

---

## 8. Environment Variables Reference

### Backend (`backend/.env` / Render Dashboard)

| Variable | Default | Required | Description |
|---|---|---|---|
| `MODEL_PATH` | `football_model.pkl` | No | Path to the trained model pickle file |
| `STATE_PATH` | `current_state.pkl` | No | Path to the ELO + team stats + H2H state file |
| `ALLOWED_ORIGINS` | `*` | **Yes (production)** | Comma-separated list of allowed frontend origins. Use your Vercel URL. |
| `PORT` | `8000` | No | Port the uvicorn server listens on |

### Frontend (`frontend/.env` / Vercel Dashboard)

| Variable | Default | Required | Description |
|---|---|---|---|
| `VITE_API_BASE_URL` | *(empty — uses Vite proxy)* | **Yes (production)** | Full URL of the deployed backend. Leave empty for local dev. |

---

## 9. Troubleshooting

### Backend issues

**Problem:** Render build fails with `pip install` errors  
**Why:** Python package compilation (e.g. numpy, scipy) requires GCC  
**Fix:** The `Dockerfile` already installs `gcc` and `g++`. If you modified the Dockerfile, ensure these lines are present:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*
```

---

**Problem:** `/health` returns `{"ready": false, "teams_loaded": 0}`  
**Why:** The model training step failed or the `.pkl` files weren't created  
**Fix:** Check Render's build logs for errors from `python model_training.py`. Common cause: the `football_data.csv` file wasn't generated by `data_generator.py` before training started. The `Dockerfile` runs them in sequence — verify the order:
```dockerfile
RUN python data_generator.py
RUN python model_training.py
```

---

**Problem:** `CORS error` in the browser console  
**Why:** The backend is rejecting requests from the frontend domain  
**Fix:** Update `ALLOWED_ORIGINS` on Render to include your exact Vercel URL:
```
https://your-app-name.vercel.app
```
Trigger a redeploy on Render after saving the change.

---

**Problem:** First request takes 30+ seconds  
**Why:** Render's free tier puts services to sleep after 15 minutes of inactivity  
**Fix (free tier):** This is expected behaviour. Add a note to your UI or use a service like [UptimeRobot](https://uptimerobot.com) (free) to ping `/health` every 14 minutes and keep it awake  
**Fix (permanent):** Upgrade to Render's Starter plan ($7/month) for always-on deployment

---

### Frontend issues

**Problem:** Teams dropdown is empty / `Could not connect to the backend` error  
**Why:** `VITE_API_BASE_URL` is missing or incorrect  
**Fix:**
1. Go to Vercel → Project → Settings → Environment Variables
2. Confirm `VITE_API_BASE_URL` is set to your full Render URL with no trailing slash:
   ```
   https://football-predictor-api.onrender.com
   ```
3. Redeploy (Deployments → Redeploy)

---

**Problem:** Page shows blank / white screen after deployment  
**Why:** JavaScript build error or missing environment variable  
**Fix:** Open browser DevTools (F12) → Console tab. Look for errors. Most common causes:
- `VITE_API_BASE_URL` not set → app tries to call `/api/teams` which doesn't exist on Vercel's static server
- A syntax error in a source file that only appears in production build

---

**Problem:** Refreshing on any page gives a 404  
**Why:** Vercel can't find a file matching the URL path  
**Fix:** Add `frontend/public/vercel.json` with the SPA rewrite rule (see Step 3.5)

---

**Problem:** `net::ERR_SSL_PROTOCOL_ERROR` or mixed content warning  
**Why:** Frontend (HTTPS) is trying to call backend over HTTP  
**Fix:** Ensure your `VITE_API_BASE_URL` starts with `https://` not `http://`

---

## 10. How It All Fits Together

Here's the complete data flow from a user clicking **Predict** to seeing results:

```
1. User opens https://your-app.vercel.app
        │
        ▼
2. Vercel CDN serves index.html + React bundle (static files)
        │
        ▼
3. React app loads in browser
   → Calls GET https://football-predictor-api.onrender.com/teams
        │
        ▼
4. Render wakes up Docker container (if sleeping)
   → FastAPI loads football_model.pkl + current_state.pkl
   → Returns list of 120 teams
        │
        ▼
5. User selects teams and clicks Predict
   → React calls POST /predict with { home_team, away_team }
        │
        ▼
6. FastAPI backend:
   → Looks up ELO ratings from current_state.pkl
   → Looks up rolling stats (form, xG, shots, etc.)
   → Looks up real H2H win rate history
   → Builds 26-feature input vector
   → Runs calibrated ML model (best of LR / RF / XGBoost)
   → Returns probabilities + ELO ratings as JSON
        │
        ▼
7. React receives JSON response
   → Updates state
   → Renders animated probability bars + ELO chips
        │
        ▼
8. User sees prediction ✅
```

### Redeployment flow (when you push code changes)

```
git push origin main
        │
        ├──► GitHub notifies Render webhook
        │         → Render rebuilds Docker image
        │         → Regenerates data + retrains model
        │         → Restarts server with new code
        │
        └──► GitHub notifies Vercel webhook
                  → Vercel runs npm run build
                  → Uploads new dist/ to CDN
                  → Live within ~90 seconds
```

---

## Quick Reference Checklist

```
BACKEND (Render)
□ GitHub repo pushed
□ New Web Service created on Render
□ Root Directory set to: backend
□ Environment set to: Docker
□ ALLOWED_ORIGINS env var set to your Vercel URL
□ First deploy successful (8–15 min)
□ /health returns {"ready": true, "teams_loaded": 120}

FRONTEND (Vercel)
□ New Project imported on Vercel
□ Root Directory set to: frontend
□ VITE_API_BASE_URL set to your Render URL
□ Deployed successfully (~2 min)
□ Teams dropdown populates
□ Prediction works end-to-end
□ vercel.json rewrite added (prevents 404 on refresh)
```

---

*Guide version: 2.1 — covers football-predictor-fixed.zip with all bug fixes applied*

# AI Football Match Predictor

An end-to-end Machine Learning football outcome prediction system. It features advanced modeling (Random Forest, XGBoost, Poisson Distributions, ELO systems) built with Python and FastAPI, served via a beautiful glassmorphism React GUI.

## Project Structure

- **/backend**: Python FastAPI application exposing the ML model endpoints.
- **/frontend**: React/Vite frontend application.

## Local Setup & Development

### Backend

1. Navigate to the `backend` directory: `cd backend`
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. Install dependencies: `pip install -r requirements.txt`
4. Generate the mock historical dataset: `python data_generator.py`
5. Train the Machine Learning models: `python model_training.py`
6. Start the FastAPI server: `uvicorn main:app --reload`
   - API will be live at `http://localhost:8000`

### Frontend

1. Navigate to the `frontend` directory: `cd frontend`
2. Install dependencies: `npm install`
3. Start the Vite development server: `npm run dev`
   - Application will be live at `http://localhost:5173`

## Deployment Instructions

### Backend (Render / Docker)

The backend is Dockerized and ready for deployment on services like Render.

1. **GitHub**: Push your code to a GitHub repository.
2. **Render Dashbaord**: Log into Render and click **New > Web Service**.
3. **Connect Repo**: Connect your GitHub repository.
4. **Environment**: Choose **Docker** as the Environment.
5. **Build & Deploy**: Render will automatically use the provided `Dockerfile` to install dependencies, generate data, train the models, and expose the FastAPI service on Port 8000.
   - _Note_: Ensure you update the frontend `fetch` URLs in `App.jsx` to point to your new Render URL once deployed.

### Frontend (Vercel)

The Vite/React frontend is optimized for zero-config Vercel deployment.

1. **Vercel Dashboard**: Log into Vercel and click **Add New > Project**.
2. **Import**: Import the GitHub repository containing this project.
3. **Framework Preset**: Vercel should automatically detect it as a **Vite** project.
4. **Root Directory**: Click "Edit" on Root Directory and select the `frontend` folder.
5. **Deploy**: Click Deploy. Vercel will run `npm run build` and publish your premium UI instantly.

> **Important Setup Note**: Before deploying the frontend to production, make sure to update the API endpoints in `frontend/src/App.jsx` from `http://localhost:8000` to your live rendered backend URL!

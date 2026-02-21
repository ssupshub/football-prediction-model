from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="Football Match Predictor API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development, can restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and latest state (elos, stats)
try:
    with open('football_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        features = model_data['features']
        
    with open('current_state.pkl', 'rb') as f:
        current_state = pickle.load(f)
        elos = current_state['elos']
        stats = current_state['stats']
except FileNotFoundError:
    print("Warning: Model or state files not found. Please run model_training.py first.")
    model, features, elos, stats = None, None, {}, {}

class MatchRequest(BaseModel):
    home_team: str
    away_team: str

class PredictionResponse(BaseModel):
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    prediction: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Football Match Predictor API. Use POST /predict"}

@app.get("/teams")
def get_teams():
    if not elos:
        return {"teams": []}
    return {"teams": sorted(list(elos.keys()))}

@app.post("/predict", response_model=PredictionResponse)
def predict_match(request: MatchRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
        
    home = request.home_team
    away = request.away_team
    
    if home not in elos or away not in elos:
         raise HTTPException(status_code=400, detail="One or both teams not found in historical data.")
         
    if home == away:
         raise HTTPException(status_code=400, detail="Home and Away teams cannot be the same.")
         
    # Engineer features for prediction based on current state
    home_elo = elos[home]
    away_elo = elos[away]
    
    home_hist = stats[home][-5:]
    away_hist = stats[away][-5:]
    
    home_form = sum([m['points'] for m in home_hist]) if home_hist else 0.0
    away_form = sum([m['points'] for m in away_hist]) if away_hist else 0.0
    
    home_avg_sc = sum([m['scored'] for m in stats[home][-10:]]) / min(10, max(1, len(stats[home])))
    home_avg_con = sum([m['conceded'] for m in stats[home][-10:]]) / min(10, max(1, len(stats[home])))
    
    away_avg_sc = sum([m['scored'] for m in stats[away][-10:]]) / min(10, max(1, len(stats[away])))
    away_avg_con = sum([m['conceded'] for m in stats[away][-10:]]) / min(10, max(1, len(stats[away])))
    
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'Home_ELO': home_elo,
        'Away_ELO': away_elo,
        'Home_Form': home_form,
        'Away_Form': away_form,
        'Home_Avg_Scored': home_avg_sc,
        'Home_Avg_Conceded': home_avg_con,
        'Away_Avg_Scored': away_avg_sc,
        'Away_Avg_Conceded': away_avg_con
    }])[features] # Ensure correct order
    
    try:
        # Prediction mapping: 0=Away, 1=Draw, 2=Home
        probabilities = model.predict_proba(input_data)[0]
        
        # Scikit-learn outputs probabilities in the order of classes seen during training (typically 0, 1, 2)
        # Verify the class order attribute if available
        classes = list(model.classes_)
        
        # Map class probabilities
        prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
        
        away_prob = float(prob_dict.get(0, 0.0))
        draw_prob = float(prob_dict.get(1, 0.0))
        home_prob = float(prob_dict.get(2, 0.0))
        
        # Determine prediction string
        max_prob_class = max(prob_dict, key=prob_dict.get)
        pred_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        prediction_str = pred_map.get(max_prob_class, "Unknown")
        
        return PredictionResponse(
            home_win_probability=home_prob,
            draw_probability=draw_prob,
            away_win_probability=away_prob,
            prediction=prediction_str
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

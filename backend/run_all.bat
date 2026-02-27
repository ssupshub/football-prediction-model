@echo off
setlocal enabledelayedexpansion

echo [1/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment. Run: python -m venv venv
    pause
    exit /b 1
)

echo [2/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo [3/4] Generating training data...
python data_generator.py
if errorlevel 1 (
    echo ERROR: data_generator.py failed.
    pause
    exit /b 1
)

echo [4/4] Training model...
python model_training.py
if errorlevel 1 (
    echo ERROR: model_training.py failed.
    pause
    exit /b 1
)

echo Starting API server on http://localhost:8000 ...
start /b uvicorn main:app --host 0.0.0.0 --port 8000

echo Done. API available at http://localhost:8000
echo Press any key to stop the server.
pause >nul
taskkill /f /im uvicorn.exe >nul 2>&1

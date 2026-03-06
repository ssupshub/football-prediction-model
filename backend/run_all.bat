@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo  Football Match Predictor — Local Setup
echo ================================================================
echo.

REM ── [1/5] Virtual environment ─────────────────────────────────────
echo [1/5] Activating virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo   venv not found — creating it now...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo        Make sure Python 3.10+ is installed and on PATH.
        pause
        exit /b 1
    )
)
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)
echo   OK.

REM ── [2/5] Dependencies ────────────────────────────────────────────
echo.
echo [2/5] Installing dependencies...
pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Check requirements.txt and your internet connection.
    pause
    exit /b 1
)
echo   OK.

REM ── [3/5] Training data ───────────────────────────────────────────
echo.
echo [3/5] Generating training data (this may take ~30 s) ...
python data_generator.py
if errorlevel 1 (
    echo ERROR: data_generator.py failed. See output above.
    pause
    exit /b 1
)
echo   OK.

REM ── [4/5] Model training ─────────────────────────────────────────
echo.
echo [4/5] Training models (this may take 3-10 min) ...
python model_training.py
if errorlevel 1 (
    echo ERROR: model_training.py failed. See output above.
    pause
    exit /b 1
)
echo   OK.

REM ── [5/5] Start API server ────────────────────────────────────────
echo.
echo [5/5] Starting API server on http://localhost:8000 ...
echo       Press Ctrl+C to stop.
echo.
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

REM uvicorn exits cleanly on Ctrl+C; errorlevel is not reliable here.
echo.
echo Server stopped.
pause

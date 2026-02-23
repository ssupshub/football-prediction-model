@echo off
call venv\Scripts\activate.bat
pip install -r requirements.txt
python data_generator.py
python model_training.py
start /b uvicorn main:app --host 0.0.0.0 --port 8000
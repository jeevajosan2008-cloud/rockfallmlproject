@echo off
TITLE Rockfall Risk AI System - Streamlit Launcher
echo ======================================================================
echo Launching Explainable AI Rockfall Risk Prediction Application...
echo ======================================================================

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo Error: Virtual environment not found in venv\
    pause
    exit /b 1
)

echo Activating Python 3.12 Virtual Environment...
call venv\Scripts\activate.bat

echo Starting Streamlit on http://localhost:8501...
start "" "http://localhost:8501"
venv\Scripts\streamlit.exe run app.py

pause

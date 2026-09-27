@echo off
TITLE Rockfall Risk AI - Dual Deployment Launcher
echo ======================================================================
echo Launching Complete System: Backend API (8000) & Frontend UI (8501)...
echo ======================================================================

REM Check virtual environment
if not exist "venv\Scripts\python.exe" (
    echo Error: Virtual environment not found in venv\
    pause
    exit /b 1
)

echo [1/2] Starting FastAPI Backend on http://localhost:8000...
start "FastAPI Backend" cmd /k "venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo [2/2] Starting Streamlit Dashboard on http://localhost:8501...
start "Streamlit Frontend" cmd /k "venv\Scripts\streamlit.exe run app.py --server.port 8501"

timeout /t 3 /nobreak >nul

echo Opening Frontends in your browser...
start "" "http://localhost:8000"
start "" "http://localhost:8501"

echo ======================================================================
echo DEPLOYMENT ACTIVE:
echo 1. Modern Web Frontend: http://localhost:8000
echo 2. Swagger OpenAPI Docs: http://localhost:8000/docs
echo 3. Streamlit Dashboard:  http://localhost:8501
echo ======================================================================
pause

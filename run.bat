@echo off
echo ===================================================
echo Starting Production Hybrid RAG Application Stack
echo ===================================================

echo [1/2] Launching FastAPI Backend on http://127.0.0.1:8000 ...
start "FastAPI Backend" cmd /k "python -m uvicorn src.main:app --host 127.0.0.1 --port 8000"

echo Waiting for backend initialization...
timeout /t 4 /nobreak >nul

echo [2/2] Launching Streamlit UI on http://127.0.0.1:8501 ...
start "Streamlit UI" cmd /k "python -m streamlit run src/app.py"

echo ===================================================
echo Done! Open http://localhost:8501 in your browser.
echo ===================================================


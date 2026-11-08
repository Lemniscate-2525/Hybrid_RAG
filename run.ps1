Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting Production Hybrid RAG Application Stack" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

Write-Host "`n[1/2] Launching FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m uvicorn src.main:app --host 127.0.0.1 --port 8000"

Start-Sleep -Seconds 4

Write-Host "[2/2] Launching Streamlit UI on http://127.0.0.1:8501 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m streamlit run src/app.py"

Write-Host "`nAll services launched! Opening http://localhost:8501 in your browser..." -ForegroundColor Yellow
Start-Process "http://localhost:8501"


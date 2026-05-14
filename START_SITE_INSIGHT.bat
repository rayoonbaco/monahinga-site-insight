@echo off
title Site Insight / Mona Work
cd /d "%~dp0"

echo ==========================================
echo Starting Site Insight / Mona Work...
echo ==========================================
echo.
echo Local URL: http://127.0.0.1:8000
echo.
echo Opening browser...
echo Keep this window open. Press CTRL+C to stop.
echo ==========================================

start "" "http://127.0.0.1:8000"

.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000

@echo off
title Site Insight / Mona Work - LOCAL SCIENCE LAB
cd /d "%~dp0"

set MONAHINGA_LOCAL_SCIENCE_LAB=1

echo ==========================================
echo Starting Site Insight / Mona Work
echo LOCAL HIGH-DETAIL SCIENCE LAB MODE
echo ==========================================
echo.
echo Render remains protected. This local mode uses bigger source/viewer payloads.
echo Local URL: http://127.0.0.1:8000
echo.
echo Opening browser...
echo Keep this window open. Press CTRL+C to stop.
echo ==========================================

start "" "http://127.0.0.1:8000"

.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000

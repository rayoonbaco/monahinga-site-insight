# Site Insight / Mona Work - Start Here

## Correct local startup

Double-click `START_SITE_INSIGHT.bat`, or run:

```cmd
cd /d "C:\Users\Owner\Desktop\ENOUGH\HUNTER (speed enhanced)\mona_work"
.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Safe cleanup

To reduce zip size without breaking the app, run:

```cmd
cd /d "C:\Users\Owner\Desktop\ENOUGH\HUNTER (speed enhanced)\mona_work"
.venv\Scripts\python.exe CLEAN_MONA_WORK_SAFE.py
```

This removes generated run outputs and Python cache only. Do not delete `.venv`, `siteinsight`, `templates`, `static`, or `requirements.txt`.

## Current archaeology synthesis status

The app has an Archaeology Layer Synthesis challenge-mode panel. A fresh run should create:

```text
reports/runs/<run_id>/archaeology_synthesis.json
```

Success signs:
- homepage loads
- run completes
- 3D viewer still loads
- old layers still appear
- archaeology synthesis panel appears
- candidate feature cards use cautious language

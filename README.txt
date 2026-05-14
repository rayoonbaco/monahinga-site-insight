Monahinga Archaeology Terrain Viewer

What is included
- Cleaned project folder only
- No virtual environment
- No old generated run folders
- No cached Copernicus DEM tile downloads
- No embedded secrets
- Live map, bbox draw flow, terrain pipeline, and 3D viewer path kept intact

What you need before running
- Python 3.12 installed
- AWS CLI installed if you want Copernicus global fallback to work

Quick start from Command Prompt opened inside this folder
1) py -3.12 -m venv .venv
2) .venv\Scriptsctivate
3) python -m pip install --upgrade pip
4) pip install -r requirements.txt
5) uvicorn app:app --reload --port 8000
6) Open http://127.0.0.1:8000

Notes
- The .env file is a blank template now.
- reports/_vendor must stay present because the generated 3D viewer depends on those local files.
- reports/runs starts empty on purpose. New runs will be generated there.

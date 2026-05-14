from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path.cwd()
SAFE_DIRS = [
    ROOT / 'reports' / 'runs',
    ROOT / '__pycache__',
    ROOT / 'siteinsight' / '__pycache__',
]
SAFE_PATTERNS = ['*.pyc', '*.pyo']

PROTECTED = {'.venv', 'siteinsight', 'templates', 'static', 'app.py', 'requirements.txt'}

print('[Mona Work Safe Cleaner]')
print(f'Project folder: {ROOT}')
print('This removes generated run outputs and Python cache only.')
print('It does NOT remove .venv, source code, static files, templates, DEM source code, or requirements.')

if not (ROOT / 'app.py').exists() or not (ROOT / 'siteinsight').exists():
    raise SystemExit('[STOP] Run this from the main mona_work folder beside app.py.')

removed = 0
for d in SAFE_DIRS:
    if d.exists() and d.is_dir():
        shutil.rmtree(d)
        print(f'[removed dir] {d}')
        removed += 1

# Keep reports folder alive.
reports = ROOT / 'reports'
reports.mkdir(exist_ok=True)
(reports / '.gitkeep').write_text('', encoding='utf-8')

for pattern in SAFE_PATTERNS:
    for p in ROOT.rglob(pattern):
        if '.venv' in p.parts:
            continue
        try:
            p.unlink()
            print(f'[removed cache] {p}')
            removed += 1
        except Exception as exc:
            print(f'[skip] {p}: {exc}')

print(f'[DONE] Safe cleanup complete. Removed {removed} generated/cache items.')
print('Next startup command: START_SITE_INSIGHT.bat')

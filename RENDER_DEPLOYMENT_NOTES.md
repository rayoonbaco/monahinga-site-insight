# Render Deployment Notes - Monahinga / Site Insight

This file was created by `patch_render_deployment_manifest.py`.

## What this pass adds

- `Procfile`
- `render.yaml`
- `.python-version`
- safer `.gitignore` rules for `.env`
- preserved `reports/_vendor/` exceptions for the 3D viewer files

## What this pass does not touch

- terrain generation
- DEM selection
- terrain conditioning
- derivative layers
- 3D viewer code
- templates
- scoring / archaeology synthesis
- generated run output

## Render start command

```text
uvicorn app:app --host 0.0.0.0 --port $PORT
```

## Health check

```text
/health
```

## Safe Git commands after local testing

Do not use `git add .`.

Use:

```cmd
cd /d "C:\Users\Owner\Desktop\ENOUGH\HUNTER (speed enhanced)\mona_work"
git status
git add .gitignore Procfile render.yaml .python-version RENDER_DEPLOYMENT_NOTES.md
git add -f reports/_vendor/three.min.js reports/_vendor/OrbitControls.js
git commit -m "Add Render deployment manifest"
git push
```

## After Render deploys

Test in this order:

1. Homepage loads.
2. `/launch` loads.
3. `/health` returns OK.
4. Static CSS loads.
5. Tiny BBox run finishes.
6. Run page loads.
7. Separate 3D viewer opens.
8. Render events show no memory restart.

## Stop if

- Render shows 502.
- Render events say memory exceeded.
- vendor files return 404.
- visuals degrade compared with the good local zip.

# Render Launch Checklist - Monahinga / Site Insight

This checklist was created by `patch_render_launch_hygiene.py`.

## What this patch changed

- Updated `.gitignore` so generated run output stays out of Git.
- Added explicit exceptions for the local 3D viewer vendor files:
  - `reports/_vendor/three.min.js`
  - `reports/_vendor/OrbitControls.js`

## What this patch did not touch

Protected systems not touched by this patch:
- terrain generation
- DEM sourcing
- derivative layers
- 3D viewer logic
- run page template
- launch page template
- archaeology synthesis
- scoring/intelligence logic
- reports/runs generated output

## Local smoke test

Run this from Command Prompt:

```cmd
cd /d "C:\Users\Owner\Desktop\ENOUGH\HUNTER (speed enhanced)\mona_work"
START_SITE_INSIGHT.bat
```

Then open:

```text
http://127.0.0.1:8010
http://127.0.0.1:8010/launch
```

## Safe Git commands after local testing

Do not use `git add .`.

Use:

```cmd
cd /d "C:\Users\Owner\Desktop\ENOUGH\HUNTER (speed enhanced)\mona_work"
git status
git add .gitignore RENDER_LAUNCH_CHECKLIST.md
git add -f reports/_vendor/three.min.js reports/_vendor/OrbitControls.js
git commit -m "Prepare Render launch vendor assets"
git push
```

## Render smoke-test ladder

1. Homepage loads.
2. `/launch` loads.
3. Static CSS loads.
4. Start a tiny BBox run.
5. Run page loads without crash screen.
6. Derivative grid looks sane.
7. Separate 3D viewer opens.
8. Render logs show no restart and no out-of-memory event.
9. Only then test a normal BBox.

## Stop conditions

Stop and inspect logs if:

- Render shows a 502.
- Render event panel says memory exceeded.
- `/reports/_vendor/three.min.js` or `/reports/_vendor/OrbitControls.js` returns 404.
- Terrain becomes speckled, spiky, corrupted, or visibly worse than local.

from __future__ import annotations

from pathlib import Path
from typing import Optional
import threading

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from siteinsight.credits import CreditStore
from siteinsight.exporter import export_run_zip
from siteinsight.global_dem import load_provider_settings, provider_ready, save_copernicus_settings
from siteinsight.pins import add_pin, load_pins
from siteinsight.pipeline import run_analysis
from siteinsight.utils import APP_NAME, DATA_DIR, REPORTS_DIR, ROOT_DIR, RUNS_DIR, SiteInsightError, ensure_dir, list_runs, load_json, make_run_id, parse_bbox_string, safe_float, save_json, utc_stamp

def _launch_background_analysis(*, run_id: str, parsed_bbox: tuple[float, float, float, float], run_name: str, persona: str, notes: str) -> None:
    run_dir = ensure_dir(RUNS_DIR / run_id)

    def worker() -> None:
        try:
            run_analysis(
                bbox=parsed_bbox,
                run_name=run_name,
                persona=persona,
                notes=notes,
                run_id=run_id,
                run_dir=run_dir,
            )
        except Exception as exc:
            manifest = load_json(run_dir / "manifest.json", {}) or {}
            manifest.update({
                "run_id": run_id,
                "run_name": run_name,
                "created_at": manifest.get("created_at") or utc_stamp(),
                "persona": persona,
                "notes": notes,
                "bbox": manifest.get("bbox") or {
                    "min_lon": parsed_bbox[0],
                    "min_lat": parsed_bbox[1],
                    "max_lon": parsed_bbox[2],
                    "max_lat": parsed_bbox[3],
                },
                "status": "error",
                "error_message": str(exc),
            })
            save_json(run_dir / "manifest.json", manifest)

    threading.Thread(target=worker, name=f"monahinga-{run_id}", daemon=True).start()

app = FastAPI(title=APP_NAME)
templates = Jinja2Templates(directory=str(ROOT_DIR / "templates"))

static_dir = ROOT_DIR / "static"
static_dir.mkdir(exist_ok=True, parents=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

credits = CreditStore(ROOT_DIR / "credits.json")



def _bytes_to_mb(value: int) -> float:
    return round(float(value or 0) / (1024 * 1024), 2)


def _safe_file_size(path: Path) -> int:
    try:
        return path.stat().st_size if path.exists() and path.is_file() else 0
    except OSError:
        return 0


def _build_run_performance_audit(run_dir: Path) -> dict:
    # Lightweight run-size audit for result-page speed decisions.
    key_names = [
        "Terrain_Texture.png",
        "Hillshade.png",
        "Slope.png",
        "LRM.png",
        "LRM_Edges.png",
        "Openness_Pos.png",
        "Openness_Neg.png",
        "SVF.png",
        "Archaeology.png",
        "Discovery.png",
        "Elevation.png",
        "viewer3d.html",
        "terrain_payload.json",
        "manifest.json",
        "archaeology_synthesis.json",
    ]

    key_files = []
    largest_file = {"name": "none", "mb": 0.0}
    total_png_bytes = 0
    png_count = 0
    direct_total_bytes = 0

    try:
        direct_files = [p for p in run_dir.iterdir() if p.is_file()]
    except OSError:
        direct_files = []

    for p in direct_files:
        size = _safe_file_size(p)
        direct_total_bytes += size
        if p.suffix.lower() == ".png":
            total_png_bytes += size
            png_count += 1
        mb = _bytes_to_mb(size)
        if mb > largest_file.get("mb", 0):
            largest_file = {"name": p.name, "mb": mb}

    for name in key_names:
        p = run_dir / name
        if p.exists():
            key_files.append({"name": name, "mb": _bytes_to_mb(_safe_file_size(p))})

    total_png_mb = _bytes_to_mb(total_png_bytes)
    direct_total_mb = _bytes_to_mb(direct_total_bytes)

    if total_png_mb >= 60 or direct_total_mb >= 100:
        status = "heavy"
        recommendation = "Use separate 3D viewer, keep thumbnails lazy, and consider smaller derivative preview exports next."
    elif total_png_mb >= 25 or direct_total_mb >= 50:
        status = "moderate"
        recommendation = "Run is acceptable, but PNG payload is worth watching before adding more layers."
    else:
        status = "light"
        recommendation = "Run payload is healthy enough for the current result page."

    return {
        "status": status,
        "recommendation": recommendation,
        "png_count": png_count,
        "total_png_mb": total_png_mb,
        "direct_total_mb": direct_total_mb,
        "largest_file": largest_file,
        "key_files": key_files,
    }


def _clean_run_asset_name(filename: str) -> str:
    cleaned = Path(filename).name
    if cleaned != filename or cleaned.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    allowed = {".png", ".jpg", ".jpeg", ".webp"}
    if Path(cleaned).suffix.lower() not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported preview file type.")
    return cleaned


def _build_cached_preview(src: Path, dest: Path, width: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime and dest.stat().st_size > 0:
        return dest

    try:
        from PIL import Image
    except Exception:
        return src

    try:
        width = max(240, min(int(width or 900), 1600))
        with Image.open(src) as im:
            im.load()
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
            if im.mode == "RGBA":
                background = Image.new("RGB", im.size, (8, 15, 28))
                background.paste(im, mask=im.getchannel("A"))
                im = background
            else:
                im = im.convert("RGB")

            im.thumbnail((width, width), Image.Resampling.LANCZOS)
            try:
                im.save(dest, "WEBP", quality=74, method=4)
            except Exception:
                jpg_dest = dest.with_suffix(".jpg")
                im.save(jpg_dest, "JPEG", quality=78, optimize=True)
                return jpg_dest
        return dest
    except Exception:
        return src

def _run_dir(run_id: str) -> Path:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found.")
    return run_dir


def _index_context(request: Request, message: str = "") -> dict:
    return {
        "request": request,
        "app_name": APP_NAME,
        "credits_balance": credits.get_balance(),
        "recent_runs": list_runs(limit=10),
        "message": message,
        "provider_ready": provider_ready(DATA_DIR),
    }



@app.get("/launch", response_class=HTMLResponse)
async def launch_preview(request: Request):
    return templates.TemplateResponse(
        "launch.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "recent_runs": list_runs(limit=3),
            "provider_ready": provider_ready(DATA_DIR),
        },
    )

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", _index_context(request))


@app.post("/analyze")
async def analyze(
    request: Request,
    bbox: str = Form(...),
    run_name: str = Form(""),
    persona: str = Form("archaeologist"),
    notes: str = Form(""),
    consume_credit: Optional[str] = Form(None),
):
    try:
        parsed_bbox = parse_bbox_string(bbox)
    except ValueError as exc:
        return templates.TemplateResponse("index.html", _index_context(request, f"Invalid bbox: {exc}"), status_code=400)

    if consume_credit and not credits.consume(1):
        return templates.TemplateResponse("index.html", _index_context(request, "Not enough credits to create a new run."), status_code=400)

    clean_run_name = run_name.strip() or "Monahinga Survey"
    clean_persona = (persona.strip() or "archaeologist").lower()
    clean_notes = notes.strip()
    run_id = make_run_id(clean_run_name)
    run_dir = ensure_dir(RUNS_DIR / run_id)

    placeholder_manifest = {
        "run_id": run_id,
        "run_name": clean_run_name,
        "created_at": utc_stamp(),
        "persona": clean_persona,
        "notes": clean_notes,
        "bbox": {
            "min_lon": parsed_bbox[0],
            "min_lat": parsed_bbox[1],
            "max_lon": parsed_bbox[2],
            "max_lat": parsed_bbox[3],
        },
        "status": "processing",
        "progress_message": "Queued terrain acquisition and derivative build.",
    }
    save_json(run_dir / "manifest.json", placeholder_manifest)
    _launch_background_analysis(
        run_id=run_id,
        parsed_bbox=parsed_bbox,
        run_name=clean_run_name,
        persona=clean_persona,
        notes=clean_notes,
    )
    return RedirectResponse(url=f"/runs/{run_id}", status_code=303)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
@app.get("/run/{run_id}", response_class=HTMLResponse)
async def run_view(request: Request, run_id: str):
    run_dir = _run_dir(run_id)
    manifest = load_json(run_dir / "manifest.json", {})
    if manifest.get("status") == "processing":
        return templates.TemplateResponse("processing.html", {"request": request, "app_name": APP_NAME, "run_id": run_id, "manifest": manifest})
    if manifest.get("status") == "error":
        return templates.TemplateResponse("processing.html", {"request": request, "app_name": APP_NAME, "run_id": run_id, "manifest": manifest})
    terrain = load_json(run_dir / "terrain_summary.json", {})
    terrain_qc = load_json(run_dir / "terrain_qc.json", {})
    archaeology = load_json(run_dir / "archaeology.json", {})
    archaeology_synthesis = load_json(run_dir / "archaeology_synthesis.json", {})
    intelligence = load_json(run_dir / "intelligence.json", {})
    pins_summary = load_json(run_dir / "pins_summary.json", {})
    brand_lab = load_json(run_dir / "brand_lab.json", {})
    source_arbitration = load_json(run_dir / "source_arbitration.json", {})
    conditioning_summary = ""
    conditioning_path = run_dir / "conditioning_summary.txt"
    if conditioning_path.exists():
        conditioning_summary = conditioning_path.read_text(encoding="utf-8", errors="ignore")
    thumb_catalog = [
        ("Terrain Texture", "Terrain_Texture.png"),
        ("Hillshade", "Hillshade.png"),
        ("Slope", "Slope.png"),
        ("Local Relief", "LRM.png"),
        ("LRM + Edges", "LRM_Edges.png"),
        ("Openness +", "Openness_Pos.png"),
        ("Openness -", "Openness_Neg.png"),
        ("SVF", "SVF.png"),
        ("Archaeology", "Archaeology.png"),
        ("Discovery", "Discovery.png"),
        ("Elevation", "Elevation.png"),
    ]
    thumb_files = [(label, filename) for label, filename in thumb_catalog if (run_dir / filename).exists()]
    missing_thumbs = [(label, filename) for label, filename in thumb_catalog if not (run_dir / filename).exists()]
    performance_audit = _build_run_performance_audit(run_dir)
    artifact_audit = load_json(run_dir / "artifact_audit.json", {})
    layer_names = ["terrain_texture", "elevation", "hillshade", "slope", "local_relief", "openness", "srv", "archaeology", "discovery"]
    artifact_truth = manifest.get("artifact_truth", {}) or {}
    inspection_mode = artifact_truth.get("inspection_mode") or manifest.get("inspection_mode") or "balanced_detail"
    browser_grid = int(artifact_truth.get("browser_grid_actual") or manifest.get("browser_grid") or 0)
    terrain_confidence = float(artifact_truth.get("terrain_confidence") or manifest.get("terrain_confidence") or 0.0)
    guarded_3d = inspection_mode == "high_detail" or browser_grid >= 1152 or terrain_confidence >= 0.78
    return templates.TemplateResponse(
        "run.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "run_id": run_id,
            "manifest": manifest,
            "terrain": terrain,
            "terrain_qc": terrain_qc,
            "archaeology": archaeology,
            "archaeology_synthesis": archaeology_synthesis,
            "intelligence": intelligence,
            "pins_summary": pins_summary,
            "brand_lab": brand_lab,
            "conditioning_summary": conditioning_summary,
            "thumb_files": thumb_files,
            "missing_thumbs": missing_thumbs,
            "performance_audit": performance_audit,
            "source_arbitration": source_arbitration,
            "artifact_truth": artifact_truth,
            "artifact_audit": artifact_audit,
            "layer_names": layer_names,
            "guarded_3d": guarded_3d,
        },
    )



@app.get("/runs/{run_id}/preview/{filename}")
async def run_preview_image(run_id: str, filename: str, width: int = 900):
    run_dir = _run_dir(run_id)
    safe_name = _clean_run_asset_name(filename)
    src = run_dir / safe_name
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=404, detail="Preview source not found.")

    width = max(240, min(int(width or 900), 1600))
    preview_dir = run_dir / "_previews"
    preview_name = f"{Path(safe_name).stem}_{width}.webp"
    preview_path = _build_cached_preview(src, preview_dir / preview_name, width)

    media_type = "image/webp" if preview_path.suffix.lower() == ".webp" else "image/jpeg" if preview_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return FileResponse(
        preview_path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )

@app.get("/owner", response_class=HTMLResponse)
async def owner_view(request: Request):
    settings = load_provider_settings(DATA_DIR)
    return templates.TemplateResponse(
        "owner.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "credits_balance": credits.get_balance(),
            "runs": list_runs(limit=100),
            "provider_ready": provider_ready(DATA_DIR),
            "provider_settings": settings,
        },
    )


@app.post("/owner/add-credits")
async def owner_add_credits(amount: str = Form(...)):
    amt = int(safe_float(amount, 0))
    if amt > 0:
        credits.add(amt)
    return RedirectResponse(url="/owner", status_code=303)


@app.post("/owner/save-provider")
async def owner_save_provider(client_id: str = Form(""), client_secret: str = Form("")):
    save_copernicus_settings(DATA_DIR, client_id=client_id, client_secret=client_secret)
    return RedirectResponse(url="/owner", status_code=303)


@app.get("/brand/{run_id}", response_class=HTMLResponse)
async def brand_lab_view(request: Request, run_id: str):
    run_dir = _run_dir(run_id)
    manifest = load_json(run_dir / "manifest.json", {})
    intelligence = load_json(run_dir / "intelligence.json", {})
    brand_lab = load_json(run_dir / "brand_lab.json", {})
    return templates.TemplateResponse(
        "brand_lab.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "run_id": run_id,
            "manifest": manifest,
            "intelligence": intelligence,
            "brand_lab": brand_lab,
        },
    )


@app.get("/runs/{run_id}/download")
async def download_run(run_id: str):
    run_dir = _run_dir(run_id)
    zip_path = export_run_zip(run_dir)
    return FileResponse(str(zip_path), filename=zip_path.name, media_type="application/zip")


@app.get("/api/runs/{run_id}/heightmap")
async def run_heightmap(run_id: str):
    run_dir = _run_dir(run_id)
    return JSONResponse(load_json(run_dir / "heightmap.json", {}))


@app.get("/api/runs/{run_id}/layers")
async def run_layers(run_id: str):
    run_dir = _run_dir(run_id)
    return JSONResponse(load_json(run_dir / "viewer_layers.json", {}))


@app.get("/api/runs/{run_id}/pins")
async def get_pins(run_id: str):
    run_dir = _run_dir(run_id)
    return JSONResponse(load_pins(run_dir))


@app.post("/api/runs/{run_id}/pins")
async def post_pin(
    run_id: str,
    label: str = Form(...),
    pin_type: str = Form(...),
    lat: str = Form(...),
    lon: str = Form(...),
    notes: str = Form(""),
):
    run_dir = _run_dir(run_id)
    add_pin(
        run_dir=run_dir,
        label=label.strip(),
        pin_type=pin_type.strip(),
        lat=safe_float(lat, 0.0),
        lon=safe_float(lon, 0.0),
        notes=notes.strip(),
    )
    return RedirectResponse(url=f"/runs/{run_id}", status_code=303)


@app.get("/api/runs/{run_id}/manifest")
@app.get("/api/run/{run_id}")
async def run_manifest(run_id: str):
    run_dir = _run_dir(run_id)
    return JSONResponse(load_json(run_dir / "manifest.json", {}))


@app.get("/health")
async def health():
    return JSONResponse({"ok": True, "app": APP_NAME})


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse({"ok": True})

from __future__ import annotations

import gc
import os
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

from siteinsight.archaeology import detect_archaeology_candidates
from siteinsight.archaeology_synthesis import build_archaeology_layer_synthesis
from siteinsight.derivatives import compute_derivatives, summarize_derivatives
from siteinsight.global_dem import fetch_dem_for_provider, list_candidate_providers
from siteinsight.heightmap import _smooth_grid, build_heightmap_payload
from siteinsight.terrain_normalization import normalize_source
from siteinsight.intelligence import build_intelligence
from siteinsight.terrain_graph import build_terrain_graph
from siteinsight.terrain_objects import extract_terrain_objects
from siteinsight.pins import save_pins
from siteinsight.rendering_policy import select_rendering_policy
from siteinsight.terrain_classification import classify_surface
from siteinsight.terrain_conditioning import condition_surface
from siteinsight.terrain_models import TerrainSource, TerrainSurface
from siteinsight.terrain_qc import qualify_source
from siteinsight.source_arbitration import SourceCandidate, build_arbitration_summary, score_candidate
from siteinsight.utils import (
    DATA_DIR,
    RUNS_DIR,
    VENDOR_DIR,
    SiteInsightError,
    bbox_area_deg,
    bbox_center,
    ensure_dir,
    make_run_id,
    save_json,
    save_text,
    utc_stamp,
)

DEFAULT_REQUEST_GRID = 896
DEFAULT_WORKING_GRID = 768
DEFAULT_BROWSER_GRID = 896

# Render Starter has a 512MB memory ceiling. Local desktop runs keep the original
# quality path; Render gets a smaller public-proof path so /analyze survives.
RENDER_SAFE_REQUEST_GRID_CAP = 448
RENDER_SAFE_WORKING_GRID_CAP = 384
RENDER_SAFE_BROWSER_GRID_CAP = 384
RENDER_SAFE_VIEWER_JSON_CAP = 320
RENDER_SAFE_LAYER_JSON_CAP = 320

# Local-only science lab lane. Render never uses this lane.
LOCAL_SCIENCE_REQUEST_GRID_CAP = 1280
LOCAL_SCIENCE_WORKING_GRID_CAP = 1024
LOCAL_SCIENCE_BROWSER_GRID_CAP = 1280
LOCAL_SCIENCE_VIEWER_JSON_CAP = 768
LOCAL_SCIENCE_LAYER_JSON_CAP = 768


def _monahinga_render_safe_mode() -> bool:
    value = (os.environ.get("MONAHINGA_RENDER_SAFE") or "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return any(os.environ.get(name) for name in (
        "RENDER",
        "RENDER_SERVICE_ID",
        "RENDER_EXTERNAL_HOSTNAME",
        "RENDER_EXTERNAL_URL",
        "RENDER_INSTANCE_ID",
    ))


def _monahinga_local_science_lab_mode() -> bool:
    value = (os.environ.get("MONAHINGA_LOCAL_SCIENCE_LAB") or "").strip().lower()
    return value in {"1", "true", "yes", "on"} and not _monahinga_render_safe_mode()


def _render_cap_grid(value: int, cap: int) -> int:
    value = max(64, int(value or 0))
    if _monahinga_render_safe_mode():
        return min(value, int(cap))
    return value


def _lab_cap_grid(value: int, cap: int) -> int:
    value = max(64, int(value or 0))
    if _monahinga_local_science_lab_mode():
        return max(value, int(cap))
    return value


def _active_layer_json_cap(default_cap: int) -> int:
    if _monahinga_render_safe_mode():
        return RENDER_SAFE_LAYER_JSON_CAP
    if _monahinga_local_science_lab_mode():
        return LOCAL_SCIENCE_LAYER_JSON_CAP
    return int(default_cap)



def _bbox_size_m(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2.0
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = max(1.0, 111320.0 * np.cos(np.deg2rad(center_lat)))
    width_m = max(0.0, (max_lon - min_lon) * meters_per_deg_lon)
    height_m = max(0.0, (max_lat - min_lat) * meters_per_deg_lat)
    return float(width_m), float(height_m)


def _adaptive_working_grid(bbox: tuple[float, float, float, float]) -> int:
    width_m, height_m = _bbox_size_m(bbox)
    dominant_m = max(width_m, height_m)
    if dominant_m < 2500:
        return 640
    if dominant_m < 6000:
        return 768
    if dominant_m < 10000:
        return 896
    if dominant_m < 15000:
        return 1024
    return 1152


def _adaptive_request_grid(working_grid: int) -> int:
    if working_grid <= 640:
        return 896
    if working_grid <= 768:
        return 1024
    if working_grid <= 896:
        return 1152
    if working_grid <= 1024:
        return 1280
    return 1536



def _adaptive_browser_grid(bbox: tuple[float, float, float, float]) -> int:
    width_m, height_m = _bbox_size_m(bbox)
    dominant_m = max(width_m, height_m)
    if dominant_m < 1600:
        return 1152
    if dominant_m < 2500:
        return 1024
    if dominant_m < 6000:
        return 960
    if dominant_m < 10000:
        return 896
    if dominant_m < 18000:
        return 768
    if dominant_m < 30000:
        return 704
    return 640


def _provider_working_grid(bbox: tuple[float, float, float, float], provider: str) -> int:
    width_m, height_m = _bbox_size_m(bbox)
    dominant_m = max(width_m, height_m)
    provider = (provider or "copernicus").lower()
    if provider == "usgs":
        if dominant_m < 2500:
            return 768
        if dominant_m < 6000:
            return 640
        if dominant_m < 10000:
            return 512
        return 448
    if dominant_m < 2500:
        return 512
    if dominant_m < 6000:
        return 448
    if dominant_m < 10000:
        return 384
    return 320


def _provider_request_grid(bbox: tuple[float, float, float, float], provider: str) -> int:
    working = _provider_working_grid(bbox, provider)
    provider = (provider or "copernicus").lower()
    if provider == "usgs":
        return min(1024, working + 192)
    return min(640, working + 128)


def _provider_browser_grid(bbox: tuple[float, float, float, float], provider: str) -> int:
    provider = (provider or "copernicus").lower()
    width_m, height_m = _bbox_size_m(bbox)
    dominant_m = max(width_m, height_m)
    if provider == "usgs":
        if dominant_m < 3500:
            return 1152
        if dominant_m < 9000:
            return 1024
        if dominant_m < 18000:
            return 896
        if dominant_m < 32000:
            return 768
        return 704
    if dominant_m < 3500:
        return 1024
    if dominant_m < 9000:
        return 896
    if dominant_m < 18000:
        return 768
    if dominant_m < 32000:
        return 704
    return 640


def _norm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _hillshade(dem: np.ndarray, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    gy, gx = np.gradient(dem)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(gx * gx + gy * gy))
    aspect = np.arctan2(-gx, gy)
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(altitude)
    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return _norm(shaded)


def _robust_norm(arr: np.ndarray, lo_q: float = 2.0, hi_q: float = 98.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    vals = arr[finite]
    lo = float(np.percentile(vals, lo_q))
    hi = float(np.percentile(vals, hi_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _unsharp(arr: np.ndarray, amount: float = 0.55, smooth_iters: int = 1) -> np.ndarray:
    base = np.asarray(arr, dtype=float)
    blur = _smooth_grid(base, max(0, int(smooth_iters)))
    sharpened = base + float(amount) * (base - blur)
    return np.clip(sharpened, 0.0, 1.0)


def _local_relief(dem: np.ndarray) -> np.ndarray:
    broad = _smooth_grid(dem, 6)
    medium = _smooth_grid(dem, 3)
    fine = _smooth_grid(dem, 1)
    relief = 0.55 * np.abs(dem - medium) + 0.30 * np.abs(dem - broad) + 0.15 * np.abs(dem - fine)
    return _robust_norm(relief, 5.0, 99.0)


def _openness(dem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    broad = _smooth_grid(dem, 5)
    medium = _smooth_grid(dem, 2)
    diff = dem - broad
    pos = np.clip(diff, 0.0, None) + 0.35 * np.clip(dem - medium, 0.0, None)
    neg = np.clip(-diff, 0.0, None) + 0.35 * np.clip(-(dem - medium), 0.0, None)
    return _robust_norm(pos, 5.0, 99.0), _robust_norm(neg, 5.0, 99.0)


def _srv_like(derivatives: dict[str, Any]) -> np.ndarray:
    ridge = _robust_norm(derivatives["ridge_strength"], 5.0, 99.0)
    valley = _robust_norm(derivatives["valley_strength"], 5.0, 99.0)
    edge = _robust_norm(derivatives["edge_strength"], 5.0, 99.0)
    return np.clip(0.42 * ridge + 0.28 * valley + 0.30 * edge, 0.0, 1.0)


def _rgb_for_layer(layer_name: str, value: float) -> tuple[int, int, int]:
    v = max(0.0, min(1.0, float(value)))
    if layer_name == "hillshade":
        g = round(20 + v * 220)
        return g, g, g
    if layer_name == "elevation":
        if v < 0.18:
            return 24, 60 + round(v * 100), 110 + round(v * 60)
        if v < 0.38:
            return 35 + round(v * 50), 90 + round(v * 90), 55
        if v < 0.62:
            return 95 + round(v * 70), 105 + round(v * 60), 55 + round(v * 25)
        if v < 0.82:
            return 145 + round(v * 55), 125 + round(v * 40), 95 + round(v * 30)
        return 220, 220, 220
    if layer_name == "slope":
        return 120 + round(v * 80), 70 + round(v * 45), 35 + round(v * 25)
    if layer_name == "local_relief":
        return 40 + round(v * 35), 95 + round(v * 80), 120 + round(v * 70)
    if layer_name == "openness":
        return 95 + round(v * 55), 135 + round(v * 55), 140 + round(v * 45)
    if layer_name == "srv":
        return 30 + round(v * 40), 30 + round(v * 40), 45 + round(v * 120)
    if layer_name == "archaeology":
        return 85 + round(v * 120), 45 + round(v * 55), 20 + round(v * 30)
    if layer_name == "discovery":
        return 30 + round(v * 40), 120 + round(v * 90), 140 + round(v * 90)
    g = round(20 + v * 220)
    return g, g, g


def _write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def _scientific_tone_curve(matrix: np.ndarray, low: float = 2.0, high: float = 98.0, gamma: float = 0.86) -> np.ndarray:
    """Balanced percentile tone curve for professional terrain products.

    Keeps relief readable without blowing ridges white or burying valleys black.
    The source matrix remains scientific input; this only changes exported display tone.
    """
    arr = np.asarray(matrix, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)
    lo = float(np.nanpercentile(finite, low))
    hi = float(np.nanpercentile(finite, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        v = np.clip(arr, 0.0, 1.0)
    else:
        v = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    v = np.power(v, gamma)
    # Lift midtones and compress the top end so ridge crests do not glare.
    v = 0.12 + 0.82 * v
    v = np.where(v > 0.88, 0.88 + (v - 0.88) * 0.45, v)
    return np.clip(v, 0.0, 1.0)


def _balanced_terrain_texture_matrix(layers_np: dict[str, np.ndarray]) -> np.ndarray:
    """Create a faster, calmer field-first terrain texture from existing layers.

    This deliberately avoids changing DEMs, derivatives, scoring, or archaeology math.
    It only creates a more readable display composite for the 3D viewer and raster card.
    """
    hill = _scientific_tone_curve(layers_np["hillshade"], gamma=0.82)
    relief = _scientific_tone_curve(layers_np["local_relief"], low=4.0, high=96.0, gamma=0.92)
    slope = _scientific_tone_curve(layers_np["slope"], low=5.0, high=97.0, gamma=1.05)
    openness = _scientific_tone_curve(layers_np.get("openness", hill), gamma=0.95)

    # Local relief adds structure, slope adds form, openness protects broad landforms.
    composite = (0.58 * hill) + (0.20 * relief) + (0.12 * slope) + (0.10 * openness)
    composite = np.clip(composite, 0.0, 1.0)
    # Gentle final midtone lift; preserve shadows but avoid the overly black surface.
    composite = 0.08 + 0.86 * np.power(composite, 0.92)
    return np.clip(composite, 0.0, 1.0)


def _matrix_to_rgb(matrix: np.ndarray, layer_name: str) -> np.ndarray:
    v = np.clip(np.asarray(matrix, dtype=float), 0.0, 1.0)
    out = np.zeros(v.shape + (3,), dtype=np.uint8)
    if layer_name in {"hillshade", "terrain_texture"}:
        # Scientific-balanced grayscale: brighter midtones, compressed white ridges.
        vv = _scientific_tone_curve(v, gamma=0.88 if layer_name == "hillshade" else 0.94)
        if layer_name == "terrain_texture":
            g = np.rint(38 + vv * 178).astype(np.uint8)
            out[..., 0] = np.minimum(g.astype(int) + 4, 232).astype(np.uint8)
            out[..., 1] = np.minimum(g.astype(int) + 3, 230).astype(np.uint8)
            out[..., 2] = np.minimum(g.astype(int) + 1, 226).astype(np.uint8)
        else:
            g = np.rint(28 + vv * 202).astype(np.uint8)
            out[..., 0] = g
            out[..., 1] = g
            out[..., 2] = g
        return out
    if layer_name == "elevation":
        m1 = v < 0.18
        m2 = (v >= 0.18) & (v < 0.38)
        m3 = (v >= 0.38) & (v < 0.62)
        m4 = (v >= 0.62) & (v < 0.82)
        m5 = v >= 0.82
        if np.any(m1):
            out[m1] = np.stack([np.full(np.count_nonzero(m1), 24), np.rint(60 + v[m1] * 100), np.rint(110 + v[m1] * 60)], axis=1).astype(np.uint8)
        if np.any(m2):
            out[m2] = np.stack([np.rint(35 + v[m2] * 50), np.rint(90 + v[m2] * 90), np.full(np.count_nonzero(m2), 55)], axis=1).astype(np.uint8)
        if np.any(m3):
            out[m3] = np.stack([np.rint(95 + v[m3] * 70), np.rint(105 + v[m3] * 60), np.rint(55 + v[m3] * 25)], axis=1).astype(np.uint8)
        if np.any(m4):
            out[m4] = np.stack([np.rint(145 + v[m4] * 55), np.rint(125 + v[m4] * 40), np.rint(95 + v[m4] * 30)], axis=1).astype(np.uint8)
        if np.any(m5):
            out[m5] = np.array([220, 220, 220], dtype=np.uint8)
        return out
    if layer_name == "slope":
        vv = _scientific_tone_curve(v, low=4.0, high=98.0, gamma=1.0)
        out[..., 0] = np.rint(112 + vv * 82).astype(np.uint8)
        out[..., 1] = np.rint(72 + vv * 55).astype(np.uint8)
        out[..., 2] = np.rint(42 + vv * 35).astype(np.uint8)
        return out
    if layer_name == "local_relief":
        vv = _scientific_tone_curve(v, low=4.0, high=96.0, gamma=0.98)
        out[..., 0] = np.rint(48 + vv * 38).astype(np.uint8)
        out[..., 1] = np.rint(96 + vv * 76).astype(np.uint8)
        out[..., 2] = np.rint(122 + vv * 76).astype(np.uint8)
        return out
    if layer_name == "openness":
        vv = _scientific_tone_curve(v, gamma=0.95)
        out[..., 0] = np.rint(96 + vv * 52).astype(np.uint8)
        out[..., 1] = np.rint(134 + vv * 58).astype(np.uint8)
        out[..., 2] = np.rint(140 + vv * 48).astype(np.uint8)
        return out
    if layer_name == "srv":
        vv = _scientific_tone_curve(v, gamma=0.95)
        out[..., 0] = np.rint(32 + vv * 42).astype(np.uint8)
        out[..., 1] = np.rint(34 + vv * 44).astype(np.uint8)
        out[..., 2] = np.rint(48 + vv * 118).astype(np.uint8)
        return out
    if layer_name == "archaeology":
        vv = _scientific_tone_curve(v, low=3.0, high=98.0, gamma=0.98)
        out[..., 0] = np.rint(82 + vv * 112).astype(np.uint8)
        out[..., 1] = np.rint(48 + vv * 60).astype(np.uint8)
        out[..., 2] = np.rint(24 + vv * 36).astype(np.uint8)
        return out
    if layer_name == "discovery":
        vv = _scientific_tone_curve(v, low=3.0, high=98.0, gamma=0.96)
        out[..., 0] = np.rint(34 + vv * 44).astype(np.uint8)
        out[..., 1] = np.rint(116 + vv * 88).astype(np.uint8)
        out[..., 2] = np.rint(138 + vv * 84).astype(np.uint8)
        return out
    g = np.rint(28 + _scientific_tone_curve(v) * 202).astype(np.uint8)
    out[..., 0] = g
    out[..., 1] = g
    out[..., 2] = g
    return out


def _read_dem_from_geotiff_bytes(tiff_bytes: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    with MemoryFile(tiff_bytes) as mem:
        with mem.open() as src:
            arr = src.read(1, masked=True).astype("float32")
            meta = {
                "transform": src.transform,
                "crs": None if src.crs is None else str(src.crs),
                "nodata": src.nodata,
                "width": src.width,
                "height": src.height,
                "res": tuple(src.res) if getattr(src, 'res', None) else None,
            }
    if np.ma.isMaskedArray(arr):
        dem = np.asarray(arr.filled(np.nan), dtype=float)
    else:
        dem = np.asarray(arr, dtype=float)
    if dem.ndim != 2 or dem.size == 0:
        raise SiteInsightError("DEM payload did not decode into a usable raster.")
    return dem, meta


def _resample_to_grid(dem: np.ndarray, out_size: int) -> np.ndarray:
    h, w = dem.shape
    transform = rasterio.transform.from_bounds(0, 0, w, h, w, h)
    profile = {"driver": "GTiff", "height": h, "width": w, "count": 1, "dtype": "float32", "transform": transform}
    with MemoryFile() as mem:
        with mem.open(**profile) as ds:
            ds.write(dem.astype("float32"), 1)
            out = ds.read(1, out_shape=(out_size, out_size), resampling=Resampling.lanczos)
    return np.asarray(out, dtype=float)


def _resample_mask_to_grid(mask: np.ndarray, out_size: int) -> np.ndarray:
    arr = _resample_to_grid(np.asarray(mask, dtype=float), out_size)
    return np.asarray(arr >= 0.5, dtype=bool)


def _resample_square_if_needed(arr: np.ndarray, out_size: int, *, is_mask: bool = False) -> np.ndarray:
    matrix = np.asarray(arr)
    if matrix.ndim != 2:
        return matrix
    if matrix.shape[0] == out_size and matrix.shape[1] == out_size:
        return matrix
    if is_mask:
        return _resample_mask_to_grid(matrix, out_size)
    return _resample_to_grid(np.asarray(matrix, dtype=float), out_size)


def _downsample_matrix_for_viewer(arr: Any, max_dim: int = 512) -> np.ndarray:
    """Create a browser-friendly matrix without changing the scientific source products.

    The app already writes PNG products for the full visual product. JSON matrices are
    only for interactive browser drawing and 3D mesh loading, so they should be much
    lighter than the canonical raster exports.
    """
    matrix = np.asarray(arr, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return matrix
    rows, cols = matrix.shape
    max_dim = max(64, int(max_dim))
    if max(rows, cols) <= max_dim:
        return matrix
    target_rows = max(2, int(round(rows * (max_dim / max(rows, cols)))))
    target_cols = max(2, int(round(cols * (max_dim / max(rows, cols)))))
    row_idx = np.linspace(0, rows - 1, target_rows).round().astype(int)
    col_idx = np.linspace(0, cols - 1, target_cols).round().astype(int)
    return matrix[np.ix_(row_idx, col_idx)]


def _build_fast_heightmap_payload(heightmap: dict[str, Any], max_dim: int = 448) -> dict[str, Any]:
    """Write a compact 3D viewer payload so the browser does not download a huge mesh JSON."""
    source_rows = int(heightmap.get('rows') or 0)
    source_cols = int(heightmap.get('cols') or 0)
    out = {k: v for k, v in heightmap.items() if k not in {'values', 'geometry_values'}}
    out['source_rows'] = source_rows
    out['source_cols'] = source_cols
    out['viewer_payload'] = 'fast_downsampled_mesh_json'
    values = _downsample_matrix_for_viewer(heightmap.get('values', []), max_dim=max_dim)
    geometry_values = _downsample_matrix_for_viewer(heightmap.get('geometry_values', heightmap.get('values', [])), max_dim=max_dim)
    if values.ndim == 2 and values.size:
        out['rows'] = int(values.shape[0])
        out['cols'] = int(values.shape[1])
        out['values'] = values.round(3).tolist()
    else:
        out['rows'] = source_rows
        out['cols'] = source_cols
        out['values'] = []
    if geometry_values.ndim == 2 and geometry_values.size:
        out['geometry_values'] = geometry_values.round(3).tolist()
    else:
        out['geometry_values'] = out['values']
    return out


def _build_browser_surface(surface: TerrainSurface, browser_grid: int) -> TerrainSurface:
    source_rows, source_cols = surface.cleaned_dem.shape
    if source_rows == browser_grid and source_cols == browser_grid:
        return surface
    return TerrainSurface(
        raw_dem=_resample_square_if_needed(surface.raw_dem, browser_grid),
        cleaned_dem=_resample_square_if_needed(surface.cleaned_dem, browser_grid),
        valid_mask=_resample_square_if_needed(surface.valid_mask, browser_grid, is_mask=True),
        provider=surface.provider,
        source_name=surface.source_name,
        bbox=surface.bbox,
        transform=surface.transform,
        crs=surface.crs,
        nodata_value=surface.nodata_value,
        pixel_size_x=surface.pixel_size_x,
        pixel_size_y=surface.pixel_size_y,
        nominal_resolution_m=surface.nominal_resolution_m,
        qc=surface.qc,
        terrain_class=surface.terrain_class,
        rendering_policy=surface.rendering_policy,
        raw_stats=surface.raw_stats,
        clean_stats=surface.clean_stats,
        slope_stats=surface.slope_stats,
        relief_stats=surface.relief_stats,
        warnings=surface.warnings,
        derivative_products=surface.derivative_products,
        terrain_objects=surface.terrain_objects,
        intelligence_graph=surface.intelligence_graph,
        normalization_summary=surface.normalization_summary,
        public_product_notes=surface.public_product_notes,
        debug={**surface.debug, 'browser_export_grid': int(browser_grid), 'analysis_grid': int(source_rows), 'analysis_cols': int(source_cols), 'browser_export_policy': 'canonical_final_export'},
    )


def _build_browser_derivatives(derivatives: dict[str, Any], browser_grid: int, source_shape: tuple[int, int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    target_shape = (int(browser_grid), int(browser_grid))
    for key, value in derivatives.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            if tuple(value.shape) == target_shape:
                out[key] = np.asarray(value, dtype=float)
            else:
                out[key] = _resample_square_if_needed(np.asarray(value, dtype=float), browser_grid)
        else:
            out[key] = value
    return out


def _artifact_audit(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = manifest.get("files", {}) or {}
    verified: dict[str, dict[str, Any]] = {}
    missing: dict[str, str] = {}
    for key, rel_name in expected.items():
        path = run_dir / rel_name
        if path.exists():
            verified[key] = {"name": rel_name, "bytes": int(path.stat().st_size)}
        else:
            missing[key] = rel_name
    thumb_names = {
        "Hillshade.png", "Slope.png", "LRM.png", "LRM_Edges.png", "Openness_Pos.png",
        "Openness_Neg.png", "SVF.png", "Archaeology.png", "Discovery.png", "Elevation.png"
    }
    thumbs_verified = sorted(name for name in thumb_names if (run_dir / name).exists())
    thumbs_missing = sorted(name for name in thumb_names if not (run_dir / name).exists())
    audit = {
        "expected_count": len(expected),
        "verified_count": len(verified),
        "missing_count": len(missing),
        "verified": verified,
        "missing": missing,
        "thumbnail_verified": thumbs_verified,
        "thumbnail_missing": thumbs_missing,
        "status": "verified" if not missing else "partial",
    }
    save_json(run_dir / "artifact_audit.json", audit)
    manifest["artifact_audit_summary"] = {
        "status": audit["status"],
        "expected_count": audit["expected_count"],
        "verified_count": audit["verified_count"],
        "missing_count": audit["missing_count"],
        "thumbnail_verified_count": len(thumbs_verified),
        "thumbnail_missing_count": len(thumbs_missing),
    }
    return audit


def _fill_nodata_nearest(dem: np.ndarray, max_iters: int = 256) -> np.ndarray:
    out = dem.copy().astype(float)
    mask = ~np.isfinite(out)
    if not mask.any():
        return out
    for _ in range(max_iters):
        if not mask.any():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros_like(out)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            rolled = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
            valid = np.isfinite(rolled)
            acc += np.where(valid, rolled, 0.0)
            cnt += valid.astype(float)
        fillable = mask & (cnt > 0)
        out[fillable] = acc[fillable] / cnt[fillable]
        mask = ~np.isfinite(out)
    if mask.any():
        fallback = float(np.nanmean(out)) if np.isfinite(np.nanmean(out)) else 0.0
        out[mask] = fallback
    return out


def _safe_percentile(values: np.ndarray, q: float, fallback: float = 0.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(fallback)
    return float(np.percentile(finite, q))


def _local_nanmedian(arr: np.ndarray) -> np.ndarray:
    stack = [
        np.roll(np.roll(arr, dy, axis=0), dx, axis=1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    ]
    return np.nanmedian(np.stack(stack, axis=0), axis=0)


def _terrain_label(confidence: float) -> str:
    if confidence >= 0.82:
        return "strong"
    if confidence >= 0.63:
        return "cleaned"
    if confidence >= 0.40:
        return "weak"
    return "insufficient"


def _recommended_exaggeration(confidence: float, spike_ratio: float, valid_ratio: float) -> float:
    ex = 1.55
    if confidence < 0.82:
        ex = 1.2
    if confidence < 0.63:
        ex = 0.95
    if confidence < 0.40:
        ex = 0.7
    if spike_ratio > 0.02:
        ex = min(ex, 0.85)
    if spike_ratio > 0.05:
        ex = min(ex, 0.6)
    if valid_ratio < 0.7:
        ex = min(ex, 0.75)
    return round(ex, 2)


def _analyze_dem_quality(dem: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(dem, dtype=float)
    finite = np.isfinite(arr)
    valid_ratio = float(finite.mean()) if arr.size else 0.0
    nan_ratio = float((~finite).mean()) if arr.size else 1.0
    stats = {
        "rows": int(arr.shape[0]),
        "cols": int(arr.shape[1]),
        "valid_ratio": round(valid_ratio, 4),
        "nan_ratio": round(nan_ratio, 4),
        "warnings": [],
    }
    if not finite.any():
        stats.update({"raw_min": None, "raw_max": None, "raw_range": 0.0, "p01": None, "p99": None, "raw_mean": None, "raw_std": None})
        stats["warnings"].append("No valid terrain pixels after DEM decode.")
        return stats

    finite_vals = arr[finite]
    raw_min = float(np.min(finite_vals))
    raw_max = float(np.max(finite_vals))
    raw_range = raw_max - raw_min
    raw_std = float(np.std(finite_vals))
    p01 = float(np.percentile(finite_vals, 1))
    p99 = float(np.percentile(finite_vals, 99))
    stats.update({
        "raw_min": round(raw_min, 3),
        "raw_max": round(raw_max, 3),
        "raw_range": round(raw_range, 3),
        "p01": round(p01, 3),
        "p99": round(p99, 3),
        "raw_mean": round(float(np.mean(finite_vals)), 3),
        "raw_std": round(raw_std, 3),
    })
    if valid_ratio < 0.8:
        stats["warnings"].append("Large nodata or masked region detected in DEM.")
    if raw_range < 3:
        stats["warnings"].append("Terrain appears nearly flat after decode.")
    if raw_min < -600 or raw_max > 10000:
        stats["warnings"].append("Suspicious raw elevation values detected.")
    return stats


def _condition_dem(dem: np.ndarray) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    arr = np.asarray(dem, dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    arr[(arr < -600.0) | (arr > 10000.0)] = np.nan

    qc = _analyze_dem_quality(arr)
    valid_initial = np.isfinite(arr)
    if not valid_initial.any():
        raise SiteInsightError("DEM contained no usable elevation values after nodata cleanup.")

    finite_vals = arr[valid_initial]
    p_low = float(np.percentile(finite_vals, 0.5))
    p_high = float(np.percentile(finite_vals, 99.5))
    clipped = arr.copy()
    clipped[valid_initial] = np.clip(clipped[valid_initial], p_low, p_high)
    outlier_mask = valid_initial & ((arr < p_low) | (arr > p_high))

    neighborhood_median = _local_nanmedian(clipped)
    diff = np.abs(clipped - neighborhood_median)
    diff[np.isnan(diff)] = 0.0
    mad = _safe_percentile(diff[np.isfinite(diff)], 75, fallback=1.0)
    spike_threshold = max(18.0, mad * 4.5)
    spike_mask = valid_initial & np.isfinite(neighborhood_median) & (diff > spike_threshold)

    cleaned = clipped.copy()
    cleaned[spike_mask] = neighborhood_median[spike_mask]

    valid_mask_before_fill = np.isfinite(cleaned).copy()
    cleaned = _fill_nodata_nearest(cleaned)
    cleaned = np.asarray(cleaned, dtype=float)

    clean_min = float(np.nanmin(cleaned))
    clean_max = float(np.nanmax(cleaned))
    clean_range = clean_max - clean_min

    outlier_ratio = float(outlier_mask.mean()) if outlier_mask.size else 0.0
    spike_ratio = float(spike_mask.mean()) if spike_mask.size else 0.0
    valid_ratio = float(valid_mask_before_fill.mean()) if valid_mask_before_fill.size else 0.0

    confidence = 1.0
    confidence -= max(0.0, (0.92 - valid_ratio)) * 1.25
    confidence -= min(0.25, outlier_ratio * 2.8)
    confidence -= min(0.32, spike_ratio * 4.8)
    if clean_range < 5.0:
        confidence -= 0.22
    if clean_range < 2.0:
        confidence -= 0.18
    confidence = max(0.0, min(1.0, confidence))

    warnings = list(qc.get("warnings", []))
    if outlier_ratio > 0.005:
        warnings.append("Extreme elevation outliers were clipped before derivative generation.")
    if spike_ratio > 0.005:
        warnings.append("Needle-like elevation spikes were repaired before 3D export.")
    if valid_ratio < 0.7:
        warnings.append("Terrain coverage is weak; downstream layers may be low-confidence.")
    if clean_range < 3.0:
        warnings.append("Terrain relief is very low after conditioning; some layers may look muted.")

    qc.update({
        "clip_low": round(p_low, 3),
        "clip_high": round(p_high, 3),
        "outlier_pixels": int(outlier_mask.sum()),
        "outlier_ratio": round(outlier_ratio, 4),
        "spike_pixels": int(spike_mask.sum()),
        "spike_ratio": round(spike_ratio, 4),
        "clean_min": round(clean_min, 3),
        "clean_max": round(clean_max, 3),
        "clean_range": round(clean_range, 3),
        "terrain_confidence": round(confidence, 4),
        "terrain_quality": _terrain_label(confidence),
        "exaggeration_recommended": _recommended_exaggeration(confidence, spike_ratio, valid_ratio),
        "warnings": warnings,
    })
    return cleaned, qc, valid_mask_before_fill


def ensure_vendor_assets() -> dict[str, bool]:
    three_path = VENDOR_DIR / "three.min.js"
    orbit_path = VENDOR_DIR / "OrbitControls.js"
    status = {"three_local": three_path.exists() and three_path.stat().st_size > 1000, "orbit_local": orbit_path.exists() and orbit_path.stat().st_size > 1000}
    status["all_local"] = bool(status["three_local"] and status["orbit_local"])
    if not status["all_local"]:
        raise SiteInsightError("Missing local 3D vendor assets in reports/_vendor. three.min.js and OrbitControls.js must exist locally for the restored viewer contract.")
    return status


def _write_context_map_html(run_dir: Path, run_id: str, bbox: tuple[float, float, float, float]) -> None:
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lon, center_lat = bbox_center(bbox)
    html = f'''<!doctype html>
<html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/><title>{run_id} - Context Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>html,body,#map{{height:100%;margin:0}} .note{{position:absolute;z-index:999;left:12px;top:12px;background:rgba(8,16,24,.8);color:#fff;padding:8px 10px;border-radius:8px;font-family:Arial}}</style></head>
<body><div class="note">{run_id}</div><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map=L.map('map').setView([{center_lat},{center_lon}],11);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'&copy; OpenStreetMap'}}).addTo(map);
const b=[[{min_lat},{min_lon}],[{max_lat},{max_lon}]]; L.rectangle(b,{{color:'#d6b36a',weight:2,fillOpacity:0.08}}).addTo(map); map.fitBounds(b);
</script></body></html>'''
    (run_dir / "context_map.html").write_text(html, encoding="utf-8")


def _write_viewer3d_html(run_dir: Path, run_id: str) -> None:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>__RUN_ID__ - Professional 3D Terrain Viewer</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top, rgba(37, 73, 120, 0.28), transparent 36%),
        linear-gradient(180deg, #07111b 0%, #081018 100%);
      color: #eef3f9;
      font-family: Arial, Helvetica, sans-serif;
    }
    .page {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .topbar {
      border-bottom: 1px solid rgba(255,255,255,.08);
      background: rgba(5, 11, 19, 0.88);
      backdrop-filter: blur(8px);
      position: sticky;
      top: 0;
      z-index: 20;
    }
    .topbar-inner {
      padding: 14px 18px 16px;
      display: grid;
      gap: 12px;
    }
    .title-row {
      display:flex;
      justify-content:space-between;
      align-items:flex-start;
      gap:12px;
      flex-wrap:wrap;
    }
    .run-title {
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .note { color:#aabacf; font-size:14px; }
    .chip-row {
      display:flex;
      gap:10px;
      flex-wrap:wrap;
    }
    .chip {
      background: rgba(255,255,255,.06);
      color:#dce9f8;
      border:1px solid rgba(255,255,255,.08);
      border-radius:999px;
      padding:8px 12px;
      font-size:13px;
    }
    .controls {
      display:flex;
      gap:10px;
      align-items:center;
      flex-wrap:wrap;
    }
    .btn, select, input[type="range"] {
      background:#182230;
      color:#eef3f9;
      border:1px solid rgba(255,255,255,.08);
      padding:10px 14px;
      border-radius:12px;
      text-decoration:none;
      font-size: 15px;
    }
    .btn {
      cursor:pointer;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .btn:hover { background:#213044; }
    .control-group { display:flex; gap:10px; align-items:center; }
    .preset-row { display:flex; gap:8px; flex-wrap:wrap; }
    .preset-btn { font-size:13px; padding:9px 11px; }
    .preset-btn.active { background:#2b3e57; border-color:rgba(231,196,106,.48); color:#f4d47a; }
    .slider-wrap { min-width: 300px; }
    .slider-wrap input[type="range"] {
      padding:0;
      width:200px;
      accent-color:#e7c46a;
      vertical-align: middle;
    }
    .slider-value {
      min-width:56px;
      display:inline-block;
      text-align:right;
      color:#f2cd70;
      font-weight:bold;
    }
    .viewer-shell {
      position: relative;
      min-height: calc(100vh - 150px);
    }
    #viewer {
      width:100%;
      height: calc(100vh - 150px);
      min-height: 620px;
      display:block;
      position:relative;
      overflow:hidden;
    }
    #viewer canvas { display:block; width:100%; height:100%; }
    #viewer::after {
      content:"";
      position:absolute;
      inset:auto 0 0 0;
      height:22%;
      pointer-events:none;
      background:linear-gradient(to top, rgba(8,16,24,0.86), rgba(8,16,24,0));
    }
    .info-panel {
      position: absolute;
      left: 18px;
      bottom: 18px;
      width: min(380px, calc(100vw - 36px));
      background: rgba(9, 17, 27, 0.84);
      border: 1px solid rgba(255,255,255,.09);
      border-radius: 18px;
      padding: 16px 16px 14px;
      z-index: 10;
      box-shadow: 0 18px 44px rgba(0,0,0,0.32);
      backdrop-filter: blur(10px);
    }
    .panel-title {
      font-size: 15px;
      font-weight: 700;
      margin-bottom: 8px;
    }
    .panel-copy {
      color: #c8d6e5;
      font-size: 14px;
      line-height: 1.45;
      margin-bottom: 12px;
    }
    .legend-grid {
      display:grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap:10px;
    }
    .legend-card {
      background: rgba(255,255,255,.04);
      border: 1px solid rgba(255,255,255,.06);
      border-radius: 14px;
      padding: 10px 12px;
    }
    .legend-card strong {
      display:block;
      font-size:13px;
      margin-bottom:4px;
      color:#eef3f9;
    }
    .legend-card span {
      color:#9fb0c3;
      font-size:12px;
      line-height:1.4;
    }
    .empty {
      padding:20px;
      color:#eef3f9;
    }
    @media (max-width: 900px) {
      .slider-wrap { min-width: 240px; }
      #viewer { min-height: 520px; height: calc(100vh - 210px); }
      .viewer-shell { min-height: calc(100vh - 210px); }
      .info-panel { position: static; width: auto; margin: 14px; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="topbar">
      <div class="topbar-inner">
        <div class="title-row">
          <div>
            <div class="run-title">__RUN_ID__ · Professional Terrain Command</div>
            <div class="note" id="viewer-note">Loading run-local terrain artifacts...</div>
          </div>
          <div class="chip-row">
            <div class="chip" id="terrain-chip">Terrain: loading</div>
            <div class="chip" id="grid-chip">Grid: loading</div>
            <div class="chip" id="mode-chip">Mode: scientific viewer</div>
          </div>
        </div>
        <div class="controls">
          <select id="texture">
            <option value="terrain_texture">Terrain Texture</option>
            <option value="crystal_clarity">RELIEF INSPECT - Crisp Terrain</option>
            <option value="microrelief_best">MICRORELIEF - Best Detail</option>
            <option value="science_composite">ARCHAEOLOGY FUSION - Signals</option>
            <option value="hillshade">Hillshade</option>
            <option value="heightmap">Heightmap</option>
            <option value="slope">Slope</option>
            <option value="lrm">Local Relief</option>
            <option value="lrm_edges">LRM + Edges</option>
            <option value="discovery">Discovery</option>
            <option value="archaeology">Archaeology</option>
          </select>
          <div class="control-group slider-wrap">
            <label for="exaggeration">Exaggeration</label>
            <input id="exaggeration" type="range" min="0" max="320" step="1" value="100" />
            <span id="exaggeration-value" class="slider-value">1.00x</span>
          </div>
          <div class="preset-row">
            <button class="btn preset-btn" data-preset="realistic" type="button">Field Realistic</button>
            <button class="btn preset-btn" data-preset="archaeology" type="button">Archaeology Detail</button>
            <button class="btn preset-btn" data-preset="mountain" type="button">Mountain Emphasis</button>
          </div>
          <button class="btn" id="reset">Reset View</button>
          <a class="btn" href="/runs/__RUN_ID__">Back to Run</a>
        </div>
      </div>
    </div>

    <div class="viewer-shell">
      <div id="viewer"></div>
      <div class="info-panel">
        <div class="panel-title">Scientific 3D Terrain Viewer</div>
        <div class="panel-copy">
          Use texture and exaggeration presets to read the landscape quickly. Terrain Texture is the field-first composite. Relief Inspect is the balanced terrain-reading mode. Microrelief is the strongest subtle-detail mode. Archaeology Fusion highlights review signals, not confirmed sites.
        </div>
        <div class="legend-grid">
          <div class="legend-card">
            <strong>Heightmap</strong>
            <span>Raw terrain shape for the 3D mesh.</span>
          </div>
          <div class="legend-card">
            <strong>Hillshade</strong>
            <span>General terrain form with light and shadow.</span>
          </div>
          <div class="legend-card">
            <strong>Local Relief</strong>
            <span>Small landform differences and subtle breaks.</span>
          </div>
          <div class="legend-card">
            <strong>Discovery / Archaeology</strong>
            <span>Interpretive overlays for anomaly hunting.</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script src="../../_vendor/three.min.js"></script>
  <script src="../../_vendor/OrbitControls.js"></script>
  <script>
    const cacheBust = `v=${Date.now()}`;
    const params = new URLSearchParams(window.location.search);
    const defaultTexture = (params.get('layer') || 'terrain_texture').toLowerCase();
    const embedMode = params.get('embed') === '1';
    const viewer = document.getElementById('viewer');
    const note = document.getElementById('viewer-note');
    const terrainChip = document.getElementById('terrain-chip');
    const gridChip = document.getElementById('grid-chip');
    const modeChip = document.getElementById('mode-chip');
    const textureSelect = document.getElementById('texture');
    const textureMap = {
      heightmap: './heightmap.png',
      terrain_texture: './Terrain_Texture.png',
      microrelief_best: './Microrelief_Best.browser-generated',
      crystal_clarity: './Crystal_Clarity.browser-generated',
      science_composite: './Science_Composite.browser-generated',
      hillshade: './Hillshade.png',
      slope: './Slope.png',
      lrm: './LRM.png',
      lrm_edges: './LRM_Edges.png',
      discovery: './Discovery.png',
      archaeology: './Archaeology.png'
    };

    if (!(window.THREE && window.THREE.OrbitControls)) {
      viewer.innerHTML = '<div class="empty">Local viewer libraries are missing or incomplete.</div>';
    } else {
      fetch(`./heightmap_viewer.json?${cacheBust}`, { cache: 'no-store' }).then(r => r.ok ? r : fetch(`./heightmap.json?${cacheBust}`, { cache: 'no-store' }))
        .then(r => r.json())
        .then(heightmap => {
          const sourceRows = Number(heightmap.rows || 0);
          const sourceCols = Number(heightmap.cols || 0);
          const values = Array.isArray(heightmap.values) ? heightmap.values : [];
          if (!sourceRows || !sourceCols || !values.length) {
            throw new Error('heightmap.json is missing rows, cols, or values.');
          }

          const exaggeration = Number(heightmap.exaggeration_default || 1.0);
          const viewerVerticalScale = Number(heightmap.viewer_vertical_scale || 1.8);
          const bboxWidthM = Number(heightmap.bbox_width_m || 0);
          const bboxHeightM = Number(heightmap.bbox_height_m || 0);
          const dominantMeters = Math.max(bboxWidthM, bboxHeightM, 1);
          const scaleMode = dominantMeters >= 22000 ? 'broad_context' : (dominantMeters >= 8000 ? 'landscape_context' : 'inspection_focus');
          const scaleBoost = scaleMode === 'broad_context' ? 1.46 : (scaleMode === 'landscape_context' ? 1.30 : 1.14);
          const effectiveViewerVerticalScale = viewerVerticalScale * scaleBoost;
          const baseHeightScale = exaggeration * effectiveViewerVerticalScale;
          const smoothingPasses = Number(heightmap.mesh_smoothing_passes || 0);
          const terrainQuality = heightmap.terrain_quality || 'unknown';
          const geometryMode = heightmap.geometry_mode || 'default';
          const sourceVerts = sourceRows * sourceCols;
          const localScienceLabActive = Boolean(heightmap.local_science_lab_mode);
          const standardMeshMax = scaleMode === 'inspection_focus' ? 420 : (scaleMode === 'landscape_context' ? 448 : 480);
          const scienceLabMeshMax = scaleMode === 'inspection_focus' ? 640 : (scaleMode === 'landscape_context' ? 672 : 704);
          const desiredMeshMax = embedMode ? 280 : (localScienceLabActive ? scienceLabMeshMax : standardMeshMax);
          const heavyRun = sourceVerts > desiredMeshMax * desiredMeshMax;
          function resampleMatrix(matrix, targetRows, targetCols) {
            const out = [];
            for (let r = 0; r < targetRows; r++) {
              const srcR = Math.min(sourceRows - 1, Math.round((r / Math.max(targetRows - 1, 1)) * (sourceRows - 1)));
              const srcRow = matrix[srcR] || [];
              const row = [];
              for (let c = 0; c < targetCols; c++) {
                const srcC = Math.min(sourceCols - 1, Math.round((c / Math.max(targetCols - 1, 1)) * (sourceCols - 1)));
                row.push(Number(srcRow[srcC] || 0));
              }
              out.push(row);
            }
            return out;
          }
          const geometryValuesRaw = Array.isArray(heightmap.geometry_values) ? heightmap.geometry_values : values;
          let meshRows = sourceRows;
          let meshCols = sourceCols;
          let meshValues = geometryValuesRaw;
          if (heavyRun) {
            meshRows = Math.min(sourceRows, desiredMeshMax);
            meshCols = Math.min(sourceCols, desiredMeshMax);
            meshValues = resampleMatrix(geometryValuesRaw, meshRows, meshCols);
          }
          note.textContent = `Fast viewer payload loaded. JSON ${sourceCols} x ${sourceRows}. Mesh ${meshCols} x ${meshRows}. Terrain ${terrainQuality}. Geometry ${geometryMode}. Base scale ${effectiveViewerVerticalScale.toFixed(2)}. ${embedMode ? 'Embedded stability mode on.' : 'Interactive performance mode on.'}`;
          terrainChip.textContent = `Terrain: ${terrainQuality}`;
          gridChip.textContent = `Grid: ${sourceCols} x ${sourceRows}${heavyRun ? ` · Mesh ${meshCols} x ${meshRows}` : ''}`;
          modeChip.textContent = `Geometry: ${geometryMode} · ${scaleMode}`;

          const scene = new THREE.Scene();
          scene.background = new THREE.Color(0x081018);
          scene.fog = new THREE.Fog(0x081018, 30, 72);

          const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 1000);

          const renderer = new THREE.WebGLRenderer({ antialias: !embedMode && !heavyRun, alpha: false, powerPreference: embedMode ? 'low-power' : 'high-performance' });
          const pixelRatioBoost = embedMode ? 0.62 : (heavyRun ? 0.70 : (scaleMode === 'inspection_focus' ? 0.88 : 0.82));
          const pixelRatioCap = embedMode ? 0.9 : (heavyRun ? 1.05 : (scaleMode === 'inspection_focus' ? 1.3 : 1.18));
          renderer.setPixelRatio(Math.min((window.devicePixelRatio || 1) * pixelRatioBoost, pixelRatioCap));
          renderer.outputColorSpace = THREE.SRGBColorSpace || undefined;
          viewer.appendChild(renderer.domElement);

          const controls = new THREE.OrbitControls(camera, renderer.domElement);
          controls.enableDamping = true;
          controls.dampingFactor = 0.08;
          controls.zoomSpeed = 1.15;
          controls.minDistance = dominantMeters >= 22000 ? 3.8 : (dominantMeters >= 8000 ? 2.8 : 1.9);
          controls.maxDistance = dominantMeters >= 22000 ? 68 : (dominantMeters >= 8000 ? 54 : 44);
          if (heavyRun) { controls.maxDistance = Math.min(controls.maxDistance, 42); }
          controls.minPolarAngle = dominantMeters >= 22000 ? 0.30 : 0.24;
          controls.maxPolarAngle = dominantMeters >= 22000 ? Math.PI * 0.34 : (dominantMeters >= 8000 ? Math.PI * 0.39 : Math.PI * 0.44);
          controls.screenSpacePanning = false;

          const hemi = new THREE.HemisphereLight(0xeaf2fb, 0x142033, 1.55);
          scene.add(hemi);
          const dir = new THREE.DirectionalLight(0xffffff, 1.35);
          dir.position.set(10, 18, 8);
          scene.add(dir);
          const fill = new THREE.DirectionalLight(0xc8d7e8, 0.55);
          fill.position.set(-12, 9, -8);
          scene.add(fill);

          const planeWidth = 22;
          const planeHeight = Math.max(7, 22 * (meshRows / Math.max(meshCols, 1)));
          const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight, meshCols - 1, meshRows - 1);
          geometry.rotateX(-Math.PI / 2);
          const position = geometry.attributes.position;

          function sampleHeight(r, c) {
            const row = meshValues[r] || [];
            const val = Number(row[c] || 0);
            return Number.isFinite(val) ? val : 0;
          }

          let i = 0;
          for (let r = 0; r < meshRows; r++) {
            for (let c = 0; c < meshCols; c++) {
              position.setY(i, sampleHeight(r, c) * baseHeightScale);
              i += 1;
            }
          }

          function smoothGeometryPass() {
            const current = [];
            let idx = 0;
            for (let r = 0; r < meshRows; r++) {
              const row = [];
              for (let c = 0; c < meshCols; c++) {
                row.push(position.getY(idx));
                idx += 1;
              }
              current.push(row);
            }
            function clamped(rr, cc) {
              const r2 = Math.max(0, Math.min(meshRows - 1, rr));
              const c2 = Math.max(0, Math.min(meshCols - 1, cc));
              return current[r2][c2];
            }
            idx = 0;
            for (let r = 0; r < meshRows; r++) {
              for (let c = 0; c < meshCols; c++) {
                let acc = 0;
                let cnt = 0;
                for (let dy = -1; dy <= 1; dy++) {
                  for (let dx = -1; dx <= 1; dx++) {
                    acc += clamped(r + dy, c + dx);
                    cnt += 1;
                  }
                }
                const center = current[r][c];
                const smooth = acc / cnt;
                position.setY(idx, center * 0.7 + smooth * 0.3);
                idx += 1;
              }
            }
          }
          for (let pass = 0; pass < smoothingPasses; pass++) {
            smoothGeometryPass();
          }

          const edgeMargin = Math.max(4, Math.round(Math.min(meshRows, meshCols) * 0.10));
          const edgeLift = 0.14;
          function edgeBlendFactor(r, c) {
            const edgeDistance = Math.min(r, c, meshRows - 1 - r, meshCols - 1 - c);
            const t = Math.max(0, Math.min(1, edgeDistance / Math.max(edgeMargin, 1)));
            return 0.42 + 0.58 * (t * t * (3 - 2 * t));
          }

          const shapedHeights = [];
          i = 0;
          for (let r = 0; r < meshRows; r++) {
            for (let c = 0; c < meshCols; c++) {
              const softened = position.getY(i) * edgeBlendFactor(r, c) + edgeLift;
              position.setY(i, softened);
              shapedHeights.push(softened);
              i += 1;
            }
          }

          function applyExaggeration(multiplier) {
            const safeMultiplier = Math.max(0, Number(multiplier || 0));
            for (let idx = 0; idx < position.count; idx++) {
              position.setY(idx, edgeLift + Math.max(0, shapedHeights[idx] - edgeLift) * safeMultiplier);
            }
            position.needsUpdate = true;
            geometry.computeVertexNormals();
          }

          position.needsUpdate = true;
          geometry.computeVertexNormals();

          const textureLoader = new THREE.TextureLoader();
          const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.84, metalness: 0.01, emissive: 0x101722, emissiveIntensity: 0.05, side: THREE.DoubleSide, bumpScale: 0.18 });
          const mesh = new THREE.Mesh(geometry, material);
          scene.add(mesh);

          const baseRadius = Math.max(planeWidth, planeHeight) * 0.72;
          const floorGeometry = new THREE.CircleGeometry(baseRadius, 72);
          floorGeometry.rotateX(-Math.PI / 2);
          const floorMaterial = new THREE.MeshBasicMaterial({ color: 0x132235, transparent: true, opacity: 0.78 });
          const floor = new THREE.Mesh(floorGeometry, floorMaterial);
          floor.position.y = 0;
          scene.add(floor);

          const haloGeometry = new THREE.RingGeometry(baseRadius * 0.82, baseRadius * 0.96, 72);
          haloGeometry.rotateX(-Math.PI / 2);
          const haloMaterial = new THREE.MeshBasicMaterial({ color: 0x27435f, transparent: true, opacity: 0.2, side: THREE.DoubleSide });
          const halo = new THREE.Mesh(haloGeometry, haloMaterial);
          halo.position.y = 0.01;
          scene.add(halo);

          const wire = new THREE.LineSegments(
            new THREE.WireframeGeometry(geometry),
            new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: embedMode ? 0.0 : 0.022 })
          );
          if (!embedMode) scene.add(wire);

          function fitCamera() {
            const box = new THREE.Box3().setFromObject(mesh);
            const size = new THREE.Vector3();
            const center = new THREE.Vector3();
            box.getSize(size);
            box.getCenter(center);
            const radius = Math.max(size.x, size.z, size.y * 1.5, 1);
            const cameraDistance = dominantMeters >= 22000 ? 1.12 : (dominantMeters >= 8000 ? 1.08 : 1.10);
            const cameraHeight = dominantMeters >= 22000 ? 0.60 : (dominantMeters >= 8000 ? 0.70 : 0.92);
            controls.target.set(center.x, Math.max(edgeLift, center.y * (dominantMeters >= 22000 ? 0.26 : (dominantMeters >= 8000 ? 0.34 : 0.55))), center.z);
            camera.position.set(center.x + radius * 0.34 * cameraDistance, center.y + radius * cameraHeight, center.z + radius * 0.94 * cameraDistance);
            camera.near = Math.max(0.1, radius / 120);
            camera.far = Math.max(200, radius * 35);
            camera.updateProjectionMatrix();
            controls.update();
          }

          let activeTextures = [];
          function releaseActiveTextures() {
            activeTextures.forEach((t) => { if (t && t.dispose) t.dispose(); });
            activeTextures = [];
          }

          function prepTexture(texture, { grayscale = false, sharp = true } = {}) {
            texture.wrapS = THREE.ClampToEdgeWrapping;
            texture.wrapT = THREE.ClampToEdgeWrapping;
            texture.generateMipmaps = true;
            texture.minFilter = THREE.LinearMipmapLinearFilter;
            texture.magFilter = THREE.LinearFilter;
            if (renderer.capabilities && renderer.capabilities.getMaxAnisotropy) {
              texture.anisotropy = Math.min(12, renderer.capabilities.getMaxAnisotropy());
            }
            texture.colorSpace = grayscale ? THREE.NoColorSpace : (THREE.SRGBColorSpace || undefined);
            texture.needsUpdate = true;
            return texture;
          }

          function makePublicationReliefTexture(mode = 'relief') {
            // Research-grade browser visualization:
            // multi-scale local relief + restrained multi-direction hillshade.
            // This makes existing DEM detail more readable without inventing data.
            const matrix = Array.isArray(heightmap.geometry_values) ? heightmap.geometry_values : values;
            const rows = sourceRows;
            const cols = sourceCols;
            const canvasMax = heightmap.local_science_lab_mode ? 1536 : 640;
            const w = Math.max(64, Math.min(canvasMax, cols));
            const h = Math.max(64, Math.min(canvasMax, rows));
            const canvas = document.createElement('canvas');
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            const image = ctx.createImageData(w, h);
            const data = image.data;

            function rawAt(r, c) {
              const rr = Math.max(0, Math.min(rows - 1, r));
              const cc = Math.max(0, Math.min(cols - 1, c));
              const row = matrix[rr] || [];
              const v = Number(row[cc] || 0);
              return Number.isFinite(v) ? v : 0;
            }

            // Resample height to the canvas grid once.
            const grid = new Float64Array(w * h);
            const sample = [];
            for (let y = 0; y < h; y++) {
              const r = Math.round((y / Math.max(h - 1, 1)) * (rows - 1));
              for (let x = 0; x < w; x++) {
                const c = Math.round((x / Math.max(w - 1, 1)) * (cols - 1));
                const v = rawAt(r, c);
                grid[y * w + x] = v;
                if ((x % 5 === 0) && (y % 5 === 0)) sample.push(v);
              }
            }

            function deTerraceGrid(sourceGrid) {
              // Edge-preserving smoothing: suppress raster stair-steps without erasing real breaks.
              // It blends only neighbors whose elevation is close to the center value.
              const out = new Float64Array(sourceGrid.length);
              const passes = heightmap.local_science_lab_mode ? 2 : 1;
              let current = sourceGrid;
              let next = out;

              for (let pass = 0; pass < passes; pass++) {
                for (let yy = 0; yy < h; yy++) {
                  for (let xx = 0; xx < w; xx++) {
                    const idx = yy * w + xx;
                    const center = current[idx];
                    let weighted = center * 2.4;
                    let weight = 2.4;

                    for (let dy2 = -1; dy2 <= 1; dy2++) {
                      for (let dx2 = -1; dx2 <= 1; dx2++) {
                        if (dx2 === 0 && dy2 === 0) continue;
                        const nx2 = Math.max(0, Math.min(w - 1, xx + dx2));
                        const ny2 = Math.max(0, Math.min(h - 1, yy + dy2));
                        const n = current[ny2 * w + nx2];
                        const diff = Math.abs(n - center);
                        const localTolerance = Math.max((sample[Math.floor(sample.length * 0.985)] || center) - (sample[Math.floor(sample.length * 0.015)] || center), 1e-9) * 0.0065;
                        const keepEdge = Math.max(0.0, 1.0 - diff / Math.max(localTolerance, 1e-9));
                        const spatial = (dx2 === 0 || dy2 === 0) ? 0.92 : 0.55;
                        const ww = keepEdge * spatial;
                        weighted += n * ww;
                        weight += ww;
                      }
                    }
                    next[idx] = weighted / Math.max(weight, 1e-9);
                  }
                }
                const temp = current;
                current = next;
                next = temp;
              }
              return current;
            }

            // Fill sample first so de-terrace has a robust local tolerance range.
            sample.sort((a, b) => a - b);
            const deTerraced = deTerraceGrid(grid);
            for (let i = 0; i < grid.length; i++) grid[i] = deTerraced[i];

            // Rebuild the sample from the de-terraced grid for more stable percentile contrast.
            sample.length = 0;
            for (let yy = 0; yy < h; yy += 5) {
              for (let xx = 0; xx < w; xx += 5) {
                sample.push(grid[yy * w + xx]);
              }
            }
            sample.sort((a, b) => a - b);
            const p = (q) => sample[Math.max(0, Math.min(sample.length - 1, Math.floor(q * (sample.length - 1))))] || 0;
            const lo = p(0.015);
            const hi = p(0.985);
            const denom = Math.max(hi - lo, 1e-9);

            // Integral image for fast local means at several radii.
            const integral = new Float64Array((w + 1) * (h + 1));
            for (let y = 1; y <= h; y++) {
              let rowSum = 0;
              for (let x = 1; x <= w; x++) {
                rowSum += grid[(y - 1) * w + (x - 1)];
                integral[y * (w + 1) + x] = integral[(y - 1) * (w + 1) + x] + rowSum;
              }
            }

            function boxMean(x, y, radius) {
              const x0 = Math.max(0, x - radius);
              const y0 = Math.max(0, y - radius);
              const x1 = Math.min(w - 1, x + radius);
              const y1 = Math.min(h - 1, y + radius);
              const A = integral[y0 * (w + 1) + x0];
              const B = integral[y0 * (w + 1) + (x1 + 1)];
              const C = integral[(y1 + 1) * (w + 1) + x0];
              const D = integral[(y1 + 1) * (w + 1) + (x1 + 1)];
              const count = Math.max(1, (x1 - x0 + 1) * (y1 - y0 + 1));
              return (D - B - C + A) / count;
            }

            function clamp01(x) {
              return Math.max(0, Math.min(1, x));
            }

            function contrast(x, mid = 0.5, amount = 1.0) {
              return clamp01((x - mid) * amount + mid);
            }

            function smoothTone(x) {
              x = clamp01(x);
              return x * x * (3 - 2 * x);
            }

            const isMicro = mode === 'micro';
            const isFusion = mode === 'fusion';
            const lights = [
              { x: -0.64, y: -0.46, z: 0.62, w: 0.36 },
              { x:  0.58, y: -0.50, z: 0.64, w: 0.32 },
              { x: -0.18, y:  0.82, z: 0.54, w: 0.32 },
            ];

            for (let y = 0; y < h; y++) {
              for (let x = 0; x < w; x++) {
                const idxGrid = y * w + x;
                const center = grid[idxGrid];

                // Wider gradient suppresses raster row/column banding.
                const span = heightmap.local_science_lab_mode ? 4 : 2;
                const left = grid[y * w + Math.max(0, x - span)];
                const right = grid[y * w + Math.min(w - 1, x + span)];
                const up = grid[Math.max(0, y - span) * w + x];
                const down = grid[Math.min(h - 1, y + span) * w + x];
                const dx = (right - left) / denom;
                const dy = (down - up) / denom;

                let nx = -dx * (isMicro ? 1.38 : 1.18);
                let ny = -dy * (isMicro ? 1.38 : 1.18);
                let nz = 1.0;
                const nLen = Math.max(Math.sqrt(nx * nx + ny * ny + nz * nz), 1e-9);
                nx /= nLen; ny /= nLen; nz /= nLen;

                let hill = 0;
                for (const light of lights) {
                  hill += light.w * Math.max(0, nx * light.x + ny * light.y + nz * light.z);
                }

                // Multi-scale local relief: small vs medium plus medium vs large.
                const smallR = heightmap.local_science_lab_mode ? 2 : 1;
                const medR = heightmap.local_science_lab_mode ? 6 : 4;
                const largeR = heightmap.local_science_lab_mode ? 16 : 10;
                const small = boxMean(x, y, smallR);
                const medium = boxMean(x, y, medR);
                const large = boxMean(x, y, largeR);
                const lrmFine = (small - medium) / (denom * 0.022);
                const lrmBroad = (medium - large) / (denom * 0.045);
                let microRelief = Math.max(-1, Math.min(1, lrmFine * 0.68 + lrmBroad * 0.32));

                // Edge is deliberately restrained; it should highlight breaks, not rows.
                const edge = clamp01(Math.sqrt(dx * dx + dy * dy) * (isMicro ? 2.05 : 1.65));
                const elevationNorm = clamp01((center - lo) / denom);

                if (isFusion) {
                  let base = 0.42 + (hill - 0.50) * 0.50 + (elevationNorm - 0.50) * 0.07;
                  base = clamp01(base);
                  const pos = clamp01(microRelief);
                  const neg = clamp01(-microRelief);
                  const ink = edge * 0.10;

                  let rr = base + pos * 0.26 - ink;
                  let gg = base + pos * 0.10 + neg * 0.08 - ink;
                  let bb = base + neg * 0.25 - ink * 0.45;

                  rr = smoothTone(contrast(rr, 0.48, 1.08));
                  gg = smoothTone(contrast(gg, 0.48, 1.04));
                  bb = smoothTone(contrast(bb, 0.48, 1.08));

                  const idx = idxGrid * 4;
                  data[idx] = Math.round(255 * rr);
                  data[idx + 1] = Math.round(255 * gg);
                  data[idx + 2] = Math.round(255 * bb);
                  data[idx + 3] = 255;
                } else {
                  let tone = isMicro ? 0.40 : 0.42;
                  tone += (hill - 0.50) * (isMicro ? 0.46 : 0.40);
                  tone += microRelief * (isMicro ? 0.42 : 0.30);
                  tone += edge * (isMicro ? 0.055 : 0.045);
                  tone += (elevationNorm - 0.50) * 0.045;
                  tone = contrast(tone, 0.48, isMicro ? 1.28 : 1.14);
                  tone = 0.10 + 0.80 * smoothTone(tone);

                  const v = Math.round(255 * clamp01(tone));
                  const idx = idxGrid * 4;
                  data[idx] = v;
                  data[idx + 1] = v;
                  data[idx + 2] = v;
                  data[idx + 3] = 255;
                }
              }
            }

            // Tiny anti-band post-process; keep it subtle so real microrelief stays.
            const antiBandBlend = isMicro ? 0.16 : 0.20;
            const original = new Uint8ClampedArray(data);
            for (let y = 1; y < h - 1; y++) {
              for (let x = 1; x < w - 1; x++) {
                const idx = (y * w + x) * 4;
                for (let ch = 0; ch < 3; ch++) {
                  const center = original[idx + ch];
                  const left = original[idx - 4 + ch];
                  const right = original[idx + 4 + ch];
                  const up = original[idx - w * 4 + ch];
                  const down = original[idx + w * 4 + ch];
                  const neighborMean = (left + right + up + down) * 0.25;
                  data[idx + ch] = Math.round(center * (1.0 - antiBandBlend) + neighborMean * antiBandBlend);
                }
              }
            }

            ctx.putImageData(image, 0, 0);
            const texture = new THREE.CanvasTexture(canvas);
            texture.wrapS = THREE.ClampToEdgeWrapping;
            texture.wrapT = THREE.ClampToEdgeWrapping;
            texture.generateMipmaps = false;
            texture.minFilter = THREE.LinearFilter;
            texture.magFilter = THREE.LinearFilter;
            if (renderer.capabilities && renderer.capabilities.getMaxAnisotropy) {
              texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
            }
            texture.colorSpace = THREE.SRGBColorSpace || undefined;
            texture.needsUpdate = true;
            return texture;
          }

          function loadTexture(url, opts = {}) {
            return new Promise((resolve, reject) => {
              textureLoader.load(`${url}?${cacheBust}`, (texture) => resolve(prepTexture(texture, opts)), undefined, reject);
            });
          }

          async function applyTexture(name) {
            const asset = textureMap[name] || textureMap.hillshade;
            releaseActiveTextures();
            try {
              if (name === 'microrelief_best') {
                const reliefTexture = makePublicationReliefTexture('micro');
                activeTextures.push(reliefTexture);
                material.map = reliefTexture;
                material.bumpMap = null;
                material.bumpScale = 0.0;
                material.roughness = 0.70;
                material.emissiveIntensity = 0.0;
                material.needsUpdate = true;
                const lane = heightmap.local_science_lab_mode ? 'Local Science Lab' : (heightmap.render_safe_mode ? 'Render Safe' : 'Local Standard');
                note.textContent = `MICRORELIEF active. De-terraced multi-scale local relief for best subtle-detail reading. Lane ${lane}. Source ${sourceCols} x ${sourceRows}. Mesh ${meshCols} x ${meshRows}.`;
                return;
              }

              if (name === 'crystal_clarity' || name === 'relief_inspect') {
                const reliefTexture = makePublicationReliefTexture('relief');
                activeTextures.push(reliefTexture);
                material.map = reliefTexture;
                material.bumpMap = null;
                material.bumpScale = 0.0;
                material.roughness = 0.72;
                material.emissiveIntensity = 0.0;
                material.needsUpdate = true;
                const lane = heightmap.local_science_lab_mode ? 'Local Science Lab' : (heightmap.render_safe_mode ? 'Render Safe' : 'Local Standard');
                note.textContent = `RELIEF INSPECT active. Balanced de-terraced terrain reading from height data. Lane ${lane}. Source ${sourceCols} x ${sourceRows}. Mesh ${meshCols} x ${meshRows}.`;
                return;
              }

              if (name === 'science_composite' || name === 'archaeology_fusion') {
                const fusionTexture = makePublicationReliefTexture('fusion');
                activeTextures.push(fusionTexture);
                material.map = fusionTexture;
                material.bumpMap = null;
                material.bumpScale = 0.0;
                material.roughness = 0.70;
                material.emissiveIntensity = 0.0;
                material.needsUpdate = true;
                const lane = heightmap.local_science_lab_mode ? 'Local Science Lab' : (heightmap.render_safe_mode ? 'Render Safe' : 'Local Standard');
                note.textContent = `ARCHAEOLOGY FUSION active. Warm/cool multi-scale relief signal for review, not confirmation. Lane ${lane}. Source ${sourceCols} x ${sourceRows}. Mesh ${meshCols} x ${meshRows}.`;
                return;
              }

              const [colorTexture, detailTexture] = await Promise.all([
                loadTexture(asset, { grayscale: false, sharp: true }),
                embedMode ? Promise.resolve(null) : loadTexture('./LRM_Edges.png', { grayscale: true, sharp: true })
              ]);
              activeTextures.push(colorTexture, detailTexture);
              material.map = colorTexture;
              material.bumpMap = embedMode ? null : detailTexture;
              material.bumpScale = embedMode ? 0.0 : Math.max(0.018, 0.040 * effectiveViewerVerticalScale);
              material.roughness = 0.78;
              material.emissiveIntensity = 0.02;
              material.needsUpdate = true;
            } catch (e) {
              material.map = null;
              material.bumpMap = null;
              material.needsUpdate = true;
            }
          }

          function resize() {
            const w = Math.max(viewer.clientWidth || window.innerWidth, 320);
            const h = Math.max(viewer.clientHeight || (window.innerHeight - 150), 280);
            renderer.setSize(w, h, false);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
          }
          window.addEventListener('resize', resize);
          resize();
          fitCamera();

          function animate() {
            controls.update();
            renderer.render(scene, camera);
            requestAnimationFrame(animate);
          }
          animate();

          const exaggerationSlider = document.getElementById('exaggeration');
          const exaggerationValue = document.getElementById('exaggeration-value');
          const presetButtons = Array.from(document.querySelectorAll('.preset-btn'));
          const defaultSliderValue = Math.max(60, Math.min(285, Math.round((baseHeightScale / Math.max(effectiveViewerVerticalScale, 0.001)) * 112)));
          exaggerationSlider.value = String(defaultSliderValue);
          function updateExaggerationLabel(sliderValue) {
            const multiplier = Number(sliderValue) / 100;
            exaggerationValue.textContent = `${multiplier.toFixed(2)}x`;
            const fidelityLane = heightmap.render_safe_mode ? 'Render Safe' : (heightmap.local_science_lab_mode ? 'Local Science Lab' : 'Local Standard');
          const sourceMeshRatio = (sourceCols && meshCols) ? (sourceCols / meshCols).toFixed(2) : 'n/a';
          note.textContent = `Dr. Source Fusion active. Lane ${fidelityLane}. Source ${sourceCols} x ${sourceRows}. Mesh ${meshCols} x ${meshRows}. Ratio ${sourceMeshRatio}:1. Desired mesh cap ${desiredMeshMax}. Sweet spot polish active. Terrain ${(typeof terrainStrength !== 'undefined' ? terrainStrength : (heightmap.terrain_strength || heightmap.terrainStrength || 'strong'))}. Geometry ${geometryMode}. Base scale ${effectiveViewerVerticalScale.toFixed(2)}. High-detail viewer on. Exaggeration ${(typeof currentExaggeration !== 'undefined' ? currentExaggeration : (Number(heightmap.exaggeration || heightmap.vertical_exaggeration || heightmap.verticalExaggeration || 1.0))).toFixed(2)}x.`;
            applyExaggeration(multiplier * effectiveViewerVerticalScale);
          }
          exaggerationSlider.addEventListener('input', (e) => {
            presetButtons.forEach((btn) => btn.classList.remove('active'));
            updateExaggerationLabel(e.target.value);
          });
          function applyPreset(name) {
            const presets = {
              realistic: Math.max(70, Math.round(defaultSliderValue * 0.82)),
              archaeology: Math.max(95, Math.round(defaultSliderValue * 1.08)),
              mountain: Math.min(320, Math.max(135, Math.round(defaultSliderValue * 1.42)))
            };
            const nextValue = presets[name] || defaultSliderValue;
            exaggerationSlider.value = String(nextValue);
            presetButtons.forEach((btn) => btn.classList.toggle('active', btn.dataset.preset === name));
            updateExaggerationLabel(nextValue);
          }
          presetButtons.forEach((btn) => btn.addEventListener('click', () => applyPreset(btn.dataset.preset)));

          textureSelect.value = textureMap[defaultTexture] ? defaultTexture : 'hillshade';
          textureSelect.addEventListener('change', (e) => applyTexture(e.target.value));
          document.getElementById('reset').addEventListener('click', () => {
            exaggerationSlider.value = String(defaultSliderValue);
            presetButtons.forEach((btn) => btn.classList.remove('active'));
            updateExaggerationLabel(defaultSliderValue);
            fitCamera();
          });

          updateExaggerationLabel(defaultSliderValue);
          applyTexture(textureSelect.value || 'hillshade');
        })
        .catch(e => {
          viewer.innerHTML = '<div class="empty">Viewer failed to load run-local terrain artifacts: ' + e.message + '</div>';
        });
    }
  </script>
</body>
</html>
"""
    (run_dir / "viewer3d.html").write_text(html.replace("__RUN_ID__", run_id), encoding="utf-8")


def _build_viewer_layers(surface, derivatives: dict[str, Any], intelligence: dict[str, Any], archaeology: dict[str, Any]) -> dict[str, Any]:
    dem = np.asarray(surface.cleaned_dem, dtype=float)
    policy = surface.rendering_policy
    target_rows, target_cols = dem.shape
    aligned_derivatives: dict[str, Any] = {}
    for key, value in derivatives.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            matrix = np.asarray(value, dtype=float)
            if matrix.shape != (target_rows, target_cols):
                out_size = max(target_rows, target_cols)
                matrix = _resample_square_if_needed(matrix, out_size)
                matrix = np.asarray(matrix, dtype=float)[:target_rows, :target_cols]
            aligned_derivatives[key] = matrix
        else:
            aligned_derivatives[key] = value
    derivatives = aligned_derivatives
    elevation = _norm(dem)

    hillshade_a = _hillshade(dem, azimuth=315.0, altitude=45.0)
    hillshade_b = _hillshade(dem, azimuth=45.0, altitude=42.0)
    hillshade_c = _hillshade(dem, azimuth=270.0, altitude=38.0)
    hillshade = np.clip(0.5 * hillshade_a + 0.3 * hillshade_b + 0.2 * hillshade_c, 0.0, 1.0)
    hillshade = _unsharp(hillshade, amount=0.4, smooth_iters=1)

    slope = _unsharp(_robust_norm(derivatives["slope"], 4.0, 99.0), amount=0.35, smooth_iters=1)
    local_relief = _unsharp(_local_relief(dem), amount=0.45, smooth_iters=1)
    broad_relief = _robust_norm(np.abs(dem - _smooth_grid(dem, 5)), 5.0, 99.0)
    fine_relief = _robust_norm(np.abs(dem - _smooth_grid(dem, 1)), 5.0, 99.0)
    ridge = _robust_norm(derivatives["ridge_strength"], 5.0, 99.0)
    valley = _robust_norm(derivatives["valley_strength"], 5.0, 99.0)
    edges = _robust_norm(derivatives["edge_strength"], 5.0, 99.0)
    multi_scale_lrm = np.clip(0.42 * local_relief + 0.28 * broad_relief + 0.18 * fine_relief + 0.12 * edges, 0.0, 1.0)
    multi_scale_lrm = _unsharp(multi_scale_lrm, amount=0.5, smooth_iters=1)

    openness_pos, openness_neg = _openness(dem)
    srv = _srv_like(derivatives)

    archaeology_layer = np.clip(
        0.34 * _robust_norm(derivatives.get("anomaly_response", derivatives["ridge_strength"]), 5.0, 99.0) +
        0.18 * _robust_norm(-derivatives["laplacian"], 5.0, 99.0) +
        0.26 * multi_scale_lrm +
        0.12 * ridge +
        0.10 * valley,
        0, 1
    )
    if policy.emphasize_local_relief:
        archaeology_layer = np.clip(0.55 * archaeology_layer + 0.45 * multi_scale_lrm, 0, 1)
    archaeology_layer = _unsharp(archaeology_layer, amount=0.45, smooth_iters=1)
    if policy.suppress_archaeology:
        archaeology_layer *= 0.5

    discovery = np.clip(
        0.22 * hillshade +
        0.16 * slope +
        0.26 * multi_scale_lrm +
        0.16 * archaeology_layer +
        0.10 * valley +
        0.10 * ridge,
        0, 1
    )
    if policy.suppress_misleading_products:
        discovery *= 0.75
    discovery = _unsharp(discovery, amount=0.35, smooth_iters=1)

    terrain_texture = np.clip(
        0.30 * hillshade +
        0.24 * multi_scale_lrm +
        0.14 * slope +
        0.10 * ridge +
        0.08 * valley +
        0.08 * openness_pos +
        0.06 * archaeology_layer +
        0.10 * elevation,
        0.0, 1.0
    )
    micro = np.clip(0.55 * multi_scale_lrm + 0.25 * ridge + 0.20 * valley, 0.0, 1.0)
    terrain_texture = np.clip(0.78 * terrain_texture + 0.22 * micro, 0.0, 1.0)
    terrain_texture = np.clip((terrain_texture - 0.5) * 1.85 + 0.5, 0.0, 1.0)
    terrain_texture = _unsharp(terrain_texture, amount=0.5, smooth_iters=1)

    return {
        "meta": {
            "terrain_signal_score": intelligence["terrain_signal_score"],
            "discovery_score": intelligence["discovery_score"],
            "archaeology_score": archaeology["archaeological_signal_score"],
            "terrain_confidence": surface.qc.overall_confidence,
            "terrain_quality": surface.qc.terrain_quality,
            "terrain_class": surface.terrain_class,
            "terrain_warnings": list(policy.viewer_warnings),
            "vertical_scale": round(float(policy.vertical_exaggeration) * 5.4, 2),
            "rendering_mode": policy.mode,
            "allow_3d": policy.allow_3d,
            "public_readiness": policy.public_readiness,
            "product_surface_style": policy.product_surface_style,
            "default_texture": "terrain_texture",
            "default_stack": ["terrain_texture", "hillshade", "local_relief"],
        },
        "default_layer": "terrain_texture",
        "default_mode": "three",
        "legend": {
            "terrain_texture": "Professional composite: hillshade, local relief, slope, ridge/valley structure, and controlled archaeology signal.",
            "hillshade": "Multi-directional shaded relief for general landform reading.",
            "elevation": "Normalized elevation tint for broad terrain position.",
            "slope": "Slope intensity for breaks, scarps, benches, and transition edges.",
            "local_relief": "Multi-scale local relief model for subtle terrain structure.",
            "openness": "Positive openness signal for ridges, convex forms, and exposed ground structure.",
            "srv": "Sky-view-style structural relief blend for terrain mass and visibility.",
            "archaeology": "Interpretive anomaly signal; use as a lead, not proof.",
            "discovery": "Combined terrain-priority layer for fast field review."
        },
        "layers": {
            "elevation": _downsample_matrix_for_viewer(elevation, _active_layer_json_cap(512)).round(5).tolist(),
            "hillshade": _downsample_matrix_for_viewer(hillshade, _active_layer_json_cap(512)).round(5).tolist(),
            "slope": _downsample_matrix_for_viewer(slope, _active_layer_json_cap(512)).round(5).tolist(),
            "local_relief": _downsample_matrix_for_viewer(multi_scale_lrm, _active_layer_json_cap(512)).round(5).tolist(),
            "openness": _downsample_matrix_for_viewer(openness_pos, _active_layer_json_cap(512)).round(5).tolist(),
            "openness_negative": _downsample_matrix_for_viewer(openness_neg, _active_layer_json_cap(512)).round(5).tolist(),
            "srv": _downsample_matrix_for_viewer(srv, _active_layer_json_cap(512)).round(5).tolist(),
            "archaeology": _downsample_matrix_for_viewer(archaeology_layer, _active_layer_json_cap(512)).round(5).tolist(),
            "discovery": _downsample_matrix_for_viewer(discovery, _active_layer_json_cap(512)).round(5).tolist(),
            "terrain_texture": _downsample_matrix_for_viewer(terrain_texture, _active_layer_json_cap(512)).round(5).tolist(),
        },
    }


def _write_validmask_png(run_dir: Path, valid_mask: np.ndarray) -> None:
    mask = np.asarray(valid_mask, dtype=bool)
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[mask] = np.array([235, 240, 245], dtype=np.uint8)
    rgb[~mask] = np.array([22, 28, 36], dtype=np.uint8)
    _write_png(run_dir / "validmask.png", rgb)


def _conditioning_summary_text(terrain_qc: dict[str, Any]) -> str:
    warnings = terrain_qc.get("warnings", []) or []
    lines = [
        "Terrain conditioning summary",
        f"- Terrain quality: {terrain_qc.get('terrain_quality', 'unknown')}",
        f"- Terrain confidence: {terrain_qc.get('terrain_confidence', 0.0)}",
        f"- Terrain class: {terrain_qc.get('terrain_class', 'unknown')}",
        f"- Valid ratio: {terrain_qc.get('valid_ratio', 0.0)}",
        f"- Raw range: {terrain_qc.get('raw_range', 0.0)}",
        f"- Clean range: {terrain_qc.get('clean_range', 0.0)}",
        f"- Outlier pixels: {terrain_qc.get('outlier_pixels', 0)}",
        f"- Spike pixels: {terrain_qc.get('spike_pixels', 0)}",
        f"- Recommended exaggeration: {terrain_qc.get('exaggeration_recommended', 1.0)}",
    ]
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in warnings)
    return "\n".join(lines)


def _brand_lab_payload(run_name: str, persona: str, intelligence: dict[str, Any]) -> dict[str, Any]:
    return {
        "front_text": run_name,
        "concept": "Neutral story/export staging area for terrain stills, captions, and shareable review material.",
        "story_panels": [
            "Terrain reveal",
            "Key landform callout",
            "Shareable still or panel export",
        ],
        "taglines": [
            "Read the ground.",
            "See the landform clearly.",
            "Terrain made reviewable.",
        ],
        "art_direction": [
            "Dark-field terrain hero renders",
            "Emboss-style relief stills",
            "Short motion reveal clips",
        ],
        "score_notes": {
            "discovery_score": intelligence.get("discovery_score", 0),
            "archaeology_score": intelligence.get("archaeological_signal_score", 0),
            "wildlife_score": intelligence.get("travel_corridor_strength", 0),
        },
        "order_handoff": {
            "status": "Concept only",
            "next_step": "Connect still exports, captions, or presentation assets later.",
        },
        "persona": persona,
    }


def run_analysis(bbox: tuple[float, float, float, float], run_name: str, persona: str, notes: str, dem_source: str = "auto", run_id: str | None = None, run_dir: Path | None = None) -> dict[str, Any]:
    ensure_vendor_assets()
    run_name = (run_name or "Monahinga Expedition").strip() or "Monahinga Expedition"
    run_id = run_id or make_run_id(run_name)
    run_dir = ensure_dir(run_dir or (RUNS_DIR / run_id))

    working_grid = _adaptive_working_grid(bbox)
    request_grid = _adaptive_request_grid(working_grid)
    browser_grid = _adaptive_browser_grid(bbox)
    bbox_width_m, bbox_height_m = _bbox_size_m(bbox)

    candidate_providers = list_candidate_providers(bbox=bbox, provider_preference=dem_source, data_dir=DATA_DIR)
    if _monahinga_render_safe_mode() and len(candidate_providers) > 1:
        # Render's Starter instance is memory-limited. Avoid holding multiple DEM
        # candidate rasters in memory during public proof runs.
        candidate_providers = candidate_providers[:1]
    candidates: list[SourceCandidate] = []
    fetch_failures: list[dict[str, str]] = []
    for provider_name in candidate_providers:
        try:
            candidate_request_grid = _render_cap_grid(_provider_request_grid(bbox, provider_name), RENDER_SAFE_REQUEST_GRID_CAP)
            candidate_working_grid = _render_cap_grid(_provider_working_grid(bbox, provider_name), RENDER_SAFE_WORKING_GRID_CAP)
            dem_bytes, provider_used = fetch_dem_for_provider(bbox=bbox, size=candidate_request_grid, provider=provider_name, data_dir=DATA_DIR)
            dem_raw, dem_meta = _read_dem_from_geotiff_bytes(dem_bytes)
            dem_resampled = _resample_to_grid(dem_raw, candidate_working_grid)
            source = TerrainSource(
                raw_dem=dem_resampled,
                provider=provider_used,
                source_name=provider_used,
                bbox=bbox,
                transform=dem_meta.get('transform'),
                crs=dem_meta.get('crs'),
                nodata_value=dem_meta.get('nodata'),
                pixel_size_x=(dem_meta.get('res') or (None, None))[0],
                pixel_size_y=(dem_meta.get('res') or (None, None))[1],
                nominal_resolution_m=(dem_meta.get('res') or (None, None))[0],
            )
            source, normalization_summary = normalize_source(source)
            qc_candidate = qualify_source(source)
            score, notes = score_candidate(source, qc_candidate)
            candidate_meta = dict(dem_meta)
            candidate_meta['normalization_summary'] = normalization_summary
            candidates.append(SourceCandidate(
                provider=provider_used,
                source_name=provider_used,
                raw_dem=source.raw_dem,
                meta=candidate_meta,
                qc=qc_candidate,
                arbitration_score=score,
                arbitration_notes=notes,
            ))
        except Exception as exc:
            fetch_failures.append({"provider": provider_name, "error": str(exc)})

    if not candidates:
        detail = '; '.join(f"{f['provider']}: {f['error']}" for f in fetch_failures) or 'No candidate sources could be fetched.'
        raise SiteInsightError(detail)

    ordered_candidates = sorted(candidates, key=lambda c: c.arbitration_score, reverse=True)
    chosen_candidate = ordered_candidates[0]
    if any(c.provider == "usgs" for c in ordered_candidates):
        best_usgs = sorted([c for c in ordered_candidates if c.provider == "usgs"], key=lambda c: c.arbitration_score, reverse=True)[0]
        if best_usgs.arbitration_score >= chosen_candidate.arbitration_score - 0.08:
            chosen_candidate = best_usgs
    provider_used = chosen_candidate.provider
    working_grid = _render_cap_grid(_provider_working_grid(bbox, provider_used), RENDER_SAFE_WORKING_GRID_CAP)
    request_grid = _render_cap_grid(_provider_request_grid(bbox, provider_used), RENDER_SAFE_REQUEST_GRID_CAP)
    browser_grid = _render_cap_grid(_provider_browser_grid(bbox, provider_used), RENDER_SAFE_BROWSER_GRID_CAP)
    if _monahinga_local_science_lab_mode():
        working_grid = min(_lab_cap_grid(working_grid, LOCAL_SCIENCE_WORKING_GRID_CAP), LOCAL_SCIENCE_WORKING_GRID_CAP)
        request_grid = min(_lab_cap_grid(request_grid, LOCAL_SCIENCE_REQUEST_GRID_CAP), LOCAL_SCIENCE_REQUEST_GRID_CAP)
        browser_grid = min(_lab_cap_grid(browser_grid, LOCAL_SCIENCE_BROWSER_GRID_CAP), LOCAL_SCIENCE_BROWSER_GRID_CAP)
    source = TerrainSource(
        raw_dem=chosen_candidate.raw_dem,
        provider=chosen_candidate.provider,
        source_name=chosen_candidate.source_name,
        bbox=bbox,
        transform=chosen_candidate.meta.get('transform'),
        crs=chosen_candidate.meta.get('crs'),
        nodata_value=chosen_candidate.meta.get('nodata'),
        pixel_size_x=(chosen_candidate.meta.get('res') or (None, None))[0],
        pixel_size_y=(chosen_candidate.meta.get('res') or (None, None))[1],
        nominal_resolution_m=(chosen_candidate.meta.get('res') or (None, None))[0],
    )
    source, normalization_summary = normalize_source(source)
    qc0 = chosen_candidate.qc
    source_arbitration = build_arbitration_summary(candidates, chosen_candidate)
    source_arbitration['normalization_summary'] = chosen_candidate.meta.get('normalization_summary', normalization_summary)
    if _monahinga_render_safe_mode():
        # Release non-chosen candidate arrays as early as possible on Render.
        candidates = [chosen_candidate]
        ordered_candidates = [chosen_candidate]
        gc.collect()
    if fetch_failures:
        source_arbitration["fetch_failures"] = fetch_failures
    surface = condition_surface(source, qc0)
    surface.normalization_summary = chosen_candidate.meta.get('normalization_summary', normalization_summary)
    surface.terrain_class = classify_surface(surface)
    surface.rendering_policy = select_rendering_policy(surface)
    surface.public_product_notes = {
        'public_readiness': surface.rendering_policy.public_readiness,
        'surface_style': surface.rendering_policy.product_surface_style,
        'export_profile': surface.rendering_policy.export_profile,
        'hero_export_ready': surface.rendering_policy.hero_export_ready,
        'diagnostic_only': surface.rendering_policy.diagnostic_only,
        'credit_gate_hook': 'Attach before /analyze execution and before premium run downloads.',
        'meter_unit': 'Prefer per full terrain run; reserve premium export credits for still/video outputs.',
        'free_preview_strategy': 'Allow web viewer previews while reserving premium still/video exports and bulk history for paid plans.',
    }
    surface.qc.diagnostics['exaggeration_recommended'] = surface.rendering_policy.vertical_exaggeration

    derivatives = compute_derivatives(surface)
    surface.derivative_products = derivatives
    terrain_objects = extract_terrain_objects(surface, derivatives, max_per_type=8)
    surface.terrain_objects = terrain_objects
    terrain_summary = summarize_derivatives(surface, derivatives)
    archaeology = detect_archaeology_candidates(surface, derivatives, top_n=8)
    archaeology_synthesis = build_archaeology_layer_synthesis(
        surface=surface,
        derivatives=derivatives,
        archaeology=archaeology,
        terrain_objects=terrain_objects,
        source_arbitration=source_arbitration,
    )
    terrain_graph = build_terrain_graph(surface, terrain_objects, archaeology=archaeology)
    surface.intelligence_graph = terrain_graph
    intelligence = build_intelligence(dem=surface.cleaned_dem, terrain_summary=terrain_summary, archaeology=archaeology, persona=persona, terrain_objects=terrain_objects, terrain_graph=terrain_graph)
    browser_surface = _build_browser_surface(surface, browser_grid)
    browser_derivatives = _build_browser_derivatives(derivatives, browser_grid, surface.cleaned_dem.shape)
    heightmap = build_heightmap_payload(browser_surface)
    heightmap['bbox_width_m'] = round(float(bbox_width_m), 2)
    heightmap['bbox_height_m'] = round(float(bbox_height_m), 2)
    viewer_layers = _build_viewer_layers(browser_surface, browser_derivatives, intelligence, archaeology)
    brand_lab = _brand_lab_payload(run_name, persona, intelligence)
    terrain_qc = surface.to_qc_dict()
    dem = surface.cleaned_dem
    valid_mask = surface.valid_mask

    center_lon, center_lat = bbox_center(bbox)
    manifest = {
        "run_id": run_id,
        "run_name": run_name,
        "created_at": utc_stamp(),
        "persona": persona,
        "notes": notes,
        "bbox": {"min_lon": bbox[0], "min_lat": bbox[1], "max_lon": bbox[2], "max_lat": bbox[3]},
        "center": {"lon": center_lon, "lat": center_lat},
        "bbox_area_deg": round(bbox_area_deg(bbox), 6),
        "provider": provider_used,
        "status": "complete",
        "headline_scores": {
            "terrain_signal_score": intelligence["terrain_signal_score"],
            "wildlife_movement_probability": intelligence["wildlife_movement_probability"],
            "archaeological_signal_score": intelligence["archaeological_signal_score"],
            "discovery_score": intelligence["discovery_score"],
        },
        "terrain_quality": terrain_qc.get("terrain_quality", "unknown"),
        "terrain_confidence": terrain_qc.get("terrain_confidence", 0.0),
        "terrain_class": terrain_qc.get("terrain_class", "unknown"),
        "browser_grid": browser_grid,
        "inspection_mode": "render_safe_public_proof" if _monahinga_render_safe_mode() else ("local_science_lab_high_detail" if _monahinga_local_science_lab_mode() else ("high_detail" if browser_grid >= 1152 else ("balanced_detail" if browser_grid >= 896 else "regional_context"))),
        "render_safe_mode": bool(_monahinga_render_safe_mode()),
        "local_science_lab_mode": bool(_monahinga_local_science_lab_mode()),
        "terrain_fidelity_specialist": {
            "role": "Dr. Source Fusion & Terrain Fidelity Specialist",
            "source_mesh_policy": "Balance source grid, working grid, viewer mesh, texture blend, and memory lane each run.",
            "render_lane": "stability first",
            "local_lab_lane": "fidelity first"
        },
        "surface_trust": terrain_qc.get("surface_trust", "mixed_surface"),
        "surface_trust_note": terrain_qc.get("surface_trust_note", "Treat this surface as contextual terrain evidence, not automatic proof."),
        "artifact_truth": {
            "artifact_stage": "final",
            "artifact_authority": "run_local_final_browser_export",
            "preview_artifacts_present": False,
            "analysis_grid": int(surface.cleaned_dem.shape[0]),
            "analysis_cols": int(surface.cleaned_dem.shape[1]),
            "browser_grid_requested": int(browser_grid),
            "browser_grid_actual": int(heightmap.get("rows", browser_grid)),
            "geometry_mode": heightmap.get("geometry_mode", "unknown"),
            "terrain_class": terrain_qc.get("terrain_class", "unknown"),
            "terrain_quality": terrain_qc.get("terrain_quality", "unknown"),
            "terrain_confidence": terrain_qc.get("terrain_confidence", 0.0),
            "inspection_mode": "high_detail" if browser_grid >= 1152 else ("balanced_detail" if browser_grid >= 896 else "regional_context"),
            "surface_trust": terrain_qc.get("surface_trust", "mixed_surface"),
            "final_export_policy": "canonical_browser_export",
            "final_export_note": "Viewer reads one final run-local export. Grid and geometry can still vary truthfully by source, bbox size, and terrain class.",
        },
        "files": {
            "viewer": "viewer3d.html",
            "context_map": "context_map.html",
            "heightmap_png": "heightmap.png",
            "heightmap_json": "heightmap.json",
            "heightmap_viewer_json": "heightmap_viewer.json",
            "validmask_png": "validmask.png",
            "terrain_qc": "terrain_qc.json",
            "terrain_surface": "terrain_surface.json",
            "terrain_contract": "terrain_contract.json",
            "product_diagnostics": "product_diagnostics.json",
            "conditioning_summary": "conditioning_summary.txt",
            "terrain_objects": "terrain_objects.json",
            "terrain_graph": "terrain_graph.json",
            "archaeology_synthesis": "archaeology_synthesis.json",
            "source_arbitration": "source_arbitration.json",
            "hillshade": "Hillshade.png",
            "slope": "Slope.png",
            "lrm": "LRM.png",
            "lrm_edges": "LRM_Edges.png",
            "openness_pos": "Openness_Pos.png",
            "openness_neg": "Openness_Neg.png",
            "svf": "SVF.png",
            "archaeology": "Archaeology.png",
            "discovery": "Discovery.png",
            "terrain_texture": "Terrain_Texture.png",
            "elevation": "Elevation.png",
        },
    }

    save_json(run_dir / "terrain_summary.json", terrain_summary)
    save_json(run_dir / "terrain_qc.json", terrain_qc)
    save_json(run_dir / "terrain_surface.json", surface.to_surface_dict())
    save_json(run_dir / "terrain_contract.json", surface.build_contract_dict())
    save_json(run_dir / "product_diagnostics.json", {
        'terrain_class': surface.terrain_class,
        'qualification_status': surface.qc.qualification_status,
        'qualification_notes': surface.qc.qualification_notes,
        'terrain_quality': surface.qc.terrain_quality,
        'terrain_confidence': surface.qc.overall_confidence,
        'rendering_policy': surface.rendering_policy.to_dict() if surface.rendering_policy else None,
        'commercial_hooks': surface.public_product_notes,
        'africa_case_guard': 'High spike or outlier burden now routes to artifact_suspicious with guarded derivatives and constrained exaggeration.',
    })
    save_json(run_dir / "archaeology.json", archaeology)
    save_json(run_dir / "archaeology_synthesis.json", archaeology_synthesis)
    save_json(run_dir / "terrain_objects.json", terrain_objects)
    save_json(run_dir / "terrain_graph.json", terrain_graph)
    save_json(run_dir / "source_arbitration.json", source_arbitration)
    save_json(run_dir / "intelligence.json", intelligence)
    if _monahinga_render_safe_mode():
        compact_heightmap = _build_fast_heightmap_payload(heightmap, max_dim=RENDER_SAFE_VIEWER_JSON_CAP)
        compact_heightmap["render_safe_mode"] = True
        compact_heightmap["render_safe_note"] = "Render Starter memory guard: compact JSON payload; local desktop quality path remains uncapped."
        save_json(run_dir / "heightmap.json", compact_heightmap)
        save_json(run_dir / "heightmap_viewer.json", compact_heightmap)
        save_json(run_dir / "viewer_layers.json", viewer_layers)
        del compact_heightmap
        gc.collect()
    elif _monahinga_local_science_lab_mode():
        lab_heightmap = _build_fast_heightmap_payload(heightmap, max_dim=LOCAL_SCIENCE_VIEWER_JSON_CAP)
        lab_heightmap["local_science_lab_mode"] = True
        lab_heightmap["science_lab_note"] = "Local high-detail science lab mode: larger viewer payload for close inspection. Not used on Render."
        save_json(run_dir / "heightmap.json", lab_heightmap)
        save_json(run_dir / "heightmap_viewer.json", lab_heightmap)
        save_json(run_dir / "viewer_layers.json", viewer_layers)
        del lab_heightmap
        gc.collect()
    else:
        save_json(run_dir / "heightmap.json", heightmap)
        save_json(run_dir / "heightmap_viewer.json", _build_fast_heightmap_payload(heightmap, max_dim=448))
        save_json(run_dir / "viewer_layers.json", viewer_layers)
    save_json(run_dir / "dem_metadata.json", {
        "provider": provider_used,
        "status": "complete",
        "mode": "provider-backed-restoration",
        "source_arbitration": source_arbitration,
        "bbox": list(bbox),
        "grid_size": working_grid,
        "pipeline_mode": "fast-first-cached-region-authority",
        "request_grid": request_grid,
        "browser_grid": browser_grid,
        "bbox_width_m": round(float(bbox_width_m), 2),
        "bbox_height_m": round(float(bbox_height_m), 2),
        "elevation_min": float(np.nanmin(dem)),
        "elevation_max": float(np.nanmax(dem)),
        "terrain_quality": terrain_qc.get("terrain_quality", "unknown"),
        "terrain_confidence": terrain_qc.get("terrain_confidence", 0.0),
        "terrain_class": terrain_qc.get("terrain_class", "unknown"),
    })
    save_json(run_dir / "brand_lab.json", brand_lab)
    save_text(run_dir / "architecture_map.txt", "\n".join([
        "Monahinga architecture map",
        "- Acquisition: global_dem.py -> provider fetchers",
        "- Qualification: terrain_qc.py evaluates coverage, voids, relief, coastal burden, and source suitability",
        "- Normalization: terrain_normalization.py standardizes nodata, numeric scale, dtype, and working-grid expectations",
        "- Conditioning: terrain_conditioning.py performs truth-preserving clipping, spike repair, coastal guards, and void fill",
        "- Classification: terrain_classification.py assigns subtle/moderate/high/artifact/water/void classes",
        "- Rendering policy: rendering_policy.py binds terrain class to 3D allowance, exaggeration, mesh profile, derivative profile, and public-readiness",
        "- Derivatives: derivatives.py adapts anomaly and relief behavior to policy profile",
        "- Objects/graph: terrain_objects.py + terrain_graph.py consume conditioned terrain products",
        "- Viewer/export: heightmap.py + pipeline.py package run-local artifacts for public-facing viewer delivery",
    ]) + "\n")
    save_text(run_dir / "commercial_strategy.txt", "\n".join([
        "Commercial strategy layer",
        "- GitHub/public-site path: keep /analyze as the clean execution boundary, and keep viewer payloads JSON-first for later embedding.",
        "- Credit logic attachment: meter successful analyze runs first; attach premium credits to export-grade stills, clip renders, and ZIP downloads later.",
        "- Free vs paid: free users get low-frequency preview runs and web viewer access; paid users unlock higher run volume, premium exports, saved project libraries, and cleaner commercial-use downloads.",
        "- Reel concept: 10-second terrain reveal showing flat raw patch -> qualified terrain class -> polished 3D hero surface with one trust caption.",
        "- Posting targets: Instagram reels, archaeology/hunting/terrain Facebook groups, Reddit communities that allow project showcases, and GitHub Pages landing demos.",
        "- Merch path: subtle-relief and moderate-relief outputs now preserve an emboss-friendly rendering mode suitable for later shirt/poster generation.",
    ]) + "\n")
    arbitration_lines = [f"- {c['provider']}: score {c['arbitration_score']}" for c in source_arbitration.get('candidates', [])]
    save_text(
        run_dir / "conditioning_summary.txt",
        _conditioning_summary_text(terrain_qc) + "\nSource arbitration:\n" + "\n".join(arbitration_lines) + "\n",
    )
    layers_np = {k: np.array(v, dtype=float) for k, v in viewer_layers["layers"].items()}
    _write_png(run_dir / "Hillshade.png", _matrix_to_rgb(layers_np["hillshade"], "hillshade"))
    _write_png(run_dir / "Slope.png", _matrix_to_rgb(layers_np["slope"], "slope"))
    _write_png(run_dir / "LRM.png", _matrix_to_rgb(layers_np["local_relief"], "local_relief"))
    _write_png(run_dir / "LRM_Edges.png", _matrix_to_rgb(np.clip(layers_np["local_relief"] * 1.2, 0, 1), "local_relief"))
    _write_png(run_dir / "Openness_Pos.png", _matrix_to_rgb(layers_np["openness"], "openness"))
    _write_png(run_dir / "Openness_Neg.png", _matrix_to_rgb(layers_np["openness_negative"], "openness"))
    _write_png(run_dir / "SVF.png", _matrix_to_rgb(layers_np["srv"], "srv"))
    _write_png(run_dir / "Archaeology.png", _matrix_to_rgb(layers_np["archaeology"], "archaeology"))
    _write_png(run_dir / "Discovery.png", _matrix_to_rgb(layers_np["discovery"], "discovery"))
    _write_png(run_dir / "Terrain_Texture.png", _matrix_to_rgb(_balanced_terrain_texture_matrix(layers_np), "terrain_texture"))
    _write_png(run_dir / "Elevation.png", _matrix_to_rgb(layers_np["elevation"], "elevation"))
    _write_png(run_dir / "heightmap.png", _matrix_to_rgb(np.clip(0.78 * layers_np["hillshade"] + 0.22 * layers_np["local_relief"], 0, 1), "hillshade"))
    _write_validmask_png(run_dir, browser_surface.valid_mask)
    _write_viewer3d_html(run_dir, run_id)
    _write_context_map_html(run_dir, run_id, bbox)
    artifact_audit = _artifact_audit(run_dir, manifest)
    manifest["status"] = "complete"
    manifest["progress_message"] = "Final terrain artifacts verified and published."
    save_json(run_dir / "manifest.json", manifest)

    initial_pins = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [center_lon, center_lat]}, "properties": {"label": "Run Center", "pin_type": "observation_point", "notes": "Auto-generated center point for this expedition run."}}]}
    save_pins(run_dir, initial_pins)
    save_text(run_dir / "summary.txt", f"Run: {run_name}\nPersona: {persona}\nProvider: {provider_used}\nDiscovery Score: {intelligence['discovery_score']}\n")
    save_json(run_dir / "intelligence_summary.json", intelligence)
    save_json(run_dir / "pam_probability.json", {"wildlife_movement_probability": intelligence["wildlife_movement_probability"], "likely_bedding_zones": intelligence["likely_bedding_zones"], "likely_feeding_zones": intelligence["likely_feeding_zones"]})
    save_json(run_dir / "routes.geojson", {"type": "FeatureCollection", "features": []})
    archaeology_features = {"type": "FeatureCollection", "features": []}
    for candidate in archaeology.get('candidates', []):
        gx = float(candidate.get('grid_x', 0)) / max(1, working_grid - 1)
        gy = float(candidate.get('grid_y', 0)) / max(1, working_grid - 1)
        lon = bbox[0] + gx * (bbox[2] - bbox[0])
        lat = bbox[3] - gy * (bbox[3] - bbox[1])
        archaeology_features['features'].append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": candidate})
    save_json(run_dir / "archaeology_features.geojson", archaeology_features)
    return {"run_id": run_id, "run_dir": str(run_dir)}

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Callable, Optional

from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject
import rasterio
import numpy as np

from .utils import SiteInsightError

REGION = "eu-central-1"
TILELIST_NAME = "tileList.txt"
TILE_PREFIX = "Copernicus_DSM_COG_30"
TILE_SUFFIX = "_DEM"
COPERNICUS_BUCKETS = {
    "glo30": "s3://copernicus-dem-30m",
    "glo90": "s3://copernicus-dem-90m",
}

ProgressCB = Optional[Callable[[str, int, str], None]]


def _cb(progress_cb: ProgressCB, step: str, pct: int, msg: str) -> None:
    if progress_cb:
        progress_cb(step, pct, msg)


def _run(cmd: list[str], timeout: int = 600) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, shell=False, timeout=timeout)
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except FileNotFoundError as e:
        return 127, "", str(e)
    except Exception as e:
        return 1, "", str(e)


def aws_cli_ready() -> bool:
    code, out, _ = _run(["aws", "--version"], timeout=20)
    return code == 0 and "aws-cli" in out.lower()


def ensure_aws_cli() -> None:
    if not aws_cli_ready():
        raise SiteInsightError(
            "AWS CLI is required for worldwide Copernicus DEM access in this build. "
            "Install AWS CLI, then retry your non-U.S. run."
        )


def _parse_bbox_tuple(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    if max_lat <= min_lat or max_lon <= min_lon:
        raise SiteInsightError("Invalid bbox passed to Copernicus DEM module.")
    return min_lon, min_lat, max_lon, max_lat


def _floor_int(x: float) -> int:
    return int(math.floor(x))


def _fmt_lat(lat_floor: int) -> str:
    hemi = "N" if lat_floor >= 0 else "S"
    deg = abs(lat_floor)
    return f"{hemi}{deg:02d}_00"


def _fmt_lon(lon_floor: int) -> str:
    hemi = "E" if lon_floor >= 0 else "W"
    deg = abs(lon_floor)
    return f"{hemi}{deg:03d}_00"


def bbox_to_tiles(bbox: tuple[float, float, float, float]) -> list[str]:
    min_lon, min_lat, max_lon, max_lat = _parse_bbox_tuple(bbox)
    eps = 1e-10
    lat_start = _floor_int(min_lat)
    lat_end = _floor_int(max_lat - eps)
    lon_start = _floor_int(min_lon)
    lon_end = _floor_int(max_lon - eps)

    tiles: list[str] = []
    for lat_deg in range(lat_start, lat_end + 1):
        for lon_deg in range(lon_start, lon_end + 1):
            tiles.append(f"{TILE_PREFIX}_{_fmt_lat(lat_deg)}_{_fmt_lon(lon_deg)}{TILE_SUFFIX}")
    return tiles


def _bucket_path(bucket_key: str) -> str:
    bucket = COPERNICUS_BUCKETS.get(bucket_key)
    if not bucket:
        raise SiteInsightError(f"Unsupported Copernicus bucket key: {bucket_key}")
    return bucket


def ensure_tilelist(cache_dir: Path, bucket_key: str, progress_cb: ProgressCB = None) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tilelist_path = cache_dir / f"{bucket_key}_{TILELIST_NAME}"
    if tilelist_path.exists() and tilelist_path.stat().st_size > 50_000:
        return tilelist_path

    ensure_aws_cli()
    bucket = _bucket_path(bucket_key)
    label = "GLO-30" if bucket_key == "glo30" else "GLO-90"
    _cb(progress_cb, "copernicus", 10, f"Downloading Copernicus {label} tile list...")
    code, out, err = _run(
        ["aws", "s3", "cp", f"{bucket}/{TILELIST_NAME}", str(tilelist_path), "--no-sign-request", "--region", REGION],
        timeout=300,
    )
    if code != 0 or not tilelist_path.exists():
        raise SiteInsightError(f"Failed to download Copernicus {label} tile list. {err or out}".strip())
    if tilelist_path.stat().st_size <= 50_000:
        raise SiteInsightError(f"Copernicus {label} tile list downloaded but looks too small.")
    return tilelist_path


def tile_exists(tilelist_path: Path, tile: str) -> bool:
    if not hasattr(tile_exists, "_cache"):
        tile_exists._cache = {}
    cache = tile_exists._cache
    key = str(tilelist_path)
    if key not in cache:
        cache[key] = {line.strip() for line in tilelist_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()}
    return tile in cache[key]


def _best_tif_name_from_listing(stdout: str) -> str | None:
    for line in stdout.splitlines():
        if ".tif" in line.lower():
            parts = line.split()
            if parts:
                fname = parts[-1].strip()
                if fname.lower().endswith(".tif"):
                    return fname
    return None


def download_tile(tile: str, tiles_dir: Path, bucket_key: str, progress_cb: ProgressCB = None) -> Path:
    tiles_dir.mkdir(parents=True, exist_ok=True)
    out_path = tiles_dir / f"{tile}_{bucket_key}.tif"
    if out_path.exists() and out_path.stat().st_size > 500_000:
        return out_path

    ensure_aws_cli()
    bucket = _bucket_path(bucket_key)
    label = "GLO-30" if bucket_key == "glo30" else "GLO-90"
    _cb(progress_cb, "copernicus", 30, f"Downloading Copernicus {label} tile {tile}...")
    primary = f"{bucket}/{tile}/{tile}.tif"
    code, out, err = _run(["aws", "s3", "cp", primary, str(out_path), "--no-sign-request", "--region", REGION], timeout=600)
    if code == 0 and out_path.exists() and out_path.stat().st_size > 500_000:
        return out_path

    _cb(progress_cb, "copernicus", 40, f"Primary path failed; listing folder for {tile}...")
    code2, out2, err2 = _run(["aws", "s3", "ls", f"{bucket}/{tile}/", "--no-sign-request", "--region", REGION], timeout=120)
    if code2 != 0:
        raise SiteInsightError(f"Failed to list Copernicus tile folder {tile}. {err2 or out2}".strip())
    tif_name = _best_tif_name_from_listing(out2)
    if not tif_name:
        raise SiteInsightError(f"Could not find a .tif file for Copernicus tile {tile}.")

    code3, out3, err3 = _run(
        ["aws", "s3", "cp", f"{bucket}/{tile}/{tif_name}", str(out_path), "--no-sign-request", "--region", REGION],
        timeout=600,
    )
    if code3 != 0 or not out_path.exists() or out_path.stat().st_size <= 500_000:
        raise SiteInsightError(f"Fallback tile download failed for Copernicus tile {tile}. {err3 or out3}".strip())
    return out_path


def resolve_and_download_tiles(bbox: tuple[float, float, float, float], cache_dir: Path, bucket_key: str, progress_cb: ProgressCB = None) -> list[Path]:
    tilelist = ensure_tilelist(cache_dir, bucket_key=bucket_key, progress_cb=progress_cb)
    tiles = bbox_to_tiles(bbox)
    missing = [t for t in tiles if not tile_exists(tilelist, t)]
    if missing:
        raise SiteInsightError(f"Copernicus {bucket_key} tiles missing from tile list: {missing}")

    tiles_dir = cache_dir / bucket_key / "tiles"
    return [download_tile(tile, tiles_dir, bucket_key=bucket_key, progress_cb=progress_cb) for tile in tiles]


def _fetch_copernicus_dem_geotiff_for_bucket(
    bbox: tuple[float, float, float, float],
    size: int,
    cache_dir: Path,
    bucket_key: str,
    progress_cb: ProgressCB = None,
) -> tuple[bytes, str]:
    min_lon, min_lat, max_lon, max_lat = _parse_bbox_tuple(bbox)
    tile_paths = resolve_and_download_tiles(bbox, cache_dir, bucket_key=bucket_key, progress_cb=progress_cb)
    if not tile_paths:
        raise SiteInsightError("No Copernicus DEM tiles were resolved for the requested bbox.")

    sources = [rasterio.open(str(p)) for p in tile_paths]
    try:
        mosaic, src_transform = merge(sources, method="first")
        src_arr = np.asarray(mosaic[0], dtype="float32")
        src_crs = sources[0].crs or CRS.from_epsg(4326)
        src_nodata = sources[0].nodata

        dst = np.full((int(size), int(size)), -9999.0, dtype="float32")
        dst_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, int(size), int(size))
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=CRS.from_epsg(4326),
            dst_nodata=-9999.0,
            resampling=Resampling.lanczos,
        )

        profile = {
            "driver": "GTiff",
            "height": int(size),
            "width": int(size),
            "count": 1,
            "dtype": "float32",
            "crs": CRS.from_epsg(4326),
            "transform": dst_transform,
            "nodata": -9999.0,
        }
        with MemoryFile() as mem:
            with mem.open(**profile) as ds:
                ds.write(dst, 1)
            label = "copernicus_30m" if bucket_key == "glo30" else "copernicus_90m"
            return mem.read(), label
    finally:
        for src in sources:
            src.close()


def fetch_copernicus_dem_geotiff(bbox: tuple[float, float, float, float], size: int, cache_dir: Path, progress_cb: ProgressCB = None) -> tuple[bytes, str]:
    errors: list[str] = []
    for bucket_key in ("glo30", "glo90"):
        try:
            return _fetch_copernicus_dem_geotiff_for_bucket(bbox, size, cache_dir, bucket_key=bucket_key, progress_cb=progress_cb)
        except Exception as exc:
            errors.append(f"{bucket_key}: {exc}")
    raise SiteInsightError("Copernicus fetch failed. " + "; ".join(errors))

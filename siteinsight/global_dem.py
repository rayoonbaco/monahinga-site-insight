from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .copernicus_dem import aws_cli_ready, fetch_copernicus_dem_geotiff
from .usgs import USGSRequest, fetch_usgs_dem_geotiff
from .utils import SiteInsightError, ensure_dir, load_json, save_json

SETTINGS_FILENAME = "provider_settings.json"


def _cache_dir(data_dir: Path) -> Path:
    return ensure_dir(data_dir / "dem_cache")


def _cache_key(bbox: tuple[float, float, float, float], size: int, provider: str) -> str:
    token = f"{provider}|{size}|{bbox[0]:.6f}|{bbox[1]:.6f}|{bbox[2]:.6f}|{bbox[3]:.6f}"
    return hashlib.sha1(token.encode("utf-8")).hexdigest()[:20]


def _cache_path(data_dir: Path, bbox: tuple[float, float, float, float], size: int, provider: str) -> Path:
    return _cache_dir(data_dir) / provider / f"{_cache_key(bbox, size, provider)}.tif"


def _read_cache(path: Path) -> bytes | None:
    if path.exists() and path.stat().st_size > 1024:
        return path.read_bytes()
    return None


def _write_cache(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _settings_path(data_dir: Path) -> Path:
    return data_dir / SETTINGS_FILENAME


def load_provider_settings(data_dir: Path) -> dict[str, Any]:
    path = _settings_path(data_dir)
    file_data = load_json(path, {})
    if not isinstance(file_data, dict):
        file_data = {}
    return {
        "copernicus": {
            "mode": "aws-open-data",
            "bucket": "s3://copernicus-dem-90m",
            "region": "eu-central-1",
            "aws_cli_ready": aws_cli_ready(),
        },
        "ui": {"show_provider_settings": False},
        "saved": file_data,
    }


def save_copernicus_settings(data_dir: Path, client_id: str, client_secret: str) -> None:
    path = _settings_path(data_dir)
    data = load_json(path, {})
    if not isinstance(data, dict):
        data = {}
    note = (client_id or client_secret or "").strip()
    if note:
        data["legacy_note"] = "Credentials are no longer required for Copernicus DEM in this build."
    save_json(path, data)


def provider_ready(data_dir: Path) -> bool:
    return aws_cli_ready()


def provider_ui_visible(data_dir: Path) -> bool:
    return True


def _bbox_looks_like_usa(bbox: tuple[float, float, float, float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    return 15.0 <= min_lat <= 75.0 and 15.0 <= max_lat <= 75.0 and -180.0 <= min_lon <= -60.0 and -180.0 <= max_lon <= -60.0




def list_candidate_providers(bbox: tuple[float, float, float, float], provider_preference: str, data_dir: Path) -> list[str]:
    pref = (provider_preference or "auto").strip().lower()
    if pref in {"usgs", "copernicus"}:
        return [pref]

    providers: list[str] = []
    if _bbox_looks_like_usa(bbox):
        providers.append("usgs")
        if aws_cli_ready():
            providers.append("copernicus")
    else:
        providers.append("copernicus")
    return providers


def fetch_dem_for_provider(bbox: tuple[float, float, float, float], size: int, provider: str, data_dir: Path) -> tuple[bytes, str]:
    provider = (provider or "copernicus").strip().lower()
    copernicus_cache_dir = ensure_dir(data_dir / "copernicus_dem")
    cache_path = _cache_path(data_dir, bbox, size, provider)
    cached = _read_cache(cache_path)
    if cached is None and provider == "copernicus":
        matches = sorted(cache_path.parent.glob(f"{cache_path.stem}_*.tif"))
        if matches:
            cached = _read_cache(matches[0])
            if cached is not None:
                label = matches[0].stem.split("_")[-2] + "_" + matches[0].stem.split("_")[-1] if matches[0].stem.endswith(("30m", "90m")) else "copernicus"
                return cached, label
    if cached is not None:
        if provider == "usgs":
            return cached, "usgs"
        if provider == "copernicus":
            return cached, "copernicus"
    if provider == "usgs":
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        payload = fetch_usgs_dem_geotiff(USGSRequest(bbox=bbox_str, size=size))
        _write_cache(cache_path, payload)
        return payload, "usgs"
    if provider == "copernicus":
        payload, label = fetch_copernicus_dem_geotiff(bbox, size, copernicus_cache_dir)
        _write_cache(cache_path.with_name(f"{cache_path.stem}_{label}.tif"), payload)
        return payload, label
    raise SiteInsightError(f"Unsupported DEM provider requested: {provider}")

def fetch_dem_geotiff_bytes(bbox: tuple[float, float, float, float], size: int, provider_preference: str, data_dir: Path) -> tuple[bytes, str]:
    pref = (provider_preference or "auto").strip().lower()
    if pref not in {"auto", "usgs", "copernicus"}:
        pref = "auto"

    if pref == "usgs":
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        return fetch_usgs_dem_geotiff(USGSRequest(bbox=bbox_str, size=size)), "usgs"

    copernicus_cache_dir = ensure_dir(data_dir / "copernicus_dem")

    if pref == "copernicus":
        return fetch_copernicus_dem_geotiff(bbox, size, copernicus_cache_dir)

    if _bbox_looks_like_usa(bbox):
        try:
            bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            return fetch_usgs_dem_geotiff(USGSRequest(bbox=bbox_str, size=size)), "usgs"
        except Exception:
            if aws_cli_ready():
                return fetch_copernicus_dem_geotiff(bbox, size, copernicus_cache_dir)
            raise SiteInsightError(
                "USGS request failed and AWS CLI is not available for Copernicus global fallback. "
                "Install AWS CLI or try a smaller USA bbox."
            )

    return fetch_copernicus_dem_geotiff(bbox, size, copernicus_cache_dir)

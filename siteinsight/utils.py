from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

APP_NAME = "Monahinga Archaeology Terrain Viewer"
ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
RUNS_DIR = REPORTS_DIR / "runs"
VENDOR_DIR = REPORTS_DIR / "_vendor"
DATA_DIR = REPORTS_DIR / "_data"

REPORTS_DIR.mkdir(exist_ok=True, parents=True)
RUNS_DIR.mkdir(exist_ok=True, parents=True)
VENDOR_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)


class SiteInsightError(RuntimeError):
    pass


def utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def slugify(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "run"


def make_run_id(run_name: str) -> str:
    return f"{utc_stamp()}_{slugify(run_name)[:40]}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_bbox_string(bbox: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("Expected 4 comma-separated numbers.")
    min_lon, min_lat, max_lon, max_lat = [float(p) for p in parts]
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("Min values must be smaller than max values.")
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise ValueError("Longitude must be between -180 and 180.")
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError("Latitude must be between -90 and 90.")
    if (max_lon - min_lon) > 2.0 or (max_lat - min_lat) > 2.0:
        raise ValueError("Bounding box is too large for this build. Keep each dimension at or below 2 degrees.")
    return min_lon, min_lat, max_lon, max_lat


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    return ((min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0)


def bbox_area_deg(bbox: tuple[float, float, float, float]) -> float:
    min_lon, min_lat, max_lon, max_lat = bbox
    return abs(max_lon - min_lon) * abs(max_lat - min_lat)


def percent_rank(value: float, values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    less = sum(1 for v in values if v <= value)
    return 100.0 * less / len(values)


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(RUNS_DIR.glob("*"), reverse=True):
        if not path.is_dir():
            continue
        manifest = load_json(path / "manifest.json", {})
        if manifest:
            runs.append(
                {
                    "run_id": path.name,
                    "run_name": manifest.get("run_name", path.name),
                    "created_at": manifest.get("created_at", ""),
                    "persona": manifest.get("persona", ""),
                    "bbox": manifest.get("bbox", ""),
                    "discovery_score": manifest.get("headline_scores", {}).get("discovery_score", 0),
                }
            )
        else:
            runs.append({"run_id": path.name, "run_name": path.name})
        if len(runs) >= limit:
            break
    return runs


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))

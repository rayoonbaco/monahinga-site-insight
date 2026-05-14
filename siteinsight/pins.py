from __future__ import annotations

from pathlib import Path
from typing import Any

from siteinsight.utils import load_json, save_json


def _pins_path(run_dir: Path) -> Path:
    return run_dir / "pins.geojson"


def _summary_path(run_dir: Path) -> Path:
    return run_dir / "pins_summary.json"


def _empty_feature_collection() -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": []}


def load_pins(run_dir: Path) -> dict[str, Any]:
    return load_json(_pins_path(run_dir), _empty_feature_collection())


def _build_summary(data: dict[str, Any]) -> dict[str, Any]:
    features = data.get("features", [])
    by_type: dict[str, int] = {}
    for feature in features:
        pin_type = feature.get("properties", {}).get("pin_type", "unknown")
        by_type[pin_type] = by_type.get(pin_type, 0) + 1
    return {
        "count": len(features),
        "by_type": by_type,
    }


def save_pins(run_dir: Path, data: dict[str, Any]) -> None:
    save_json(_pins_path(run_dir), data)
    save_json(_summary_path(run_dir), _build_summary(data))


def add_pin(
    run_dir: Path,
    label: str,
    pin_type: str,
    lat: float,
    lon: float,
    notes: str = "",
) -> dict[str, Any]:
    data = load_pins(run_dir)
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat],
        },
        "properties": {
            "label": label,
            "pin_type": pin_type,
            "notes": notes,
        },
    }
    data.setdefault("features", []).append(feature)
    save_pins(run_dir, data)
    return feature
from __future__ import annotations

from typing import Any

import numpy as np

from .terrain_models import TerrainSurface


def _safe_percentile(values: np.ndarray, q: float, fallback: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(fallback)
    return float(np.percentile(arr, q))


def _window_stats(arr: np.ndarray, y: int, x: int, radius: int = 4) -> tuple[float, float, float]:
    rows, cols = arr.shape
    y0 = max(0, y - radius)
    y1 = min(rows, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(cols, x + radius + 1)
    local = np.asarray(arr[y0:y1, x0:x1], dtype=float)
    finite = local[np.isfinite(local)]
    if finite.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.nanmean(finite)), float(np.nanstd(finite)), float(np.nanmax(finite) - np.nanmin(finite))


def extract_terrain_objects(surface: TerrainSurface, derivatives: dict[str, Any], max_per_type: int = 8) -> dict[str, Any]:
    dem = np.asarray(surface.cleaned_dem, dtype=float)
    slope = np.asarray(derivatives['slope'], dtype=float)
    ridge = np.asarray(derivatives['ridge_strength'], dtype=float)
    valley = np.asarray(derivatives['valley_strength'], dtype=float)
    local_relief = np.asarray(derivatives['relief_local'], dtype=float)
    anomaly = np.asarray(derivatives.get('anomaly_response', np.zeros_like(dem)), dtype=float)

    rows, cols = dem.shape
    ridge_thr = _safe_percentile(ridge, 92, 0.0)
    valley_thr = _safe_percentile(valley, 92, 0.0)
    bench_slope_thr = _safe_percentile(slope, 35 if surface.terrain_class in {'subtle_relief', 'subtle_archaeology'} else 28, 0.0)
    anomaly_thr = _safe_percentile(anomaly, 95 if surface.terrain_class in {'artifact_suspicious'} else 90, 0.0)
    relief_thr = _safe_percentile(local_relief, 65 if surface.terrain_class in {'subtle_relief', 'subtle_archaeology'} else 75, 0.0)

    features: list[dict[str, Any]] = []

    def add_candidates(mask: np.ndarray, kind: str, strength_source: np.ndarray, stride: int = 6) -> None:
        used: set[tuple[int, int]] = set()
        flat = np.argsort(strength_source.flatten())[::-1]
        count = 0
        for idx in flat:
            y, x = divmod(int(idx), cols)
            if not bool(mask[y, x]):
                continue
            key = (y // stride, x // stride)
            if key in used:
                continue
            used.add(key)
            mean_local, std_local, rise_local = _window_stats(dem, y, x, radius=4)
            features.append({
                'id': f'{kind}_{count+1}',
                'feature_type': kind,
                'grid_x': int(x),
                'grid_y': int(y),
                'strength': round(float(strength_source[y, x]), 4),
                'local_mean_elevation': round(mean_local, 3),
                'local_std_elevation': round(std_local, 3),
                'local_rise': round(rise_local, 3),
            })
            count += 1
            if count >= max_per_type:
                break

    ridge_mask = ridge >= ridge_thr
    valley_mask = valley >= valley_thr
    bench_mask = (slope <= bench_slope_thr) & (local_relief >= relief_thr)
    anomaly_mask = anomaly >= anomaly_thr

    add_candidates(ridge_mask, 'ridge_node', ridge)
    add_candidates(valley_mask, 'valley_node', valley)
    add_candidates(bench_mask, 'bench_candidate', local_relief)
    add_candidates(anomaly_mask, 'anomaly_candidate', anomaly, stride=5)

    type_counts: dict[str, int] = {}
    for feature in features:
        type_counts[feature['feature_type']] = type_counts.get(feature['feature_type'], 0) + 1

    ridge_density = round(type_counts.get('ridge_node', 0) / max(1, rows * cols) * 1000, 4)
    anomaly_density = round(type_counts.get('anomaly_candidate', 0) / max(1, rows * cols) * 1000, 4)

    return {
        'terrain_class': surface.terrain_class,
        'feature_count': len(features),
        'feature_type_counts': type_counts,
        'ridge_density': ridge_density,
        'anomaly_density': anomaly_density,
        'features': features,
        'summary': {
            'topography_mode': 'quiet_object_mode' if surface.terrain_class in {'subtle_relief', 'subtle_archaeology'} else 'standard_object_mode',
            'notes': 'Terrain objects are heuristic nodes extracted from derivative peaks and quiet benches. They are evidence anchors, not validated landform claims.',
        },
    }

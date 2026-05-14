from __future__ import annotations

from typing import Any

import numpy as np

from .terrain_models import TerrainSurface


def _smooth_grid(values: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    for _ in range(max(0, iterations)):
        acc = np.zeros_like(out)
        count = 0.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += np.roll(np.roll(out, dy, axis=0), dx, axis=1)
                count += 1.0
        out = acc / max(count, 1.0)
    return out


def _local_contrast_preserve(arr: np.ndarray, finite: np.ndarray, weight: float = 0.22) -> np.ndarray:
    broad = _smooth_grid(arr, 3)
    detail = arr - broad
    vals = detail[finite]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=float)
    lo, hi = np.percentile(vals, [3, 97])
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-9:
        return np.zeros_like(arr, dtype=float)
    local = np.clip((detail - lo) / (hi - lo), 0.0, 1.0)
    return np.clip((1.0 - weight) * 0.5 + weight * local, 0.0, 1.0)


def _normalize(dem: np.ndarray, mode: str = 'robust_percentile', valid_mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(dem, dtype=float).copy()
    finite = np.isfinite(arr)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)

    vals = arr[finite]
    if mode == 'local_relief':
        broad = _smooth_grid(arr, 2)
        work = np.clip(arr - broad, np.percentile(arr[finite] - broad[finite], 2), np.percentile(arr[finite] - broad[finite], 98))
        lo, hi = np.percentile(work[finite], [2, 98])
    elif mode == 'land_robust':
        low = np.percentile(vals, 8)
        hi = np.percentile(vals, 98)
        lo = np.percentile(vals[vals >= low], 5) if np.any(vals >= low) else low
    else:
        lo, hi = np.percentile(vals, [2, 98])

    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _viewer_vertical_scale(surface: TerrainSurface) -> float:
    """Choose the viewer-only vertical scale.

    This does not change the DEM or scientific derivatives. It only controls how strongly
    the browser presents the terrain mesh so broad-context runs read like real landforms
    instead of flat grayscale sheets.
    """
    relief_p95 = float(surface.relief_stats.get('relief_p95', 0.0))
    relief_p75 = float(surface.relief_stats.get('relief_p50', 0.0))
    qc = surface.qc
    terrain_class = surface.terrain_class

    base = float(surface.rendering_policy.vertical_exaggeration)

    if terrain_class in {'artifact_suspicious', 'insufficient_surface', 'void_dominated'}:
        base = min(base, 0.78)
    elif terrain_class in {'subtle_relief', 'subtle_archaeology'}:
        base = max(base, 1.18)
    elif terrain_class in {'water_influenced', 'coastal_transition'}:
        base = max(base, 1.05)
    elif terrain_class == 'moderate_relief':
        base = max(base, 1.28)
    else:
        base = max(base, 1.42)

    if relief_p95 < 0.35:
        base *= 1.18
    elif relief_p95 < 0.75:
        base *= 1.14
    elif relief_p75 < 1.5:
        base *= 1.10
    else:
        base *= 1.06

    if qc.spike_ratio > 0.01:
        base = min(base, 0.86)
    if qc.spike_ratio > 0.03:
        base = min(base, 0.70)

    if terrain_class in {'artifact_suspicious', 'insufficient_surface', 'void_dominated'}:
        return round(float(max(0.58, min(base, 0.86))), 2)
    return round(float(max(1.05, min(base, 2.15))), 2)


def _build_geometry_values(surface: TerrainSurface) -> tuple[np.ndarray, str, int]:
    dem = np.asarray(surface.cleaned_dem, dtype=float)
    valid_mask = np.asarray(surface.valid_mask, dtype=bool)
    policy = surface.rendering_policy
    terrain_class = surface.terrain_class

    base = _normalize(dem, 'land_robust' if policy.z_normalization == 'land_robust' else 'robust_percentile', valid_mask=valid_mask)
    detail = _normalize(dem, 'local_relief', valid_mask=valid_mask)

    if policy.mesh_profile == 'guarded' or terrain_class in {'artifact_suspicious', 'insufficient_surface', 'void_dominated'}:
        geometry = np.clip(_smooth_grid(base, 3), 0.0, 1.0)
        return geometry, 'guarded_smoothed', 3
    if policy.mesh_profile == 'coastal_guarded' or terrain_class in {'water_influenced', 'coastal_transition'}:
        broad = _smooth_grid(base, 3)
        geometry = np.clip(0.95 * broad + 0.05 * detail, 0.0, 1.0)
        return geometry, 'coastal_guarded_surface', 2
    if policy.mesh_profile == 'subtle' or terrain_class in {'subtle_relief', 'subtle_archaeology'}:
        broad = _smooth_grid(base, 2)
        fine = np.clip(0.72 * detail + 0.28 * _smooth_grid(detail, 1), 0.0, 1.0)
        geometry = np.clip(0.82 * broad + 0.18 * fine, 0.0, 1.0)
        return geometry, 'subtle_relief_surface', 1
    if terrain_class == 'moderate_relief':
        broad = _smooth_grid(base, 2)
        fine = np.clip(0.62 * detail + 0.38 * _smooth_grid(detail, 1), 0.0, 1.0)
        geometry = np.clip(0.88 * broad + 0.12 * fine, 0.0, 1.0)
        return geometry, 'moderate_relief_surface', 1
    geometry = np.clip(0.92 * _smooth_grid(base, 1) + 0.08 * detail, 0.0, 1.0)
    return geometry, 'terrain_elevation', 0


def build_heightmap_payload(surface_or_dem: TerrainSurface | np.ndarray, exaggeration_default: float = 1.6, qc: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(surface_or_dem, TerrainSurface):
        surface = surface_or_dem
        dem = np.asarray(surface.cleaned_dem, dtype=float)
        policy = surface.rendering_policy
        qc = surface.to_qc_dict()
        exaggeration_default = policy.vertical_exaggeration
        normalized = _normalize(dem, policy.z_normalization, valid_mask=surface.valid_mask)
        geometry_values, geometry_mode, smooth_iterations = _build_geometry_values(surface)
        warnings = list(policy.viewer_warnings)
        terrain_quality = qc.get('terrain_quality', 'unknown')
        terrain_class = surface.terrain_class
        allow_3d = policy.allow_3d
        rendering_policy = policy.to_dict()
        public_product_notes = surface.public_product_notes
        viewer_vertical_scale = _viewer_vertical_scale(surface)
    else:
        dem = np.asarray(surface_or_dem, dtype=float)
        normalized = _normalize(dem, 'robust_percentile')
        geometry_values = normalized.copy()
        geometry_mode = 'fallback_elevation'
        smooth_iterations = 1
        warnings = list((qc or {}).get('warnings', []))
        terrain_quality = (qc or {}).get('terrain_quality', 'unknown')
        terrain_class = (qc or {}).get('terrain_class', 'unknown')
        allow_3d = True
        rendering_policy = None
        public_product_notes = {}
        viewer_vertical_scale = 1.8

    rows, cols = dem.shape
    zmin = float(np.nanmin(dem)) if np.isfinite(dem).any() else 0.0
    zmax = float(np.nanmax(dem)) if np.isfinite(dem).any() else 0.0

    return {
        'rows': rows,
        'cols': cols,
        'zmin': zmin,
        'zmax': zmax,
        'exaggeration_default': round(float(exaggeration_default), 2),
        'viewer_vertical_scale': round(float(viewer_vertical_scale), 2),
        'mesh_smoothing_passes': int(smooth_iterations),
        'geometry_mode': geometry_mode,
        'terrain_confidence': round(float((qc or {}).get('terrain_confidence', 0.0)), 4),
        'terrain_quality': terrain_quality,
        'terrain_class': terrain_class,
        'allow_3d': allow_3d,
        'warnings': warnings,
        'rendering_policy': rendering_policy,
        'public_product_notes': public_product_notes,
        'values': normalized.round(3).tolist(),
        'geometry_values': geometry_values.round(3).tolist(),
    }

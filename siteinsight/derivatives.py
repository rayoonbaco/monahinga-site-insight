from __future__ import annotations

from typing import Any

import numpy as np

from .terrain_models import TerrainSurface


def _safe_percentile(arr: np.ndarray, q: float, fallback: float = 0.0) -> float:
    finite = np.asarray(arr, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(fallback)
    return float(np.percentile(finite, q))


def _as_surface(surface_or_dem: TerrainSurface | np.ndarray) -> tuple[np.ndarray, TerrainSurface | None]:
    if isinstance(surface_or_dem, TerrainSurface):
        return np.asarray(surface_or_dem.cleaned_dem, dtype=float), surface_or_dem
    return np.asarray(surface_or_dem, dtype=float), None


def compute_derivatives(surface_or_dem: TerrainSurface | np.ndarray) -> dict[str, Any]:
    dem, surface = _as_surface(surface_or_dem)
    valid_mask = np.ones_like(dem, dtype=bool) if surface is None else np.asarray(surface.valid_mask, dtype=bool)

    gy, gx = np.gradient(dem)
    slope = np.sqrt(gx**2 + gy**2)
    aspect = np.arctan2(-gy, gx)

    lap = (
        -4 * dem
        + np.roll(dem, 1, axis=0)
        + np.roll(dem, -1, axis=0)
        + np.roll(dem, 1, axis=1)
        + np.roll(dem, -1, axis=1)
    )

    relief_small = (
        np.maximum.reduce([dem, np.roll(dem, 1, 0), np.roll(dem, -1, 0), np.roll(dem, 1, 1), np.roll(dem, -1, 1)])
        - np.minimum.reduce([dem, np.roll(dem, 1, 0), np.roll(dem, -1, 0), np.roll(dem, 1, 1), np.roll(dem, -1, 1)])
    )
    relief_medium = (
        np.maximum.reduce([dem, np.roll(dem, 3, 0), np.roll(dem, -3, 0), np.roll(dem, 3, 1), np.roll(dem, -3, 1)])
        - np.minimum.reduce([dem, np.roll(dem, 3, 0), np.roll(dem, -3, 0), np.roll(dem, 3, 1), np.roll(dem, -3, 1)])
    )

    ridge_strength = np.clip(-lap, 0, None)
    valley_strength = np.clip(lap, 0, None)
    edge_strength = np.clip(slope - np.nanmedian(slope), 0, None)

    terrain_class = surface.terrain_class if surface else 'unknown'
    derivative_profile = surface.rendering_policy.derivative_profile if surface and surface.rendering_policy else 'balanced'

    if derivative_profile == 'micro_relief':
        local_weight, medium_weight, slope_weight, anomaly_scale = 0.62, 0.23, 0.15, 1.08
    elif derivative_profile == 'land_bias':
        local_weight, medium_weight, slope_weight, anomaly_scale = 0.46, 0.34, 0.20, 0.9
    elif derivative_profile == 'subtle_land':
        local_weight, medium_weight, slope_weight, anomaly_scale = 0.55, 0.27, 0.18, 0.92
    elif derivative_profile == 'guarded':
        local_weight, medium_weight, slope_weight, anomaly_scale = 0.34, 0.36, 0.30, 0.7
    elif derivative_profile == 'suppressed':
        local_weight, medium_weight, slope_weight, anomaly_scale = 0.25, 0.35, 0.40, 0.45
    else:
        local_weight, medium_weight, slope_weight, anomaly_scale = 0.35, 0.45, 0.20, 1.0

    anomaly_response = np.clip(
        anomaly_scale * (
            local_weight * (relief_small / max(_safe_percentile(relief_small, 95, 1.0), 1e-9))
            + medium_weight * (relief_medium / max(_safe_percentile(relief_medium, 95, 1.0), 1e-9))
            + slope_weight * (edge_strength / max(_safe_percentile(edge_strength, 95, 1.0), 1e-9))
        ),
        0,
        1,
    )

    if terrain_class in {'artifact_suspicious', 'insufficient_surface', 'void_dominated'}:
        anomaly_response *= 0.7
        ridge_strength *= 0.78
        valley_strength *= 0.78
    elif terrain_class in {'coastal_transition', 'water_influenced'}:
        anomaly_response *= 0.88
        relief_medium *= 0.88

    for layer in (slope, aspect, lap, relief_small, relief_medium, ridge_strength, valley_strength, edge_strength, anomaly_response):
        layer[~valid_mask] = 0.0

    return {
        'slope': slope,
        'aspect': aspect,
        'laplacian': lap,
        'relief_local': relief_small,
        'relief_medium': relief_medium,
        'ridge_strength': ridge_strength,
        'valley_strength': valley_strength,
        'edge_strength': edge_strength,
        'anomaly_response': anomaly_response,
        'derivative_profile': derivative_profile,
    }


def summarize_derivatives(surface_or_dem: TerrainSurface | np.ndarray, d: dict[str, Any], qc: dict[str, Any] | None = None) -> dict[str, Any]:
    dem, surface = _as_surface(surface_or_dem)
    if surface is not None:
        qc = surface.to_qc_dict()
    slope = np.asarray(d['slope'], dtype=float)
    relief_local = np.asarray(d['relief_local'], dtype=float)
    ridge_strength = np.asarray(d['ridge_strength'], dtype=float)
    valley_strength = np.asarray(d['valley_strength'], dtype=float)
    edge_strength = np.asarray(d['edge_strength'], dtype=float)

    steep_ratio = float((slope > _safe_percentile(slope, 75)).mean())
    ridge_ratio = float((ridge_strength > _safe_percentile(ridge_strength, 80)).mean())
    valley_ratio = float((valley_strength > _safe_percentile(valley_strength, 80)).mean())
    edge_ratio = float((edge_strength > _safe_percentile(edge_strength, 80)).mean())

    summary = {
        'elevation_range': round(float(np.nanmax(dem) - np.nanmin(dem)), 2),
        'mean_elevation': round(float(np.nanmean(dem)), 2),
        'mean_slope': round(float(np.nanmean(slope)), 2),
        'max_slope': round(float(np.nanmax(slope)), 2),
        'mean_local_relief': round(float(np.nanmean(relief_local)), 2),
        'steep_ratio': round(steep_ratio, 4),
        'ridge_ratio': round(ridge_ratio, 4),
        'valley_ratio': round(valley_ratio, 4),
        'edge_ratio': round(edge_ratio, 4),
        'derivative_profile': d.get('derivative_profile', 'balanced'),
    }
    if qc:
        summary.update({
            'terrain_confidence': round(float(qc.get('terrain_confidence', 0.0)), 4),
            'terrain_quality': qc.get('terrain_quality', 'unknown'),
            'valid_ratio': round(float(qc.get('valid_ratio', 0.0)), 4),
            'outlier_pixels': int(qc.get('outlier_pixels', 0)),
            'spike_pixels': int(qc.get('spike_pixels', 0)),
            'terrain_warnings': list(qc.get('warnings', [])),
            'terrain_class': qc.get('terrain_class', 'unknown'),
            'public_readiness': ((qc.get('rendering_policy') or {}).get('public_readiness') if isinstance(qc.get('rendering_policy'), dict) else None),
        })
    return summary

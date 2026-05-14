from __future__ import annotations

import numpy as np

from .terrain_models import TerrainQC, TerrainSource


def _safe_percentile(values: np.ndarray, q: float, fallback: float = 0.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(fallback)
    return float(np.percentile(finite, q))


def _local_relief(arr: np.ndarray) -> np.ndarray:
    work = np.asarray(arr, dtype=float)
    return (
        np.maximum.reduce([work, np.roll(work, 1, 0), np.roll(work, -1, 0), np.roll(work, 1, 1), np.roll(work, -1, 1)])
        - np.minimum.reduce([work, np.roll(work, 1, 0), np.roll(work, -1, 0), np.roll(work, 1, 1), np.roll(work, -1, 1)])
    )


def qualify_source(source: TerrainSource) -> TerrainQC:
    arr = np.asarray(source.raw_dem, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    arr[(arr < -600.0) | (arr > 10000.0)] = np.nan
    finite = np.isfinite(arr)
    valid_ratio = float(finite.mean()) if arr.size else 0.0
    void_ratio = 1.0 - valid_ratio

    if not finite.any():
        return TerrainQC(
            valid_ratio=valid_ratio,
            void_ratio=void_ratio,
            edge_void_ratio=1.0,
            raw_min=None,
            raw_max=None,
            raw_range=0.0,
            raw_std=0.0,
            p01=None,
            p99=None,
            slope_p50=0.0,
            slope_p95=0.0,
            local_relief_p50=0.0,
            local_relief_p95=0.0,
            outlier_ratio=0.0,
            spike_ratio=0.0,
            lowland_ratio=0.0,
            edge_lowland_ratio=0.0,
            water_influence_ratio=0.0,
            land_dominance_ratio=0.0,
            source_confidence=0.0,
            overall_confidence=0.0,
            terrain_quality='insufficient',
            qualification_status='rejected',
            qualification_notes=['No valid terrain pixels after DEM decode.'],
            warnings=['No valid terrain pixels after DEM decode.'],
            diagnostics={
                'coverage_ratio': 0.0,
                'resolution_appropriateness': 'unknown',
                'source_suitability': 'rejected',
            },
        )

    finite_vals = arr[finite]
    raw_min = float(np.nanmin(finite_vals))
    raw_max = float(np.nanmax(finite_vals))
    raw_range = raw_max - raw_min
    raw_std = float(np.nanstd(finite_vals))
    p01 = _safe_percentile(finite_vals, 1, raw_min)
    p99 = _safe_percentile(finite_vals, 99, raw_max)

    work = arr.copy()
    fallback = float(np.nanmedian(finite_vals))
    work[~finite] = fallback
    gy, gx = np.gradient(work)
    slope = np.sqrt(gx * gx + gy * gy)
    relief = _local_relief(work)

    edge = np.zeros_like(work, dtype=bool)
    edge[[0, 1, -2, -1], :] = True
    edge[:, [0, 1, -2, -1]] = True
    edge_void_ratio = float((~np.isfinite(arr[edge])).mean()) if edge.any() else 0.0

    lowland_threshold = max(2.0, raw_min + max(0.75, raw_range * 0.06))
    lowland_mask = finite & (arr <= lowland_threshold)
    edge_lowland_ratio = float(lowland_mask[edge].mean()) if edge.any() else 0.0
    lowland_ratio = float(lowland_mask.mean()) if arr.size else 0.0

    smooth_lowland_mask = lowland_mask & (slope <= _safe_percentile(slope[finite], 35, 0.0)) & (relief <= _safe_percentile(relief[finite], 30, 0.0))
    water_influence_ratio = float(smooth_lowland_mask.mean()) if arr.size else 0.0
    land_dominance_ratio = max(0.0, min(1.0, 1.0 - edge_lowland_ratio * 0.9))

    nominal_resolution = source.nominal_resolution_m
    if nominal_resolution is None:
        resolution_appropriateness = 'unknown'
    elif nominal_resolution <= 35:
        resolution_appropriateness = 'good'
    elif nominal_resolution <= 90:
        resolution_appropriateness = 'usable'
    else:
        resolution_appropriateness = 'coarse'

    qualification_notes: list[str] = []
    warnings: list[str] = []
    if valid_ratio >= 0.98:
        qualification_notes.append('Coverage is nearly complete.')
    elif valid_ratio >= 0.9:
        qualification_notes.append('Coverage is acceptable for downstream conditioning.')
    else:
        qualification_notes.append('Coverage is weaker than preferred and may reduce trust.')
        warnings.append('Large nodata or masked region detected in DEM.')

    if raw_range < 3.0:
        qualification_notes.append('Decoded terrain range is extremely quiet.')
        warnings.append('Terrain appears nearly flat after decode.')
    elif raw_range < 10.0:
        qualification_notes.append('Decoded terrain range is subtle and needs restrained presentation.')
    else:
        qualification_notes.append('Decoded terrain shows usable relief separation.')

    if edge_lowland_ratio > 0.55 and raw_range > 20.0:
        qualification_notes.append('Edge-lowland burden suggests coastal or slab contamination risk.')
        warnings.append('Patch appears coastal or edge-lowland dominated; viewer scaling will de-emphasize sea-level slabs.')
    elif water_influence_ratio > 0.45 and raw_range > 5.0 and edge_lowland_ratio < 0.35:
        qualification_notes.append('Smooth lowland dominance present, but inland terrain signal remains meaningful.')
        warnings.append('Low-relief smooth inland terrain detected; preserving subtle land signal instead of assuming water dominance.')

    if raw_min < -600 or raw_max > 10000:
        warnings.append('Suspicious raw elevation values detected.')

    confidence = 1.0
    confidence -= max(0.0, (0.92 - valid_ratio)) * 1.1
    if raw_range < 5.0:
        confidence -= 0.12
    if raw_range < 2.0:
        confidence -= 0.12
    confidence -= min(0.18, max(0.0, edge_void_ratio - 0.05) * 0.8)
    if edge_lowland_ratio > 0.6 and raw_range > 20.0:
        confidence -= 0.08
    if resolution_appropriateness == 'coarse':
        confidence -= 0.06
    confidence = max(0.0, min(1.0, confidence))

    if confidence >= 0.82:
        quality = 'strong'
        qualification_status = 'qualified'
        source_suitability = 'strong'
    elif confidence >= 0.63:
        quality = 'cleaned'
        qualification_status = 'qualified_with_guards'
        source_suitability = 'usable'
    elif confidence >= 0.40:
        quality = 'weak'
        qualification_status = 'guarded'
        source_suitability = 'guarded'
    else:
        quality = 'insufficient'
        qualification_status = 'rejected'
        source_suitability = 'weak'

    return TerrainQC(
        valid_ratio=valid_ratio,
        void_ratio=void_ratio,
        edge_void_ratio=edge_void_ratio,
        raw_min=raw_min,
        raw_max=raw_max,
        raw_range=raw_range,
        raw_std=raw_std,
        p01=p01,
        p99=p99,
        slope_p50=_safe_percentile(slope[finite], 50, 0.0),
        slope_p95=_safe_percentile(slope[finite], 95, 0.0),
        local_relief_p50=_safe_percentile(relief[finite], 50, 0.0),
        local_relief_p95=_safe_percentile(relief[finite], 95, 0.0),
        outlier_ratio=0.0,
        spike_ratio=0.0,
        lowland_ratio=lowland_ratio,
        edge_lowland_ratio=edge_lowland_ratio,
        water_influence_ratio=water_influence_ratio,
        land_dominance_ratio=land_dominance_ratio,
        source_confidence=confidence,
        overall_confidence=confidence,
        terrain_quality=quality,
        qualification_status=qualification_status,
        qualification_notes=qualification_notes,
        warnings=warnings,
        diagnostics={
            'lowland_threshold': round(float(lowland_threshold), 3),
            'coverage_ratio': round(valid_ratio, 4),
            'resolution_appropriateness': resolution_appropriateness,
            'source_suitability': source_suitability,
            'nominal_resolution_m': nominal_resolution,
            'raw_dynamic_range': round(raw_range, 3),
            'water_influence_flag': bool(water_influence_ratio > 0.45),
        },
    )

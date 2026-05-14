from __future__ import annotations

from dataclasses import replace

import numpy as np

from .terrain_models import TerrainQC, TerrainSource, TerrainSurface


def _safe_percentile(values: np.ndarray, q: float, fallback: float = 0.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(fallback)
    return float(np.percentile(finite, q))


def _local_nanmedian(arr: np.ndarray) -> np.ndarray:
    stack = [np.roll(np.roll(arr, dy, axis=0), dx, axis=1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    return np.nanmedian(np.stack(stack, axis=0), axis=0)


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


def _local_relief(arr: np.ndarray) -> np.ndarray:
    work = np.asarray(arr, dtype=float)
    return (
        np.maximum.reduce([work, np.roll(work, 1, 0), np.roll(work, -1, 0), np.roll(work, 1, 1), np.roll(work, -1, 1)])
        - np.minimum.reduce([work, np.roll(work, 1, 0), np.roll(work, -1, 0), np.roll(work, 1, 1), np.roll(work, -1, 1)])
    )


def condition_surface(source: TerrainSource, qc: TerrainQC) -> TerrainSurface:
    arr = np.asarray(source.raw_dem, dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    arr[(arr < -600.0) | (arr > 10000.0)] = np.nan
    valid_initial = np.isfinite(arr)
    if not valid_initial.any():
        raise ValueError('DEM contained no usable elevation values after nodata cleanup.')

    finite_vals = arr[valid_initial]
    clip_low = float(np.percentile(finite_vals, 0.5))
    clip_high = float(np.percentile(finite_vals, 99.5))
    clipped = arr.copy()
    clipped[valid_initial] = np.clip(clipped[valid_initial], clip_low, clip_high)
    outlier_mask = valid_initial & ((arr < clip_low) | (arr > clip_high))

    neighborhood_median = _local_nanmedian(clipped)
    diff = np.abs(clipped - neighborhood_median)
    diff[np.isnan(diff)] = 0.0
    mad = _safe_percentile(diff[np.isfinite(diff)], 75, fallback=1.0)
    spike_threshold = max(12.0 if qc.raw_range < 20 else 18.0, mad * 4.5)
    spike_mask = valid_initial & np.isfinite(neighborhood_median) & (diff > spike_threshold)

    cleaned = clipped.copy()
    cleaned[spike_mask] = neighborhood_median[spike_mask]

    lowland_threshold = qc.diagnostics.get('lowland_threshold', None)
    coastal_adjustment_applied = False
    if lowland_threshold is not None and qc.edge_lowland_ratio > 0.55 and qc.raw_range > 20.0:
        edge = np.zeros_like(cleaned, dtype=bool)
        edge[[0, 1, -2, -1], :] = True
        edge[:, [0, 1, -2, -1]] = True
        coastal_mask = edge & (cleaned <= float(lowland_threshold))
        if coastal_mask.any():
            coastal_fill = _safe_percentile(cleaned[valid_initial], 10, fallback=float(np.nanmedian(cleaned[valid_initial])))
            cleaned[coastal_mask] = coastal_fill
            coastal_adjustment_applied = True

    valid_mask = np.isfinite(cleaned).copy()
    cleaned = _fill_nodata_nearest(cleaned)
    cleaned = np.asarray(cleaned, dtype=float)

    relief = _local_relief(cleaned)
    gy, gx = np.gradient(cleaned)
    slope = np.sqrt(gx * gx + gy * gy)
    clean_range = float(np.nanmax(cleaned) - np.nanmin(cleaned))
    outlier_ratio = float(outlier_mask.mean()) if outlier_mask.size else 0.0
    spike_ratio = float(spike_mask.mean()) if spike_mask.size else 0.0

    confidence = float(qc.source_confidence)
    confidence -= min(0.25, outlier_ratio * 2.5)
    confidence -= min(0.28, spike_ratio * 4.5)
    if clean_range < 5.0:
        confidence -= 0.08
    if clean_range < 2.0:
        confidence -= 0.08
    confidence = max(0.0, min(1.0, confidence))

    warnings = list(dict.fromkeys(list(qc.warnings)))
    if outlier_ratio > 0.005:
        warnings.append('Extreme elevation outliers were clipped before derivative generation.')
    if spike_ratio > 0.005:
        warnings.append('Needle-like elevation spikes were repaired before 3D export.')
    if clean_range < 3.0:
        warnings.append('Terrain relief is very low after conditioning; some layers may look muted.')
    if coastal_adjustment_applied:
        warnings.append('Edge-connected lowland slab was downweighted before visualization to preserve inland terrain signal.')
    if qc.water_influence_ratio > 0.45 and qc.edge_lowland_ratio < 0.35:
        warnings.append('Subtle inland terrain preserved with land-first conditioning; smooth canopy is not treated as open water by default.')

    if confidence >= 0.82:
        quality = 'strong'
    elif confidence >= 0.63:
        quality = 'cleaned'
    elif confidence >= 0.40:
        quality = 'weak'
    else:
        quality = 'insufficient'

    conditioning_summary = {
        'truth_preserving_stage': True,
        'clip_low': round(clip_low, 3),
        'clip_high': round(clip_high, 3),
        'spike_threshold': round(float(spike_threshold), 3),
        'outlier_pixels': int(outlier_mask.sum()),
        'outlier_ratio': round(outlier_ratio, 4),
        'spike_pixels': int(spike_mask.sum()),
        'spike_ratio': round(spike_ratio, 4),
        'coastal_adjustment_applied': bool(coastal_adjustment_applied),
        'void_fill_strategy': 'iterative_neighbor_average',
        'visual_enhancement_applied': False,
        'terrain_truth_contract': 'cleaned_dem is conditioning output; presentation decisions occur downstream in rendering policy.',
    }

    qc2 = replace(
        qc,
        outlier_ratio=outlier_ratio,
        spike_ratio=spike_ratio,
        overall_confidence=confidence,
        terrain_quality=quality,
        warnings=warnings,
        diagnostics={
            **qc.diagnostics,
            **conditioning_summary,
            'exaggeration_recommended': 1.0,
        },
    )

    return TerrainSurface(
        raw_dem=np.asarray(source.raw_dem, dtype=float),
        cleaned_dem=cleaned,
        valid_mask=valid_mask,
        provider=source.provider,
        source_name=source.source_name,
        bbox=source.bbox,
        transform=source.transform,
        crs=source.crs,
        nodata_value=source.nodata_value,
        pixel_size_x=source.pixel_size_x,
        pixel_size_y=source.pixel_size_y,
        nominal_resolution_m=source.nominal_resolution_m,
        qc=qc2,
        terrain_class='unclassified',
        rendering_policy=None,
        raw_stats={
            'raw_min': float(np.nanmin(arr[valid_initial])),
            'raw_max': float(np.nanmax(arr[valid_initial])),
            'raw_range': float(np.nanmax(arr[valid_initial]) - np.nanmin(arr[valid_initial])),
        },
        clean_stats={
            'clean_min': float(np.nanmin(cleaned)),
            'clean_max': float(np.nanmax(cleaned)),
            'clean_range': clean_range,
        },
        slope_stats={
            'slope_p50': _safe_percentile(slope, 50, 0.0),
            'slope_p95': _safe_percentile(slope, 95, 0.0),
        },
        relief_stats={
            'relief_p50': _safe_percentile(relief, 50, 0.0),
            'relief_p95': _safe_percentile(relief, 95, 0.0),
        },
        warnings=warnings,
        debug={
            'conditioning_summary': conditioning_summary,
            'terrain_truth_ready': confidence >= 0.4,
        },
    )

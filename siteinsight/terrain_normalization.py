from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .terrain_models import TerrainSource


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if not np.isfinite(out):
            return None
        return out
    except Exception:
        return None


def normalize_source(source: TerrainSource) -> tuple[TerrainSource, dict[str, Any]]:
    """Normalize provider outputs into a consistent terrain contract.

    This stage is intentionally conservative. It standardizes dtype, nodata
    handling, sign/scale sanity, and working-grid expectations without trying
    to beautify the terrain.
    """
    raw = np.asarray(source.raw_dem, dtype=np.float32)
    arr = raw.astype(float, copy=True)

    nodata_value = _safe_float(source.nodata_value)
    if nodata_value is not None:
        arr[np.isclose(arr, nodata_value, atol=1e-6)] = np.nan

    arr[~np.isfinite(arr)] = np.nan
    arr[(arr < -12000.0) | (arr > 15000.0)] = np.nan

    finite = np.isfinite(arr)
    vertical_scale_fixed = False
    if finite.any():
        p01 = float(np.nanpercentile(arr, 1))
        p99 = float(np.nanpercentile(arr, 99))
        spread = p99 - p01
        # Some sources may arrive effectively in centimeters.
        if spread > 20000.0:
            arr[finite] = arr[finite] / 100.0
            vertical_scale_fixed = True

    pixel_x = abs(_safe_float(source.pixel_size_x) or 0.0) or None
    pixel_y = abs(_safe_float(source.pixel_size_y) or 0.0) or None
    nominal_resolution_m = _safe_float(source.nominal_resolution_m)

    if nominal_resolution_m is None:
        fallback_res = pixel_x or pixel_y
        if fallback_res is not None:
            nominal_resolution_m = float(fallback_res)

    normalized = replace(
        source,
        raw_dem=np.asarray(arr, dtype=float),
        nodata_value=nodata_value,
        pixel_size_x=pixel_x,
        pixel_size_y=pixel_y,
        nominal_resolution_m=nominal_resolution_m,
        debug={
            **(source.debug or {}),
            'normalization_applied': True,
            'vertical_scale_fixed': vertical_scale_fixed,
        },
    )

    summary = {
        'dtype': str(raw.dtype),
        'working_dtype': 'float64',
        'vertical_scale_fixed': bool(vertical_scale_fixed),
        'nodata_value': nodata_value,
        'crs': source.crs,
        'nominal_resolution_m': nominal_resolution_m,
        'pixel_size_x': pixel_x,
        'pixel_size_y': pixel_y,
        'working_grid_shape': [int(arr.shape[0]), int(arr.shape[1])],
        'valid_ratio_after_normalization': round(float(np.isfinite(arr).mean()) if arr.size else 0.0, 4),
    }
    return normalized, summary

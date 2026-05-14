from __future__ import annotations

from typing import Any

import numpy as np

from .terrain_models import TerrainSurface


def detect_archaeology_candidates(
    dem_or_surface: TerrainSurface | np.ndarray,
    derivatives: dict[str, Any],
    top_n: int = 8,
) -> dict[str, Any]:
    if isinstance(dem_or_surface, TerrainSurface):
        surface = dem_or_surface
        dem = surface.cleaned_dem
        terrain_class = surface.terrain_class
        suppress = surface.rendering_policy.suppress_archaeology
    else:
        surface = None
        dem = np.asarray(dem_or_surface, dtype=float)
        terrain_class = 'unknown'
        suppress = False

    terrain_objects = (surface.terrain_objects if surface is not None else {}) or {}
    object_features = terrain_objects.get('features', [])

    slope = np.asarray(derivatives['slope'], dtype=float)
    lap = np.asarray(derivatives['laplacian'], dtype=float)
    ridge = np.asarray(derivatives['ridge_strength'], dtype=float)
    anomaly = np.asarray(derivatives.get('anomaly_response', np.zeros_like(slope)), dtype=float)

    smooth_mask = slope < np.percentile(slope, 55 if terrain_class in {'subtle_relief', 'subtle_archaeology'} else 45)
    convex_mask = lap < np.percentile(lap, 35 if terrain_class in {'subtle_relief', 'subtle_archaeology'} else 25)
    ridge_mask = ridge > np.percentile(ridge, 62 if terrain_class in {'subtle_relief', 'subtle_archaeology'} else 70)

    anomaly_score = (
        smooth_mask.astype(float) * (0.22 if terrain_class in {'subtle_relief', 'subtle_archaeology'} else 0.35)
        + convex_mask.astype(float) * 0.28
        + ridge_mask.astype(float) * (0.18 if terrain_class in {'subtle_relief', 'subtle_archaeology'} else 0.30)
        + anomaly * (0.32 if terrain_class in {'subtle_relief', 'subtle_archaeology'} else 0.07)
    )
    if suppress:
        anomaly_score *= 0.5

    candidates: list[dict[str, Any]] = []
    flat_scores = anomaly_score.flatten()
    flat_indices = np.argsort(flat_scores)[::-1]

    used = set()
    rows, cols = dem.shape

    for flat_idx in flat_indices:
        y, x = divmod(int(flat_idx), cols)
        key = (y // 6, x // 6)
        if key in used:
            continue
        used.add(key)

        local_radius = 4
        y0 = max(0, y - local_radius)
        y1 = min(rows, y + local_radius + 1)
        x0 = max(0, x - local_radius)
        x1 = min(cols, x + local_radius + 1)
        local = dem[y0:y1, x0:x1]
        local_anomaly = anomaly[y0:y1, x0:x1]

        rise = float(local.max() - local.min())
        symmetry = float(1.0 - abs(local.mean() - np.flipud(local).mean()) / max(1.0, local.std() + 1e-6))
        geometric_hint = float(np.clip((rise / (15.0 if terrain_class in {'subtle_relief', 'subtle_archaeology'} else 40.0)) + symmetry / 2.0 + float(np.nanmean(local_anomaly)) * 0.6, 0, 1))
        confidence = float(np.clip(flat_scores[flat_idx] * 100, 0, 100))

        label = 'mound_candidate'
        if geometric_hint > 0.78:
            label = 'ring_or_embankment_candidate'
        elif rise < (10 if terrain_class in {'subtle_relief', 'subtle_archaeology'} else 18):
            label = 'terrace_candidate'

        candidates.append(
            {
                'grid_x': int(x),
                'grid_y': int(y),
                'feature_type': label,
                'confidence': round(confidence, 2),
                'local_rise': round(rise, 2),
                'symmetry_hint': round(symmetry, 3),
                'geometric_hint': round(geometric_hint, 3),
            }
        )
        if len(candidates) >= top_n:
            break

    object_bonus = min(12.0, sum(1 for f in object_features if f.get('feature_type') == 'anomaly_candidate') * 1.8)
    signal_score = round(float(np.mean([c['confidence'] for c in candidates])) if candidates else 0.0, 2)
    if terrain_class == 'subtle_archaeology':
        signal_score = round(min(100.0, signal_score * 1.12 + object_bonus), 2)
    if suppress:
        signal_score = round(signal_score * 0.55, 2)
    structure_probability = round(min(100.0, signal_score * 0.9 + len(candidates) * 2.5), 2)

    return {
        'archaeological_signal_score': signal_score,
        'structure_probability': structure_probability,
        'candidate_count': len(candidates),
        'terrain_object_support': len(object_features),
        'terrain_class': terrain_class,
        'candidates': candidates,
        'summary': 'Heuristic archaeological screening only. These are candidate shapes and terrain anomalies, not proof claims.',
    }

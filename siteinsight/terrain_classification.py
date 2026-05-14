from __future__ import annotations

from .terrain_models import TerrainSurface


def classify_surface(surface: TerrainSurface) -> str:
    qc = surface.qc
    clean_range = float(surface.clean_stats.get('clean_range', 0.0))
    relief_p95 = float(surface.relief_stats.get('relief_p95', 0.0))
    slope_p95 = float(surface.slope_stats.get('slope_p95', 0.0))

    if qc.valid_ratio < 0.35 or qc.qualification_status == 'rejected':
        return 'void_dominated'
    if qc.overall_confidence < 0.25:
        return 'insufficient_surface'
    if qc.spike_ratio > 0.02 or qc.outlier_ratio > 0.03:
        return 'artifact_suspicious'
    if qc.edge_lowland_ratio > 0.45 and qc.lowland_ratio > 0.20 and clean_range > 20.0:
        return 'coastal_transition'
    if qc.water_influence_ratio > 0.55 and qc.edge_lowland_ratio > 0.40:
        return 'water_influenced'
    if clean_range < 12.0 and relief_p95 < 3.5 and qc.edge_lowland_ratio < 0.35 and qc.land_dominance_ratio > 0.65:
        return 'subtle_archaeology'
    if clean_range < 4.0 or relief_p95 < 1.2:
        return 'subtle_relief'
    if clean_range < 60.0 and slope_p95 < 6.0:
        return 'moderate_relief'
    return 'high_relief'

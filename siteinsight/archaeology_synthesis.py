from __future__ import annotations

from typing import Any

import numpy as np

from .terrain_models import TerrainSurface


def _finite_window(arr: np.ndarray, y: int, x: int, radius: int = 5) -> np.ndarray:
    rows, cols = arr.shape
    y0 = max(0, int(y) - radius)
    y1 = min(rows, int(y) + radius + 1)
    x0 = max(0, int(x) - radius)
    x1 = min(cols, int(x) + radius + 1)
    win = np.asarray(arr[y0:y1, x0:x1], dtype=float)
    win = win[np.isfinite(win)]
    return win if win.size else np.asarray([0.0], dtype=float)


def _percentile_score(value: float, arr: np.ndarray, q_low: float = 50.0, q_high: float = 95.0) -> float:
    finite = np.asarray(arr, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    lo = float(np.percentile(finite, q_low))
    hi = float(np.percentile(finite, q_high))
    if hi <= lo:
        return 0.0
    return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))


def _band(conf: float) -> str:
    if conf >= 0.80:
        return 'strong pre-field package'
    if conf >= 0.65:
        return 'strong candidate for expert/field review'
    if conf >= 0.50:
        return 'preliminary candidate feature'
    if conf >= 0.30:
        return 'interesting visual anomaly'
    return 'insufficient terrain evidence'


def _safe_round(value: Any, places: int = 3) -> float:
    try:
        if value is None or not np.isfinite(float(value)):
            return 0.0
        return round(float(value), places)
    except Exception:
        return 0.0


def build_archaeology_layer_synthesis(
    surface: TerrainSurface,
    derivatives: dict[str, Any],
    archaeology: dict[str, Any],
    terrain_objects: dict[str, Any] | None = None,
    source_arbitration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an archaeology decision-support layer without changing DEM math.

    This is a cautious synthesis product. It makes layer agreement, contradictions,
    source quality, and field-review needs visible to the operator.
    """
    dem = np.asarray(surface.cleaned_dem, dtype=float)
    slope = np.asarray(derivatives.get('slope', np.zeros_like(dem)), dtype=float)
    relief = np.asarray(derivatives.get('relief_local', np.zeros_like(dem)), dtype=float)
    edge = np.asarray(derivatives.get('edge_strength', np.zeros_like(dem)), dtype=float)
    lap = np.asarray(derivatives.get('laplacian', np.zeros_like(dem)), dtype=float)
    anomaly = np.asarray(derivatives.get('anomaly_response', np.zeros_like(dem)), dtype=float)

    qc = surface.to_qc_dict()
    terrain_conf = float(qc.get('terrain_confidence') or qc.get('overall_confidence') or 0.0)
    terrain_class = qc.get('terrain_class') or getattr(surface, 'terrain_class', 'unknown')
    provider = getattr(surface, 'provider', None) or getattr(surface, 'source_name', 'unknown')
    source_arbitration = source_arbitration or {}
    terrain_objects = terrain_objects or {}

    candidates: list[dict[str, Any]] = []
    for idx, cand in enumerate((archaeology or {}).get('candidates', [])[:8], start=1):
        x = int(cand.get('grid_x') or 0)
        y = int(cand.get('grid_y') or 0)
        local_relief_mean = float(np.mean(_finite_window(relief, y, x)))
        local_edge_mean = float(np.mean(_finite_window(edge, y, x)))
        local_slope_mean = float(np.mean(_finite_window(slope, y, x)))
        local_lap_mean = float(np.mean(_finite_window(np.abs(lap), y, x)))
        local_anomaly_mean = float(np.mean(_finite_window(anomaly, y, x)))

        lrm_score = _percentile_score(local_relief_mean, relief)
        edge_score = _percentile_score(local_edge_mean, edge)
        slope_score = _percentile_score(local_slope_mean, slope)
        openness_svf_proxy = _percentile_score(local_lap_mean, np.abs(lap))
        anomaly_score = max(0.0, min(1.0, local_anomaly_mean))

        layer_scores = {
            'lrm_microrelief': _safe_round(lrm_score),
            'slope_break': _safe_round(slope_score),
            'lrm_edges_boundary': _safe_round(edge_score),
            'openness_svf_shape_proxy': _safe_round(openness_svf_proxy),
            'discovery_anomaly': _safe_round(anomaly_score),
            'three_d_context': _safe_round(min(1.0, max(0.25, terrain_conf))),
        }
        agree_count = sum(1 for v in layer_scores.values() if float(v) >= 0.45)
        weak_count = sum(1 for v in layer_scores.values() if float(v) < 0.25)
        consensus = agree_count / max(1, len(layer_scores))
        contradiction = min(1.0, weak_count / max(1, len(layer_scores)) + max(0.0, 0.55 - terrain_conf) * 0.65)
        base_conf = float(cand.get('confidence') or 0.0) / 100.0
        synthesis_conf = max(0.0, min(0.89, (base_conf * 0.30) + (consensus * 0.38) + (terrain_conf * 0.22) + (anomaly_score * 0.10) - contradiction * 0.18))
        if terrain_conf < 0.63:
            synthesis_conf = min(synthesis_conf, 0.64)
        if source_arbitration.get('selected_provider', '').lower().startswith('copernicus'):
            synthesis_conf = min(synthesis_conf, 0.74)

        false_positive_risks = []
        if slope_score > 0.75 and lrm_score < 0.35:
            false_positive_risks.append('slope edge without enough microrelief support')
        if edge_score > 0.80 and anomaly_score < 0.35:
            false_positive_risks.append('sharp edge may be modern track, seam, or processing artifact')
        if terrain_conf < 0.63:
            false_positive_risks.append('source quality is guarded; cap interpretation confidence')
        if not false_positive_risks:
            false_positive_risks.append('natural drainage, erosion, vegetation artifact, or modern disturbance still must be checked')

        candidates.append({
            'rank': idx,
            'candidate_name': f"Candidate {idx}: {str(cand.get('feature_type', 'terrain anomaly')).replace('_', ' ').title()}",
            'decision_status': _band(synthesis_conf),
            'confidence': _safe_round(synthesis_conf, 2),
            'grid_x': x,
            'grid_y': y,
            'observed_pattern': f"Existing heuristic detected {cand.get('feature_type', 'terrain anomaly')} with local rise {cand.get('local_rise', 'unknown')} and geometric hint {cand.get('geometric_hint', 'unknown')}.",
            'layer_evidence': layer_scores,
            'layer_agreement': _safe_round(consensus, 2),
            'contradiction_score': _safe_round(contradiction, 2),
            'why_it_matters': 'A candidate becomes more interesting when microrelief, boundary, slope-break, and 3D context agree instead of relying on one attractive raster.',
            'false_positive_risks': false_positive_risks,
            'what_can_be_decided_now': 'Rank this as a non-invasive review target and compare it across Hillshade, LRM, LRM_Edges, Openness/SVF, Discovery, and 3D.',
            'what_must_not_be_decided_yet': 'Do not call this a site, structure, mound, burial, or cultural resource without expert and field confirmation.',
            'fastest_confidence_upgrade': 'Add historical maps, field photos, higher-resolution bare-earth LiDAR/DEM, or expert review notes for this exact AOI.',
            'field_verification_task': 'Check the candidate from the safest legal access point; photograph the landform, note modern disturbance, drainage, vegetation, and GPS position.',
            'best_screenshot_angle': '3D oblique Terrain Texture view plus side-by-side LRM_Edges and Discovery crop at the same candidate center.',
        })

    combined_conf = 0.0
    if candidates:
        combined_conf = float(np.mean([c['confidence'] for c in candidates[: min(5, len(candidates))]]))
    combined_conf = min(combined_conf, terrain_conf if terrain_conf < 0.75 else 0.89)

    provider_lower = str(provider).lower()
    if provider_lower == 'usgs':
        source_doctrine = 'Default to USGS/3DEP where available for U.S. archaeology because it is closer to bare-earth terrain support. Keep Copernicus as a global/worldview fallback, not a U.S. archaeology replacement.'
    elif provider_lower.startswith('copernicus'):
        source_doctrine = 'Copernicus is useful for global continuity and outside-USA reconnaissance. Treat it as contextual terrain unless better bare-earth LiDAR/DEM exists.'
    else:
        source_doctrine = 'Source is mixed or unknown; make source quality and fallback status visible before raising archaeology confidence.'

    return {
        'mode': 'archaeology_layer_synthesis_challenge_mode',
        'decision_status': _band(combined_conf),
        'combined_confidence': _safe_round(combined_conf, 2),
        'command_summary': 'Layer synthesis added: candidates are now judged by agreement, contradiction, source confidence, and field-verification path rather than a single archaeology score.',
        'terrain_quality_chief': {
            'decision': 'Protect raw DEM truth; tune display and layer fusion without rewriting terrain acquisition in this pass.',
            'confidence': _safe_round(terrain_conf, 2),
            'terrain_class': terrain_class,
        },
        'source_mesh_fallback_architect': {
            'decision': source_doctrine,
            'selected_provider': source_arbitration.get('selected_provider') or provider,
            'candidate_count': source_arbitration.get('candidate_count', 1),
        },
        'canopy_rainforest_recon_lead': {
            'decision': 'For jungle pyramid hunting, true tree-penetrating archaeology needs bare-earth LiDAR where available. Canopy height, GEDI, SAR, and optical imagery are reconnaissance layers, not proof layers.',
            'next_layer_targets': ['bare-earth lidar/dtm', 'canopy height model', 'gedi footprints', 'sar texture/change', 'historical maps', 'hydrology/wetness index'],
        },
        'unique_ai_moves': [
            'Layer Consensus Heat: rank where LRM, slope breaks, LRM edges, openness/SVF proxy, anomaly response, and 3D source confidence agree.',
            'Contradiction Map: penalize attractive shapes that only appear in one layer or are weakened by guarded source quality.',
            'False-Positive Duel: force every candidate to survive natural drainage, modern track, vegetation, processing seam, and erosion explanations.',
            'Confidence Upgrade Queue: state the fastest data or expert review that would raise confidence.',
        ],
        'candidate_cards': candidates,
        'candidate_count': len(candidates),
        'layer_synthesis_matrix': [
            {'layer': 'Hillshade / Terrain Texture', 'contribution': 'human-readable landform story and best screenshot context', 'cannot_prove': 'cultural origin because lighting can mislead'},
            {'layer': 'Slope', 'contribution': 'scarps, terrace edges, banks, ditch walls, sharp breaks', 'cannot_prove': 'whether break is natural or modern'},
            {'layer': 'LRM', 'contribution': 'microrelief after broad terrain is suppressed', 'cannot_prove': 'site status without context'},
            {'layer': 'LRM_Edges', 'contribution': 'candidate outlines and rims', 'cannot_prove': 'validity when edges are noise or seams'},
            {'layer': 'Openness/SVF', 'contribution': 'shape confirmation with less lighting bias', 'cannot_prove': 'anthropogenic cause'},
            {'layer': '3D/Elevation', 'contribution': 'ridge, bench, water, pass, route, and presentation context', 'cannot_prove': 'archaeology by visual drama alone'},
            {'layer': 'Discovery/Archaeology Blend', 'contribution': 'hypothesis view and review priority', 'cannot_prove': 'raw evidence unless badges show what drove it'},
        ],
        'protected_systems': [
            'DEM acquisition and cache behavior',
            'existing derivative PNG generation',
            'fast heightmap_viewer.json path',
            'viewer3d.html and Terrain_Texture default',
            'cautious archaeology language',
        ],
        'next_surgical_pass': 'Add a visual Layer Consensus Heat raster and candidate AOI crop links after this JSON/UI pass is verified.',
    }

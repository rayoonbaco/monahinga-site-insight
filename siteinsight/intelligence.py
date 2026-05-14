from __future__ import annotations

from typing import Any

from siteinsight.utils import clamp


def _score(value: float) -> int:
    return int(round(clamp(value, 0, 100)))


def build_intelligence(
    dem,
    terrain_summary: dict[str, Any],
    archaeology: dict[str, Any],
    persona: str,
    terrain_objects: dict[str, Any] | None = None,
    terrain_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elevation_range = float(terrain_summary['elevation_range'])
    mean_slope = float(terrain_summary['mean_slope'])
    local_relief = float(terrain_summary['mean_local_relief'])
    ridge_ratio = float(terrain_summary['ridge_ratio'])
    valley_ratio = float(terrain_summary['valley_ratio'])
    edge_ratio = float(terrain_summary['edge_ratio'])
    terrain_class = terrain_summary.get('terrain_class', 'unknown')

    terrain_signal_score = _score(
        elevation_range * (0.12 if terrain_class in {'subtle_relief', 'subtle_archaeology'} else 0.18)
        + mean_slope * (1.4 if terrain_class in {'subtle_relief', 'subtle_archaeology'} else 2.2)
        + local_relief * (1.4 if terrain_class in {'subtle_relief', 'subtle_archaeology'} else 0.9)
        + ridge_ratio * 40
        + valley_ratio * 30
        + edge_ratio * 20
    )

    terrain_flow_score = _score(valley_ratio * 40 + ridge_ratio * 25 + edge_ratio * 30 + mean_slope * 0.8)
    bench_shelter_score = _score(edge_ratio * 35 + ridge_ratio * 30 + (100 - mean_slope * 2.5))
    lowland_transition_score = _score(valley_ratio * 35 + edge_ratio * 20 + local_relief * 1.1)

    terrain_objects = terrain_objects or {}
    terrain_graph = terrain_graph or {}
    object_count = int(terrain_objects.get('feature_count', 0))
    graph_signal = float(terrain_graph.get('graph_signal_score', 0.0))
    archaeology_score = _score(float(archaeology.get('archaeological_signal_score', 0)))
    review_priority_score = _score(
        terrain_signal_score * 0.34
        + archaeology_score * 0.34
        + terrain_flow_score * 0.16
        + object_count * 1.4
        + graph_signal * 0.16
    )

    narratives = {
        'explorer': 'This landscape reads as a structured terrain review area with visible relief breaks, flow lines, and a few places that merit slower inspection.',
        'hunter': 'This terrain has clear movement structure, but this build now reports it as landform flow and access rather than wildlife prediction.',
        'archaeologist': 'The anomaly pattern is strong enough for follow-up review. The right next step is non-invasive inspection of candidate landforms and context layers.',
        'survey': 'This run is tuned for terrain review. Use the relief, slope, openness, and archaeology layers together before drawing conclusions.',
    }
    if terrain_class == 'subtle_archaeology':
        narratives['archaeologist'] = 'This patch reads as subtle archaeological terrain: low-amplitude landforms, muted canopy-smoothed relief, and structured anomaly pockets worth careful follow-up.'
    elif terrain_class == 'coastal_transition':
        narratives['explorer'] = 'This patch mixes coastal lowland and inland relief. Viewer scaling is land-first so offshore or lowland slabs do not dominate the read.'

    top_takeaways = [
        'Terrain flow strengthens where ridge, valley, and edge signals overlap.',
        'Candidate anomaly zones are present, but they remain review targets rather than confirmed sites.',
        'The run is well suited for pins, revisit workflows, and future field-note or media attachment.',
    ]
    if terrain_class == 'subtle_archaeology':
        top_takeaways[1] = 'Subtle anomaly zones survive conditioning because this terrain is being treated as quiet inland landform rather than water-dominated flatness.'

    if object_count:
        top_takeaways.append(f'Terrain object extraction found {object_count} anchors for graph-based follow-up.')

    candidate_review_zones = [
        {
            'label': 'Primary Ridge-Edge Review Zone',
            'why': 'Strong ridge and edge interaction creates a clear place to inspect landform change.',
            'score': _score(terrain_flow_score * 0.55 + bench_shelter_score * 0.45),
        },
        {
            'label': 'Secondary Valley Transition Zone',
            'why': 'Valley transition with manageable relief and structured movement through the patch.',
            'score': _score(lowland_transition_score * 0.5 + terrain_flow_score * 0.5),
        },
        {
            'label': 'Wide-View Observation Point',
            'why': 'Balanced access and broad terrain visibility for field review.',
            'score': _score(terrain_signal_score * 0.5 + review_priority_score * 0.5),
        },
    ]

    return {
        'persona': persona,
        'terrain_signal_score': terrain_signal_score,
        'wildlife_movement_probability': terrain_flow_score,
        'travel_corridor_strength': terrain_flow_score,
        'likely_bedding_zones': bench_shelter_score,
        'likely_feeding_zones': lowland_transition_score,
        'archaeological_signal_score': archaeology_score,
        'discovery_score': review_priority_score,
        'terrain_class': terrain_class,
        'primary_routes': [
            'Ridge shoulder to saddle compression',
            'Valley edge transition',
            'Bench-like lateral movement lane',
        ],
        'stand_recommendation_candidates': candidate_review_zones,
        'top_takeaways': top_takeaways,
        'narrative': narratives.get(persona, narratives['survey']),
        'terrain_object_count': object_count,
        'terrain_graph_signal_score': round(graph_signal, 2),
        'future_hooks': {
            'field_note_attachment': 'Not implemented yet; runtime path can support it later.',
            'real_dem_provider_fetch': 'Implemented through provider-backed terrain acquisition.',
            'shareable_story_exports': 'Partially implemented through run ZIP export.',
        },
    }

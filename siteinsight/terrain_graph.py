from __future__ import annotations

from typing import Any
import math

from .terrain_models import TerrainSurface


def _distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(float(a['grid_x']) - float(b['grid_x']), float(a['grid_y']) - float(b['grid_y']))


def build_terrain_graph(surface: TerrainSurface, terrain_objects: dict[str, Any], archaeology: dict[str, Any] | None = None) -> dict[str, Any]:
    features = list(terrain_objects.get('features', []))
    nodes: list[dict[str, Any]] = []
    for feature in features:
        nodes.append({
            'id': feature['id'],
            'kind': feature['feature_type'],
            'grid_x': feature['grid_x'],
            'grid_y': feature['grid_y'],
            'strength': feature.get('strength', 0.0),
        })

    edges: list[dict[str, Any]] = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            d = _distance(a, b)
            if d > 18.0:
                continue
            relation = 'adjacent'
            if {'ridge_node', 'valley_node'} == {a['kind'], b['kind']}:
                relation = 'ridge_valley_pair'
            elif 'anomaly_candidate' in {a['kind'], b['kind']}:
                relation = 'anomaly_context'
            elif a['kind'] == b['kind']:
                relation = 'same_type_cluster'
            edges.append({
                'source': a['id'],
                'target': b['id'],
                'distance': round(d, 3),
                'relation': relation,
            })

    anomaly_nodes = [n for n in nodes if n['kind'] == 'anomaly_candidate']
    archaeological_candidates = archaeology.get('candidate_count', 0) if archaeology else 0
    graph_signal = min(100.0, len(edges) * 2.2 + len(anomaly_nodes) * 5.0 + archaeological_candidates * 4.0)

    return {
        'terrain_class': surface.terrain_class,
        'node_count': len(nodes),
        'edge_count': len(edges),
        'graph_signal_score': round(graph_signal, 2),
        'nodes': nodes,
        'edges': edges,
        'summary': {
            'notes': 'The terrain graph links extracted terrain objects into local spatial context for higher-level reasoning.',
            'dominant_relations': sorted({e['relation'] for e in edges}),
        },
    }

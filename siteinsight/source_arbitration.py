from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .terrain_models import TerrainQC, TerrainSource


@dataclass
class SourceCandidate:
    provider: str
    source_name: str
    raw_dem: np.ndarray
    meta: dict[str, Any]
    qc: TerrainQC
    arbitration_score: float
    arbitration_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "source_name": self.source_name,
            "arbitration_score": round(float(self.arbitration_score), 4),
            "arbitration_notes": list(self.arbitration_notes),
            "qc": self.qc.to_dict(),
            "meta": {
                "crs": self.meta.get("crs"),
                "width": self.meta.get("width"),
                "height": self.meta.get("height"),
                "res": list(self.meta.get("res")) if self.meta.get("res") else None,
                "nodata": self.meta.get("nodata"),
            },
        }


def score_candidate(source: TerrainSource, qc: TerrainQC) -> tuple[float, list[str]]:
    score = float(qc.overall_confidence)
    notes: list[str] = []

    relief_bonus = min(0.08, float(qc.local_relief_p95) / 60.0)
    score += relief_bonus
    if relief_bonus > 0.0:
        notes.append("Retains usable local-relief signal.")

    slope_bonus = min(0.04, float(qc.slope_p95) / 80.0)
    score += slope_bonus
    if slope_bonus > 0.0:
        notes.append("Shows non-flat slope variation.")

    if qc.valid_ratio >= 0.98:
        score += 0.03
        notes.append("Coverage is nearly complete.")
    elif qc.valid_ratio < 0.9:
        score -= 0.05
        notes.append("Coverage is weaker than preferred.")

    if qc.raw_range < 2.0:
        score -= 0.12
        notes.append("Decoded elevation range is extremely weak.")
    elif qc.raw_range < 8.0:
        score -= 0.05
        notes.append("Decoded elevation range is quiet for this patch.")

    if qc.edge_lowland_ratio > 0.45 and qc.raw_range > 20.0:
        score -= 0.08
        notes.append("Edge-lowland burden may indicate coastal or slab contamination.")

    if qc.water_influence_ratio > 0.5 and qc.edge_lowland_ratio > 0.35:
        score -= 0.06
        notes.append("Smooth lowland dominance remains suspicious.")

    provider = (source.provider or '').lower()
    if provider == 'usgs':
        score += 0.03
        notes.append("USGS source gets a mild preference where available.")
    elif provider.startswith('copernicus'):
        notes.append("Copernicus remains the global fallback baseline.")

    return max(0.0, min(1.0, score)), notes


def build_arbitration_summary(candidates: list[SourceCandidate], chosen: SourceCandidate) -> dict[str, Any]:
    ordered = sorted(candidates, key=lambda c: c.arbitration_score, reverse=True)
    return {
        "selected_provider": chosen.provider,
        "selected_source_name": chosen.source_name,
        "selected_score": round(float(chosen.arbitration_score), 4),
        "candidate_count": len(ordered),
        "selection_reason": chosen.arbitration_notes[:],
        "candidates": [c.to_dict() for c in ordered],
    }

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TerrainSource:
    raw_dem: np.ndarray
    provider: str
    source_name: str
    bbox: tuple[float, float, float, float]
    transform: Any | None = None
    crs: str | None = None
    nodata_value: float | None = None
    pixel_size_x: float | None = None
    pixel_size_y: float | None = None
    nominal_resolution_m: float | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerrainQC:
    valid_ratio: float
    void_ratio: float
    edge_void_ratio: float
    raw_min: float | None
    raw_max: float | None
    raw_range: float
    raw_std: float
    p01: float | None
    p99: float | None
    slope_p50: float
    slope_p95: float
    local_relief_p50: float
    local_relief_p95: float
    outlier_ratio: float
    spike_ratio: float
    lowland_ratio: float
    edge_lowland_ratio: float
    water_influence_ratio: float
    land_dominance_ratio: float
    source_confidence: float
    overall_confidence: float
    terrain_quality: str
    qualification_status: str = 'unchecked'
    qualification_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'valid_ratio': round(float(self.valid_ratio), 4),
            'nan_ratio': round(float(self.void_ratio), 4),
            'edge_void_ratio': round(float(self.edge_void_ratio), 4),
            'raw_min': None if self.raw_min is None else round(float(self.raw_min), 3),
            'raw_max': None if self.raw_max is None else round(float(self.raw_max), 3),
            'raw_range': round(float(self.raw_range), 3),
            'raw_std': round(float(self.raw_std), 3),
            'p01': None if self.p01 is None else round(float(self.p01), 3),
            'p99': None if self.p99 is None else round(float(self.p99), 3),
            'slope_p50': round(float(self.slope_p50), 4),
            'slope_p95': round(float(self.slope_p95), 4),
            'local_relief_p50': round(float(self.local_relief_p50), 4),
            'local_relief_p95': round(float(self.local_relief_p95), 4),
            'outlier_ratio': round(float(self.outlier_ratio), 4),
            'spike_ratio': round(float(self.spike_ratio), 4),
            'lowland_ratio': round(float(self.lowland_ratio), 4),
            'edge_lowland_ratio': round(float(self.edge_lowland_ratio), 4),
            'water_influence_ratio': round(float(self.water_influence_ratio), 4),
            'land_dominance_ratio': round(float(self.land_dominance_ratio), 4),
            'source_confidence': round(float(self.source_confidence), 4),
            'terrain_confidence': round(float(self.overall_confidence), 4),
            'terrain_quality': self.terrain_quality,
            'qualification_status': self.qualification_status,
            'qualification_notes': list(self.qualification_notes),
            'warnings': list(self.warnings),
            **self.diagnostics,
        }


@dataclass
class RenderingPolicy:
    allow_3d: bool
    mode: str
    vertical_exaggeration: float
    z_normalization: str
    mesh_smoothing: float
    clip_percentiles: tuple[float, float] | None
    emphasize_local_relief: bool
    suppress_archaeology: bool
    suppress_misleading_products: bool
    viewer_warnings: list[str] = field(default_factory=list)
    fallback_message: str | None = None
    product_surface_style: str = 'terrain_standard'
    export_profile: str = 'web'
    public_readiness: str = 'internal'
    derivative_profile: str = 'balanced'
    mesh_profile: str = 'standard'
    texture_profile: str = 'sharp'
    hero_export_ready: bool = False
    diagnostic_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            'allow_3d': bool(self.allow_3d),
            'mode': self.mode,
            'vertical_exaggeration': round(float(self.vertical_exaggeration), 2),
            'z_normalization': self.z_normalization,
            'mesh_smoothing': round(float(self.mesh_smoothing), 3),
            'clip_percentiles': list(self.clip_percentiles) if self.clip_percentiles else None,
            'emphasize_local_relief': bool(self.emphasize_local_relief),
            'suppress_archaeology': bool(self.suppress_archaeology),
            'suppress_misleading_products': bool(self.suppress_misleading_products),
            'viewer_warnings': list(self.viewer_warnings),
            'fallback_message': self.fallback_message,
            'product_surface_style': self.product_surface_style,
            'export_profile': self.export_profile,
            'public_readiness': self.public_readiness,
            'derivative_profile': self.derivative_profile,
            'mesh_profile': self.mesh_profile,
            'texture_profile': self.texture_profile,
            'hero_export_ready': bool(self.hero_export_ready),
            'diagnostic_only': bool(self.diagnostic_only),
        }


@dataclass
class TerrainSurface:
    raw_dem: np.ndarray
    cleaned_dem: np.ndarray
    valid_mask: np.ndarray
    provider: str
    source_name: str
    bbox: tuple[float, float, float, float]
    transform: Any | None
    crs: str | None
    nodata_value: float | None
    pixel_size_x: float | None
    pixel_size_y: float | None
    nominal_resolution_m: float | None
    qc: TerrainQC
    terrain_class: str
    rendering_policy: RenderingPolicy | None
    raw_stats: dict[str, Any] = field(default_factory=dict)
    clean_stats: dict[str, Any] = field(default_factory=dict)
    slope_stats: dict[str, Any] = field(default_factory=dict)
    relief_stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    derivative_products: dict[str, Any] = field(default_factory=dict)
    terrain_objects: dict[str, Any] = field(default_factory=dict)
    intelligence_graph: dict[str, Any] = field(default_factory=dict)
    normalization_summary: dict[str, Any] = field(default_factory=dict)
    public_product_notes: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def _safe_min(self, arr: np.ndarray) -> float:
        finite = np.asarray(arr, dtype=float)
        if not np.isfinite(finite).any():
            return 0.0
        return float(np.nanmin(finite))

    def _safe_max(self, arr: np.ndarray) -> float:
        finite = np.asarray(arr, dtype=float)
        if not np.isfinite(finite).any():
            return 0.0
        return float(np.nanmax(finite))

    def build_contract_dict(self) -> dict[str, Any]:
        clean_finite = np.asarray(self.cleaned_dem, dtype=float)
        raw_finite = np.asarray(self.raw_dem, dtype=float)
        return {
            'provider': self.provider,
            'source_name': self.source_name,
            'bbox': list(self.bbox),
            'crs': self.crs,
            'nodata_value': self.nodata_value,
            'nominal_resolution_m': self.nominal_resolution_m,
            'pixel_size_x': self.pixel_size_x,
            'pixel_size_y': self.pixel_size_y,
            'rows': int(clean_finite.shape[0]),
            'cols': int(clean_finite.shape[1]),
            'raw_elevation_stats': {
                'min': round(self._safe_min(raw_finite), 3),
                'max': round(self._safe_max(raw_finite), 3),
                'range': round(float(self.raw_stats.get('raw_range', self._safe_max(raw_finite) - self._safe_min(raw_finite))), 3),
            },
            'clean_elevation_stats': {
                'min': round(self._safe_min(clean_finite), 3),
                'max': round(self._safe_max(clean_finite), 3),
                'range': round(float(self.clean_stats.get('clean_range', self._safe_max(clean_finite) - self._safe_min(clean_finite))), 3),
            },
            'slope_stats': self.slope_stats,
            'local_relief_stats': self.relief_stats,
            'valid_pixels': int(np.asarray(self.valid_mask, dtype=bool).sum()),
            'raw_valid_ratio': round(float(np.isfinite(raw_finite).mean()) if raw_finite.size else 0.0, 4),
            'qualified_surface': self.qc.to_dict(),
            'terrain_class': self.terrain_class,
            'rendering_policy': self.rendering_policy.to_dict() if self.rendering_policy else None,
            'normalization_summary': self.normalization_summary,
            'public_product_notes': self.public_product_notes,
            'debug': self.debug,
        }

    def to_qc_dict(self) -> dict[str, Any]:
        payload = self.qc.to_dict()
        clean_finite = np.asarray(self.cleaned_dem, dtype=float)
        payload.update({
            'terrain_class': self.terrain_class,
            'rendering_policy': self.rendering_policy.to_dict() if self.rendering_policy else None,
            'valid_pixels': int(np.asarray(self.valid_mask, dtype=bool).sum()),
            'rows': int(self.cleaned_dem.shape[0]),
            'cols': int(self.cleaned_dem.shape[1]),
            'clean_min': round(self._safe_min(clean_finite), 3),
            'clean_max': round(self._safe_max(clean_finite), 3),
            'clean_range': round(float(self.clean_stats.get('clean_range', self._safe_max(clean_finite) - self._safe_min(clean_finite))), 3),
            'terrain_object_count': int(len((self.terrain_objects or {}).get('features', []))),
            'graph_node_count': int((self.intelligence_graph or {}).get('node_count', 0)),
            'graph_edge_count': int((self.intelligence_graph or {}).get('edge_count', 0)),
            'normalization_summary': self.normalization_summary,
            'public_product_notes': self.public_product_notes,
        })
        return payload

    def to_surface_dict(self) -> dict[str, Any]:
        return {
            'provider': self.provider,
            'source_name': self.source_name,
            'bbox': list(self.bbox),
            'crs': self.crs,
            'nominal_resolution_m': self.nominal_resolution_m,
            'terrain_class': self.terrain_class,
            'warnings': list(self.warnings),
            'raw_stats': self.raw_stats,
            'clean_stats': self.clean_stats,
            'slope_stats': self.slope_stats,
            'relief_stats': self.relief_stats,
            'qc': self.qc.to_dict(),
            'rendering_policy': self.rendering_policy.to_dict() if self.rendering_policy else None,
            'normalization_summary': self.normalization_summary,
            'public_product_notes': self.public_product_notes,
            'debug': self.debug,
            'terrain_contract': self.build_contract_dict(),
        }

from __future__ import annotations

from .terrain_models import RenderingPolicy, TerrainSurface


def select_rendering_policy(surface: TerrainSurface) -> RenderingPolicy:
    qc = surface.qc
    terrain_class = surface.terrain_class
    warnings = list(dict.fromkeys((surface.warnings or []) + qc.qualification_notes))

    if terrain_class in {'void_dominated', 'insufficient_surface'}:
        return RenderingPolicy(
            allow_3d=False,
            mode='suppressed',
            vertical_exaggeration=0.6,
            z_normalization='robust_percentile',
            mesh_smoothing=0.15,
            clip_percentiles=(2.0, 98.0),
            emphasize_local_relief=False,
            suppress_archaeology=True,
            suppress_misleading_products=True,
            viewer_warnings=warnings,
            fallback_message='Terrain quality is too weak for reliable 3D interpretation.',
            product_surface_style='diagnostic_flat',
            export_profile='diagnostic',
            public_readiness='blocked',
            derivative_profile='suppressed',
            mesh_profile='low_truth',
            texture_profile='muted',
            hero_export_ready=False,
            diagnostic_only=True,
        )

    if terrain_class == 'artifact_suspicious':
        return RenderingPolicy(
            allow_3d=True,
            mode='diagnostic',
            vertical_exaggeration=0.68,
            z_normalization='robust_percentile',
            mesh_smoothing=0.22,
            clip_percentiles=(4.0, 96.0),
            emphasize_local_relief=False,
            suppress_archaeology=True,
            suppress_misleading_products=True,
            viewer_warnings=warnings + ['3D is constrained because artifact burden is elevated.'],
            fallback_message='Terrain constrained because artifact burden is elevated.',
            product_surface_style='diagnostic_wire',
            export_profile='diagnostic',
            public_readiness='review_only',
            derivative_profile='guarded',
            mesh_profile='guarded',
            texture_profile='muted',
            hero_export_ready=False,
            diagnostic_only=True,
        )

    if terrain_class == 'coastal_transition':
        return RenderingPolicy(
            allow_3d=True,
            mode='coastal_land_focus',
            vertical_exaggeration=0.86,
            z_normalization='land_robust',
            mesh_smoothing=0.12,
            clip_percentiles=(5.0, 97.0),
            emphasize_local_relief=True,
            suppress_archaeology=False,
            suppress_misleading_products=False,
            viewer_warnings=warnings,
            product_surface_style='terrain_coastal',
            export_profile='web_marketing',
            public_readiness='guarded',
            derivative_profile='land_bias',
            mesh_profile='coastal_guarded',
            texture_profile='sharp',
            hero_export_ready=True,
            diagnostic_only=False,
        )

    if terrain_class == 'water_influenced':
        return RenderingPolicy(
            allow_3d=True,
            mode='water_guarded',
            vertical_exaggeration=0.84,
            z_normalization='land_robust',
            mesh_smoothing=0.13,
            clip_percentiles=(4.0, 97.0),
            emphasize_local_relief=True,
            suppress_archaeology=False,
            suppress_misleading_products=False,
            viewer_warnings=warnings,
            product_surface_style='terrain_hybrid',
            export_profile='web',
            public_readiness='guarded',
            derivative_profile='subtle_land',
            mesh_profile='guarded',
            texture_profile='sharp',
            hero_export_ready=True,
            diagnostic_only=False,
        )

    if terrain_class in {'subtle_relief', 'subtle_archaeology'}:
        return RenderingPolicy(
            allow_3d=True,
            mode='subtle_relief',
            vertical_exaggeration=1.05 if terrain_class == 'subtle_archaeology' else 0.92,
            z_normalization='local_relief',
            mesh_smoothing=0.08,
            clip_percentiles=(1.0, 99.0),
            emphasize_local_relief=True,
            suppress_archaeology=False,
            suppress_misleading_products=False,
            viewer_warnings=warnings,
            product_surface_style='terrain_emboss_candidate',
            export_profile='web_marketing',
            public_readiness='good',
            derivative_profile='micro_relief',
            mesh_profile='subtle',
            texture_profile='sharp',
            hero_export_ready=True,
            diagnostic_only=False,
        )

    if terrain_class == 'moderate_relief':
        return RenderingPolicy(
            allow_3d=True,
            mode='normal',
            vertical_exaggeration=1.08,
            z_normalization='robust_percentile',
            mesh_smoothing=0.06,
            clip_percentiles=(1.0, 99.0),
            emphasize_local_relief=False,
            suppress_archaeology=False,
            suppress_misleading_products=False,
            viewer_warnings=warnings,
            product_surface_style='terrain_standard',
            export_profile='web_marketing',
            public_readiness='good',
            derivative_profile='balanced',
            mesh_profile='standard',
            texture_profile='sharp',
            hero_export_ready=True,
            diagnostic_only=False,
        )

    return RenderingPolicy(
        allow_3d=True,
        mode='high_relief',
        vertical_exaggeration=0.88 if qc.edge_lowland_ratio > 0.45 else 1.0,
        z_normalization='robust_percentile',
        mesh_smoothing=0.05,
        clip_percentiles=(2.0, 98.0),
        emphasize_local_relief=False,
        suppress_archaeology=False,
        suppress_misleading_products=False,
        viewer_warnings=warnings,
        product_surface_style='terrain_hero',
        export_profile='web_marketing',
        public_readiness='good',
        derivative_profile='balanced',
        mesh_profile='hero',
        texture_profile='sharp',
        hero_export_ready=True,
        diagnostic_only=False,
    )

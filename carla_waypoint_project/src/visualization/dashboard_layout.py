"""Dashboard panel rectangle calculation."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from config.settings import CAMERA, DASHBOARD, DISPLAY


@dataclass(frozen=True)
class DashboardLayout:
    """Resolved dashboard panel rectangles in pygame screen coordinates."""

    main_view_rect: pygame.Rect
    map_rect: pygame.Rect
    lidar_rect: pygame.Rect
    sensor_panel_rect: pygame.Rect


def build_dashboard_layout(
    width: int = DISPLAY.width,
    height: int = DISPLAY.height,
) -> DashboardLayout:
    """Build panel rectangles while keeping the camera image at native size."""
    margin = DASHBOARD.margin_px
    gap = DASHBOARD.gap_px
    right_width = DASHBOARD.right_column_width
    bottom_height = DASHBOARD.bottom_panel_height

    main_view_rect = pygame.Rect(
        margin,
        margin,
        CAMERA.image_width,
        CAMERA.image_height,
    )

    right_x = main_view_rect.right + gap
    right_height = main_view_rect.height
    map_height = (right_height - gap) // 2
    lidar_height = right_height - gap - map_height

    map_rect = pygame.Rect(right_x, margin, right_width, map_height)
    lidar_rect = pygame.Rect(right_x, map_rect.bottom + gap, right_width, lidar_height)

    sensor_panel_rect = pygame.Rect(
        margin,
        main_view_rect.bottom + gap,
        width - 2 * margin,
        bottom_height,
    )

    if sensor_panel_rect.bottom > height - margin:
        sensor_panel_rect.height = max(120, height - margin - sensor_panel_rect.top)

    return DashboardLayout(
        main_view_rect=main_view_rect,
        map_rect=map_rect,
        lidar_rect=lidar_rect,
        sensor_panel_rect=sensor_panel_rect,
    )

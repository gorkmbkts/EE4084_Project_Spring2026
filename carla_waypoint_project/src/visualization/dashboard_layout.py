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
    control_panel_rect: pygame.Rect


def build_dashboard_layout(
    width: int = DISPLAY.width,
    height: int = DISPLAY.height,
) -> DashboardLayout:
    """Build panel rectangles while keeping the camera image aspect stable."""
    margin = DASHBOARD.margin_px
    gap = DASHBOARD.gap_px

    available_width = max(320, width - 2 * margin)
    available_height = max(300, height - 2 * margin)
    right_width = min(
        max(DASHBOARD.right_column_width, available_width - CAMERA.image_width - gap),
        max(DASHBOARD.right_column_width, available_width // 2),
    )
    left_width = max(320, available_width - right_width - gap)

    camera_aspect = CAMERA.image_width / CAMERA.image_height
    main_width = min(CAMERA.image_width, left_width)
    max_main_height = max(180, available_height - DASHBOARD.bottom_panel_height - gap)
    main_height = min(CAMERA.image_height, max_main_height)
    if main_width / main_height > camera_aspect:
        main_width = int(main_height * camera_aspect)
    else:
        main_height = int(main_width / camera_aspect)

    main_view_rect = pygame.Rect(
        margin,
        margin,
        main_width,
        main_height,
    )

    right_x = margin + left_width + gap
    right_height = available_height
    map_height = min(353, max(190, int(right_height * 0.36)))
    lidar_height = min(300, max(170, int(right_height * 0.26)))
    control_height = right_height - map_height - lidar_height - 2 * gap
    if control_height < 180:
        control_height = 180
        remaining = max(240, right_height - control_height - 2 * gap)
        map_height = max(140, remaining // 2)
        lidar_height = max(120, remaining - map_height)

    map_rect = pygame.Rect(right_x, margin, right_width, map_height)
    lidar_rect = pygame.Rect(right_x, map_rect.bottom + gap, right_width, lidar_height)
    control_panel_rect = pygame.Rect(
        right_x,
        lidar_rect.bottom + gap,
        right_width,
        max(120, height - margin - (lidar_rect.bottom + gap)),
    )

    bottom_height = max(120, height - margin - main_view_rect.bottom - gap)
    sensor_panel_rect = pygame.Rect(
        margin,
        main_view_rect.bottom + gap,
        left_width,
        bottom_height,
    )

    if sensor_panel_rect.bottom > height - margin:
        sensor_panel_rect.height = max(120, height - margin - sensor_panel_rect.top)
    if control_panel_rect.bottom > height - margin:
        control_panel_rect.height = max(120, height - margin - control_panel_rect.top)

    return DashboardLayout(
        main_view_rect=main_view_rect,
        map_rect=map_rect,
        lidar_rect=lidar_rect,
        sensor_panel_rect=sensor_panel_rect,
        control_panel_rect=control_panel_rect,
    )

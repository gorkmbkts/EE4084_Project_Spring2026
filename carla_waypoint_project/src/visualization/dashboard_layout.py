"""Dashboard panel rectangle calculation."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from config.settings import CAMERA, DASHBOARD, DISPLAY


@dataclass(frozen=True)
class DashboardLayout:
    """Resolved dashboard panel rectangles in pygame screen coordinates."""

    main_view_rect: pygame.Rect
    workspace_rect: pygame.Rect
    behavior_tuning_rect: pygame.Rect
    control_visual_rect: pygame.Rect
    driving_state_rect: pygame.Rect
    map_rect: pygame.Rect
    lidar_rect: pygame.Rect
    tab_panel_rect: pygame.Rect
    status_bar_rect: pygame.Rect


def build_dashboard_layout(
    width: int = DISPLAY.width,
    height: int = DISPLAY.height,
) -> DashboardLayout:
    """Build panel rectangles for the KalmanLab dashboard."""
    margin = DASHBOARD.margin_px
    gap = DASHBOARD.gap_px
    status_height = getattr(DASHBOARD, "status_bar_height_px", 34)

    available_width = max(320, width - 2 * margin)
    available_height = max(300, height - 2 * margin)
    content_height = max(240, available_height - status_height - gap)
    min_left_width = 320 if available_width >= 780 else 240
    min_right_width = 300 if available_width >= 780 else 220
    desired_right_width = max(DASHBOARD.right_column_width, available_width // 4)
    right_width = min(desired_right_width, max(min_right_width, available_width - min_left_width - gap))
    right_width = max(min_right_width, right_width)
    left_width = available_width - right_width - gap
    if left_width < min_left_width:
        left_width = max(220, available_width - min_right_width - gap)
        right_width = max(180, available_width - left_width - gap)
    if left_width + right_width + gap > available_width:
        total_columns_width = max(1, available_width - gap)
        right_width = max(140, int(total_columns_width * 0.42))
        left_width = max(140, total_columns_width - right_width)

    camera_aspect = CAMERA.image_width / CAMERA.image_height
    desired_main_height = max(1, int(left_width / camera_aspect))
    min_workspace_height = 160 if content_height >= 560 else 100
    max_main_height = max(120, content_height - gap - min_workspace_height)
    main_height = min(desired_main_height, max_main_height)
    workspace_height = max(80, content_height - main_height - gap)

    main_view_rect = pygame.Rect(
        margin,
        margin,
        left_width,
        main_height,
    )
    workspace_rect = pygame.Rect(
        margin,
        main_view_rect.bottom + gap,
        left_width,
        workspace_height,
    )
    bottom_gap = max(8, gap)
    bottom_panel_width = max(60, (workspace_rect.width - 2 * bottom_gap) // 3)
    bottom_group_width = 3 * bottom_panel_width + 2 * bottom_gap
    bottom_group_left = workspace_rect.left + max(0, (workspace_rect.width - bottom_group_width) // 2)
    behavior_tuning_rect = pygame.Rect(
        bottom_group_left,
        workspace_rect.top,
        bottom_panel_width,
        workspace_rect.height,
    )
    control_visual_rect = pygame.Rect(
        behavior_tuning_rect.right + bottom_gap,
        workspace_rect.top,
        bottom_panel_width,
        workspace_rect.height,
    )
    driving_state_rect = pygame.Rect(
        control_visual_rect.right + bottom_gap,
        workspace_rect.top,
        bottom_panel_width,
        workspace_rect.height,
    )

    right_x = margin + left_width + gap
    right_height = content_height
    map_height = max(150, int(right_height * 0.34))
    lidar_height = max(135, int(right_height * 0.27))
    tab_height = right_height - map_height - lidar_height - 2 * gap
    if tab_height < 190:
        tab_height = 190
        remaining = max(220, right_height - tab_height - 2 * gap)
        map_height = max(110, int(remaining * 0.56))
        lidar_height = max(100, remaining - map_height)

    map_rect = pygame.Rect(right_x, margin, right_width, map_height)
    lidar_rect = pygame.Rect(right_x, map_rect.bottom + gap, right_width, lidar_height)
    tab_panel_rect = pygame.Rect(
        right_x,
        lidar_rect.bottom + gap,
        right_width,
        max(120, margin + content_height - (lidar_rect.bottom + gap)),
    )
    status_bar_rect = pygame.Rect(
        margin,
        margin + content_height + gap,
        available_width,
        status_height,
    )

    if tab_panel_rect.bottom > margin + content_height:
        tab_panel_rect.height = max(120, margin + content_height - tab_panel_rect.top)
    if status_bar_rect.bottom > height - margin:
        status_bar_rect.height = max(24, height - margin - status_bar_rect.top)

    return DashboardLayout(
        main_view_rect=main_view_rect,
        workspace_rect=workspace_rect,
        behavior_tuning_rect=behavior_tuning_rect,
        control_visual_rect=control_visual_rect,
        driving_state_rect=driving_state_rect,
        map_rect=map_rect,
        lidar_rect=lidar_rect,
        tab_panel_rect=tab_panel_rect,
        status_bar_rect=status_bar_rect,
    )

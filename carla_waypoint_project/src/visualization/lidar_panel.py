"""Mini LiDAR bird's-eye-view panel rendering."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pygame

from config.settings import DASHBOARD, LIDAR_PANEL
from src.sensors.lidar_sensor import LidarMeasurement


class LidarPanelRenderer:
    """Render a lightweight local XY point-cloud projection."""

    def __init__(self) -> None:
        self._font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size)

    def draw(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        measurement: Optional[LidarMeasurement],
    ) -> None:
        pygame.draw.rect(
            surface,
            LIDAR_PANEL.background_color,
            rect,
            border_radius=DASHBOARD.panel_radius_px,
        )

        content = pygame.Rect(
            rect.left + DASHBOARD.panel_padding_px,
            rect.top + 28,
            rect.width - 2 * DASHBOARD.panel_padding_px,
            rect.height - 42,
        )
        old_clip = surface.get_clip()
        surface.set_clip(rect)

        center = content.center
        scale = min(content.width, content.height) / (2.0 * max(1.0, LIDAR_PANEL.range_m))
        self._draw_grid(surface, content, center, scale)

        if measurement is None or measurement.point_count == 0:
            self._draw_empty(surface, content)
        else:
            self._draw_points(surface, content, center, scale, measurement.points)
            self._draw_stats(surface, rect, measurement)

        pygame.draw.circle(surface, LIDAR_PANEL.ego_color, center, 4)
        pygame.draw.line(surface, LIDAR_PANEL.ego_color, center, (center[0], center[1] - 13), 2)
        surface.set_clip(old_clip)

    def _draw_grid(
        self,
        surface: pygame.Surface,
        content: pygame.Rect,
        center: tuple[int, int],
        scale: float,
    ) -> None:
        for ring_index in range(1, LIDAR_PANEL.ring_count + 1):
            radius_m = LIDAR_PANEL.range_m * ring_index / LIDAR_PANEL.ring_count
            radius_px = int(radius_m * scale)
            pygame.draw.circle(surface, LIDAR_PANEL.grid_color, center, radius_px, width=1)

        pygame.draw.line(surface, LIDAR_PANEL.axis_color, (center[0], content.top), (center[0], content.bottom), 1)
        pygame.draw.line(surface, LIDAR_PANEL.axis_color, (content.left, center[1]), (content.right, center[1]), 1)

    def _draw_points(
        self,
        surface: pygame.Surface,
        content: pygame.Rect,
        center: tuple[int, int],
        scale: float,
        points: np.ndarray,
    ) -> None:
        xy = points[:, :2]
        finite_mask = np.isfinite(xy).all(axis=1)
        xy = xy[finite_mask]
        if xy.size == 0:
            return

        distances = np.linalg.norm(xy, axis=1)
        in_range = distances <= LIDAR_PANEL.range_m
        xy = xy[in_range]
        distances = distances[in_range]
        if xy.size == 0:
            return

        if len(xy) > LIDAR_PANEL.max_points:
            stride = int(math.ceil(len(xy) / LIDAR_PANEL.max_points))
            xy = xy[::stride]
            distances = distances[::stride]

        screen_x = center[0] + xy[:, 1] * scale
        screen_y = center[1] - xy[:, 0] * scale
        near_color = np.array(LIDAR_PANEL.near_point_color, dtype=np.float32)
        far_color = np.array(LIDAR_PANEL.far_point_color, dtype=np.float32)
        ratios = np.clip(distances / max(1.0, LIDAR_PANEL.range_m), 0.0, 1.0)
        colors = near_color * (1.0 - ratios[:, None]) + far_color * ratios[:, None]

        for x, y, color in zip(screen_x.astype(int), screen_y.astype(int), colors.astype(np.uint8)):
            if not content.collidepoint(int(x), int(y)):
                continue
            if LIDAR_PANEL.point_radius_px <= 1:
                surface.set_at((int(x), int(y)), tuple(int(c) for c in color))
            else:
                pygame.draw.circle(surface, tuple(int(c) for c in color), (int(x), int(y)), LIDAR_PANEL.point_radius_px)

    def _draw_empty(self, surface: pygame.Surface, content: pygame.Rect) -> None:
        text = self._font.render("Waiting for LiDAR frame...", True, DASHBOARD.muted_text_color)
        surface.blit(text, (content.left + 8, content.top + 8))

    def _draw_stats(self, surface: pygame.Surface, rect: pygame.Rect, measurement: LidarMeasurement) -> None:
        text = f"frame {measurement.frame} | {measurement.point_count} pts"
        text_surface = self._font.render(text, True, DASHBOARD.muted_text_color)
        surface.blit(text_surface, (rect.left + DASHBOARD.panel_padding_px, rect.bottom - 20))

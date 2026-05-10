"""pygame dashboard display wrapper."""

from __future__ import annotations

from typing import Optional

import pygame

from config.settings import DASHBOARD, DISPLAY
from src.visualization.dashboard_layout import DashboardLayout, build_dashboard_layout


class PygameDisplay:
    """Own the pygame window and dashboard panel layout."""

    def __init__(
        self,
        width: int = DISPLAY.width,
        height: int = DISPLAY.height,
        title: str = DISPLAY.title,
    ) -> None:
        pygame.init()
        self._surface = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self._clear_color = DISPLAY.clear_color
        self._layout = build_dashboard_layout(width=width, height=height)
        self._title_font = pygame.font.SysFont("consolas", DASHBOARD.title_font_size, bold=True)
        self._status_font = pygame.font.SysFont("consolas", DASHBOARD.text_font_size)

    @property
    def surface(self) -> pygame.Surface:
        return self._surface

    @property
    def layout(self) -> DashboardLayout:
        return self._layout

    @property
    def main_view_rect(self) -> pygame.Rect:
        return self._layout.main_view_rect

    @property
    def map_rect(self) -> pygame.Rect:
        return self._layout.map_rect

    @property
    def lidar_rect(self) -> pygame.Rect:
        return self._layout.lidar_rect

    @property
    def sensor_panel_rect(self) -> pygame.Rect:
        return self._layout.sensor_panel_rect

    def begin_frame(self, camera_surface: Optional[pygame.Surface]) -> None:
        """Clear the dashboard and blit the current camera frame at 1:1 scale."""
        self._surface.fill(self._clear_color)
        for rect in (
            self._layout.main_view_rect,
            self._layout.map_rect,
            self._layout.lidar_rect,
            self._layout.sensor_panel_rect,
        ):
            pygame.draw.rect(
                self._surface,
                DASHBOARD.panel_background_color,
                rect,
                border_radius=DASHBOARD.panel_radius_px,
            )

        main_rect = self._layout.main_view_rect
        if camera_surface is None:
            waiting = self._status_font.render("Waiting for camera frame...", True, DASHBOARD.muted_text_color)
            self._surface.blit(waiting, (main_rect.left + 16, main_rect.top + 16))
        else:
            old_clip = self._surface.get_clip()
            self._surface.set_clip(main_rect)
            self._surface.blit(camera_surface, main_rect.topleft)
            self._surface.set_clip(old_clip)

    def end_frame(self) -> None:
        """Present the composed frame to the screen."""
        self.draw_panel_chrome()
        pygame.display.flip()

    def draw_panel_chrome(self) -> None:
        """Draw panel borders and compact titles after panel contents render."""
        self._draw_panel_frame(self._layout.main_view_rect, "Game View")
        self._draw_panel_frame(self._layout.map_rect, "2D Map")
        self._draw_panel_frame(self._layout.lidar_rect, "LiDAR")
        self._draw_panel_frame(self._layout.sensor_panel_rect, "Sensor Data")

    def _draw_panel_frame(self, rect: pygame.Rect, title: str) -> None:
        pygame.draw.rect(
            self._surface,
            DASHBOARD.panel_border_color,
            rect,
            width=DASHBOARD.panel_border_width_px,
            border_radius=DASHBOARD.panel_radius_px,
        )
        title_surface = self._title_font.render(title, True, DASHBOARD.title_color)
        label_rect = pygame.Rect(
            rect.left + 8,
            rect.top + 4,
            title_surface.get_width() + 10,
            title_surface.get_height() + 2,
        )
        pygame.draw.rect(self._surface, DASHBOARD.panel_background_color, label_rect)
        self._surface.blit(title_surface, (label_rect.left + 5, label_rect.top + 1))

    def shutdown(self) -> None:
        """Close pygame resources."""
        pygame.quit()

"""pygame dashboard display wrapper."""

from __future__ import annotations

from typing import Optional

import pygame

from config.settings import DASHBOARD, DISPLAY
from src.visualization.dashboard_layout import DashboardLayout, build_dashboard_layout
from src.visualization.windowing import configure_native_window, create_display_surface, display_flags_from_settings


class PygameDisplay:
    """Own the pygame window and dashboard panel layout."""

    def __init__(
        self,
        width: int = DISPLAY.width,
        height: int = DISPLAY.height,
        title: str = DISPLAY.title,
        existing_surface: Optional[pygame.Surface] = None,
        ) -> None:
        pygame.init()
        self._display_flags = display_flags_from_settings()
        current_surface = pygame.display.get_surface()
        if existing_surface is not None and current_surface is not None:
            self._surface = current_surface
            pygame.display.set_caption(title)
            configure_native_window()
        else:
            self._surface = create_display_surface(width=width, height=height, title=title)
        pygame.display.set_caption(title)
        self._clear_color = DISPLAY.clear_color
        actual_width, actual_height = self._surface.get_size()
        self._layout = build_dashboard_layout(width=actual_width, height=actual_height)
        self._camera_content_rect = self._layout.main_view_rect.copy()
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
    def camera_content_rect(self) -> pygame.Rect:
        return self._camera_content_rect.copy()

    @property
    def workspace_rect(self) -> pygame.Rect:
        return self._layout.workspace_rect

    @property
    def behavior_tuning_rect(self) -> pygame.Rect:
        return self._layout.behavior_tuning_rect

    @property
    def control_visual_rect(self) -> pygame.Rect:
        return self._layout.control_visual_rect

    @property
    def driving_state_rect(self) -> pygame.Rect:
        return self._layout.driving_state_rect

    @property
    def map_rect(self) -> pygame.Rect:
        return self._layout.map_rect

    @property
    def lidar_rect(self) -> pygame.Rect:
        return self._layout.lidar_rect

    @property
    def tab_panel_rect(self) -> pygame.Rect:
        return self._layout.tab_panel_rect

    @property
    def status_bar_rect(self) -> pygame.Rect:
        return self._layout.status_bar_rect

    @property
    def sensor_panel_rect(self) -> pygame.Rect:
        return self._layout.status_bar_rect

    @property
    def control_panel_rect(self) -> pygame.Rect:
        return self._layout.tab_panel_rect

    def resize(self, width: int, height: int) -> None:
        """Rebuild the window surface and dashboard layout after a resize."""
        if DISPLAY.fullscreen:
            return
        self._surface = pygame.display.set_mode((width, height), self._display_flags)
        actual_width, actual_height = self._surface.get_size()
        self._layout = build_dashboard_layout(width=actual_width, height=actual_height)
        self._camera_content_rect = self._layout.main_view_rect.copy()

    def begin_frame(self, camera_surface: Optional[pygame.Surface]) -> pygame.Rect:
        """Clear the dashboard and blit the current camera frame without distorting it."""
        self._surface.fill(self._clear_color)
        for rect in (
            self._layout.main_view_rect,
            self._layout.behavior_tuning_rect,
            self._layout.control_visual_rect,
            self._layout.driving_state_rect,
            self._layout.map_rect,
            self._layout.lidar_rect,
            self._layout.tab_panel_rect,
            self._layout.status_bar_rect,
        ):
            pygame.draw.rect(
                self._surface,
                DASHBOARD.panel_background_color,
                rect,
                border_radius=DASHBOARD.panel_radius_px,
            )

        main_rect = self._layout.main_view_rect
        if camera_surface is None:
            self._camera_content_rect = main_rect.copy()
            waiting = self._status_font.render("Waiting for camera frame...", True, DASHBOARD.muted_text_color)
            self._surface.blit(waiting, (main_rect.left + 16, main_rect.top + 16))
        else:
            old_clip = self._surface.get_clip()
            self._surface.set_clip(main_rect)
            camera_rect = self._fit_surface_rect(camera_surface, main_rect)
            self._camera_content_rect = camera_rect.copy()
            if camera_rect.size == camera_surface.get_size():
                self._surface.blit(camera_surface, camera_rect.topleft)
            else:
                scaled = pygame.transform.scale(camera_surface, camera_rect.size)
                self._surface.blit(scaled, camera_rect.topleft)
            self._surface.set_clip(old_clip)
        return self._camera_content_rect.copy()

    def end_frame(self) -> None:
        """Present the composed frame to the screen."""
        self.draw_panel_chrome()
        pygame.display.flip()

    def draw_panel_chrome(self) -> None:
        """Draw panel borders and compact titles after panel contents render."""
        self._draw_panel_frame(self._layout.main_view_rect, "Game View")
        self._draw_panel_frame(self._layout.behavior_tuning_rect, "Behavior Tuning")
        self._draw_panel_frame(self._layout.control_visual_rect, "Applied Controls")
        self._draw_panel_frame(self._layout.driving_state_rect, "Driving State")
        self._draw_panel_frame(self._layout.map_rect, "2D Map")
        self._draw_panel_frame(self._layout.lidar_rect, "LiDAR")
        self._draw_panel_border(self._layout.tab_panel_rect)
        self._draw_panel_border(self._layout.status_bar_rect)

    @staticmethod
    def _fit_surface_rect(surface: pygame.Surface, target: pygame.Rect) -> pygame.Rect:
        source_width, source_height = surface.get_size()
        if source_width <= 0 or source_height <= 0:
            return target.copy()

        scale = min(target.width / source_width, target.height / source_height)
        width = max(1, int(source_width * scale))
        height = max(1, int(source_height * scale))
        return pygame.Rect(
            target.left + (target.width - width) // 2,
            target.top + (target.height - height) // 2,
            width,
            height,
        )

    def _draw_panel_frame(self, rect: pygame.Rect, title: str) -> None:
        self._draw_panel_border(rect)
        title_surface = self._title_font.render(title, True, DASHBOARD.title_color)
        label_rect = pygame.Rect(
            rect.left + 8,
            rect.top + 4,
            title_surface.get_width() + 10,
            title_surface.get_height() + 2,
        )
        pygame.draw.rect(self._surface, DASHBOARD.panel_background_color, label_rect)
        self._surface.blit(title_surface, (label_rect.left + 5, label_rect.top + 1))

    def _draw_panel_border(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(
            self._surface,
            DASHBOARD.panel_border_color,
            rect,
            width=DASHBOARD.panel_border_width_px,
            border_radius=DASHBOARD.panel_radius_px,
        )

    def shutdown(self) -> None:
        """Close pygame resources."""
        pygame.quit()

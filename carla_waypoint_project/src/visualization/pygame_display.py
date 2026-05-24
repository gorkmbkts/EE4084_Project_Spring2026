"""pygame dashboard display wrapper."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys
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
        self._display_flags = self._display_flags_from_settings()
        self._surface = pygame.display.set_mode((width, height), self._display_flags)
        pygame.display.set_caption(title)
        self._configure_native_window()
        self._clear_color = DISPLAY.clear_color
        actual_width, actual_height = self._surface.get_size()
        self._layout = build_dashboard_layout(width=actual_width, height=actual_height)
        self._title_font = pygame.font.SysFont("consolas", DASHBOARD.title_font_size, bold=True)
        self._status_font = pygame.font.SysFont("consolas", DASHBOARD.text_font_size)

    @staticmethod
    def _display_flags_from_settings() -> int:
        flags = 0
        if DISPLAY.fullscreen and not DISPLAY.maximized:
            flags |= pygame.FULLSCREEN
        if DISPLAY.resizable or DISPLAY.maximized:
            flags |= pygame.RESIZABLE
        return flags

    def _configure_native_window(self) -> None:
        """Apply Windows window styles and maximize without pygame fullscreen."""
        if sys.platform != "win32":
            return

        window_info = pygame.display.get_wm_info()
        hwnd = window_info.get("window")
        if not hwnd:
            return

        hwnd_handle = wintypes.HWND(hwnd)
        if DISPLAY.borderless:
            self._make_window_borderless_resizable(hwnd_handle)
        if DISPLAY.maximized:
            ctypes.windll.user32.ShowWindow(hwnd_handle, 3)
        pygame.time.wait(50)
        pygame.event.pump()
        self._surface = pygame.display.get_surface() or self._surface

    @staticmethod
    def _make_window_borderless_resizable(hwnd: wintypes.HWND) -> None:
        user32 = ctypes.windll.user32
        gwl_style = -16
        swp_nosize = 0x0001
        swp_nomove = 0x0002
        swp_nozorder = 0x0004
        swp_framechanged = 0x0020
        ws_caption = 0x00C00000
        ws_sysmenu = 0x00080000
        ws_thickframe = 0x00040000
        ws_minimizebox = 0x00020000
        ws_maximizebox = 0x00010000

        style = user32.GetWindowLongW(hwnd, gwl_style)
        style &= ~ws_caption
        style |= ws_sysmenu | ws_thickframe | ws_minimizebox | ws_maximizebox
        user32.SetWindowLongW(hwnd, gwl_style, style)
        user32.SetWindowPos(
            hwnd,
            wintypes.HWND(0),
            0,
            0,
            0,
            0,
            swp_nomove | swp_nosize | swp_nozorder | swp_framechanged,
        )

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

    @property
    def control_panel_rect(self) -> pygame.Rect:
        return self._layout.control_panel_rect

    def resize(self, width: int, height: int) -> None:
        """Rebuild the window surface and dashboard layout after a resize."""
        if DISPLAY.fullscreen:
            return
        self._surface = pygame.display.set_mode((width, height), self._display_flags)
        actual_width, actual_height = self._surface.get_size()
        self._layout = build_dashboard_layout(width=actual_width, height=actual_height)

    def begin_frame(self, camera_surface: Optional[pygame.Surface]) -> None:
        """Clear the dashboard and blit the current camera frame without distorting it."""
        self._surface.fill(self._clear_color)
        for rect in (
            self._layout.main_view_rect,
            self._layout.map_rect,
            self._layout.lidar_rect,
            self._layout.sensor_panel_rect,
            self._layout.control_panel_rect,
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
            camera_rect = self._fit_surface_rect(camera_surface, main_rect)
            if camera_rect.size == camera_surface.get_size():
                self._surface.blit(camera_surface, camera_rect.topleft)
            else:
                scaled = pygame.transform.scale(camera_surface, camera_rect.size)
                self._surface.blit(scaled, camera_rect.topleft)
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
        self._draw_panel_frame(self._layout.control_panel_rect, "Controls")

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

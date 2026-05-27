"""Compact bottom status bar for the pygame dashboard."""

from __future__ import annotations

import pygame

from config.settings import DASHBOARD


class StatusBar:
    """Render a cached one-line status summary."""

    def __init__(self) -> None:
        self._font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size, bold=True)
        self._text = ""
        self._surface: pygame.Surface | None = None

    def set_text(self, text: str) -> None:
        if text == self._text:
            return
        self._text = text
        self._surface = self._font.render(text, True, DASHBOARD.text_color)

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(
            surface,
            DASHBOARD.panel_inner_color,
            rect,
            border_radius=DASHBOARD.panel_radius_px,
        )
        pygame.draw.rect(
            surface,
            DASHBOARD.panel_border_color,
            rect,
            width=DASHBOARD.panel_border_width_px,
            border_radius=DASHBOARD.panel_radius_px,
        )
        if self._surface is None:
            return
        old_clip = surface.get_clip()
        surface.set_clip(rect.inflate(-12, 0))
        y = rect.centery - self._surface.get_height() // 2
        surface.blit(self._surface, (rect.left + 10, y))
        surface.set_clip(old_clip)

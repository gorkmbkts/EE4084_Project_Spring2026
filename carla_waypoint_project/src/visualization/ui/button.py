"""Lightweight pygame button widget."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import pygame

from config.settings import DASHBOARD


class Button:
    """Clickable rectangular button with cached label rendering."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        callback: Callable[[], None],
        font: pygame.font.Font,
        enabled: bool = True,
        active: bool = False,
    ) -> None:
        self.rect = rect.copy()
        self.label = label
        self.callback = callback
        self.enabled = enabled
        self.active = active
        self.hovered = False
        self._font = font
        self._label_enabled: Optional[pygame.Surface] = None
        self._label_disabled: Optional[pygame.Surface] = None
        self._render_labels()

    def set_rect(self, rect: pygame.Rect) -> None:
        self.rect = rect.copy()

    def set_state(
        self,
        enabled: Optional[bool] = None,
        active: Optional[bool] = None,
    ) -> None:
        if enabled is not None:
            self.enabled = enabled
            if not enabled:
                self.hovered = False
        if active is not None:
            self.active = active

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Update hover state and invoke the callback on left-click."""
        if hasattr(event, "pos"):
            self.hovered = self.enabled and self.rect.collidepoint(event.pos)

        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return False
        if not self.enabled or not self.rect.collidepoint(event.pos):
            return False

        self.callback()
        return True

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the button without allocating text surfaces per frame."""
        background = self._background_color()
        border = self._border_color()
        pygame.draw.rect(surface, background, self.rect, border_radius=4)
        pygame.draw.rect(surface, border, self.rect, width=1, border_radius=4)
        if self.active and self.enabled:
            indicator = pygame.Rect(self.rect.left + 2, self.rect.top + 4, 4, self.rect.height - 8)
            pygame.draw.rect(surface, DASHBOARD.success_color, indicator, border_radius=2)

        label_surface = self._label_enabled if self.enabled else self._label_disabled
        if label_surface is None:
            return

        label_rect = label_surface.get_rect(center=self.rect.center)
        old_clip = surface.get_clip()
        surface.set_clip(self.rect.inflate(-8, 0))
        surface.blit(label_surface, label_rect)
        surface.set_clip(old_clip)

    def _render_labels(self) -> None:
        self._label_enabled = self._font.render(self.label, True, DASHBOARD.text_color)
        self._label_disabled = self._font.render(self.label, True, DASHBOARD.muted_text_color)

    def _background_color(self) -> tuple[int, int, int]:
        if not self.enabled:
            return (38, 42, 50)
        if self.active:
            return (26, 74, 54)
        if self.hovered:
            return (48, 56, 70)
        return DASHBOARD.panel_inner_color

    def _border_color(self) -> tuple[int, int, int]:
        if not self.enabled:
            return (56, 62, 72)
        if self.active:
            return DASHBOARD.success_color
        if self.hovered:
            return DASHBOARD.title_color
        return DASHBOARD.panel_border_color


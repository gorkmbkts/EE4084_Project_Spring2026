"""Control panel composed of reusable buttons and cached status text."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional

import pygame

from config.settings import DASHBOARD
from src.visualization.ui.button import Button


class ControlPanel:
    """Manage dashboard buttons and a compact status readout."""

    def __init__(self, rect: pygame.Rect) -> None:
        self._rect = rect.copy()
        self._font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size)
        self._status_font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size)
        self._buttons: list[Button] = []
        self._buttons_by_label: dict[str, Button] = {}
        self._status_lines: tuple[str, ...] = ()
        self._status_surfaces: tuple[pygame.Surface, ...] = ()
        self._status_area_height = 96

    @property
    def rect(self) -> pygame.Rect:
        return self._rect

    def add_button(
        self,
        label: str,
        callback: Callable[[], None],
        enabled: bool = True,
        active: bool = False,
    ) -> Button:
        button = Button(
            rect=pygame.Rect(0, 0, 1, 1),
            label=label,
            callback=callback,
            font=self._font,
            enabled=enabled,
            active=active,
        )
        self._buttons.append(button)
        self._buttons_by_label[label] = button
        self._layout_buttons()
        return button

    def set_rect(self, rect: pygame.Rect) -> None:
        if self._rect == rect:
            return
        self._rect = rect.copy()
        self._layout_buttons()

    def set_button_state(
        self,
        label: str,
        enabled: Optional[bool] = None,
        active: Optional[bool] = None,
    ) -> None:
        button = self._buttons_by_label.get(label)
        if button is not None:
            button.set_state(enabled=enabled, active=active)

    def set_status_text(self, text: str) -> None:
        self.set_status_lines([text] if text else [])

    def set_status_lines(self, lines: Sequence[str]) -> None:
        normalized = tuple(line for line in lines if line)
        if normalized == self._status_lines:
            return
        self._status_lines = normalized
        self._status_surfaces = tuple(
            self._status_font.render(line, True, DASHBOARD.text_color)
            for line in self._status_lines
        )
        self._layout_buttons()

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return True when the panel consumes a mouse event."""
        if not hasattr(event, "pos"):
            return False

        inside_panel = self._rect.collidepoint(event.pos)
        if not inside_panel:
            if event.type == pygame.MOUSEMOTION:
                for button in self._buttons:
                    button.hovered = False
            return False

        for button in self._buttons:
            if button.handle_event(event):
                return True

        return event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL)

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(
            surface,
            DASHBOARD.panel_inner_color,
            self._rect,
            border_radius=DASHBOARD.panel_radius_px,
        )

        old_clip = surface.get_clip()
        surface.set_clip(self._rect)
        x = self._rect.left + DASHBOARD.panel_padding_px
        y = self._rect.top + 28
        status_bottom = y + self._status_area_height
        for status_surface in self._status_surfaces:
            if y + status_surface.get_height() > status_bottom:
                break
            surface.blit(status_surface, (x, y))
            y += 14

        for button in self._buttons:
            button.draw(surface)
        surface.set_clip(old_clip)

    def _layout_buttons(self) -> None:
        if not self._buttons:
            return

        padding = DASHBOARD.panel_padding_px
        compact = self._rect.height < 380
        very_compact = self._rect.height < 300
        gap = 3 if very_compact else (4 if compact else 6)
        if very_compact and self._rect.width >= 420:
            columns = 3
        else:
            columns = 2 if self._rect.width >= 360 else 1
        button_height = 22 if very_compact else (24 if compact else 27)
        status_height = 74 if very_compact else min(106, 12 + len(self._status_lines) * 14)
        self._status_area_height = status_height
        top = self._rect.top + 28 + status_height
        available_width = self._rect.width - 2 * padding - (columns - 1) * gap
        button_width = max(80, available_width // columns)

        for index, button in enumerate(self._buttons):
            row = index // columns
            column = index % columns
            rect = pygame.Rect(
                self._rect.left + padding + column * (button_width + gap),
                top + row * (button_height + gap),
                button_width,
                button_height,
            )
            button.set_rect(rect)

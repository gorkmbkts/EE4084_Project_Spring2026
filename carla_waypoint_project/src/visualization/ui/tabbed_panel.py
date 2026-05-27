"""Tabbed pygame panel with cached buttons and active-tab text rendering."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional

import pygame

from config.settings import DASHBOARD
from src.visualization.ui.button import Button


class TabbedPanel:
    """Draw and route events for a compact tabbed dashboard panel."""

    def __init__(self, rect: pygame.Rect, tabs: Sequence[str]) -> None:
        if not tabs:
            raise ValueError("TabbedPanel requires at least one tab.")
        self._rect = rect.copy()
        self._tabs = tuple(tabs)
        self._active_tab = self._tabs[0]
        self._font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size)
        self._tab_font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size, bold=True)
        self._text_font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size)
        self._tab_buttons: dict[str, Button] = {}
        self._content_buttons: dict[str, list[Button]] = {tab: [] for tab in self._tabs}
        self._buttons_by_label: dict[str, list[Button]] = {}
        self._text_lines: dict[str, tuple[str, ...]] = {tab: () for tab in self._tabs}
        self._text_surfaces: dict[str, tuple[pygame.Surface, ...]] = {tab: () for tab in self._tabs}

        for tab in self._tabs:
            self._tab_buttons[tab] = Button(
                rect=pygame.Rect(0, 0, 1, 1),
                label=tab,
                callback=lambda selected=tab: self.set_active_tab(selected),
                font=self._tab_font,
                active=(tab == self._active_tab),
            )
        self._layout()

    @property
    def rect(self) -> pygame.Rect:
        return self._rect

    @property
    def active_tab(self) -> str:
        return self._active_tab

    def set_rect(self, rect: pygame.Rect) -> None:
        if self._rect == rect:
            return
        self._rect = rect.copy()
        self._layout()

    def set_active_tab(self, tab: str) -> None:
        if tab not in self._content_buttons or tab == self._active_tab:
            return
        self._active_tab = tab
        for name, button in self._tab_buttons.items():
            button.set_state(active=(name == self._active_tab))
        self._layout()

    def add_button(
        self,
        tab: str,
        label: str,
        callback: Callable[[], None],
        enabled: bool = True,
        active: bool = False,
    ) -> Button:
        if tab not in self._content_buttons:
            raise KeyError(f"Unknown tab: {tab}")
        button = Button(
            rect=pygame.Rect(0, 0, 1, 1),
            label=label,
            callback=callback,
            font=self._font,
            enabled=enabled,
            active=active,
        )
        self._content_buttons[tab].append(button)
        self._buttons_by_label.setdefault(label, []).append(button)
        self._layout()
        return button

    def set_button_state(
        self,
        label: str,
        enabled: Optional[bool] = None,
        active: Optional[bool] = None,
    ) -> None:
        for button in self._buttons_by_label.get(label, ()):
            button.set_state(enabled=enabled, active=active)

    def set_text_lines(self, tab: str, lines: Sequence[str]) -> None:
        if tab not in self._text_lines:
            raise KeyError(f"Unknown tab: {tab}")
        normalized = tuple(line for line in lines if line)
        if normalized == self._text_lines[tab]:
            return
        self._text_lines[tab] = normalized
        self._text_surfaces[tab] = tuple(
            self._text_font.render(line, True, self._color_for_line(line))
            for line in normalized
        )

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not hasattr(event, "pos"):
            return False

        inside_panel = self._rect.collidepoint(event.pos)
        if not inside_panel:
            if event.type == pygame.MOUSEMOTION:
                self._clear_hover()
            return False

        for button in self._tab_buttons.values():
            if button.handle_event(event):
                return True

        for button in self._content_buttons[self._active_tab]:
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
        for button in self._tab_buttons.values():
            button.draw(surface)
        for button in self._content_buttons[self._active_tab]:
            button.draw(surface)

        self._draw_active_text(surface)
        surface.set_clip(old_clip)

    def _draw_active_text(self, surface: pygame.Surface) -> None:
        content_left = self._rect.left + DASHBOARD.panel_padding_px
        y = self._text_top()
        bottom = self._rect.bottom - DASHBOARD.panel_padding_px
        for text_surface in self._text_surfaces.get(self._active_tab, ()):
            if y + text_surface.get_height() > bottom:
                break
            surface.blit(text_surface, (content_left, y))
            y += 16

    def _layout(self) -> None:
        self._layout_tab_buttons()
        self._layout_content_buttons()

    def _layout_tab_buttons(self) -> None:
        padding = DASHBOARD.panel_padding_px
        gap = 4
        tab_count = len(self._tabs)
        available_width = self._rect.width - 2 * padding - (tab_count - 1) * gap
        tab_width = max(38, available_width // tab_count)
        tab_height = 24
        y = self._rect.top + 8
        for index, tab in enumerate(self._tabs):
            rect = pygame.Rect(
                self._rect.left + padding + index * (tab_width + gap),
                y,
                tab_width,
                tab_height,
            )
            self._tab_buttons[tab].set_rect(rect)

    def _layout_content_buttons(self) -> None:
        padding = DASHBOARD.panel_padding_px
        gap = 4
        top = self._content_top()
        for tab, buttons in self._content_buttons.items():
            columns = 2 if self._rect.width >= 350 else 1
            if tab == "Filters":
                columns = 1 if self._rect.width < 380 else 2
            button_height = 24
            available_width = self._rect.width - 2 * padding - (columns - 1) * gap
            button_width = max(90, available_width // columns)
            for index, button in enumerate(buttons):
                row = index // columns
                column = index % columns
                rect = pygame.Rect(
                    self._rect.left + padding + column * (button_width + gap),
                    top + row * (button_height + gap),
                    button_width,
                    button_height,
                )
                button.set_rect(rect)

    def _content_top(self) -> int:
        return self._rect.top + 40

    def _text_top(self) -> int:
        active_buttons = self._content_buttons[self._active_tab]
        if not active_buttons:
            return self._content_top()
        return max(button.rect.bottom for button in active_buttons) + 8

    def _clear_hover(self) -> None:
        for button in self._tab_buttons.values():
            button.hovered = False
        for buttons in self._content_buttons.values():
            for button in buttons:
                button.hovered = False

    @staticmethod
    def _color_for_line(line: str) -> tuple[int, int, int]:
        lowered = line.lower()
        if "warning" in lowered or "unsafe" in lowered or "failed" in lowered or "error" in lowered:
            return DASHBOARD.warning_color
        if lowered.startswith("active") or "completed" in lowered:
            return DASHBOARD.success_color
        if line.endswith(":"):
            return DASHBOARD.title_color
        return DASHBOARD.text_color

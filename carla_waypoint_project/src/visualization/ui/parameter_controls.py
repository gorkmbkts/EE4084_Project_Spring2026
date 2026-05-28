"""Reusable pygame slider, numeric input, and preset controls."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Optional

import pygame

from config.settings import DASHBOARD
from src.evaluation.benchmark_config import ParameterSpec


class ParameterEditor:
    """Render and edit a dictionary of numeric values with sliders and inputs."""

    def __init__(
        self,
        specs: Sequence[ParameterSpec],
        values: dict[str, float],
        presets: dict[str, dict[str, float]],
        active_preset: str,
        title: str = "",
        on_commit: Optional[Callable[[dict[str, float], str], None]] = None,
        max_rows: Optional[int] = None,
    ) -> None:
        self._specs = tuple(specs)
        self._values = {spec.key: spec.clamp(values.get(spec.key, spec.minimum)) for spec in self._specs}
        self._presets = {name: dict(preset) for name, preset in presets.items()}
        self._active_preset = active_preset if active_preset in presets else "Custom"
        self._title = title
        self._on_commit = on_commit
        self._max_rows = max_rows
        self._font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size)
        self._small_font = pygame.font.SysFont("consolas", 11)
        self._bold_font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size, bold=True)
        self._track_rects: dict[str, pygame.Rect] = {}
        self._input_rects: dict[str, pygame.Rect] = {}
        self._preset_rects: dict[str, pygame.Rect] = {}
        self._drag_key: Optional[str] = None
        self._active_input_key: Optional[str] = None
        self._input_text = ""
        self._scroll_y = 0
        self._last_rect = pygame.Rect(0, 0, 1, 1)
        self._last_content_rect = pygame.Rect(0, 0, 1, 1)
        self._dirty = False
        self.status_text = ""

    @property
    def active_preset(self) -> str:
        return self._active_preset

    def values(self) -> dict[str, float]:
        committed = dict(self._values)
        if self._active_input_key is not None:
            self._commit_active_input(call_callback=False)
            committed = dict(self._values)
        return committed

    def set_values(self, values: dict[str, object], active_preset: str = "Custom", commit: bool = False) -> None:
        for spec in self._specs:
            self._values[spec.key] = spec.clamp(values.get(spec.key, self._values.get(spec.key, spec.minimum)))
        self._active_preset = active_preset
        self._active_input_key = None
        self._input_text = ""
        self._dirty = False
        if commit:
            self._commit()

    def apply_preset(self, preset_name: str, commit: bool = True) -> None:
        values = self._presets.get(preset_name)
        if values is None:
            return
        for spec in self._specs:
            if spec.key in values:
                self._values[spec.key] = spec.clamp(values[spec.key])
        self._active_preset = preset_name
        self._active_input_key = None
        self._input_text = ""
        self._dirty = False
        self.status_text = f"Preset: {preset_name}"
        if commit:
            self._commit()

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and self._active_input_key is not None:
            return self._handle_key_down(event)

        if self._drag_key is not None:
            if event.type == pygame.MOUSEMOTION and hasattr(event, "pos"):
                self._set_value_from_x(self._drag_key, event.pos[0], commit=False)
                return True
            if event.type == pygame.MOUSEBUTTONUP and getattr(event, "button", None) == 1:
                self._set_value_from_x(self._drag_key, event.pos[0], commit=True)
                self._drag_key = None
                return True

        if not hasattr(event, "pos"):
            return False

        position = event.pos
        if not self._last_rect.collidepoint(position):
            if event.type == pygame.MOUSEBUTTONDOWN and self._active_input_key is not None:
                self._commit_active_input(call_callback=True)
            return False

        if event.type == pygame.MOUSEWHEEL:
            self._scroll_y = self._clamp_int(self._scroll_y - event.y * 42, 0, self._max_scroll_y())
            return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for name, rect in self._preset_rects.items():
                if rect.collidepoint(position):
                    self.apply_preset(name, commit=True)
                    return True

            for key, rect in self._input_rects.items():
                if rect.collidepoint(position):
                    self._activate_input(key)
                    return True

            if self._active_input_key is not None:
                self._commit_active_input(call_callback=True)

            for key, rect in self._track_rects.items():
                if rect.inflate(0, 14).collidepoint(position):
                    self._drag_key = key
                    self._set_value_from_x(key, position[0], commit=False)
                    return True
            return True

        return event.type in (pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION)

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._last_rect = rect.copy()
        pygame.draw.rect(surface, DASHBOARD.panel_inner_color, rect, border_radius=DASHBOARD.panel_radius_px)
        self._track_rects.clear()
        self._input_rects.clear()
        self._preset_rects.clear()

        content = pygame.Rect(
            rect.left + DASHBOARD.panel_padding_px,
            rect.top + DASHBOARD.panel_padding_px,
            rect.width - 2 * DASHBOARD.panel_padding_px,
            rect.height - 2 * DASHBOARD.panel_padding_px,
        )
        self._draw_header(surface, content)
        presets_bottom = self._draw_presets(surface, content)

        editor_top = presets_bottom + 8
        if self._title:
            title = self._bold_font.render(self._title, True, DASHBOARD.title_color)
            surface.blit(title, (content.left, editor_top))
            editor_top += title.get_height() + 6

        editor_rect = pygame.Rect(content.left, editor_top, content.width, max(20, content.bottom - editor_top))
        self._last_content_rect = editor_rect.copy()
        old_clip = surface.get_clip()
        surface.set_clip(editor_rect)

        row_height = 28
        y = editor_rect.top - self._scroll_y
        visible_specs = self._specs[: self._max_rows] if self._max_rows is not None else self._specs
        previous_group = None
        for spec in visible_specs:
            if spec.group and spec.group != previous_group:
                if y + 16 >= editor_rect.top and y <= editor_rect.bottom:
                    group_surface = self._small_font.render(spec.group.upper(), True, DASHBOARD.muted_text_color)
                    surface.blit(group_surface, (editor_rect.left, y + 2))
                y += 17
                previous_group = spec.group

            row = pygame.Rect(editor_rect.left, y, editor_rect.width, row_height)
            if row.bottom >= editor_rect.top and row.top <= editor_rect.bottom:
                self._draw_parameter_row(surface, row, spec)
            y += row_height

        surface.set_clip(old_clip)
        self._draw_scrollbar(surface, editor_rect)

    def _draw_header(self, surface: pygame.Surface, content: pygame.Rect) -> None:
        if not self.status_text:
            return
        color = DASHBOARD.warning_color if "failed" in self.status_text.lower() or "error" in self.status_text.lower() else DASHBOARD.muted_text_color
        text = self._fit_text(self.status_text, self._small_font, content.width)
        rendered = self._small_font.render(text, True, color)
        surface.blit(rendered, (content.left, content.top))

    def _draw_presets(self, surface: pygame.Surface, content: pygame.Rect) -> int:
        y = content.top + (16 if self.status_text else 0)
        if not self._presets:
            return y
        x = content.left
        button_height = 24
        gap = 6
        for name in (*self._presets.keys(), "Custom"):
            width = min(142, max(74, self._bold_font.size(name)[0] + 18))
            if x + width > content.right:
                x = content.left
                y += button_height + gap
            rect = pygame.Rect(x, y, width, button_height)
            self._preset_rects[name] = rect
            active = name == self._active_preset
            hovered = rect.collidepoint(pygame.mouse.get_pos())
            if active:
                background = (35, 73, 53)
                border = DASHBOARD.success_color
            elif hovered:
                background = (34, 42, 54)
                border = (116, 188, 255)
            else:
                background = (24, 30, 39)
                border = DASHBOARD.panel_border_color
            pygame.draw.rect(surface, background, rect, border_radius=4)
            pygame.draw.rect(surface, border, rect, width=1, border_radius=4)
            label = self._fit_text(name, self._small_font, rect.width - 8)
            rendered = self._small_font.render(label, True, DASHBOARD.title_color if active else DASHBOARD.text_color)
            surface.blit(rendered, rendered.get_rect(center=rect.center))
            x += width + gap
        return y + button_height

    def _draw_parameter_row(self, surface: pygame.Surface, row: pygame.Rect, spec: ParameterSpec) -> None:
        value = self._values.get(spec.key, spec.minimum)
        label_width = min(132, max(82, int(row.width * 0.34)))
        input_width = min(86, max(62, int(row.width * 0.22)))
        track_left = row.left + label_width + 8
        track_right = row.right - input_width - 10
        if track_right <= track_left + 34:
            track_left = row.left
            track_right = row.right - input_width - 8
            label_y = row.top
        else:
            label_y = row.top + 6

        label = self._fit_text(spec.label, self._font, label_width)
        surface.blit(self._font.render(label, True, DASHBOARD.text_color), (row.left, label_y))

        track_rect = pygame.Rect(track_left, row.top + 13, max(30, track_right - track_left), 5)
        self._track_rects[spec.key] = track_rect
        normalized = self._normalized(spec, value)
        pygame.draw.rect(surface, (49, 56, 68), track_rect, border_radius=2)
        fill_rect = track_rect.copy()
        fill_rect.width = max(2, int(track_rect.width * normalized))
        pygame.draw.rect(surface, (95, 175, 240), fill_rect, border_radius=2)
        knob_x = track_rect.left + int(track_rect.width * normalized)
        knob = pygame.Rect(0, 0, 9, 14)
        knob.center = (knob_x, track_rect.centery)
        pygame.draw.rect(
            surface,
            (224, 236, 248) if self._drag_key == spec.key else (151, 206, 255),
            knob,
            border_radius=3,
        )

        input_rect = pygame.Rect(row.right - input_width, row.top + 2, input_width, 22)
        self._input_rects[spec.key] = input_rect
        active = self._active_input_key == spec.key
        pygame.draw.rect(surface, (14, 18, 24), input_rect, border_radius=3)
        pygame.draw.rect(
            surface,
            (116, 188, 255) if active else DASHBOARD.panel_border_color,
            input_rect,
            width=1,
            border_radius=3,
        )
        if active:
            text = self._input_text
        else:
            text = self._format_value(value, spec)
        rendered = self._small_font.render(self._fit_text(text, self._small_font, input_rect.width - 8), True, DASHBOARD.title_color)
        surface.blit(rendered, (input_rect.left + 4, input_rect.top + 5))

        if spec.unit and row.width >= 355:
            unit = self._small_font.render(spec.unit, True, DASHBOARD.muted_text_color)
            surface.blit(unit, (input_rect.right + 4, input_rect.top + 5))

    def _draw_scrollbar(self, surface: pygame.Surface, content: pygame.Rect) -> None:
        max_scroll = self._max_scroll_y()
        if max_scroll <= 0:
            return
        bar = pygame.Rect(content.right - 4, content.top, 3, content.height)
        pygame.draw.rect(surface, (42, 48, 58), bar, border_radius=2)
        thumb_h = max(20, int(content.height * content.height / max(1, self._content_height())))
        thumb_y = content.top + int((content.height - thumb_h) * self._scroll_y / max_scroll)
        pygame.draw.rect(surface, (116, 188, 255), pygame.Rect(bar.left, thumb_y, 3, thumb_h), border_radius=2)

    def _handle_key_down(self, event: pygame.event.Event) -> bool:
        assert self._active_input_key is not None
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._commit_active_input(call_callback=True)
            return True
        if event.key == pygame.K_ESCAPE:
            self._active_input_key = None
            self._input_text = ""
            return True
        if event.key == pygame.K_BACKSPACE:
            self._input_text = self._input_text[:-1]
            return True
        char = getattr(event, "unicode", "")
        if char and char in "0123456789.-+eE":
            self._input_text += char
            return True
        return True

    def _activate_input(self, key: str) -> None:
        if self._active_input_key is not None:
            self._commit_active_input(call_callback=True)
        spec = self._spec_for_key(key)
        if spec is None:
            return
        self._active_input_key = key
        self._input_text = self._format_value(self._values.get(key, spec.minimum), spec, include_unit=False)

    def _commit_active_input(self, call_callback: bool) -> None:
        key = self._active_input_key
        if key is None:
            return
        spec = self._spec_for_key(key)
        if spec is not None:
            try:
                value = float(self._input_text)
            except ValueError:
                value = self._values.get(key, spec.minimum)
                self.status_text = f"Invalid value for {spec.label}; kept previous value"
            self._values[key] = spec.clamp(value)
            self._mark_custom()
        self._active_input_key = None
        self._input_text = ""
        if call_callback:
            self._commit()

    def _set_value_from_x(self, key: str, x: int, commit: bool) -> None:
        spec = self._spec_for_key(key)
        track = self._track_rects.get(key)
        if spec is None or track is None:
            return
        ratio = (x - track.left) / max(1, track.width)
        self._values[key] = spec.clamp(spec.minimum + max(0.0, min(1.0, ratio)) * (spec.maximum - spec.minimum))
        self._mark_custom()
        if commit:
            self._commit()

    def _commit(self) -> None:
        if self._active_input_key is not None:
            self._commit_active_input(call_callback=False)
        self._dirty = False
        if self._on_commit is not None:
            self._on_commit(dict(self._values), self._active_preset)

    def _mark_custom(self) -> None:
        if self._active_preset != "Custom":
            self._active_preset = "Custom"
        self._dirty = True

    def _spec_for_key(self, key: str) -> Optional[ParameterSpec]:
        for spec in self._specs:
            if spec.key == key:
                return spec
        return None

    def _content_height(self) -> int:
        visible_specs = self._specs[: self._max_rows] if self._max_rows is not None else self._specs
        groups = []
        for spec in visible_specs:
            if spec.group and spec.group not in groups:
                groups.append(spec.group)
        return len(visible_specs) * 28 + len(groups) * 17

    def _max_scroll_y(self) -> int:
        return max(0, self._content_height() - max(1, self._last_content_rect.height))

    @staticmethod
    def _normalized(spec: ParameterSpec, value: float) -> float:
        if spec.maximum <= spec.minimum:
            return 0.0
        return max(0.0, min(1.0, (float(value) - spec.minimum) / (spec.maximum - spec.minimum)))

    @staticmethod
    def _format_value(value: float, spec: ParameterSpec, include_unit: bool = False) -> str:
        if not math.isfinite(float(value)):
            number = "0"
        elif spec.decimals <= 0:
            number = f"{float(value):.0f}"
        else:
            number = f"{float(value):.{spec.decimals}f}"
        return f"{number}{spec.unit if include_unit else ''}"

    @staticmethod
    def _fit_text(text: str, font: pygame.font.Font, max_width: int) -> str:
        if font.size(text)[0] <= max_width:
            return text
        ellipsis = "..."
        available = max(0, max_width - font.size(ellipsis)[0])
        fitted = ""
        for char in text:
            if font.size(fitted + char)[0] > available:
                break
            fitted += char
        return fitted.rstrip() + ellipsis

    @staticmethod
    def _clamp_int(value: int, low: int, high: int) -> int:
        return max(low, min(high, int(value)))

"""Pygame panels used while automated benchmark mode is active."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

import pygame

from config.settings import DASHBOARD


class TestProgressPanel:
    """Replacement for behavior tuning while benchmark settings are locked."""

    def __init__(self, rect: pygame.Rect) -> None:
        self._rect = rect.copy()
        self._font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size)
        self._bold_font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size, bold=True)
        self._small_font = pygame.font.SysFont("consolas", 11)

    def set_rect(self, rect: pygame.Rect) -> None:
        self._rect = rect.copy()

    def draw(self, surface: pygame.Surface, lines: Iterable[str]) -> None:
        pygame.draw.rect(surface, DASHBOARD.panel_inner_color, self._rect, border_radius=DASHBOARD.panel_radius_px)
        content = pygame.Rect(
            self._rect.left + DASHBOARD.panel_padding_px,
            self._rect.top + 29,
            self._rect.width - 2 * DASHBOARD.panel_padding_px,
            self._rect.height - 38,
        )
        y = content.top
        for index, line in enumerate(lines):
            if y + 15 > content.bottom:
                break
            font = self._bold_font if index == 0 or line.endswith(":") else self._font
            color = self._line_color(line)
            rendered = font.render(self._fit_text(line, font, content.width), True, color)
            surface.blit(rendered, (content.left, y))
            y += 17

    @staticmethod
    def _line_color(line: str) -> tuple[int, int, int]:
        lowered = line.lower()
        if "error" in lowered or "failed" in lowered or "aborted" in lowered:
            return DASHBOARD.warning_color
        if "test mode: on" in lowered or "running" in lowered or "completed" in lowered:
            return DASHBOARD.success_color
        if line.endswith(":"):
            return DASHBOARD.title_color
        return DASHBOARD.text_color

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


class LiveEvaluationPanel:
    """Replacement for the tab panel during automated benchmark mode."""

    def __init__(self, rect: pygame.Rect) -> None:
        self._rect = rect.copy()
        self._font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size)
        self._bold_font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size, bold=True)
        self._small_font = pygame.font.SysFont("consolas", 11)
        self._position_errors: deque[float] = deque(maxlen=120)
        self._raw_errors: deque[float] = deque(maxlen=120)
        self._speed_actual: deque[float] = deque(maxlen=120)
        self._speed_estimated: deque[float] = deque(maxlen=120)

    def set_rect(self, rect: pygame.Rect) -> None:
        self._rect = rect.copy()

    def update_histories(
        self,
        position_error_m: Optional[float],
        raw_error_m: Optional[float],
        actual_speed_mps: Optional[float],
        estimated_speed_mps: Optional[float],
    ) -> None:
        self._append_if_finite(self._position_errors, position_error_m)
        self._append_if_finite(self._raw_errors, raw_error_m)
        self._append_if_finite(self._speed_actual, actual_speed_mps)
        self._append_if_finite(self._speed_estimated, estimated_speed_mps)

    def draw(self, surface: pygame.Surface, lines: Iterable[str]) -> None:
        pygame.draw.rect(surface, DASHBOARD.panel_inner_color, self._rect, border_radius=DASHBOARD.panel_radius_px)
        content = pygame.Rect(
            self._rect.left + DASHBOARD.panel_padding_px,
            self._rect.top + 30,
            self._rect.width - 2 * DASHBOARD.panel_padding_px,
            self._rect.height - 40,
        )
        y = content.top

        graph_height = 44 if content.height >= 250 else 0
        graph_area_top = content.bottom - graph_height * 3 - 16 if graph_height else content.bottom
        text_bottom = max(y, graph_area_top - 8)
        for line in lines:
            if y + 15 > text_bottom:
                break
            color = self._line_color(line)
            rendered = self._font.render(self._fit_text(line, self._font, content.width), True, color)
            surface.blit(rendered, (content.left, y))
            y += 16

        if graph_height <= 0:
            return
        graph_width = content.width
        graph_rect = pygame.Rect(content.left, graph_area_top, graph_width, graph_height)
        self._draw_series(surface, graph_rect, "pos error", self._position_errors, DASHBOARD.success_color)
        graph_rect.y += graph_height + 6
        self._draw_dual_series(surface, graph_rect, "raw vs filtered", self._raw_errors, self._position_errors)
        graph_rect.y += graph_height + 6
        self._draw_dual_series(surface, graph_rect, "speed gt vs est", self._speed_actual, self._speed_estimated)

    @staticmethod
    def _append_if_finite(history: deque[float], value: Optional[float]) -> None:
        if isinstance(value, (int, float)) and value == value and value not in (float("inf"), float("-inf")):
            history.append(float(value))

    def _draw_series(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        values: deque[float],
        color: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surface, (14, 18, 24), rect, border_radius=4)
        pygame.draw.rect(surface, (54, 63, 78), rect, width=1, border_radius=4)
        surface.blit(self._small_font.render(label, True, DASHBOARD.muted_text_color), (rect.left + 7, rect.top + 4))
        if len(values) < 2:
            return
        self._draw_polyline(surface, rect.inflate(-12, -18).move(0, 7), tuple(values), color)

    def _draw_dual_series(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        first: deque[float],
        second: deque[float],
    ) -> None:
        pygame.draw.rect(surface, (14, 18, 24), rect, border_radius=4)
        pygame.draw.rect(surface, (54, 63, 78), rect, width=1, border_radius=4)
        surface.blit(self._small_font.render(label, True, DASHBOARD.muted_text_color), (rect.left + 7, rect.top + 4))
        plot = rect.inflate(-12, -18).move(0, 7)
        if len(first) >= 2:
            self._draw_polyline(surface, plot, tuple(first), (255, 145, 64))
        if len(second) >= 2:
            self._draw_polyline(surface, plot, tuple(second), (116, 188, 255))

    @staticmethod
    def _draw_polyline(
        surface: pygame.Surface,
        rect: pygame.Rect,
        values: tuple[float, ...],
        color: tuple[int, int, int],
    ) -> None:
        if len(values) < 2:
            return
        max_value = max(1.0e-6, max(abs(value) for value in values))
        points = []
        for index, value in enumerate(values):
            x = rect.left + int(index * rect.width / max(1, len(values) - 1))
            normalized = max(0.0, min(1.0, float(value) / max_value))
            y = rect.bottom - int(normalized * rect.height)
            points.append((x, y))
        pygame.draw.lines(surface, color, False, points, width=2)

    @staticmethod
    def _line_color(line: str) -> tuple[int, int, int]:
        lowered = line.lower()
        if "n/a" in lowered:
            return DASHBOARD.muted_text_color
        if "rmse" in lowered or "improvement" in lowered:
            return DASHBOARD.success_color
        if "max" in lowered or "error" in lowered:
            return DASHBOARD.warning_color
        return DASHBOARD.text_color

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

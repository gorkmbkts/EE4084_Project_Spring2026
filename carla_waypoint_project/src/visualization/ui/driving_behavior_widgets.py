"""Pygame widgets for autonomous driving tuning and diagnostics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Optional

import pygame

from config.settings import DASHBOARD
from src.control.driving_behavior import DrivingBehaviorConfig, SpeedPlan
from src.localization.state_estimator import EgoState


@dataclass(frozen=True)
class _SliderSpec:
    label: str
    attribute: str
    minimum: float
    maximum: float
    unit: str
    decimals: int = 1


class BehaviorTuningPanel:
    """Compact live sliders for the shared driving behavior config."""

    def __init__(self, rect: pygame.Rect, config: DrivingBehaviorConfig) -> None:
        self._rect = rect.copy()
        self._config = config
        self._font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size)
        self._value_font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size, bold=True)
        self._small_font = pygame.font.SysFont("consolas", 11)
        self._scroll_y = 0
        self._drag_attribute: Optional[str] = None
        self._track_rects: dict[str, pygame.Rect] = {}
        self._sliders = (
            _SliderSpec("Max speed", "max_speed_mps", 2.0, 14.0, "m/s", 1),
            _SliderSpec("Min curve speed", "min_curve_speed_mps", 0.8, 7.0, "m/s", 1),
            _SliderSpec("Max accel", "max_forward_accel_mps2", 0.3, 4.0, "m/s2", 1),
            _SliderSpec("Max braking", "max_braking_decel_mps2", 0.5, 7.0, "m/s2", 1),
            _SliderSpec("Max steer rate", "max_steer_rate_per_s", 0.4, 4.0, "/s", 1),
            _SliderSpec("Curve lookahead", "curve_lookahead_m", 8.0, 55.0, "m", 0),
            _SliderSpec("Curvature sens", "curvature_sensitivity", 0.2, 3.0, "x", 2),
            _SliderSpec("Safe cornering", "safe_cornering_factor", 0.6, 1.8, "x", 2),
            _SliderSpec("Speed aggress", "speed_change_aggressiveness", 0.2, 2.5, "x", 2),
            _SliderSpec("Throttle smooth", "throttle_smoothing", 0.0, 0.9, "", 2),
            _SliderSpec("Brake smooth", "brake_smoothing", 0.0, 0.9, "", 2),
            _SliderSpec("Steer smooth", "steering_smoothing", 0.0, 0.9, "", 2),
            _SliderSpec("Actuator delay", "actuator_delay_s", 0.0, 0.35, "s", 2),
            _SliderSpec("Imperfection", "actuator_noise", 0.0, 0.05, "", 3),
        )

    def set_rect(self, rect: pygame.Rect) -> None:
        self._rect = rect.copy()
        self._scroll_y = min(self._scroll_y, self._max_scroll_y())

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self._drag_attribute is not None:
            if event.type == pygame.MOUSEMOTION and hasattr(event, "pos"):
                self._set_value_from_position(self._drag_attribute, event.pos[0])
                return True
            if event.type == pygame.MOUSEBUTTONUP and getattr(event, "button", None) == 1:
                self._drag_attribute = None
                return True

        position = event.pos if hasattr(event, "pos") else pygame.mouse.get_pos()
        if not self._rect.collidepoint(position):
            return False

        if event.type == pygame.MOUSEWHEEL:
            self._scroll_y = self._clamp_int(
                self._scroll_y - event.y * 42,
                0,
                self._max_scroll_y(),
            )
            return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for attribute, track_rect in self._track_rects.items():
                hit_rect = track_rect.inflate(0, 12)
                if hit_rect.collidepoint(position):
                    self._drag_attribute = attribute
                    self._set_value_from_position(attribute, position[0])
                    return True
            return True

        return event.type in (pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION)

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(
            surface,
            DASHBOARD.panel_inner_color,
            self._rect,
            border_radius=DASHBOARD.panel_radius_px,
        )
        content = pygame.Rect(
            self._rect.left + DASHBOARD.panel_padding_px,
            self._rect.top + 27,
            self._rect.width - 2 * DASHBOARD.panel_padding_px,
            self._rect.height - 36,
        )

        old_clip = surface.get_clip()
        surface.set_clip(content)
        self._track_rects.clear()

        row_height = 20
        y = content.top - self._scroll_y
        for spec in self._sliders:
            row = pygame.Rect(content.left, y, content.width, row_height)
            if row.bottom >= content.top and row.top <= content.bottom:
                self._draw_slider(surface, row, spec)
            y += row_height

        surface.set_clip(old_clip)
        self._draw_scrollbar(surface, content)

    def _draw_slider(self, surface: pygame.Surface, row: pygame.Rect, spec: _SliderSpec) -> None:
        value = float(getattr(self._config, spec.attribute))
        normalized = self._normalized(spec, value)
        label_width = min(124, max(84, row.width // 3))
        value_width = 66
        track_left = row.left + label_width + 8
        track_right = row.right - value_width - 8
        if track_right <= track_left + 24:
            track_left = row.left
            track_right = row.right - value_width - 8
            label_y = row.top - 1
        else:
            label_y = row.top + 1

        label = self._fit_text(spec.label, self._font, max(40, label_width))
        label_surface = self._font.render(label, True, DASHBOARD.text_color)
        surface.blit(label_surface, (row.left, label_y))

        track_rect = pygame.Rect(track_left, row.centery - 2, max(24, track_right - track_left), 4)
        self._track_rects[spec.attribute] = track_rect
        pygame.draw.rect(surface, (50, 57, 69), track_rect, border_radius=2)
        fill_rect = track_rect.copy()
        fill_rect.width = max(2, int(track_rect.width * normalized))
        pygame.draw.rect(surface, (88, 177, 255), fill_rect, border_radius=2)
        knob_x = track_rect.left + int(track_rect.width * normalized)
        knob_rect = pygame.Rect(0, 0, 8, 12)
        knob_rect.center = (knob_x, track_rect.centery)
        knob_color = (222, 235, 250) if self._drag_attribute == spec.attribute else (155, 205, 255)
        pygame.draw.rect(surface, knob_color, knob_rect, border_radius=3)

        value_text = self._format_value(value, spec)
        value_surface = self._value_font.render(value_text, True, DASHBOARD.title_color)
        surface.blit(value_surface, (row.right - value_surface.get_width(), row.top + 1))

    def _draw_scrollbar(self, surface: pygame.Surface, content: pygame.Rect) -> None:
        max_scroll = self._max_scroll_y()
        if max_scroll <= 0:
            return
        bar_rect = pygame.Rect(content.right - 4, content.top, 3, content.height)
        pygame.draw.rect(surface, (42, 48, 58), bar_rect, border_radius=2)
        thumb_h = max(20, int(content.height * content.height / self._content_height()))
        thumb_y = content.top + int((content.height - thumb_h) * (self._scroll_y / max_scroll))
        pygame.draw.rect(surface, (116, 188, 255), pygame.Rect(bar_rect.left, thumb_y, 3, thumb_h), border_radius=2)

    def _set_value_from_position(self, attribute: str, x: int) -> None:
        spec = next((item for item in self._sliders if item.attribute == attribute), None)
        track = self._track_rects.get(attribute)
        if spec is None or track is None:
            return
        ratio = (x - track.left) / max(1, track.width)
        value = spec.minimum + self._clamp(ratio, 0.0, 1.0) * (spec.maximum - spec.minimum)
        setattr(self._config, spec.attribute, float(value))

        if self._config.min_curve_speed_mps > self._config.max_speed_mps:
            self._config.min_curve_speed_mps = self._config.max_speed_mps

    def _normalized(self, spec: _SliderSpec, value: float) -> float:
        if spec.maximum <= spec.minimum:
            return 0.0
        return self._clamp((value - spec.minimum) / (spec.maximum - spec.minimum), 0.0, 1.0)

    def _content_height(self) -> int:
        return len(self._sliders) * 20

    def _max_scroll_y(self) -> int:
        viewport = max(1, self._rect.height - 36)
        return max(0, self._content_height() - viewport)

    @staticmethod
    def _format_value(value: float, spec: _SliderSpec) -> str:
        if spec.decimals == 0:
            number = f"{value:.0f}"
        else:
            number = f"{value:.{spec.decimals}f}"
        return f"{number}{spec.unit}"

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
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def _clamp_int(value: int, low: int, high: int) -> int:
        return max(low, min(high, int(value)))


class ControlVisualizationWidget:
    """Vector-style applied steering, throttle, and brake visualization."""

    def __init__(self, rect: pygame.Rect) -> None:
        self._rect = rect.copy()
        self._font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size, bold=True)
        self._small_font = pygame.font.SysFont("consolas", 11)

    def set_rect(self, rect: pygame.Rect) -> None:
        self._rect = rect.copy()

    def draw(
        self,
        surface: pygame.Surface,
        applied_control: Optional[object],
        requested_control: Optional[object] = None,
    ) -> None:
        pygame.draw.rect(
            surface,
            DASHBOARD.panel_inner_color,
            self._rect,
            border_radius=DASHBOARD.panel_radius_px,
        )
        content = pygame.Rect(
            self._rect.left + DASHBOARD.panel_padding_px,
            self._rect.top + 30,
            self._rect.width - 2 * DASHBOARD.panel_padding_px,
            self._rect.height - 42,
        )
        if applied_control is None:
            self._draw_waiting(surface, content)
            return

        steer = self._control_value(applied_control, "steer", 0.0)
        throttle = self._control_value(applied_control, "throttle", 0.0)
        brake = self._control_value(applied_control, "brake", 0.0)
        requested_throttle = self._control_value(requested_control, "throttle", throttle)
        requested_brake = self._control_value(requested_control, "brake", brake)

        wheel_center = (
            content.left + max(58, int(content.width * 0.31)),
            content.top + int(content.height * 0.46),
        )
        wheel_radius = max(32, min(58, content.height // 3, content.width // 5))
        self._draw_steering_wheel(surface, wheel_center, wheel_radius, steer)
        self._draw_centered_text(
            surface,
            f"STEER {steer:+.2f}",
            (wheel_center[0], min(content.bottom - 16, wheel_center[1] + wheel_radius + 18)),
            DASHBOARD.text_color,
        )

        bars_left = content.left + int(content.width * 0.58)
        bar_width = max(20, min(34, (content.right - bars_left - 28) // 2))
        bar_height = max(72, min(content.height - 28, 132))
        bar_top = content.top + max(4, (content.height - bar_height) // 2 - 2)
        throttle_rect = pygame.Rect(bars_left, bar_top, bar_width, bar_height)
        brake_rect = pygame.Rect(bars_left + bar_width + 28, bar_top, bar_width, bar_height)
        self._draw_vertical_bar(
            surface,
            throttle_rect,
            throttle,
            requested_throttle,
            "THR",
            (84, 222, 132),
        )
        self._draw_vertical_bar(
            surface,
            brake_rect,
            brake,
            requested_brake,
            "BRK",
            (255, 196, 87),
        )

    def _draw_steering_wheel(
        self,
        surface: pygame.Surface,
        center: tuple[int, int],
        radius: int,
        steer: float,
    ) -> None:
        # CARLA steer value is usually in [-1.0, +1.0].
        # Clamp it so the wheel drawing never goes wild if an unexpected value arrives.
        steer = self._clamp(float(steer), -1.0, 1.0)

        # Positive steer should visually rotate the wheel to the right.
        # In pygame, +y goes downward, so the sign is inverted for natural rotation.
        angle = -steer * math.radians(90.0)

        rim_dark = (52, 61, 75)
        rim_light = (122, 139, 164)
        hub_dark = (18, 22, 28)
        hub_blue = (116, 188, 255)
        spoke_color = (158, 174, 198)
        marker_color = (84, 222, 132)

        # Outer steering wheel rim
        pygame.draw.circle(surface, rim_dark, center, radius, width=7)
        pygame.draw.circle(surface, rim_light, center, radius, width=2)

        # Hub
        hub_radius = max(8, radius // 5)
        pygame.draw.circle(surface, hub_dark, center, hub_radius)
        pygame.draw.circle(surface, hub_blue, center, max(5, radius // 8))

        # More natural 3-spoke layout:
        # 12 o'clock, 5 o'clock, 7 o'clock positions, all rotated by steer angle.
        spoke_angles = (
            angle - math.pi / 2.0,        # top spoke
            angle + math.radians(30.0),   # lower-right spoke
            angle + math.radians(150.0),  # lower-left spoke
        )

        spoke_start_radius = hub_radius + 2
        spoke_end_radius = radius * 0.72

        for spoke_angle in spoke_angles:
            start = (
                int(center[0] + math.cos(spoke_angle) * spoke_start_radius),
                int(center[1] + math.sin(spoke_angle) * spoke_start_radius),
            )
            end = (
                int(center[0] + math.cos(spoke_angle) * spoke_end_radius),
                int(center[1] + math.sin(spoke_angle) * spoke_end_radius),
            )
            pygame.draw.line(surface, spoke_color, start, end, width=3)

        # Top marker, slightly inside the outer rim so it does not look detached.
        marker_radius = radius - 2
        marker_angle = angle - math.pi / 2.0
        marker = (
            int(center[0] + math.cos(marker_angle) * marker_radius),
            int(center[1] + math.sin(marker_angle) * marker_radius),
        )
        pygame.draw.circle(surface, marker_color, marker, 4)

    def _draw_vertical_bar(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        value: float,
        requested_value: float,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        value = self._clamp(value, 0.0, 1.0)
        requested_value = self._clamp(requested_value, 0.0, 1.0)
        pygame.draw.rect(surface, (28, 34, 43), rect, border_radius=5)
        pygame.draw.rect(surface, (74, 84, 101), rect, width=1, border_radius=5)
        fill_height = int(rect.height * value)
        if fill_height > 0:
            fill_rect = pygame.Rect(
                rect.left + 3,
                rect.bottom - fill_height + 1,
                rect.width - 6,
                max(1, fill_height - 2),
            )
            pygame.draw.rect(surface, color, fill_rect, border_radius=4)
        marker_y = rect.bottom - int(rect.height * requested_value)
        pygame.draw.line(surface, (224, 229, 237), (rect.left - 3, marker_y), (rect.right + 3, marker_y), width=1)

        label_surface = self._small_font.render(label, True, DASHBOARD.muted_text_color)
        surface.blit(label_surface, (rect.centerx - label_surface.get_width() // 2, rect.bottom + 4))
        value_surface = self._font.render(f"{value:.2f}", True, DASHBOARD.title_color)
        surface.blit(value_surface, (rect.centerx - value_surface.get_width() // 2, rect.top - 20))

    def _draw_waiting(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        text = self._font.render("Waiting for control data", True, DASHBOARD.muted_text_color)
        surface.blit(text, text.get_rect(center=rect.center))

    def _draw_centered_text(
        self,
        surface: pygame.Surface,
        text: str,
        center: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        rendered = self._font.render(text, True, color)
        surface.blit(rendered, rendered.get_rect(center=center))

    @staticmethod
    def _control_value(control: Optional[object], attribute: str, fallback: float) -> float:
        if control is None:
            return fallback
        return float(getattr(control, attribute, fallback))

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))


class DrivingDiagnosticsWidget:
    """Render speed-planner state and a small actual-vs-target history."""

    def __init__(self, rect: pygame.Rect, config: DrivingBehaviorConfig) -> None:
        self._rect = rect.copy()
        self._config = config
        self._font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size)
        self._title_font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size, bold=True)
        self._small_font = pygame.font.SysFont("consolas", 11)
        self._actual_speed_history: Deque[float] = deque(maxlen=100)
        self._target_speed_history: Deque[float] = deque(maxlen=100)
        self._last_state_timestamp: Optional[float] = None

    def set_rect(self, rect: pygame.Rect) -> None:
        self._rect = rect.copy()

    def draw(
        self,
        surface: pygame.Surface,
        state: Optional[EgoState],
        speed_plan: Optional[SpeedPlan],
        applied_control: Optional[object],
    ) -> None:
        pygame.draw.rect(
            surface,
            DASHBOARD.panel_inner_color,
            self._rect,
            border_radius=DASHBOARD.panel_radius_px,
        )
        content = pygame.Rect(
            self._rect.left + DASHBOARD.panel_padding_px,
            self._rect.top + 29,
            self._rect.width - 2 * DASHBOARD.panel_padding_px,
            self._rect.height - 40,
        )
        self._update_history(state, speed_plan)

        actual_speed = state.speed if state is not None else 0.0
        target_speed = speed_plan.target_speed_mps if speed_plan is not None else 0.0
        mode = speed_plan.mode if speed_plan is not None else "WAITING"
        curvature = speed_plan.curvature_score if speed_plan is not None else 0.0
        lookahead = speed_plan.lookahead_distance_m if speed_plan is not None else 0.0

        rows = [
            ("Actual speed", f"{actual_speed:4.1f} m/s", DASHBOARD.text_color),
            ("Target speed", f"{target_speed:4.1f} m/s", DASHBOARD.success_color),
            ("Mode", mode[:18], DASHBOARD.warning_color if "CURVE" in mode else DASHBOARD.success_color),
            ("Curve score", f"{curvature:4.2f}", DASHBOARD.text_color),
            ("Lookahead", f"{lookahead:4.0f} m", DASHBOARD.text_color),
        ]
        if applied_control is not None:
            rows.extend(
                [
                    ("Throttle", f"{float(getattr(applied_control, 'throttle', 0.0)):4.2f}", DASHBOARD.text_color),
                    ("Brake", f"{float(getattr(applied_control, 'brake', 0.0)):4.2f}", DASHBOARD.text_color),
                    ("Steer", f"{float(getattr(applied_control, 'steer', 0.0)):+4.2f}", DASHBOARD.text_color),
                ]
            )

        left_width = max(130, int(content.width * 0.46))
        y = content.top
        for label, value, color in rows:
            if y + 15 > content.bottom:
                break
            label_surface = self._font.render(label, True, DASHBOARD.muted_text_color)
            value_surface = self._title_font.render(value, True, color)
            surface.blit(label_surface, (content.left, y))
            surface.blit(value_surface, (content.left + left_width, y))
            y += 17

        graph_top = min(content.bottom - 78, y + 8)
        graph_rect = pygame.Rect(content.left, graph_top, content.width, max(58, content.bottom - graph_top))
        self._draw_speed_graph(surface, graph_rect)

    def _update_history(self, state: Optional[EgoState], speed_plan: Optional[SpeedPlan]) -> None:
        if state is None or speed_plan is None:
            return
        timestamp = float(state.timestamp)
        if self._last_state_timestamp == timestamp:
            return
        self._last_state_timestamp = timestamp
        self._actual_speed_history.append(max(0.0, float(state.speed)))
        self._target_speed_history.append(max(0.0, float(speed_plan.target_speed_mps)))

    def _draw_speed_graph(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        if rect.width < 40 or rect.height < 36:
            return
        pygame.draw.rect(surface, (14, 18, 24), rect, border_radius=4)
        pygame.draw.rect(surface, (54, 63, 78), rect, width=1, border_radius=4)
        label = self._small_font.render("speed history", True, DASHBOARD.muted_text_color)
        surface.blit(label, (rect.left + 8, rect.top + 5))

        plot = pygame.Rect(rect.left + 8, rect.top + 24, rect.width - 16, rect.height - 32)
        if len(self._actual_speed_history) < 2:
            return
        max_speed = max(1.0, self._config.max_speed_mps, max(self._target_speed_history, default=0.0))
        self._draw_polyline(surface, plot, tuple(self._target_speed_history), max_speed, (84, 222, 132))
        self._draw_polyline(surface, plot, tuple(self._actual_speed_history), max_speed, (116, 188, 255))

    @staticmethod
    def _draw_polyline(
        surface: pygame.Surface,
        rect: pygame.Rect,
        values: tuple[float, ...],
        max_value: float,
        color: tuple[int, int, int],
    ) -> None:
        if len(values) < 2:
            return
        count = len(values)
        points: list[tuple[int, int]] = []
        for index, value in enumerate(values):
            x = rect.left + int(index * rect.width / max(1, count - 1))
            normalized = max(0.0, min(1.0, value / max_value))
            y = rect.bottom - int(normalized * rect.height)
            points.append((x, y))
        pygame.draw.lines(surface, color, False, points, width=2)

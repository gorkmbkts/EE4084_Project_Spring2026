"""Text diagnostics panel for live simulation and sensor data."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import pygame

from config.settings import DASHBOARD
from src.control.waypoint_tracker import TrackingStatus
from src.localization.state_estimator import EgoState
from src.sensors.gnss_sensor import GnssMeasurement
from src.sensors.imu_sensor import ImuMeasurement
from src.sensors.lidar_sensor import LidarMeasurement


@dataclass(frozen=True)
class SensorPanelData:
    """All dashboard values rendered by the bottom diagnostics panel."""

    drive_mode: str
    map_selection_active: bool
    sync_status: str
    fixed_delta_seconds: float
    pygame_frame_dt_seconds: float
    planner_status: str
    route_size: int
    ego_state: Optional[EgoState]
    tracking: TrackingStatus
    gnss: Optional[GnssMeasurement]
    imu: Optional[ImuMeasurement]
    lidar: Optional[LidarMeasurement]


class SensorPanelRenderer:
    """Render structured text rows for state, sensors, and tracking."""

    def __init__(self) -> None:
        self._section_font = pygame.font.SysFont("consolas", DASHBOARD.text_font_size, bold=True)
        self._row_font = pygame.font.SysFont("consolas", DASHBOARD.small_font_size)

    def draw(self, surface: pygame.Surface, rect: pygame.Rect, data: SensorPanelData) -> None:
        pygame.draw.rect(
            surface,
            DASHBOARD.panel_inner_color,
            rect,
            border_radius=DASHBOARD.panel_radius_px,
        )

        content = pygame.Rect(
            rect.left + DASHBOARD.panel_padding_px,
            rect.top + 28,
            rect.width - 2 * DASHBOARD.panel_padding_px,
            rect.height - 36,
        )
        column_gap = 18
        column_width = (content.width - 3 * column_gap) // 4
        columns = [
            pygame.Rect(content.left + index * (column_width + column_gap), content.top, column_width, content.height)
            for index in range(4)
        ]

        self._draw_section(surface, columns[0], "Simulation / State", self._simulation_rows(data))
        self._draw_section(surface, columns[1], "GNSS", self._gnss_rows(data.gnss))
        self._draw_section(surface, columns[2], "IMU / LiDAR", self._imu_lidar_rows(data.imu, data.lidar))
        self._draw_section(surface, columns[3], "Route Tracking", self._route_rows(data))

    def _draw_section(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        title: str,
        rows: list[tuple[str, tuple[int, int, int]]],
    ) -> None:
        title_surface = self._section_font.render(title, True, DASHBOARD.title_color)
        surface.blit(title_surface, rect.topleft)
        y = rect.top + 21
        for text, color in rows:
            row_surface = self._row_font.render(text, True, color)
            surface.blit(row_surface, (rect.left, y))
            y += 17
            if y > rect.bottom - 14:
                break

    def _simulation_rows(self, data: SensorPanelData) -> list[tuple[str, tuple[int, int, int]]]:
        state = data.ego_state
        rows = [
            (f"Mode: {data.drive_mode}", DASHBOARD.text_color),
            (f"Map select: {'ON' if data.map_selection_active else 'OFF'}", DASHBOARD.text_color),
            (data.sync_status[:46], DASHBOARD.success_color if "ON" in data.sync_status else DASHBOARD.warning_color),
            (f"Sim dt: {data.fixed_delta_seconds:.3f} s", DASHBOARD.text_color),
            (f"Pygame dt: {data.pygame_frame_dt_seconds:.3f} s", DASHBOARD.muted_text_color),
        ]
        if state is None:
            rows.append(("GT pose: waiting", DASHBOARD.warning_color))
            return rows

        rows.extend(
            [
                (f"GT x/y/z: {state.x:7.2f} {state.y:7.2f} {state.z:5.2f}", DASHBOARD.text_color),
                (f"GT yaw: {state.yaw:7.2f} deg", DASHBOARD.text_color),
                (f"Speed: {state.speed:5.2f} m/s ({state.speed * 3.6:5.1f} km/h)", DASHBOARD.text_color),
            ]
        )
        return rows

    def _gnss_rows(self, gnss: Optional[GnssMeasurement]) -> list[tuple[str, tuple[int, int, int]]]:
        if gnss is None:
            return [("Waiting for GNSS frame", DASHBOARD.warning_color)]
        return [
            (f"Lat: {gnss.latitude:.8f}", DASHBOARD.text_color),
            (f"Lon: {gnss.longitude:.8f}", DASHBOARD.text_color),
            (f"Alt: {gnss.altitude:.3f} m", DASHBOARD.text_color),
            (f"Frame: {gnss.frame}", DASHBOARD.muted_text_color),
            (f"Stamp: {gnss.timestamp:.3f} s", DASHBOARD.muted_text_color),
        ]

    def _imu_lidar_rows(
        self,
        imu: Optional[ImuMeasurement],
        lidar: Optional[LidarMeasurement],
    ) -> list[tuple[str, tuple[int, int, int]]]:
        rows: list[tuple[str, tuple[int, int, int]]] = []
        if imu is None:
            rows.append(("IMU: waiting", DASHBOARD.warning_color))
        else:
            ax, ay, az = imu.accelerometer
            gx, gy, gz = imu.gyroscope
            rows.extend(
                [
                    (f"Acc:  {ax:6.2f} {ay:6.2f} {az:6.2f}", DASHBOARD.text_color),
                    (f"Gyro: {gx:6.2f} {gy:6.2f} {gz:6.2f}", DASHBOARD.text_color),
                    (f"Compass: {imu.compass:7.3f} rad", DASHBOARD.text_color),
                    (f"IMU frame: {imu.frame}", DASHBOARD.muted_text_color),
                ]
            )

        if lidar is None:
            rows.append(("LiDAR: waiting", DASHBOARD.warning_color))
        else:
            rows.extend(
                [
                    (f"LiDAR pts: {lidar.point_count}", DASHBOARD.text_color),
                    (f"LiDAR frame: {lidar.frame}", DASHBOARD.muted_text_color),
                    (f"LiDAR stamp: {lidar.timestamp:.3f} s", DASHBOARD.muted_text_color),
                ]
            )
        return rows

    def _route_rows(self, data: SensorPanelData) -> list[tuple[str, tuple[int, int, int]]]:
        tracking = data.tracking
        heading = self._format_optional_float(tracking.heading_error_deg, "deg")
        return [
            (f"Route: {data.route_size} wp", DASHBOARD.text_color),
            (f"Done: {'YES' if tracking.completed else 'NO'}", DASHBOARD.success_color if tracking.completed else DASHBOARD.text_color),
            (f"Closest: {tracking.closest_index}", DASHBOARD.text_color),
            (f"Target:  {tracking.target_index}", DASHBOARD.text_color),
            (f"CTE: {self._format_float(tracking.cross_track_error_m, 'm')}", DASHBOARD.text_color),
            (f"Goal dist: {self._format_float(tracking.distance_to_goal_m, 'm')}", DASHBOARD.text_color),
            (f"Heading err: {heading}", DASHBOARD.text_color),
            (data.planner_status[:44], DASHBOARD.muted_text_color),
        ]

    @staticmethod
    def _format_float(value: float, suffix: str) -> str:
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.2f} {suffix}"

    @staticmethod
    def _format_optional_float(value: Optional[float], suffix: str) -> str:
        if value is None or not math.isfinite(value):
            return "n/a"
        return f"{value:.2f} {suffix}"

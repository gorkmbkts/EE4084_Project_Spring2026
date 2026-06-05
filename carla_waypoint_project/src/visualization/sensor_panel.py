"""Text diagnostics panel for live simulation, localization, and sensors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import pygame

from config.settings import DASHBOARD, GNSS, IMU
from src.control.waypoint_tracker import TrackingStatus
from src.core.localization_status import LocalizationStatus
from src.core.vehicle_state import VehicleState
from src.localization.gnss_projection import GnssDiagnostics
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
    ego_state: Optional[VehicleState]
    ground_truth_state: Optional[VehicleState]
    estimated_state: Optional[VehicleState]
    localization_status: Optional[LocalizationStatus]
    route_activation_state: str
    stabilization_active: bool
    stabilization_error_m: Optional[float]
    stabilization_stable_ticks: int
    stabilization_required_ticks: int
    stabilization_elapsed_seconds: float
    stabilization_timeout_seconds: float
    route_generation_blocked: bool
    tracking: TrackingStatus
    gnss: Optional[GnssMeasurement]
    gnss_diagnostics: Optional[GnssDiagnostics]
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

        self._draw_section(surface, columns[0], "Simulation / GT", self._simulation_rows(data))
        self._draw_section(surface, columns[1], "Filter Estimate", self._estimate_rows(data))
        self._draw_section(surface, columns[2], "GNSS / IMU", self._gnss_imu_rows(data))
        self._draw_section(surface, columns[3], "LiDAR / Route", self._lidar_route_rows(data))

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
        gt = data.ground_truth_state
        control_source = "Estimate" if data.drive_mode == "AUTO" else "Ground truth"
        rows = [
            (f"Mode: {data.drive_mode}", DASHBOARD.text_color),
            (f"Control state: {control_source}", DASHBOARD.text_color),
            (f"Map select: {'ON' if data.map_selection_active else 'OFF'}", DASHBOARD.text_color),
            (data.sync_status[:44], DASHBOARD.success_color if "ON" in data.sync_status else DASHBOARD.warning_color),
            (f"Sim/Pygame dt: {data.fixed_delta_seconds:.3f}/{data.pygame_frame_dt_seconds:.3f}", DASHBOARD.text_color),
        ]
        if gt is None:
            rows.append(("GT pose: waiting", DASHBOARD.warning_color))
            return rows

        rows.extend(
            [
                (f"GT x/y/z: {gt.x:7.2f} {gt.y:7.2f} {gt.z:5.2f}", DASHBOARD.text_color),
                (f"GT yaw: {gt.yaw:7.2f} deg", DASHBOARD.text_color),
                (f"GT speed: {gt.speed:5.2f} m/s", DASHBOARD.text_color),
            ]
        )
        if data.ego_state is None:
            rows.append(("Route state: waiting", DASHBOARD.warning_color))
        else:
            rows.append((f"Route state t: {data.ego_state.timestamp:7.2f}", DASHBOARD.muted_text_color))
        return rows

    def _estimate_rows(self, data: SensorPanelData) -> list[tuple[str, tuple[int, int, int]]]:
        status = data.localization_status
        estimate = data.estimated_state
        if status is None:
            return [("Estimator: waiting", DASHBOARD.warning_color)]

        rows = [
            (f"Filter: {status.filter_name}", DASHBOARD.text_color),
            (
                f"Initialized: {'YES' if status.initialized else 'NO'}",
                DASHBOARD.success_color if status.initialized else DASHBOARD.warning_color,
            ),
        ]

        if estimate is None:
            rows.append(("Est pose: waiting for GNSS", DASHBOARD.warning_color))
        else:
            rows.extend(
                [
                    (f"Est x/y: {estimate.x:7.2f} {estimate.y:7.2f}", DASHBOARD.text_color),
                    (f"Est yaw: {estimate.yaw:7.2f} deg", DASHBOARD.text_color),
                    (f"Est speed: {estimate.speed:5.2f} m/s", DASHBOARD.text_color),
                    (f"Model: {(estimate.model_type or 'n/a')[:18]}", DASHBOARD.muted_text_color),
                    (f"Source: {(estimate.source_filter_id or 'n/a')[:17]}", DASHBOARD.muted_text_color),
                    (f"Caps: {(','.join(estimate.capabilities()) or 'basic')[:20]}", DASHBOARD.muted_text_color),
                ]
            )

        if status.gnss_local is None:
            rows.append(("GNSS local: waiting", DASHBOARD.warning_color))
        else:
            rows.append((f"GNSS x/y: {status.gnss_local.x:7.2f} {status.gnss_local.y:7.2f}", DASHBOARD.text_color))

        rows.extend(
            [
                (f"Pos err GT: {self._format_optional_float(status.position_error_m, 'm')}", DASHBOARD.warning_color),
                (f"GNSS frame: {self._format_optional_int(status.last_gnss_frame)}", DASHBOARD.muted_text_color),
                (f"IMU frame:  {self._format_optional_int(status.last_imu_frame)}", DASHBOARD.muted_text_color),
            ]
        )
        return rows

    def _gnss_imu_rows(self, data: SensorPanelData) -> list[tuple[str, tuple[int, int, int]]]:
        rows: list[tuple[str, tuple[int, int, int]]] = []
        gnss = data.gnss
        diagnostics = data.gnss_diagnostics
        if gnss is None:
            rows.append(("GNSS: waiting", DASHBOARD.warning_color))
        else:
            rows.extend(
                [
                    (f"Lat: {gnss.latitude:.8f}", DASHBOARD.text_color),
                    (f"Lon: {gnss.longitude:.8f}", DASHBOARD.text_color),
                    (f"Alt: {gnss.altitude:.2f} m", DASHBOARD.text_color),
                ]
            )
            if diagnostics is not None:
                rows.extend(
                    [
                        (f"GNSS dx/dy: {diagnostics.dx_m:+5.2f} {diagnostics.dy_m:+5.2f}", DASHBOARD.warning_color),
                        (f"GNSS horiz: {diagnostics.horizontal_error_m:5.2f} m", DASHBOARD.warning_color),
                    ]
                )
            rows.append((f"Noise ll: {GNSS.noise_lat_stddev_deg:.1e}/{GNSS.noise_lon_stddev_deg:.1e}", DASHBOARD.muted_text_color))

        imu = data.imu
        if imu is None:
            rows.append(("IMU: waiting", DASHBOARD.warning_color))
            return rows

        ax, ay, az = imu.accelerometer
        gx, gy, gz = imu.gyroscope
        gt_compass = self._gt_compass_deg(data.ground_truth_state)
        compass_deg = math.degrees(imu.compass)
        compass_error = self._angle_error_deg(compass_deg, gt_compass) if gt_compass is not None else None
        rows.extend(
            [
                (f"Acc:  {ax:+6.2f} {ay:+6.2f} {az:+6.2f}", DASHBOARD.text_color),
                (f"Gyro: {gx:+6.3f} {gy:+6.3f} {gz:+6.3f}", DASHBOARD.text_color),
                (f"Compass: {compass_deg:7.2f} deg", DASHBOARD.text_color),
                (f"Comp err: {self._format_optional_float(compass_error, 'deg')}", DASHBOARD.warning_color),
                (f"Acc noise: {IMU.noise_accel_stddev_x:.2f}/{IMU.noise_accel_stddev_y:.2f}", DASHBOARD.muted_text_color),
            ]
        )
        return rows

    def _lidar_route_rows(self, data: SensorPanelData) -> list[tuple[str, tuple[int, int, int]]]:
        lidar = data.lidar
        rows: list[tuple[str, tuple[int, int, int]]] = []
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

        tracking = data.tracking
        heading = self._format_optional_float(tracking.heading_error_deg, "deg")
        rows.extend(
            [
                (f"Route: {data.route_size} wp", DASHBOARD.text_color),
                (f"Init: {data.route_activation_state[:22]}", DASHBOARD.text_color),
                (
                    f"Route blocked: {'YES' if data.route_generation_blocked else 'NO'}",
                    DASHBOARD.warning_color if data.route_generation_blocked else DASHBOARD.success_color,
                ),
                (
                    f"Stable ticks: {data.stabilization_stable_ticks}/{data.stabilization_required_ticks}",
                    DASHBOARD.warning_color if data.stabilization_active else DASHBOARD.muted_text_color,
                ),
                (
                    f"Init err: {self._format_optional_float(data.stabilization_error_m, 'm')}",
                    DASHBOARD.warning_color if data.stabilization_active else DASHBOARD.muted_text_color,
                ),
                (
                    f"Init t: {data.stabilization_elapsed_seconds:.1f}/{data.stabilization_timeout_seconds:.1f}s",
                    DASHBOARD.warning_color if data.stabilization_active else DASHBOARD.muted_text_color,
                ),
                (f"Done: {'YES' if tracking.completed else 'NO'}", DASHBOARD.success_color if tracking.completed else DASHBOARD.text_color),
                (f"Closest: {tracking.closest_index}", DASHBOARD.text_color),
                (f"Target:  {tracking.target_index}", DASHBOARD.text_color),
                (f"Search:  {tracking.search_start_index}-{tracking.search_end_index}", DASHBOARD.muted_text_color),
                (f"CTE: {self._format_float(tracking.cross_track_error_m, 'm')}", DASHBOARD.text_color),
                (f"Goal dist: {self._format_float(tracking.distance_to_goal_m, 'm')}", DASHBOARD.text_color),
                (f"Heading err: {heading}", DASHBOARD.text_color),
                (data.planner_status[:44], DASHBOARD.muted_text_color),
            ]
        )
        return rows

    @staticmethod
    def _gt_compass_deg(state: Optional[VehicleState]) -> Optional[float]:
        if state is None:
            return None
        return (state.yaw + 90.0) % 360.0

    @staticmethod
    def _angle_error_deg(measured_deg: float, reference_deg: Optional[float]) -> Optional[float]:
        if reference_deg is None:
            return None
        error = measured_deg - reference_deg
        while error > 180.0:
            error -= 360.0
        while error < -180.0:
            error += 360.0
        return error

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

    @staticmethod
    def _format_optional_int(value: Optional[int]) -> str:
        if value is None:
            return "n/a"
        return str(value)

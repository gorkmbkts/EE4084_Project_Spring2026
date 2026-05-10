"""Centralized project settings and constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

ColorRGB = Tuple[int, int, int]


@dataclass(frozen=True)
class CarlaSettings:
    host: str = "localhost"
    port: int = 2000
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class DisplaySettings:
    width: int = 1280
    height: int = 720
    fps: int = 30
    title: str = "CARLA Route Selection + Ground Truth Following"
    clear_color: ColorRGB = (15, 15, 15)


@dataclass(frozen=True)
class VehicleSettings:
    primary_blueprint_filter: str = "cybertruck*"
    fallback_blueprint_filter: str = "vehicle.*"
    spawn_point_index: int | None = 0
    teleport_z_offset_m: float = 0.7


@dataclass(frozen=True)
class CameraSettings:
    blueprint_id: str = "sensor.camera.rgb"
    image_width: int = 1280
    image_height: int = 720
    fov_deg: float = 90.0
    relative_x: float = -8.0
    relative_y: float = 0.0
    relative_z: float = 3.0
    relative_pitch: float = -15.0
    relative_yaw: float = 0.0
    relative_roll: float = 0.0


@dataclass(frozen=True)
class WaypointOverlaySettings:
    count: int = 30
    step_distance_m: float = 2.0
    target_index: int = 4
    height_offset_m: float = 0.5
    full_path_color: ColorRGB = (255, 0, 0)
    target_color: ColorRGB = (0, 255, 0)
    full_path_radius_px: int = 4
    target_radius_px: int = 6


@dataclass(frozen=True)
class ManualControlSettings:
    throttle: float = 0.7
    brake: float = 1.0
    steer: float = 0.6


@dataclass(frozen=True)
class RoutePlannerSettings:
    sampling_resolution_m: float = 2.0
    snap_search_radius_m: float = 3.0


@dataclass(frozen=True)
class WaypointTrackerSettings:
    lookahead_base_m: float = 3.0
    lookahead_gain_s: float = 0.20
    search_backtrack_count: int = 6
    search_forward_count: int = 90
    completion_distance_m: float = 4.0


@dataclass(frozen=True)
class AutonomousControlSettings:
    target_speed_mps: float = 5.0
    turn_speed_mps: float = 2.8
    wheel_base_m: float = 2.8
    max_steer: float = 1.0
    max_steer_angle_deg: float = 55.0
    steering_gain: float = 1.20
    turn_slowdown_steer_threshold: float = 0.25
    sharp_turn_steer_threshold: float = 0.70
    max_throttle: float = 0.45
    max_brake: float = 0.8
    speed_kp: float = 0.25
    brake_kp: float = 0.35
    stop_distance_m: float = 3.0


@dataclass(frozen=True)
class TopDownMapSettings:
    panel_width: int = 430
    panel_height: int = 430
    margin_px: int = 14
    world_margin_m: float = 20.0
    min_zoom: float = 0.35
    max_zoom: float = 6.0
    zoom_step: float = 1.15
    background_color: ColorRGB = (18, 20, 24)
    border_color: ColorRGB = (210, 210, 210)
    road_color: ColorRGB = (94, 98, 104)
    route_color: ColorRGB = (255, 216, 0)
    start_color: ColorRGB = (0, 220, 95)
    goal_color: ColorRGB = (50, 145, 255)
    vehicle_color: ColorRGB = (0, 230, 230)
    target_color: ColorRGB = (255, 64, 220)
    text_color: ColorRGB = (235, 235, 235)
    muted_text_color: ColorRGB = (170, 174, 180)


CARLA = CarlaSettings()
DISPLAY = DisplaySettings()
VEHICLE = VehicleSettings()
CAMERA = CameraSettings()
WAYPOINT = WaypointOverlaySettings()
MANUAL_CONTROL = ManualControlSettings()
ROUTE_PLANNER = RoutePlannerSettings()
WAYPOINT_TRACKER = WaypointTrackerSettings()
AUTONOMOUS_CONTROL = AutonomousControlSettings()
TOPDOWN_MAP = TopDownMapSettings()

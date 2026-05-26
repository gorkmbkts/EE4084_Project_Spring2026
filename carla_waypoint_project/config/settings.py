"""Centralized project settings and constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

ColorRGB = Tuple[int, int, int]


@dataclass(frozen=True)
class CarlaSettings:
    host: str = "localhost"
    port: int = 2000
    timeout_seconds: float = 20.0
    connection_attempts: int = 3
    retry_delay_seconds: float = 2.0


@dataclass(frozen=True)
class DisplaySettings:
    width: int = 1742
    height: int = 1022
    fps: int = 30
    title: str = "CARLA KF Localization Dashboard"
    clear_color: ColorRGB = (10, 12, 16)
    fullscreen: bool = False
    resizable: bool = True
    maximized: bool = True
    borderless: bool = True


@dataclass(frozen=True)
class SimulationSettings:
    synchronous_mode: bool = True
    fixed_delta_seconds: float = 0.05


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
class GnssSettings:
    blueprint_id: str = "sensor.other.gnss"
    sensor_tick: float = 0.05
    noise_lat_stddev_deg: float = 0.000004 #TODO: 0.000004 normalde ama too good to be true
    noise_lon_stddev_deg: float = 0.000004
    noise_alt_stddev_m: float = 0.45
    noise_lat_bias_deg: float = 0.0
    noise_lon_bias_deg: float = 0.0
    noise_alt_bias_m: float = 0.0
    noise_seed: int = 4084
    relative_x: float = 0.0
    relative_y: float = 0.0
    relative_z: float = 2.2


@dataclass(frozen=True)
class ImuSettings:
    blueprint_id: str = "sensor.other.imu"
    sensor_tick: float = 0.05
    noise_accel_stddev_x: float = 0.08
    noise_accel_stddev_y: float = 0.08
    noise_accel_stddev_z: float = 0.10
    noise_gyro_stddev_x: float = 0.004
    noise_gyro_stddev_y: float = 0.004
    noise_gyro_stddev_z: float = 0.006
    noise_gyro_bias_x: float = 0.0005
    noise_gyro_bias_y: float = -0.0005
    noise_gyro_bias_z: float = 0.0008
    noise_seed: int = 8408
    relative_x: float = 0.0
    relative_y: float = 0.0
    relative_z: float = 2.0


@dataclass(frozen=True)
class LidarSettings:
    blueprint_id: str = "sensor.lidar.ray_cast"
    sensor_tick: float = 0.05
    channels: int = 32
    range_m: float = 50.0
    points_per_second: int = 56000
    rotation_frequency_hz: float = 20.0
    upper_fov_deg: float = 10.0
    lower_fov_deg: float = -30.0
    relative_x: float = 0.0
    relative_y: float = 0.0
    relative_z: float = 2.4


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
    lookahead_gain_s: float = 0.35
    search_backtrack_count: int = 4
    search_forward_count: int = 12
    max_closest_index_advance_per_update: int = 3
    max_target_index_ahead_of_closest: int = 6
    completion_distance_m: float = 4.0
    completion_index_window: int = 5


@dataclass(frozen=True)
class RouteInitializationSettings:
    position_error_threshold_m: float = 2.5
    timeout_position_error_threshold_m: float = 6.0
    estimated_speed_threshold_mps: float = 0.8
    stable_ticks_required: int = 8
    max_wait_seconds: float = 20.0
    hold_brake: float = 1.0


@dataclass(frozen=True)
class BenchmarkSettings:
    max_pass_duration_s: float = 180.0
    output_root: str = "logs/filter_tests"
    generate_plots_on_completion: bool = True
    collect_stabilization_samples: bool = True
    route_completion_required: bool = True


@dataclass(frozen=True)
class AutonomousControlSettings:
    target_speed_mps: float = 4.8
    turn_speed_mps: float = 2.6 #TODO: tune this better 2.6->0.6
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
class LocalizationSettings:
    estimator_name: str = "KF-CA" #TODO: !!!!!!!!!!! we will tune this later, maybe even compare multiple estimators
    min_prediction_dt_s: float = 1.0e-4
    max_prediction_dt_s: float = 0.20
    process_jerk_stddev_mps3: float = 1.20
    gnss_position_stddev_m: float = 1.25
    imu_accel_stddev_mps2: float = 0.45
    initial_position_stddev_m: float = 4.0
    initial_velocity_stddev_mps: float = 3.0
    initial_accel_stddev_mps2: float = 1.5
    yaw_from_velocity_min_speed_mps: float = 0.35


@dataclass(frozen=True)
class DashboardSettings:
    margin_px: int = 14
    gap_px: int = 14
    right_column_width: int = 420
    bottom_panel_height: int = 260
    panel_radius_px: int = 4
    panel_border_width_px: int = 1
    panel_padding_px: int = 10
    title_font_size: int = 16
    text_font_size: int = 15
    small_font_size: int = 13
    background_color: ColorRGB = (10, 12, 16)
    panel_background_color: ColorRGB = (19, 22, 28)
    panel_inner_color: ColorRGB = (13, 16, 21)
    panel_border_color: ColorRGB = (82, 91, 108)
    title_color: ColorRGB = (235, 238, 244)
    text_color: ColorRGB = (224, 229, 237)
    muted_text_color: ColorRGB = (154, 162, 176)
    warning_color: ColorRGB = (255, 196, 87)
    success_color: ColorRGB = (84, 222, 132)


@dataclass(frozen=True)
class TopDownMapSettings:
    panel_width: int = 420
    panel_height: int = 353
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
    estimated_vehicle_color: ColorRGB = (84, 222, 132)
    target_color: ColorRGB = (255, 64, 220)
    gnss_color: ColorRGB = (255, 145, 64)
    gnss_trail_color: ColorRGB = (255, 145, 64)
    gnss_trail_length: int = 45
    text_color: ColorRGB = (235, 235, 235)
    muted_text_color: ColorRGB = (170, 174, 180)


@dataclass(frozen=True)
class LidarPanelSettings:
    range_m: float = 50.0
    max_points: int = 4500
    point_radius_px: int = 1
    ring_count: int = 4
    background_color: ColorRGB = (8, 11, 15)
    grid_color: ColorRGB = (45, 54, 68)
    axis_color: ColorRGB = (85, 96, 115)
    near_point_color: ColorRGB = (101, 235, 188)
    far_point_color: ColorRGB = (75, 145, 255)
    ego_color: ColorRGB = (255, 255, 255)


CARLA = CarlaSettings()
DISPLAY = DisplaySettings()
SIMULATION = SimulationSettings()
VEHICLE = VehicleSettings()
CAMERA = CameraSettings()
GNSS = GnssSettings()
IMU = ImuSettings()
LIDAR = LidarSettings()
WAYPOINT = WaypointOverlaySettings()
MANUAL_CONTROL = ManualControlSettings()
ROUTE_PLANNER = RoutePlannerSettings()
WAYPOINT_TRACKER = WaypointTrackerSettings()
ROUTE_INITIALIZATION = RouteInitializationSettings()
BENCHMARK = BenchmarkSettings()
AUTONOMOUS_CONTROL = AutonomousControlSettings()
LOCALIZATION = LocalizationSettings()
DASHBOARD = DashboardSettings()
TOPDOWN_MAP = TopDownMapSettings()
LIDAR_PANEL = LidarPanelSettings()

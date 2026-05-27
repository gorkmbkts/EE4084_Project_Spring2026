"""Single-run localization filter benchmark metadata construction."""

from __future__ import annotations

from typing import Optional, Sequence

from config.settings import (
    AUTONOMOUS_CONTROL,
    BENCHMARK,
    DISPLAY,
    GNSS,
    IMU,
    SIMULATION,
    WAYPOINT_TRACKER,
)
from src.evaluation.test_route_store import SavedTestRoute
from src.utils.carla_import import ensure_carla_import
from src.utils.map_names import normalize_map_name

carla = ensure_carla_import()


def build_benchmark_metadata(
    benchmark_id: str,
    timestamp: str,
    route: SavedTestRoute,
    start_waypoint: "carla.Waypoint",
    goal_waypoint: "carla.Waypoint",
    route_waypoints: Sequence["carla.Waypoint"],
    map_name: Optional[str],
    selected_map_load_name: Optional[str] = None,
    active_map_id: Optional[str] = None,
    weather: Optional[dict[str, object]] = None,
    vehicle_blueprint: Optional[str] = None,
    active_filter_info: Optional[dict[str, object]] = None,
    active_filter_tune: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    """Build benchmark-level metadata from settings and route context."""
    filter_info = dict(active_filter_info or {})
    filter_tune = dict(active_filter_tune or {})
    active_filter_id = str(filter_info.get("id") or "unknown")
    active_filter_name = str(filter_info.get("name") or active_filter_id)
    active_filter_type = str(filter_info.get("type") or "unknown")
    active_filter_state_vector = str(filter_info.get("state_vector") or "n/a")
    active_filter_process_model = str(filter_info.get("process_model") or "n/a")
    active_filter_measurement_model = str(filter_info.get("measurement_model") or "n/a")
    active_filter_description = str(filter_info.get("description") or "")
    raw_gnss_note = "Raw noisy GNSS is logged as a localization baseline and is not the default closed-loop control filter."
    normalized_active_map_id = active_map_id or normalize_map_name(map_name)
    normalized_route_map_id = normalize_map_name(route.map_name)
    return {
        "active_filter_id": active_filter_id,
        "active_filter_name": active_filter_name,
        "active_filter_type": active_filter_type,
        "active_filter_state_vector": active_filter_state_vector,
        "active_filter_process_model": active_filter_process_model,
        "active_filter_measurement_model": active_filter_measurement_model,
        "active_filter_description": active_filter_description,
        "active_filter_tune": filter_tune,
        "raw_gnss_baseline_note": raw_gnss_note,
        "active_filter": {
            "id": active_filter_id,
            "name": active_filter_name,
            "type": active_filter_type,
            "state_vector": active_filter_state_vector,
            "process_model": active_filter_process_model,
            "measurement_model": active_filter_measurement_model,
            "description": active_filter_description,
            "safe_for_autonomous_control": bool(filter_info.get("safe_for_autonomous_control", True)),
            "tune": filter_tune,
        },
        "general": {
            "benchmark_id": benchmark_id,
            "timestamp": timestamp,
            "route_name": route.name,
            "map_name": map_name,
            "active_carla_map_name": map_name,
            "normalized_active_map_id": normalized_active_map_id,
            "selected_map_load_name": selected_map_load_name,
            "route_map_name": route.map_name,
            "normalized_route_map_id": normalized_route_map_id,
            "route_start": _location_dict(start_waypoint.transform.location),
            "route_goal": _location_dict(goal_waypoint.transform.location),
            "route_length_m": _route_length(route_waypoints),
            "route_waypoint_count": len(route_waypoints),
            "route_points": [_location_xy_dict(waypoint.transform.location) for waypoint in route_waypoints],
            "carla_fixed_delta_seconds": SIMULATION.fixed_delta_seconds,
            "pygame_fps": DISPLAY.fps,
            "weather": weather,
            "vehicle_blueprint": vehicle_blueprint,
            "benchmark_settings": {
                "max_pass_duration_s": BENCHMARK.max_pass_duration_s,
                "generate_plots_on_completion": BENCHMARK.generate_plots_on_completion,
                "collect_stabilization_samples": BENCHMARK.collect_stabilization_samples,
                "route_completion_required": BENCHMARK.route_completion_required,
                "max_kalman_plot_error_m": BENCHMARK.max_kalman_plot_error_m,
                "max_filtered_plot_error_m": BENCHMARK.max_kalman_plot_error_m,
                "max_trajectory_jump_m": BENCHMARK.max_trajectory_jump_m,
                "route_bounds_margin_m": BENCHMARK.route_bounds_margin_m,
                "metrics_use_driving_phase_only": BENCHMARK.metrics_use_driving_phase_only,
            },
        },
        "kalman_filter": {
            "filter_type": active_filter_name,
            "state_vector": active_filter_state_vector,
            "process_model": active_filter_process_model,
            "measurement_models": [active_filter_measurement_model],
            "tune": filter_tune,
        },
        "sensor_configuration": {
            "gnss": {
                "sensor_tick": GNSS.sensor_tick,
                "noise_lat_stddev_deg": GNSS.noise_lat_stddev_deg,
                "noise_lon_stddev_deg": GNSS.noise_lon_stddev_deg,
                "noise_alt_stddev_m": GNSS.noise_alt_stddev_m,
                "noise_lat_bias_deg": GNSS.noise_lat_bias_deg,
                "noise_lon_bias_deg": GNSS.noise_lon_bias_deg,
                "noise_alt_bias_m": GNSS.noise_alt_bias_m,
                "noise_seed": GNSS.noise_seed,
            },
            "imu": {
                "sensor_tick": IMU.sensor_tick,
                "noise_accel_stddev_x": IMU.noise_accel_stddev_x,
                "noise_accel_stddev_y": IMU.noise_accel_stddev_y,
                "noise_accel_stddev_z": IMU.noise_accel_stddev_z,
                "noise_gyro_stddev_x": IMU.noise_gyro_stddev_x,
                "noise_gyro_stddev_y": IMU.noise_gyro_stddev_y,
                "noise_gyro_stddev_z": IMU.noise_gyro_stddev_z,
                "noise_gyro_bias_x": IMU.noise_gyro_bias_x,
                "noise_gyro_bias_y": IMU.noise_gyro_bias_y,
                "noise_gyro_bias_z": IMU.noise_gyro_bias_z,
                "noise_seed": IMU.noise_seed,
            },
        },
        "controller_configuration": {
            "target_speed_mps": AUTONOMOUS_CONTROL.target_speed_mps,
            "turn_speed_mps": AUTONOMOUS_CONTROL.turn_speed_mps,
            "steering_gain": AUTONOMOUS_CONTROL.steering_gain,
            "max_throttle": AUTONOMOUS_CONTROL.max_throttle,
            "max_brake": AUTONOMOUS_CONTROL.max_brake,
            "speed_kp": AUTONOMOUS_CONTROL.speed_kp,
            "brake_kp": AUTONOMOUS_CONTROL.brake_kp,
            "stop_distance_m": AUTONOMOUS_CONTROL.stop_distance_m,
            "waypoint_tracker": {
                "lookahead_base_m": WAYPOINT_TRACKER.lookahead_base_m,
                "lookahead_gain_s": WAYPOINT_TRACKER.lookahead_gain_s,
                "search_backtrack_count": WAYPOINT_TRACKER.search_backtrack_count,
                "search_forward_count": WAYPOINT_TRACKER.search_forward_count,
                "max_closest_index_advance_per_update": WAYPOINT_TRACKER.max_closest_index_advance_per_update,
                "max_target_index_ahead_of_closest": WAYPOINT_TRACKER.max_target_index_ahead_of_closest,
                "completion_distance_m": WAYPOINT_TRACKER.completion_distance_m,
                "completion_index_window": WAYPOINT_TRACKER.completion_index_window,
            },
        },
        "notes": {
            "raw_gnss_baseline": raw_gnss_note,
            "single_run_benchmark": (
                "Vehicle control uses the active KalmanLab filter estimate. "
                "Ground truth, active filter estimate, raw GNSS, and route tracking are logged in one run."
            ),
        },
    }


def _location_dict(location: "carla.Location") -> dict[str, float]:
    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(location.z),
    }


def _location_xy_dict(location: "carla.Location") -> dict[str, float]:
    return {
        "x": float(location.x),
        "y": float(location.y),
    }


def _route_length(route_waypoints: Sequence["carla.Waypoint"]) -> Optional[float]:
    if len(route_waypoints) < 2:
        return None
    total = 0.0
    for previous, current in zip(route_waypoints, route_waypoints[1:]):
        total += previous.transform.location.distance(current.transform.location)
    return float(total)

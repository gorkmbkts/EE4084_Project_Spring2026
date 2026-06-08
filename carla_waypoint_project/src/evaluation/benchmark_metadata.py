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
    tracking_mode: str = "passive",
    active_control_input_used: bool = False,
    sensor_noise_config: Optional[dict[str, object]] = None,
    vehicle_behavior_config: Optional[dict[str, object]] = None,
    actuator_realism_config: Optional[dict[str, object]] = None,
    random_seed: Optional[int] = None,
    run_id: Optional[str] = None,
    route_index: Optional[int] = None,
    route_count: Optional[int] = None,
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
    active_filter_model_type = str(filter_info.get("model_type") or "n/a")
    active_filter_safe = bool(filter_info.get("safe_for_autonomous_control", True))
    active_filter_active_tracking = bool(filter_info.get("active_tracking_supported", False))
    active_filter_benchmark_selectable = bool(filter_info.get("benchmark_selectable", active_filter_safe))
    active_filter_experimental = bool(filter_info.get("experimental", False))
    active_filter_requires_raw_imu = bool(filter_info.get("requires_raw_imu", False))
    raw_gnss_note = "Raw noisy GNSS is logged as a localization baseline and is not the default closed-loop control filter."
    normalized_active_map_id = active_map_id or normalize_map_name(map_name)
    normalized_route_map_id = normalize_map_name(route.map_name)
    sensor_config = dict(sensor_noise_config or {})
    behavior_config = dict(vehicle_behavior_config or {})
    actuator_config = dict(actuator_realism_config or {})
    gnss_config = {
        "sensor_tick": sensor_config.get("gnss_sensor_tick", GNSS.sensor_tick),
        "noise_lat_stddev_deg": sensor_config.get("gnss_noise_lat_stddev_deg", GNSS.noise_lat_stddev_deg),
        "noise_lon_stddev_deg": sensor_config.get("gnss_noise_lon_stddev_deg", GNSS.noise_lon_stddev_deg),
        "noise_alt_stddev_m": sensor_config.get("gnss_noise_alt_stddev_m", GNSS.noise_alt_stddev_m),
        "noise_lat_bias_deg": sensor_config.get("gnss_noise_lat_bias_deg", GNSS.noise_lat_bias_deg),
        "noise_lon_bias_deg": sensor_config.get("gnss_noise_lon_bias_deg", GNSS.noise_lon_bias_deg),
        "noise_alt_bias_m": sensor_config.get("gnss_noise_alt_bias_m", GNSS.noise_alt_bias_m),
        "noise_seed": sensor_config.get("gnss_noise_seed", GNSS.noise_seed),
    }
    imu_config = {
        "sensor_tick": sensor_config.get("imu_sensor_tick", IMU.sensor_tick),
        "noise_accel_stddev_x": sensor_config.get("imu_noise_accel_stddev_x", IMU.noise_accel_stddev_x),
        "noise_accel_stddev_y": sensor_config.get("imu_noise_accel_stddev_y", IMU.noise_accel_stddev_y),
        "noise_accel_stddev_z": sensor_config.get("imu_noise_accel_stddev_z", IMU.noise_accel_stddev_z),
        "noise_gyro_stddev_x": sensor_config.get("imu_noise_gyro_stddev_x", IMU.noise_gyro_stddev_x),
        "noise_gyro_stddev_y": sensor_config.get("imu_noise_gyro_stddev_y", IMU.noise_gyro_stddev_y),
        "noise_gyro_stddev_z": sensor_config.get("imu_noise_gyro_stddev_z", IMU.noise_gyro_stddev_z),
        "noise_gyro_bias_x": sensor_config.get("imu_noise_gyro_bias_x", IMU.noise_gyro_bias_x),
        "noise_gyro_bias_y": sensor_config.get("imu_noise_gyro_bias_y", IMU.noise_gyro_bias_y),
        "noise_gyro_bias_z": sensor_config.get("imu_noise_gyro_bias_z", IMU.noise_gyro_bias_z),
        "noise_seed": sensor_config.get("imu_noise_seed", IMU.noise_seed),
    }
    return {
        "run_id": run_id,
        "random_seed": random_seed,
        "route_index": route_index,
        "route_count": route_count,
        "selected_filter": active_filter_id,
        "tracking_mode": tracking_mode,
        "active_control_input_used_by_filter": bool(active_control_input_used),
        "active_filter_id": active_filter_id,
        "active_filter_name": active_filter_name,
        "active_filter_type": active_filter_type,
        "active_filter_model_type": active_filter_model_type,
        "active_filter_state_vector": active_filter_state_vector,
        "active_filter_process_model": active_filter_process_model,
        "active_filter_measurement_model": active_filter_measurement_model,
        "active_filter_description": active_filter_description,
        "active_filter_safe_for_autonomous_control": active_filter_safe,
        "active_filter_active_tracking_supported": active_filter_active_tracking,
        "active_filter_benchmark_selectable": active_filter_benchmark_selectable,
        "active_filter_experimental": active_filter_experimental,
        "active_filter_requires_raw_imu": active_filter_requires_raw_imu,
        "active_filter_tune": filter_tune,
        "raw_gnss_baseline_note": raw_gnss_note,
        "active_filter": {
            "id": active_filter_id,
            "name": active_filter_name,
            "type": active_filter_type,
            "model_type": active_filter_model_type,
            "state_vector": active_filter_state_vector,
            "process_model": active_filter_process_model,
            "measurement_model": active_filter_measurement_model,
            "description": active_filter_description,
            "safe_for_autonomous_control": active_filter_safe,
            "active_tracking_supported": active_filter_active_tracking,
            "benchmark_selectable": active_filter_benchmark_selectable,
            "experimental": active_filter_experimental,
            "requires_raw_imu": active_filter_requires_raw_imu,
            "provided_state_fields": tuple(filter_info.get("provided_state_fields", ())),
            "tracking_mode": tracking_mode,
            "active_control_input_used": bool(active_control_input_used),
            "recommendation_applied": bool(filter_info.get("recommendation_applied", False)),
            "tune": filter_tune,
        },
        "general": {
            "benchmark_id": benchmark_id,
            "run_id": run_id,
            "timestamp": timestamp,
            "route_name": route.name,
            "route_index": route_index,
            "route_count": route_count,
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
            "tracking_mode": tracking_mode,
            "active_control_input_used_by_filter": bool(active_control_input_used),
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
                "teleport_settle_seconds": BENCHMARK.teleport_settle_seconds,
                "sensor_warmup_seconds": BENCHMARK.sensor_warmup_seconds,
                "filter_warmup_seconds": BENCHMARK.filter_warmup_seconds,
                "offline_metric_warmup_seconds": BENCHMARK.offline_metric_warmup_seconds,
                "min_fresh_sensor_frames_after_teleport": BENCHMARK.min_fresh_sensor_frames_after_teleport,
                "max_valid_imu_accel_mps2": BENCHMARK.max_valid_imu_accel_mps2,
                "divergence_error_threshold_m": BENCHMARK.divergence_error_threshold_m,
            },
            "teleport_transient_handling": (
                "Saved route tests begin by relocating the ego vehicle to the route start. "
                "Startup stabilization samples are diagnostic; valid_for_metrics/eval metrics "
                "exclude those non-physical transients from route performance comparison."
            ),
        },
        "kalman_filter": {
            "filter_type": active_filter_name,
            "state_vector": active_filter_state_vector,
            "process_model": active_filter_process_model,
            "measurement_models": [active_filter_measurement_model],
            "tracking_mode": tracking_mode,
            "active_control_input_used": bool(active_control_input_used),
            "tune": filter_tune,
        },
        "sensor_configuration": {
            "gnss": gnss_config,
            "imu": imu_config,
            "raw_config": sensor_config,
        },
        "vehicle_behavior_config": behavior_config,
        "actuator_realism_config": actuator_config,
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
                "Vehicle control uses the selected KalmanLab filter estimate. "
                "Ground truth, selected filter estimate, raw GNSS, and route tracking are logged in one run."
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

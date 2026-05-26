"""Single-run Kalman benchmark metadata construction."""

from __future__ import annotations

from typing import Optional, Sequence

from config.settings import (
    AUTONOMOUS_CONTROL,
    BENCHMARK,
    DISPLAY,
    GNSS,
    IMU,
    LOCALIZATION,
    SIMULATION,
    WAYPOINT_TRACKER,
)
from src.evaluation.test_route_store import SavedTestRoute
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


def build_benchmark_metadata(
    benchmark_id: str,
    timestamp: str,
    route: SavedTestRoute,
    start_waypoint: "carla.Waypoint",
    goal_waypoint: "carla.Waypoint",
    route_waypoints: Sequence["carla.Waypoint"],
    map_name: Optional[str],
    weather: Optional[dict[str, object]] = None,
    vehicle_blueprint: Optional[str] = None,
) -> dict[str, object]:
    """Build benchmark-level metadata from settings and route context."""
    return {
        "general": {
            "benchmark_id": benchmark_id,
            "timestamp": timestamp,
            "route_name": route.name,
            "map_name": map_name,
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
            },
        },
        "kalman_filter": {
            "filter_type": LOCALIZATION.estimator_name,
            "state_vector": "[px, py, vx, vy, ax, ay]^T",
            "process_model": "Constant Acceleration",
            "measurement_models": [
                "GNSS position x/y",
                "IMU acceleration ax/ay",
            ],
            "process_noise_parameters": {
                "process_jerk_stddev_mps3": LOCALIZATION.process_jerk_stddev_mps3,
            },
            "measurement_noise_parameters": {
                "gnss_position_stddev_m": LOCALIZATION.gnss_position_stddev_m,
                "imu_accel_stddev_mps2": LOCALIZATION.imu_accel_stddev_mps2,
            },
            "initial_covariance_parameters": {
                "initial_position_stddev_m": LOCALIZATION.initial_position_stddev_m,
                "initial_velocity_stddev_mps": LOCALIZATION.initial_velocity_stddev_mps,
                "initial_accel_stddev_mps2": LOCALIZATION.initial_accel_stddev_mps2,
            },
            "yaw_from_velocity_min_speed_mps": LOCALIZATION.yaw_from_velocity_min_speed_mps,
            "min_prediction_dt_s": LOCALIZATION.min_prediction_dt_s,
            "max_prediction_dt_s": LOCALIZATION.max_prediction_dt_s,
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
            "raw_gnss_baseline": (
                "Raw noisy GNSS is evaluated as a localization baseline only and is not used for closed-loop control."
            )
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

"""Shared configuration model for automated filter evaluation benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
from typing import Optional, Sequence

from config.settings import GNSS, IMU
from src.evaluation.test_route_store import SavedTestRoute, TestRouteStore
from src.utils.map_names import maps_compatible, normalize_map_name

_VALID_TRACKING_MODES = ("passive", "active")


def _normalize_tracking_mode(value: object) -> str:
    text = str(value or "passive").strip().lower()
    return text if text in _VALID_TRACKING_MODES else "passive"


@dataclass(frozen=True)
class ParameterSpec:
    """UI and validation metadata for one numeric benchmark parameter."""

    key: str
    label: str
    minimum: float
    maximum: float
    unit: str = ""
    decimals: int = 2
    group: str = ""

    def clamp(self, value: object) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = self.minimum
        if not math.isfinite(number):
            number = self.minimum
        return max(self.minimum, min(self.maximum, number))


@dataclass
class SensorNoiseConfig:
    """Mutable sensor noise/error values applied to CARLA sensor blueprints."""

    gnss_sensor_tick: float = GNSS.sensor_tick
    gnss_noise_lat_stddev_deg: float = GNSS.noise_lat_stddev_deg
    gnss_noise_lon_stddev_deg: float = GNSS.noise_lon_stddev_deg
    gnss_noise_alt_stddev_m: float = GNSS.noise_alt_stddev_m
    gnss_noise_lat_bias_deg: float = GNSS.noise_lat_bias_deg
    gnss_noise_lon_bias_deg: float = GNSS.noise_lon_bias_deg
    gnss_noise_alt_bias_m: float = GNSS.noise_alt_bias_m
    imu_sensor_tick: float = IMU.sensor_tick
    imu_noise_accel_stddev_x: float = IMU.noise_accel_stddev_x
    imu_noise_accel_stddev_y: float = IMU.noise_accel_stddev_y
    imu_noise_accel_stddev_z: float = IMU.noise_accel_stddev_z
    imu_noise_gyro_stddev_x: float = IMU.noise_gyro_stddev_x
    imu_noise_gyro_stddev_y: float = IMU.noise_gyro_stddev_y
    imu_noise_gyro_stddev_z: float = IMU.noise_gyro_stddev_z
    imu_noise_gyro_bias_x: float = IMU.noise_gyro_bias_x
    imu_noise_gyro_bias_y: float = IMU.noise_gyro_bias_y
    imu_noise_gyro_bias_z: float = IMU.noise_gyro_bias_z
    gnss_noise_seed: int = GNSS.noise_seed
    imu_noise_seed: int = IMU.noise_seed
    preset_name: str = "Medium Noise"

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SensorNoiseConfig":
        values = {field.name: getattr(cls(), field.name) for field in fields(cls)}
        for key in values:
            if key in data:
                values[key] = data[key]
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


SENSOR_NOISE_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec("gnss_sensor_tick", "GNSS tick", 0.01, 0.50, "s", 2, "GNSS"),
    ParameterSpec("gnss_noise_lat_stddev_deg", "GNSS lat std", 0.0, 0.00005, "deg", 7, "GNSS"),
    ParameterSpec("gnss_noise_lon_stddev_deg", "GNSS lon std", 0.0, 0.00005, "deg", 7, "GNSS"),
    ParameterSpec("gnss_noise_alt_stddev_m", "GNSS alt std", 0.0, 3.0, "m", 2, "GNSS"),
    ParameterSpec("gnss_noise_lat_bias_deg", "GNSS lat bias", -0.00005, 0.00005, "deg", 7, "GNSS"),
    ParameterSpec("gnss_noise_lon_bias_deg", "GNSS lon bias", -0.00005, 0.00005, "deg", 7, "GNSS"),
    ParameterSpec("gnss_noise_alt_bias_m", "GNSS alt bias", -5.0, 5.0, "m", 2, "GNSS"),
    ParameterSpec("imu_sensor_tick", "IMU tick", 0.01, 0.50, "s", 2, "IMU"),
    ParameterSpec("imu_noise_accel_stddev_x", "IMU accel X std", 0.0, 1.0, "m/s2", 3, "IMU"),
    ParameterSpec("imu_noise_accel_stddev_y", "IMU accel Y std", 0.0, 1.0, "m/s2", 3, "IMU"),
    ParameterSpec("imu_noise_accel_stddev_z", "IMU accel Z std", 0.0, 1.5, "m/s2", 3, "IMU"),
    ParameterSpec("imu_noise_gyro_stddev_x", "IMU gyro X std", 0.0, 0.08, "rad/s", 4, "IMU"),
    ParameterSpec("imu_noise_gyro_stddev_y", "IMU gyro Y std", 0.0, 0.08, "rad/s", 4, "IMU"),
    ParameterSpec("imu_noise_gyro_stddev_z", "IMU gyro Z std", 0.0, 0.08, "rad/s", 4, "IMU"),
    ParameterSpec("imu_noise_gyro_bias_x", "IMU gyro X bias", -0.04, 0.04, "rad/s", 4, "IMU"),
    ParameterSpec("imu_noise_gyro_bias_y", "IMU gyro Y bias", -0.04, 0.04, "rad/s", 4, "IMU"),
    ParameterSpec("imu_noise_gyro_bias_z", "IMU gyro Z bias", -0.04, 0.04, "rad/s", 4, "IMU"),
)


def default_sensor_noise_values() -> dict[str, float]:
    config = SensorNoiseConfig()
    return {spec.key: float(getattr(config, spec.key)) for spec in SENSOR_NOISE_SPECS}


SENSOR_NOISE_PRESETS: dict[str, dict[str, float]] = {
    "Low Noise": {
        **default_sensor_noise_values(),
        "gnss_noise_lat_stddev_deg": 0.0000015,
        "gnss_noise_lon_stddev_deg": 0.0000015,
        "gnss_noise_alt_stddev_m": 0.20,
        "imu_noise_accel_stddev_x": 0.03,
        "imu_noise_accel_stddev_y": 0.03,
        "imu_noise_accel_stddev_z": 0.05,
        "imu_noise_gyro_stddev_x": 0.0015,
        "imu_noise_gyro_stddev_y": 0.0015,
        "imu_noise_gyro_stddev_z": 0.0020,
    },
    "Medium Noise": default_sensor_noise_values(),
    "High Noise": {
        **default_sensor_noise_values(),
        "gnss_noise_lat_stddev_deg": 0.000012,
        "gnss_noise_lon_stddev_deg": 0.000012,
        "gnss_noise_alt_stddev_m": 1.30,
        "imu_noise_accel_stddev_x": 0.25,
        "imu_noise_accel_stddev_y": 0.25,
        "imu_noise_accel_stddev_z": 0.35,
        "imu_noise_gyro_stddev_x": 0.018,
        "imu_noise_gyro_stddev_y": 0.018,
        "imu_noise_gyro_stddev_z": 0.024,
    },
    "GNSS Degraded": {
        **default_sensor_noise_values(),
        "gnss_noise_lat_stddev_deg": 0.000025,
        "gnss_noise_lon_stddev_deg": 0.000025,
        "gnss_noise_alt_stddev_m": 2.50,
        "gnss_noise_lat_bias_deg": 0.000008,
        "gnss_noise_lon_bias_deg": -0.000008,
        "gnss_noise_alt_bias_m": 1.20,
    },
    "IMU Degraded": {
        **default_sensor_noise_values(),
        "imu_noise_accel_stddev_x": 0.45,
        "imu_noise_accel_stddev_y": 0.45,
        "imu_noise_accel_stddev_z": 0.65,
        "imu_noise_gyro_stddev_x": 0.035,
        "imu_noise_gyro_stddev_y": 0.035,
        "imu_noise_gyro_stddev_z": 0.045,
        "imu_noise_gyro_bias_x": 0.010,
        "imu_noise_gyro_bias_y": -0.010,
        "imu_noise_gyro_bias_z": 0.014,
    },
}


BEHAVIOR_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec("max_speed_mps", "Max speed", 2.0, 14.0, "m/s", 1, "Speed planner"),
    ParameterSpec("min_curve_speed_mps", "Min curve speed", 0.8, 7.0, "m/s", 1, "Speed planner"),
    ParameterSpec("max_forward_accel_mps2", "Max accel", 0.3, 4.0, "m/s2", 1, "Speed planner"),
    ParameterSpec("max_braking_decel_mps2", "Max braking", 0.5, 7.0, "m/s2", 1, "Speed planner"),
    ParameterSpec("curve_lookahead_m", "Curve lookahead", 8.0, 55.0, "m", 0, "Speed planner"),
    ParameterSpec("curvature_sensitivity", "Curvature sens", 0.2, 3.0, "x", 2, "Speed planner"),
    ParameterSpec("safe_cornering_factor", "Safe cornering", 0.6, 1.8, "x", 2, "Speed planner"),
    ParameterSpec("speed_change_aggressiveness", "Speed aggress", 0.2, 2.5, "x", 2, "Speed planner"),
    ParameterSpec("enable_model_aware_control", "Model-aware ctrl", 0.0, 1.0, "", 0, "Model-aware control"),
    ParameterSpec("yaw_rate_feedforward_gain", "Yaw-rate FF", 0.0, 1.0, "x", 2, "Model-aware control"),
    ParameterSpec("yaw_rate_feedback_gain", "Yaw-rate FB", 0.0, 1.0, "x", 2, "Model-aware control"),
    ParameterSpec("max_model_steer_correction", "Model steer cap", 0.0, 0.5, "", 2, "Model-aware control"),
    ParameterSpec("min_model_control_speed_mps", "Model min speed", 0.1, 5.0, "m/s", 1, "Model-aware control"),
    ParameterSpec("model_state_lowpass_alpha", "State alpha", 0.02, 1.0, "", 2, "Model-aware control"),
    ParameterSpec("max_abs_motion_yaw_rate_radps", "Motion yaw cap", 0.2, 5.0, "rad/s", 2, "Model-aware control"),
    ParameterSpec("enable_model_speed_guard", "Model speed guard", 0.0, 1.0, "", 0, "Model-aware control"),
    ParameterSpec("model_curvature_speed_factor", "Model speed factor", 0.1, 1.5, "x", 2, "Model-aware control"),
    ParameterSpec("enable_acceleration_feedforward", "Accel FF", 0.0, 1.0, "", 0, "Model-aware control"),
    ParameterSpec("acceleration_feedforward_gain", "Accel FF gain", 0.0, 1.0, "x", 2, "Model-aware control"),
    ParameterSpec("max_acceleration_feedforward_delta", "Accel FF cap", 0.0, 0.5, "", 2, "Model-aware control"),
)


ACTUATOR_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec("throttle_smoothing", "Throttle smooth", 0.0, 0.9, "", 2, "Actuator lag"),
    ParameterSpec("brake_smoothing", "Brake smooth", 0.0, 0.9, "", 2, "Actuator lag"),
    ParameterSpec("steering_smoothing", "Steer smooth", 0.0, 0.9, "", 2, "Actuator lag"),
    ParameterSpec("actuator_delay_s", "Actuator delay", 0.0, 0.35, "s", 2, "Delay"),
    ParameterSpec("actuator_noise", "Command noise", 0.0, 0.05, "", 3, "Noise"),
    ParameterSpec("throttle_response_gain", "Throttle gain", 0.4, 1.6, "x", 2, "Response gain"),
    ParameterSpec("brake_response_gain", "Brake gain", 0.4, 1.6, "x", 2, "Response gain"),
    ParameterSpec("steering_response_gain", "Steering gain", 0.4, 1.8, "x", 2, "Response gain"),
    ParameterSpec("max_throttle_rate_per_s", "Throttle rate", 0.2, 6.0, "/s", 1, "Rate limits"),
    ParameterSpec("max_brake_rate_per_s", "Brake rate", 0.2, 8.0, "/s", 1, "Rate limits"),
    ParameterSpec("max_steer_rate_per_s", "Steer rate", 0.4, 8.0, "/s", 1, "Rate limits"),
)


def behavior_values_from_config(config: object) -> dict[str, float]:
    return {spec.key: float(getattr(config, spec.key)) for spec in BEHAVIOR_SPECS}


def actuator_values_from_config(config: object) -> dict[str, float]:
    return {spec.key: float(getattr(config, spec.key)) for spec in ACTUATOR_SPECS}


def driving_behavior_from_values(
    values: dict[str, object],
    preset_name: str = "Balanced",
) -> dict[str, object]:
    result = _base_behavior_values()
    for spec in BEHAVIOR_SPECS:
        result[spec.key] = spec.clamp(values.get(spec.key, result[spec.key]))
    if result["min_curve_speed_mps"] > result["max_speed_mps"]:
        result["min_curve_speed_mps"] = result["max_speed_mps"]
    result["preset_name"] = preset_name
    return result


def apply_behavior_values(config: object, values: dict[str, object]) -> None:
    for spec in BEHAVIOR_SPECS:
        setattr(config, spec.key, spec.clamp(values.get(spec.key, getattr(config, spec.key))))
    if config.min_curve_speed_mps > config.max_speed_mps:
        config.min_curve_speed_mps = config.max_speed_mps
    # Backward compatibility: older configs stored actuator fields in vehicle_behavior_config.
    if any(spec.key in values for spec in ACTUATOR_SPECS):
        apply_actuator_values(config, values)


def default_actuator_realism_values() -> dict[str, float]:
    return dict(_base_actuator_values())


def actuator_realism_from_values(
    values: dict[str, object],
    preset_name: str = "Realistic",
) -> dict[str, object]:
    result = _base_actuator_values()
    for spec in ACTUATOR_SPECS:
        result[spec.key] = spec.clamp(values.get(spec.key, result[spec.key]))
    result["preset_name"] = preset_name
    result["enabled"] = True
    return result


def apply_actuator_values(config: object, values: dict[str, object]) -> None:
    for spec in ACTUATOR_SPECS:
        setattr(config, spec.key, spec.clamp(values.get(spec.key, getattr(config, spec.key))))


def _base_behavior_values() -> dict[str, float]:
    return {
        "max_speed_mps": 8.0,
        "min_curve_speed_mps": 2.6,
        "max_forward_accel_mps2": 1.6,
        "max_braking_decel_mps2": 3.2,
        "curve_lookahead_m": 28.0,
        "curvature_sensitivity": 1.15,
        "speed_change_aggressiveness": 1.0,
        "safe_cornering_factor": 1.0,
        "enable_model_aware_control": 0.0,
        "yaw_rate_feedforward_gain": 0.25,
        "yaw_rate_feedback_gain": 0.0,
        "max_model_steer_correction": 0.15,
        "min_model_control_speed_mps": 1.0,
        "model_state_lowpass_alpha": 0.25,
        "max_abs_motion_yaw_rate_radps": 2.5,
        "enable_model_speed_guard": 0.0,
        "model_curvature_speed_factor": 0.5,
        "enable_acceleration_feedforward": 0.0,
        "acceleration_feedforward_gain": 0.0,
        "max_acceleration_feedforward_delta": 0.15,
    }


def _base_actuator_values() -> dict[str, float]:
    return {
        "throttle_smoothing": 0.35,
        "brake_smoothing": 0.28,
        "steering_smoothing": 0.30,
        "actuator_noise": 0.008,
        "actuator_delay_s": 0.08,
        "throttle_response_gain": 1.0,
        "brake_response_gain": 1.0,
        "steering_response_gain": 1.0,
        "max_throttle_rate_per_s": 1.8,
        "max_brake_rate_per_s": 2.4,
        "max_steer_rate_per_s": 1.8,
    }


BEHAVIOR_PRESETS: dict[str, dict[str, float]] = {
    "Balanced": _base_behavior_values(),
    "Conservative": {
        **_base_behavior_values(),
        "max_speed_mps": 5.2,
        "min_curve_speed_mps": 1.8,
        "max_forward_accel_mps2": 0.9,
        "max_braking_decel_mps2": 2.4,
        "curvature_sensitivity": 1.55,
        "safe_cornering_factor": 1.25,
        "speed_change_aggressiveness": 0.65,
    },
    "Aggressive": {
        **_base_behavior_values(),
        "max_speed_mps": 10.5,
        "min_curve_speed_mps": 3.2,
        "max_forward_accel_mps2": 2.8,
        "max_braking_decel_mps2": 5.5,
        "curvature_sensitivity": 0.85,
        "safe_cornering_factor": 0.85,
        "speed_change_aggressiveness": 1.65,
    },
}


ACTUATOR_REALISM_PRESETS: dict[str, dict[str, float]] = {
    "Perfect Actuator": {
        **_base_actuator_values(),
        "throttle_smoothing": 0.0,
        "brake_smoothing": 0.0,
        "steering_smoothing": 0.0,
        "actuator_delay_s": 0.0,
        "actuator_noise": 0.0,
        "max_throttle_rate_per_s": 6.0,
        "max_brake_rate_per_s": 8.0,
        "max_steer_rate_per_s": 8.0,
    },
    "Mild Realistic": {
        **_base_actuator_values(),
        "throttle_smoothing": 0.22,
        "brake_smoothing": 0.18,
        "steering_smoothing": 0.20,
        "actuator_delay_s": 0.04,
        "actuator_noise": 0.004,
        "max_throttle_rate_per_s": 2.8,
        "max_brake_rate_per_s": 3.6,
        "max_steer_rate_per_s": 2.8,
    },
    "Realistic": _base_actuator_values(),
    "Delayed / Harsh Realistic": {
        **_base_actuator_values(),
        "throttle_smoothing": 0.58,
        "brake_smoothing": 0.50,
        "steering_smoothing": 0.54,
        "actuator_delay_s": 0.18,
        "actuator_noise": 0.018,
        "throttle_response_gain": 0.88,
        "brake_response_gain": 1.12,
        "steering_response_gain": 0.82,
        "max_throttle_rate_per_s": 0.9,
        "max_brake_rate_per_s": 1.2,
        "max_steer_rate_per_s": 1.0,
    },
}


@dataclass
class BenchmarkConfig:
    """Complete reproducible configuration for an automated test run."""

    selected_filter: str
    selected_routes: tuple[SavedTestRoute, ...]
    sensor_noise_config: SensorNoiseConfig
    vehicle_behavior_config: dict[str, object]
    actuator_realism_config: dict[str, object] | None = None
    selected_filter_tune: dict[str, object] | None = None
    tracking_mode: str = "passive"
    sensor_noise_preset: str = "Medium Noise"
    vehicle_behavior_preset: str = "Balanced"
    actuator_realism_preset: str = "Realistic"
    random_seed: int = 4084
    output_root: str = "benchmark_results"
    run_id: str = ""
    created_at: str = ""
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec="seconds")
        if self.metadata is None:
            self.metadata = {}
        if self.selected_filter_tune is None:
            self.selected_filter_tune = {}
        if self.actuator_realism_config is None:
            legacy_actuator_values = {
                spec.key: self.vehicle_behavior_config[spec.key]
                for spec in ACTUATOR_SPECS
                if spec.key in self.vehicle_behavior_config
            }
            if legacy_actuator_values:
                preset = str(self.vehicle_behavior_config.get("preset_name") or self.actuator_realism_preset)
                self.actuator_realism_config = actuator_realism_from_values(legacy_actuator_values, preset_name=preset)
            else:
                self.actuator_realism_config = actuator_realism_from_values(
                    ACTUATOR_REALISM_PRESETS.get(self.actuator_realism_preset, ACTUATOR_REALISM_PRESETS["Realistic"]),
                    preset_name=self.actuator_realism_preset,
                )
        self.tracking_mode = _normalize_tracking_mode(self.tracking_mode)

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_filter": self.selected_filter,
            "selected_routes": [route.to_dict() for route in self.selected_routes],
            "sensor_noise_config": self.sensor_noise_config.to_dict(),
            "vehicle_behavior_config": dict(self.vehicle_behavior_config),
            "actuator_realism_config": dict(self.actuator_realism_config or {}),
            "selected_filter_tune": dict(self.selected_filter_tune or {}),
            "tracking_mode": self.tracking_mode,
            "sensor_noise_preset": self.sensor_noise_preset,
            "vehicle_behavior_preset": self.vehicle_behavior_preset,
            "actuator_realism_preset": self.actuator_realism_preset,
            "random_seed": self.random_seed,
            "output_root": self.output_root,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "metadata": dict(self.metadata or {}),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@dataclass(frozen=True)
class RouteListItem:
    """Saved route plus lightweight metadata for setup UIs."""

    index: int
    route: SavedTestRoute
    straight_line_length_m: Optional[float]
    compatible_with_available_maps: bool
    map_id: Optional[str]


def sensor_noise_config_from_values(values: dict[str, object], preset_name: str = "Custom") -> SensorNoiseConfig:
    base = SensorNoiseConfig()
    data = base.to_dict()
    for spec in SENSOR_NOISE_SPECS:
        data[spec.key] = spec.clamp(values.get(spec.key, data[spec.key]))
    data["preset_name"] = preset_name
    return SensorNoiseConfig.from_dict(data)


def load_available_test_routes(
    available_maps: Sequence[str] = (),
    store_path: Optional[Path] = None,
) -> list[RouteListItem]:
    store = TestRouteStore(path=store_path)
    normalized_available = {normalize_map_name(name) for name in available_maps if normalize_map_name(name)}
    result: list[RouteListItem] = []
    for index, route in enumerate(store.all_routes):
        route_map_id = normalize_map_name(route.map_name)
        compatible = not normalized_available or route_map_id in normalized_available
        result.append(
            RouteListItem(
                index=index,
                route=route,
                straight_line_length_m=_straight_line_distance(route),
                compatible_with_available_maps=compatible,
                map_id=route_map_id,
            )
        )
    return result


def validate_benchmark_config(
    config: BenchmarkConfig,
    valid_filter_ids: Sequence[str],
    available_maps: Sequence[str] = (),
) -> list[str]:
    errors: list[str] = []
    if not config.selected_filter:
        errors.append("Select a filter.")
    elif config.selected_filter not in set(valid_filter_ids):
        errors.append(f"Filter is not benchmark-selectable: {config.selected_filter}.")
    if not config.selected_routes:
        errors.append("Select at least one saved test route.")
    elif len(config.selected_routes) != 1:
        errors.append("Closed-loop benchmark requires exactly one selected route.")
    if _normalize_tracking_mode(config.tracking_mode) != config.tracking_mode:
        errors.append(f"Unsupported tracking mode: {config.tracking_mode}.")

    available = [name for name in available_maps if name]
    for route in config.selected_routes:
        if route.map_name and available and not any(maps_compatible(candidate, route.map_name) for candidate in available):
            errors.append(f"Route map unavailable: {route.name} -> {route.map_name}.")

    for spec in SENSOR_NOISE_SPECS:
        value = getattr(config.sensor_noise_config, spec.key)
        if spec.clamp(value) != float(value):
            errors.append(f"{spec.label} is outside [{spec.minimum}, {spec.maximum}].")
    for spec in BEHAVIOR_SPECS:
        value = config.vehicle_behavior_config.get(spec.key)
        if spec.clamp(value) != float(value):
            errors.append(f"{spec.label} is outside [{spec.minimum}, {spec.maximum}].")
    actuator_config = config.actuator_realism_config or {}
    for spec in ACTUATOR_SPECS:
        value = actuator_config.get(spec.key)
        if spec.clamp(value) != float(value):
            errors.append(f"{spec.label} is outside [{spec.minimum}, {spec.maximum}].")
    return errors


def benchmark_output_root(output_root: str) -> Path:
    root = Path(output_root)
    if root.is_absolute():
        return root
    return Path(__file__).resolve().parents[2] / root


def project_commit_hash() -> Optional[str]:
    project_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _straight_line_distance(route: SavedTestRoute) -> Optional[float]:
    try:
        return math.sqrt(
            (route.goal.x - route.start.x) ** 2
            + (route.goal.y - route.start.y) ** 2
            + (route.goal.z - route.start.z) ** 2
        )
    except (TypeError, ValueError):
        return None

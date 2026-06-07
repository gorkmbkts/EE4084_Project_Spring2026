"""Map selected sensor-noise profiles into locked filter measurement-noise tune values."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

from src.KalmanLab.tune_advisor import gnss_horizontal_bias_m, gnss_horizontal_stddev_m
from src.evaluation.consistency_metrics import MEASUREMENT_NOISE_TUNE_KEYS


@dataclass(frozen=True)
class LockedSensorNoiseValues:
    values: dict[str, float]
    sources: dict[str, str]
    signature: str
    representative_config: dict[str, object]


class SensorNoiseTuneMapper:
    """Build tune overrides that lock measurement noise from a sensor-noise config."""

    @staticmethod
    def locked_values(
        filter_id: str,
        base_tune: dict[str, object],
        sensor_noise_config: object,
        tune_specs: tuple[object, ...] = (),
    ) -> LockedSensorNoiseValues:
        sensor = _sensor_dict(sensor_noise_config)
        specs_by_key = {str(getattr(spec, "key", "")): spec for spec in tune_specs if getattr(spec, "key", "")}
        values: dict[str, float] = {}
        sources: dict[str, str] = {}

        _maybe_set(
            "gnss_position_stddev_m",
            _first_float(sensor, "gnss_position_stddev_m"),
            "sensor_noise_config.gnss_position_stddev_m",
            values,
            sources,
        )
        if "gnss_position_stddev_m" not in values:
            gnss_stddev = max(0.25, gnss_horizontal_stddev_m(sensor))
            gnss_bias = gnss_horizontal_bias_m(sensor)
            if gnss_bias > 0.0:
                gnss_stddev = max(gnss_stddev, gnss_stddev + 0.65 * gnss_bias)
            values["gnss_position_stddev_m"] = gnss_stddev
            sources["gnss_position_stddev_m"] = "derived_from_gnss_lat_lon_noise"

        _maybe_set(
            "imu_yaw_stddev_deg",
            _first_float(sensor, "imu_yaw_stddev_deg", "imu_compass_stddev_deg"),
            "sensor_noise_config.imu_yaw_or_compass",
            values,
            sources,
        )
        _maybe_set(
            "imu_yaw_rate_stddev_radps",
            _first_float(sensor, "imu_yaw_rate_stddev_radps", "imu_gyro_stddev_radps", "imu_noise_gyro_stddev_z"),
            "sensor_noise_config.imu_gyro_z",
            values,
            sources,
        )
        _maybe_set(
            "imu_accel_stddev_mps2",
            _first_float(sensor, "imu_accel_stddev_mps2"),
            "sensor_noise_config.imu_accel_stddev_mps2",
            values,
            sources,
        )
        if "imu_accel_stddev_mps2" not in values:
            accel_x = _first_float(sensor, "imu_noise_accel_stddev_x")
            accel_y = _first_float(sensor, "imu_noise_accel_stddev_y")
            if accel_x is not None or accel_y is not None:
                ax = accel_x if accel_x is not None else accel_y or 0.0
                ay = accel_y if accel_y is not None else accel_x or 0.0
                values["imu_accel_stddev_mps2"] = math.sqrt(0.5 * (ax * ax + ay * ay))
                sources["imu_accel_stddev_mps2"] = "derived_from_imu_accel_xy_noise"

        for key in sorted(MEASUREMENT_NOISE_TUNE_KEYS):
            if key not in base_tune:
                values.pop(key, None)
                sources.pop(key, None)
                continue
            if key not in values:
                fallback = _optional_float(base_tune.get(key))
                if fallback is not None:
                    values[key] = fallback
                    sources[key] = "base_tune_fallback_no_profile_field"

        for key, value in list(values.items()):
            spec = specs_by_key.get(key)
            if spec is not None and hasattr(spec, "clamp"):
                values[key] = float(spec.clamp(value))

        return LockedSensorNoiseValues(
            values=values,
            sources=sources,
            signature=noise_signature(sensor),
            representative_config=sensor,
        )

    @staticmethod
    def apply_locked_values(
        filter_id: str,
        base_tune: dict[str, object],
        sensor_noise_config: object,
        tune_specs: tuple[object, ...] = (),
    ) -> dict[str, object]:
        locked = SensorNoiseTuneMapper.locked_values(filter_id, base_tune, sensor_noise_config, tune_specs)
        merged = dict(base_tune)
        merged.update(locked.values)
        return merged


def process_only_auto_tune_profile(profile: dict[str, object]) -> dict[str, object]:
    """Return an auto-tune profile with measurement-noise parameters removed."""
    result = dict(profile)
    for group in ("primary", "secondary"):
        params = profile.get(group)
        if not isinstance(params, list):
            result[group] = []
            continue
        result[group] = [
            dict(param)
            for param in params
            if isinstance(param, dict) and str(param.get("key") or "") not in MEASUREMENT_NOISE_TUNE_KEYS
        ]
    return result


def noise_signature(sensor_noise_config: object) -> str:
    sensor = _sensor_dict(sensor_noise_config)
    return json.dumps(sensor, sort_keys=True, separators=(",", ":"))


def _sensor_dict(sensor_noise_config: object) -> dict[str, object]:
    if isinstance(sensor_noise_config, dict):
        return dict(sensor_noise_config)
    to_dict = getattr(sensor_noise_config, "to_dict", None)
    if callable(to_dict):
        try:
            data = to_dict()
        except Exception:
            data = {}
        if isinstance(data, dict):
            return dict(data)
    return {
        key: getattr(sensor_noise_config, key)
        for key in dir(sensor_noise_config)
        if key.startswith(("gnss_", "imu_")) and not key.startswith("_")
    }


def _maybe_set(
    key: str,
    value: Optional[float],
    source: str,
    values: dict[str, float],
    sources: dict[str, str],
) -> None:
    if value is not None:
        values[key] = value
        sources[key] = source


def _first_float(sensor: dict[str, object], *keys: str) -> Optional[float]:
    for key in keys:
        value = _optional_float(sensor.get(key))
        if value is not None:
            return value
    return None


def _optional_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

"""Rule-based filter tune recommendations from selected sensor noise."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from src.KalmanLab.filter_base import TRACKING_MODE_ACTIVE, normalize_tracking_mode


DEG_TO_M_LAT = 111_320.0
DEG_TO_M_LON = 111_320.0


@dataclass(frozen=True)
class TuneRecommendation:
    """Deterministic recommendation payload for UI and benchmark metadata."""

    filter_id: str
    tracking_mode: str
    values: dict[str, float]
    messages: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def has_values(self) -> bool:
        return bool(self.values)


def recommend_filter_tune(
    filter_id: str,
    sensor_noise_config: object,
    tracking_mode: str,
    current_tune: dict[str, object],
    tune_specs: Sequence[object] = (),
) -> TuneRecommendation:
    """Return simple deterministic tune recommendations for the selected filter."""
    mode = normalize_tracking_mode(tracking_mode)
    sensor = _sensor_dict(sensor_noise_config)
    current = dict(current_tune or {})
    specs_by_key = {str(getattr(spec, "key", "")): spec for spec in tune_specs if getattr(spec, "key", "")}

    gnss_noise_m = gnss_horizontal_stddev_m(sensor)
    gnss_bias_m = gnss_horizontal_bias_m(sensor)
    imu_accel_noise = _rms(
        _float(sensor.get("imu_noise_accel_stddev_x"), 0.08),
        _float(sensor.get("imu_noise_accel_stddev_y"), 0.08),
    )

    recommended: dict[str, float] = {}
    messages: list[str] = []
    warnings: list[str] = []

    if "gnss_position_stddev_m" in current or "gnss_position_stddev_m" in specs_by_key:
        gnss_recommendation = max(0.25, gnss_noise_m)
        if gnss_bias_m > 0.15:
            gnss_recommendation = max(gnss_recommendation, gnss_noise_m + 0.65 * gnss_bias_m)
            warnings.append("GNSS bias is nonzero; increase GNSS measurement noise.")
        if gnss_noise_m >= 2.0:
            warnings.append("GNSS appears degraded; avoid over-trusting position fixes.")
        recommended["gnss_position_stddev_m"] = _clamp_by_spec(
            "gnss_position_stddev_m",
            gnss_recommendation,
            specs_by_key,
        )
        messages.append(
            f"GNSS horizontal noise is about {gnss_noise_m:.2f} m; "
            f"recommend gnss_position_stddev_m = {recommended['gnss_position_stddev_m']:.2f} m."
        )

    if filter_id == "ca_kf":
        if "imu_accel_stddev_mps2" in current or "imu_accel_stddev_mps2" in specs_by_key:
            imu_rec = max(0.05, imu_accel_noise * 1.25)
            recommended["imu_accel_stddev_mps2"] = _clamp_by_spec("imu_accel_stddev_mps2", imu_rec, specs_by_key)
            messages.append(
                f"IMU accel noise is about {imu_accel_noise:.2f} m/s2; "
                f"recommend imu_accel_stddev_mps2 = {recommended['imu_accel_stddev_mps2']:.2f}."
            )
        if "process_jerk_stddev_mps3" in current or "process_jerk_stddev_mps3" in specs_by_key:
            base = 0.65 + 0.20 * gnss_noise_m + 0.75 * imu_accel_noise
            if mode == TRACKING_MODE_ACTIVE:
                base *= 0.82
                base = max(base, 0.45)
            else:
                base = max(base, 0.70)
            recommended["process_jerk_stddev_mps3"] = _clamp_by_spec("process_jerk_stddev_mps3", base, specs_by_key)
            messages.append(
                "Active tracking can use slightly lower jerk noise."
                if mode == TRACKING_MODE_ACTIVE
                else "Passive tracking needs enough jerk noise to absorb unmodeled maneuvers."
            )
        if "command_accel_stddev_mps2" in current or "command_accel_stddev_mps2" in specs_by_key:
            command_std = 1.0 + 0.5 * imu_accel_noise + (0.25 if mode != TRACKING_MODE_ACTIVE else 0.0)
            recommended["command_accel_stddev_mps2"] = _clamp_by_spec("command_accel_stddev_mps2", command_std, specs_by_key)

    elif filter_id == "cv_kf":
        if "process_accel_stddev_mps2" in current or "process_accel_stddev_mps2" in specs_by_key:
            imu_enabled = bool(current.get("use_imu_acceleration_control", False))
            base = 0.75 + 0.35 * gnss_noise_m
            if imu_enabled:
                base += 0.90 * imu_accel_noise
            else:
                base += 0.35
            if mode == TRACKING_MODE_ACTIVE:
                base *= 0.78
                base = max(base, 0.45)
            else:
                base = max(base, 0.85)
            recommended["process_accel_stddev_mps2"] = _clamp_by_spec("process_accel_stddev_mps2", base, specs_by_key)
            messages.append(
                "Active command input lowers the recommended CV process acceleration noise."
                if mode == TRACKING_MODE_ACTIVE
                else "Passive CV needs process acceleration noise for unmodeled turns and speed changes."
            )
        if "imu_accel_control_stddev_mps2" in current or "imu_accel_control_stddev_mps2" in specs_by_key:
            imu_control_std = max(0.08, imu_accel_noise * 1.35)
            recommended["imu_accel_control_stddev_mps2"] = _clamp_by_spec(
                "imu_accel_control_stddev_mps2",
                imu_control_std,
                specs_by_key,
            )
        if "command_accel_stddev_mps2" in current or "command_accel_stddev_mps2" in specs_by_key:
            command_std = 1.1 + 0.45 * imu_accel_noise + (0.25 if mode != TRACKING_MODE_ACTIVE else 0.0)
            recommended["command_accel_stddev_mps2"] = _clamp_by_spec("command_accel_stddev_mps2", command_std, specs_by_key)

    elif filter_id == "raw_gnss":
        warnings.append("Raw GNSS remains unsafe for autonomous control; recommendations only affect display smoothing.")

    return TuneRecommendation(
        filter_id=filter_id,
        tracking_mode=mode,
        values=recommended,
        messages=tuple(messages[:4]),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def gnss_horizontal_stddev_m(sensor_noise_config: object) -> float:
    sensor = _sensor_dict(sensor_noise_config)
    lat_m = abs(_float(sensor.get("gnss_noise_lat_stddev_deg"), 0.0)) * DEG_TO_M_LAT
    lon_m = abs(_float(sensor.get("gnss_noise_lon_stddev_deg"), 0.0)) * DEG_TO_M_LON
    return float(math.sqrt(0.5 * (lat_m * lat_m + lon_m * lon_m)))


def gnss_horizontal_bias_m(sensor_noise_config: object) -> float:
    sensor = _sensor_dict(sensor_noise_config)
    lat_m = abs(_float(sensor.get("gnss_noise_lat_bias_deg"), 0.0)) * DEG_TO_M_LAT
    lon_m = abs(_float(sensor.get("gnss_noise_lon_bias_deg"), 0.0)) * DEG_TO_M_LON
    return float(math.hypot(lat_m, lon_m))


def _sensor_dict(sensor_noise_config: object) -> dict[str, object]:
    if isinstance(sensor_noise_config, dict):
        return dict(sensor_noise_config)
    to_dict = getattr(sensor_noise_config, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
        except Exception:
            result = {}
        if isinstance(result, dict):
            return result
    return {
        key: getattr(sensor_noise_config, key)
        for key in dir(sensor_noise_config)
        if key.startswith(("gnss_", "imu_")) and not key.startswith("_")
    }


def _float(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return float(number)


def _rms(first: float, second: float) -> float:
    return math.sqrt(0.5 * (first * first + second * second))


def _clamp_by_spec(key: str, value: float, specs_by_key: dict[str, object]) -> float:
    spec = specs_by_key.get(key)
    if spec is not None and hasattr(spec, "clamp"):
        return float(spec.clamp(value))
    return float(value)

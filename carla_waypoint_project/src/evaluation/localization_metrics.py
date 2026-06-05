"""Metrics for offline localization replay results."""

from __future__ import annotations

import math
from typing import Iterable, Optional


def compute_localization_metrics(
    samples: Iterable[dict[str, object]],
    raw_gnss_rmse_m: Optional[float] = None,
) -> dict[str, object]:
    """Compute report-friendly localization metrics from estimate samples."""
    rows = list(samples)
    position_errors = _finite_values(row.get("position_error_m") for row in rows)
    yaw_errors = [abs(value) for value in _finite_values(row.get("yaw_error_deg") for row in rows)]
    speed_errors = _finite_values(row.get("speed_error_mps") for row in rows)
    velocity_errors = _finite_values(row.get("velocity_error_mps") for row in rows)
    nis_values = _finite_values(row.get("nis") for row in rows)
    nees_values = _finite_values(row.get("nees") for row in rows)
    valid_count = len(position_errors)
    total_count = len(rows)
    rmse = _rmse(position_errors)
    final_error = _last_finite(row.get("position_error_m") for row in rows)
    improvement = None
    if raw_gnss_rmse_m is not None and raw_gnss_rmse_m > 0.0 and rmse is not None:
        improvement = 100.0 * (raw_gnss_rmse_m - rmse) / raw_gnss_rmse_m

    return {
        "position_rmse_m": rmse,
        "position_mae_m": _mean(position_errors),
        "max_position_error_m": max(position_errors) if position_errors else None,
        "final_position_error_m": final_error,
        "mean_position_error_m": _mean(position_errors),
        "position_error_std_m": _stddev(position_errors),
        "improvement_over_raw_gnss_percent": improvement,
        "valid_estimate_count": valid_count,
        "missing_or_invalid_estimate_count": max(0, total_count - valid_count),
        "total_sample_count": total_count,
        "yaw_rmse_deg": _rmse(yaw_errors),
        "speed_rmse_mps": _rmse(speed_errors),
        "velocity_rmse_mps": _rmse(velocity_errors),
        "mean_nis": _mean(nis_values) if nis_values else None,
        "mean_nees": _mean(nees_values) if nees_values else None,
        "nis_available": bool(nis_values),
        "nees_available": bool(nees_values),
    }


def position_error(x_value: object, y_value: object, gt_x: object, gt_y: object) -> Optional[float]:
    x = _optional_float(x_value)
    y = _optional_float(y_value)
    x_gt = _optional_float(gt_x)
    y_gt = _optional_float(gt_y)
    if x is None or y is None or x_gt is None or y_gt is None:
        return None
    return math.hypot(x - x_gt, y - y_gt)


def yaw_error_deg(yaw_value: object, gt_yaw_value: object) -> Optional[float]:
    yaw = _optional_float(yaw_value)
    gt_yaw = _optional_float(gt_yaw_value)
    if yaw is None or gt_yaw is None:
        return None
    delta = yaw - gt_yaw
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def nees_xy(
    x_error_m: object,
    y_error_m: object,
    covariance_diagonal: object,
) -> Optional[float]:
    x_error = _optional_float(x_error_m)
    y_error = _optional_float(y_error_m)
    if x_error is None or y_error is None:
        return None
    if not isinstance(covariance_diagonal, (list, tuple)) or len(covariance_diagonal) < 2:
        return None
    var_x = _optional_float(covariance_diagonal[0])
    var_y = _optional_float(covariance_diagonal[1])
    if var_x is None or var_y is None or var_x <= 1.0e-9 or var_y <= 1.0e-9:
        return None
    return x_error * x_error / var_x + y_error * y_error / var_y


def _finite_values(values: Iterable[object]) -> list[float]:
    result: list[float] = []
    for value in values:
        number = _optional_float(value)
        if number is not None:
            result.append(number)
    return result


def _optional_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rmse(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _stddev(values: list[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _last_finite(values: Iterable[object]) -> Optional[float]:
    latest = None
    for value in values:
        number = _optional_float(value)
        if number is not None:
            latest = number
    return latest

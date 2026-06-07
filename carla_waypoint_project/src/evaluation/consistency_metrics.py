"""Consistency metric helpers for NIS and position NEES."""

from __future__ import annotations

import json
import math
from typing import Iterable, Optional


NIS_EXPECTED_DIMENSIONS: dict[str, int] = {
    "gnss_position": 2,
    "imu_yaw": 1,
    "imu_yaw_rate": 1,
    "imu_longitudinal_accel": 1,
    "imu_acceleration": 2,
    "command_acceleration": 2,
}


MEASUREMENT_NOISE_TUNE_KEYS = frozenset(
    {
        "gnss_position_stddev_m",
        "imu_yaw_stddev_deg",
        "imu_yaw_rate_stddev_radps",
        "imu_accel_stddev_mps2",
    }
)


def position_covariance_2x2_from_matrix(covariance: object) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    """Return the x/y 2x2 covariance block when available and finite."""
    try:
        pxx = float(covariance[0][0])  # type: ignore[index]
        pxy = float(covariance[0][1])  # type: ignore[index]
        pyx = float(covariance[1][0])  # type: ignore[index]
        pyy = float(covariance[1][1])  # type: ignore[index]
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(value) for value in (pxx, pxy, pyx, pyy)):
        return None
    return ((pxx, pxy), (pyx, pyy))


def position_nees(
    x_error_m: object,
    y_error_m: object,
    position_covariance_2x2: object = None,
    covariance_diagonal: object = None,
) -> dict[str, object]:
    """Compute full 2D position NEES, falling back to a labeled diagonal approximation."""
    x_error = optional_float(x_error_m)
    y_error = optional_float(y_error_m)
    if x_error is None or y_error is None:
        return _empty_nees()

    block = _position_covariance_block(position_covariance_2x2)
    if block is not None:
        (pxx, pxy), (pyx, pyy) = block
        det = pxx * pyy - pxy * pyx
        if math.isfinite(det) and abs(det) > 1.0e-12:
            inv00 = pyy / det
            inv01 = -pxy / det
            inv10 = -pyx / det
            inv11 = pxx / det
            value = x_error * (inv00 * x_error + inv01 * y_error) + y_error * (inv10 * x_error + inv11 * y_error)
            if math.isfinite(value) and value >= 0.0:
                return {
                    "position_nees": float(value),
                    "position_nees_diagonal_approx": None,
                    "position_nees_source": "full_2x2",
                    "nees": float(value),
                }

    approx = position_nees_diagonal_approx(x_error, y_error, covariance_diagonal)
    if approx is None:
        return _empty_nees()
    return {
        "position_nees": None,
        "position_nees_diagonal_approx": approx,
        "position_nees_source": "diagonal_approx",
        "nees": approx,
    }


def position_nees_diagonal_approx(x_error: float, y_error: float, covariance_diagonal: object) -> Optional[float]:
    if not isinstance(covariance_diagonal, (list, tuple)) or len(covariance_diagonal) < 2:
        return None
    var_x = optional_float(covariance_diagonal[0])
    var_y = optional_float(covariance_diagonal[1])
    if var_x is None or var_y is None or var_x <= 1.0e-9 or var_y <= 1.0e-9:
        return None
    value = x_error * x_error / var_x + y_error * y_error / var_y
    return float(value) if math.isfinite(value) and value >= 0.0 else None


def summarize_values(values: Iterable[object]) -> dict[str, object]:
    finite = [value for value in (optional_float(item) for item in values) if value is not None]
    return {
        "mean": mean(finite),
        "median": percentile(finite, 50.0),
        "p95": percentile(finite, 95.0),
        "p99": percentile(finite, 99.0),
        "sample_count": len(finite),
    }


def summarize_nis_by_type(rows: Iterable[dict[str, object]], field: str = "nis_by_type") -> dict[str, dict[str, object]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        for update_type, value in _dict_value(row.get(field)).items():
            number = optional_float(value)
            if number is not None:
                grouped.setdefault(str(update_type), []).append(number)
    return {
        update_type: {
            **summarize_values(values),
            "expected_dimension": NIS_EXPECTED_DIMENSIONS.get(update_type),
        }
        for update_type, values in sorted(grouped.items())
    }


def summarize_position_nees(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    row_list = list(rows)
    full = summarize_values(row.get("position_nees") for row in row_list)
    approx = summarize_values(row.get("position_nees_diagonal_approx") for row in row_list)
    legacy = summarize_values(row.get("nees") for row in row_list)
    source = "unavailable"
    if full["sample_count"]:
        source = "full_2x2"
    elif approx["sample_count"]:
        source = "diagonal_approx"
    return {
        "position_nees_summary": full,
        "position_nees_diagonal_approx_summary": approx,
        "legacy_nees_summary": legacy,
        "position_nees_source": source,
        "mean_position_nees": full["mean"],
        "mean_position_nees_diagonal_approx": approx["mean"],
        "mean_nees": legacy["mean"],
        "nees_available": bool(legacy["sample_count"]),
        "position_nees_available": bool(full["sample_count"]),
        "position_nees_diagonal_approx_available": bool(approx["sample_count"]),
    }


def optional_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values: list[float], percentile_value: float) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    index = int(math.ceil((percentile_value / 100.0) * len(sorted_values))) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def _empty_nees() -> dict[str, object]:
    return {
        "position_nees": None,
        "position_nees_diagonal_approx": None,
        "position_nees_source": "unavailable",
        "nees": None,
    }


def _position_covariance_block(value: object) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        value = parsed
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        row0 = value[0]
        row1 = value[1]
        if not isinstance(row0, (list, tuple)) or not isinstance(row1, (list, tuple)):
            return None
        pxx = float(row0[0])
        pxy = float(row0[1])
        pyx = float(row1[0])
        pyy = float(row1[1])
    except (TypeError, ValueError, IndexError):
        return None
    if pxx <= 1.0e-12 or pyy <= 1.0e-12:
        return None
    if not all(math.isfinite(item) for item in (pxx, pxy, pyx, pyy)):
        return None
    return ((pxx, pxy), (pyx, pyy))


def _dict_value(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}

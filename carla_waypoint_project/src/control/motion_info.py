"""Optional model-aware motion information for autonomous control."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional


@dataclass(frozen=True)
class MotionInfo:
    """Extra motion-model data that may be supplied by advanced filters.

    ``EgoState`` remains the mandatory controller input.  This dataclass is an
    optional side channel; callers and controllers must tolerate it being absent.
    """

    source_filter_id: str = ""
    model_type: str = ""
    yaw_rate_radps: Optional[float] = None
    longitudinal_accel_mps2: Optional[float] = None
    lateral_accel_mps2: Optional[float] = None
    curvature_1pm: Optional[float] = None
    confidence: Optional[float] = None
    covariance_diagonal: Optional[tuple[float, ...]] = None


def motion_info_from_diagnostics(diagnostics: dict[str, Any]) -> Optional[MotionInfo]:
    """Build ``MotionInfo`` from filter diagnostics when useful fields exist."""
    if not isinstance(diagnostics, dict):
        return None

    filter_id = str(diagnostics.get("filter_id") or "")
    model_type = str(diagnostics.get("model_type") or _model_type_from_filter_id(filter_id))
    yaw_rate = _finite_or_none(diagnostics.get("yaw_rate_radps"))
    longitudinal_accel = _finite_or_none(
        diagnostics.get("longitudinal_accel_mps2", diagnostics.get("acceleration_mps2"))
    )
    lateral_accel = _finite_or_none(diagnostics.get("lateral_accel_mps2"))
    curvature = _finite_or_none(diagnostics.get("curvature_1pm"))
    confidence = _finite_or_none(diagnostics.get("confidence"))
    covariance_diagonal = _finite_tuple_or_none(diagnostics.get("covariance_diagonal"))

    if filter_id == "ca_kf" and longitudinal_accel is None:
        longitudinal_accel = _finite_or_none(diagnostics.get("latest_command_longitudinal_accel_mps2"))

    if yaw_rate is None and longitudinal_accel is None and lateral_accel is None and curvature is None:
        return None

    return MotionInfo(
        source_filter_id=filter_id,
        model_type=model_type,
        yaw_rate_radps=yaw_rate,
        longitudinal_accel_mps2=longitudinal_accel,
        lateral_accel_mps2=lateral_accel,
        curvature_1pm=curvature,
        confidence=confidence,
        covariance_diagonal=covariance_diagonal,
    )


def _model_type_from_filter_id(filter_id: str) -> str:
    if filter_id == "ctrv_ekf":
        return "CTRV"
    if filter_id == "ctra_ekf":
        return "CTRA"
    if filter_id == "ca_kf":
        return "CA"
    if filter_id == "cv_kf":
        return "CV"
    if filter_id == "ego_kinematic_ekf":
        return "EGO_KINEMATIC"
    if filter_id == "raw_gnss":
        return "RAW_GNSS"
    return filter_id


def _finite_or_none(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_tuple_or_none(value: object) -> Optional[tuple[float, ...]]:
    if not isinstance(value, (tuple, list)):
        return None
    result: list[float] = []
    for item in value:
        number = _finite_or_none(item)
        if number is None:
            return None
        result.append(number)
    return tuple(result)

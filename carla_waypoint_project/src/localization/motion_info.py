"""Optional model-aware motion information produced by localization filters."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional


@dataclass(frozen=True)
class MotionInfo:
    """Extra motion-model data that may accompany an ``EgoState`` estimate."""

    source_filter_id: str = ""
    model_type: str = ""
    yaw_rate_radps: Optional[float] = None
    longitudinal_accel_mps2: Optional[float] = None
    lateral_accel_mps2: Optional[float] = None
    curvature_1pm: Optional[float] = None
    confidence: Optional[float] = None
    covariance_diagonal: Optional[tuple[float, ...]] = None


def motion_info_from_diagnostics(
    diagnostics: dict[str, Any],
    filter_info: Optional[dict[str, Any]] = None,
) -> Optional[MotionInfo]:
    """Build ``MotionInfo`` from plugin diagnostics and metadata.

    Model type is supplied by the plugin via diagnostics or ``FILTER_INFO``.
    ``filter_id`` is kept only as source identity; it is not interpreted here.
    """
    if not isinstance(diagnostics, dict):
        return None
    info = filter_info if isinstance(filter_info, dict) else {}

    filter_id = str(diagnostics.get("filter_id") or info.get("id") or "")
    model_type = str(diagnostics.get("model_type") or info.get("model_type") or "")
    yaw_rate = _finite_or_none(diagnostics.get("yaw_rate_radps"))
    longitudinal_accel = _finite_or_none(
        diagnostics.get("longitudinal_accel_mps2", diagnostics.get("acceleration_mps2"))
    )
    lateral_accel = _finite_or_none(diagnostics.get("lateral_accel_mps2"))
    curvature = _finite_or_none(diagnostics.get("curvature_1pm"))
    confidence = _finite_or_none(diagnostics.get("confidence"))
    covariance_diagonal = _finite_tuple_or_none(diagnostics.get("covariance_diagonal"))

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

"""No-server checks for VehicleState capability handling in control."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.control.driving_behavior import DrivingBehaviorConfig
from src.control.vehicle_controller import VehicleController
from src.core.vehicle_state import VehicleState


@dataclass(frozen=True)
class _Location:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class _Transform:
    location: _Location


@dataclass(frozen=True)
class _Waypoint:
    transform: _Transform


def _waypoint(x: float, y: float) -> _Waypoint:
    return _Waypoint(_Transform(_Location(x, y)))


def _state(**overrides: object) -> VehicleState:
    values = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "yaw": 0.0,
        "speed": 5.0,
        "timestamp": 1.0,
        "source_filter_id": "test_filter",
        "model_type": "TEST",
    }
    values.update(overrides)
    return VehicleState(**values)


def _controller(config: DrivingBehaviorConfig | None = None) -> VehicleController:
    return VehicleController(behavior_config=config or DrivingBehaviorConfig())


def test_basic_state_falls_back_without_model_fields() -> None:
    controller = _controller()
    control = controller.compute_control(_state(), _waypoint(12.0, 0.0), target_speed_mps=6.0)
    diagnostics = controller.latest_model_control_diagnostics
    if diagnostics["model_state_used"]:
        raise AssertionError("basic state unexpectedly used model-aware control")
    if diagnostics["model_state_ignored_reason"] != "model-aware control disabled":
        raise AssertionError(f"unexpected reason: {diagnostics['model_state_ignored_reason']}")
    if abs(control.steer) > 1.0e-9:
        raise AssertionError(f"straight target produced steer {control.steer}")


def test_curvature_field_is_preferred_when_enabled() -> None:
    config = DrivingBehaviorConfig(enable_model_aware_control=1.0, model_state_lowpass_alpha=1.0)
    controller = _controller(config)
    controller.compute_control(
        _state(curvature_1pm=0.04),
        _waypoint(12.0, 0.0),
        target_speed_mps=6.0,
    )
    diagnostics = controller.latest_model_control_diagnostics
    if not diagnostics["model_state_used"]:
        raise AssertionError(f"curvature was ignored: {diagnostics}")
    if diagnostics["state_used_fields"] != ("curvature_1pm",):
        raise AssertionError(f"unexpected used fields: {diagnostics['state_used_fields']}")


def test_yaw_rate_derives_curvature_when_curvature_missing() -> None:
    config = DrivingBehaviorConfig(enable_model_aware_control=1.0, model_state_lowpass_alpha=1.0)
    controller = _controller(config)
    controller.compute_control(
        _state(yaw_rate_radps=0.2),
        _waypoint(12.0, 0.0),
        target_speed_mps=6.0,
    )
    diagnostics = controller.latest_model_control_diagnostics
    if diagnostics["state_used_fields"] != ("yaw_rate_radps", "speed"):
        raise AssertionError(f"yaw-rate fallback not used: {diagnostics}")


def test_unsafe_state_rejected_for_model_aware_control() -> None:
    config = DrivingBehaviorConfig(enable_model_aware_control=1.0)
    controller = _controller(config)
    controller.compute_control(
        _state(curvature_1pm=0.04, safe_for_autonomous_control=False),
        _waypoint(12.0, 0.0),
        target_speed_mps=6.0,
    )
    diagnostics = controller.latest_model_control_diagnostics
    if diagnostics["model_state_used"]:
        raise AssertionError("unsafe state was used")
    if diagnostics["model_state_ignored_reason"] != "state source not marked safe for autonomous control":
        raise AssertionError(f"unexpected reason: {diagnostics['model_state_ignored_reason']}")


def test_non_finite_optional_values_are_sanitized() -> None:
    state = _state(
        yaw_rate_radps=float("nan"),
        curvature_1pm=float("inf"),
        acceleration_mps2=-math.inf,
    )
    if state.has_yaw_rate or state.has_curvature or state.has_acceleration:
        raise AssertionError(f"non-finite optional fields survived: {state}")


def run_all() -> None:
    test_basic_state_falls_back_without_model_fields()
    test_curvature_field_is_preferred_when_enabled()
    test_yaw_rate_derives_curvature_when_curvature_missing()
    test_unsafe_state_rejected_for_model_aware_control()
    test_non_finite_optional_values_are_sanitized()


if __name__ == "__main__":
    run_all()
    print("vehicle state controller checks passed")

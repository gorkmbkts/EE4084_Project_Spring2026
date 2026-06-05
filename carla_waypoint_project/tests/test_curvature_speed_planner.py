"""No-server checks for waypoint-only route curvature scoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.control.driving_behavior import CurvatureSpeedPlanner, DrivingBehaviorConfig
from src.core.vehicle_state import VehicleState


@dataclass(frozen=True)
class _Location:
    x: float
    y: float


@dataclass(frozen=True)
class _Transform:
    location: _Location


@dataclass(frozen=True)
class _Waypoint:
    transform: _Transform


def _waypoint(x: float, y: float) -> _Waypoint:
    return _Waypoint(_Transform(_Location(x, y)))


def _state(x: float = 0.0, y: float = 0.0) -> VehicleState:
    return VehicleState(x=x, y=y, z=0.0, yaw=0.0, speed=5.0, timestamp=1.0)


def _planner() -> CurvatureSpeedPlanner:
    config = DrivingBehaviorConfig(
        max_speed_mps=8.0,
        min_curve_speed_mps=2.0,
        curve_lookahead_m=30.0,
        curvature_sensitivity=1.0,
        speed_change_aggressiveness=10.0,
    )
    return CurvatureSpeedPlanner(config)


def test_straight_route_ignores_laterally_offset_state() -> None:
    planner = _planner()
    route = [_waypoint(float(x), 0.0) for x in range(0, 40, 5)]
    plan = planner.plan(
        state=_state(x=0.0, y=4.0),
        preview_waypoints=route,
        route_completed=False,
        dt_seconds=1.0,
    )
    if plan.mode != "STRAIGHT":
        raise AssertionError(f"straight route classified as {plan.mode}")
    if plan.curvature_score >= 0.18:
        raise AssertionError(f"straight route curvature too high: {plan.curvature_score}")


def test_curved_route_scores_higher_than_straight() -> None:
    planner = _planner()
    curved = [
        _waypoint(0.0, 0.0),
        _waypoint(5.0, 0.0),
        _waypoint(10.0, 1.0),
        _waypoint(14.0, 4.0),
        _waypoint(17.0, 8.0),
        _waypoint(19.0, 13.0),
    ]
    plan = planner.plan(_state(), curved, route_completed=False, dt_seconds=1.0)
    if plan.curvature_score <= 0.18:
        raise AssertionError(f"curved route score too low: {plan.curvature_score}")
    if plan.curvature_rad_per_m <= 0.0:
        raise AssertionError(f"curved route curvature not positive: {plan.curvature_rad_per_m}")


def test_duplicate_waypoints_do_not_create_false_curvature() -> None:
    planner = _planner()
    route = [
        _waypoint(0.0, 0.0),
        _waypoint(0.01, 0.0),
        _waypoint(5.0, 0.0),
        _waypoint(5.02, 0.0),
        _waypoint(10.0, 0.0),
        _waypoint(15.0, 0.0),
    ]
    plan = planner.plan(_state(y=3.0), route, route_completed=False, dt_seconds=1.0)
    if plan.mode != "STRAIGHT":
        raise AssertionError(f"duplicate straight route classified as {plan.mode}")
    if not math.isfinite(plan.lookahead_distance_m) or plan.lookahead_distance_m <= 0.0:
        raise AssertionError(f"invalid lookahead distance: {plan.lookahead_distance_m}")


def test_too_short_preview_returns_zero_curvature() -> None:
    planner = _planner()
    plan = planner.plan(
        state=_state(),
        preview_waypoints=[_waypoint(0.0, 0.0), _waypoint(1.0, 0.0), _waypoint(2.0, 0.0)],
        route_completed=False,
        dt_seconds=1.0,
    )
    if plan.curvature_score != 0.0 or plan.curvature_rad_per_m != 0.0:
        raise AssertionError(f"short preview should be zero curvature: {plan}")
    if plan.mode != "STRAIGHT":
        raise AssertionError(f"short preview classified as {plan.mode}")


def run_all() -> None:
    test_straight_route_ignores_laterally_offset_state()
    test_curved_route_scores_higher_than_straight()
    test_duplicate_waypoints_do_not_create_false_curvature()
    test_too_short_preview_returns_zero_curvature()


if __name__ == "__main__":
    run_all()
    print("curvature speed planner checks passed")

"""Small sanity checks for CTRV/CTRA EKF motion-model helpers."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.KalmanLab.filters import ctra_ekf, ctrv_ekf
from src.KalmanLab.registry import discover_filters
from src.localization.gnss_projection import LocalGnssMeasurement
from src.localization.motion_info import motion_info_from_diagnostics


def assert_close(actual: float, expected: float, tolerance: float = 1.0e-6) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual!r} != {expected!r} within {tolerance}")


def test_angle_wrapping() -> None:
    assert_close(ctrv_ekf.normalize_angle_rad(3.0 * math.pi), math.pi)
    assert_close(ctra_ekf.normalize_angle_deg(540.0), 180.0)


def test_ctrv_straight_motion() -> None:
    state = [1.0, 2.0, 0.0, 4.0, 0.0]
    predicted = ctrv_ekf.process_model(state, dt=0.5, turn_rate_epsilon_radps=1.0e-4)
    assert_close(predicted[0], 3.0)
    assert_close(predicted[1], 2.0)
    assert_close(predicted[2], 0.0)
    assert_close(predicted[3], 4.0)


def test_ctrv_turning_prediction() -> None:
    state = [0.0, 0.0, 0.0, 10.0, 0.5]
    predicted = ctrv_ekf.process_model(state, dt=1.0, turn_rate_epsilon_radps=1.0e-4)
    assert_close(predicted[0], 10.0 / 0.5 * math.sin(0.5), 1.0e-6)
    assert_close(predicted[1], 10.0 / 0.5 * (-math.cos(0.5) + 1.0), 1.0e-6)
    assert_close(predicted[2], 0.5)


def test_ctra_zero_yaw_rate_positive_accel() -> None:
    state = [0.0, 0.0, 0.0, 2.0, 1.5, 0.0]
    predicted = ctra_ekf.process_model(state, dt=2.0, turn_rate_epsilon_radps=1.0e-4)
    assert_close(predicted[0], 2.0 * 2.0 + 0.5 * 1.5 * 4.0)
    assert_close(predicted[1], 0.0)
    assert_close(predicted[3], 5.0)
    assert_close(predicted[4], 1.5)


def test_wrapped_yaw_update() -> None:
    core = ctrv_ekf._CtrvEkfCore(dict(ctrv_ekf.TUNE))
    core.initialize((0.0, 0.0), math.radians(179.0), 0.0, 0.0, 0.0)
    core.update_yaw(math.radians(-179.0))
    if core.last_innovation is None:
        raise AssertionError("yaw update did not store innovation")
    assert_close(math.degrees(core.last_innovation[0]), 2.0, 1.0e-6)


def test_gyro_yaw_rate_update() -> None:
    core = ctrv_ekf._CtrvEkfCore(dict(ctrv_ekf.TUNE))
    core.initialize((0.0, 0.0), 0.0, 0.0, 0.0, 0.0)
    core.update_yaw_rate(0.3)
    if core.last_update_type != "imu_yaw_rate":
        raise AssertionError(f"unexpected update type: {core.last_update_type}")
    assert core.snapshot() is not None
    assert core.snapshot().yaw_rate_radps > 0.0


def test_filter_wrappers_initialize_from_gnss_and_imu() -> None:
    class Projector:
        def project(self, gnss: object) -> LocalGnssMeasurement:
            return LocalGnssMeasurement(
                x=float(gnss.x),
                y=float(gnss.y),
                z=0.0,
                latitude=0.0,
                longitude=0.0,
                altitude=0.0,
                frame=int(gnss.frame),
                timestamp=float(gnss.timestamp),
            )

    imu = SimpleNamespace(
        accelerometer=(1.2, 0.0, 0.0),
        gyroscope=(0.0, 0.0, 0.2),
        compass=math.pi / 2.0,
        frame=1,
        timestamp=0.0,
    )
    gnss = SimpleNamespace(x=5.0, y=2.0, frame=2, timestamp=0.01)
    for module in (ctrv_ekf, ctra_ekf):
        filt = module.Filter(Projector())
        filt.process_imu(imu)
        state = filt.process_gnss(gnss)
        if state is None:
            raise AssertionError(f"{module.__name__} did not initialize")
        diagnostics = filt.get_diagnostics()
        info = motion_info_from_diagnostics(diagnostics)
        if info is None or info.yaw_rate_radps is None:
            raise AssertionError(f"{module.__name__} did not expose MotionInfo")


def test_registry_capabilities() -> None:
    records = {record.filter_id: record for record in discover_filters() if record.valid}
    for filter_id in ("ctrv_ekf", "ctra_ekf"):
        record = records.get(filter_id)
        if record is None:
            raise AssertionError(f"{filter_id} not discovered")
        if not record.benchmark_selectable:
            raise AssertionError(f"{filter_id} not benchmark selectable")
        if not record.active_tracking_supported:
            raise AssertionError(f"{filter_id} does not advertise active tracking")
        if record.filter_class is None or not hasattr(record.filter_class, "process_control"):
            raise AssertionError(f"{filter_id} has no process_control")


def test_startup_setup_does_not_hide_experimental_filters() -> None:
    startup_source = (PROJECT_ROOT / "src" / "visualization" / "startup_map_selector.py").read_text(encoding="utf-8")
    if "record.valid and record.safe_for_autonomous_control" in startup_source:
        raise AssertionError("startup setup still filters by autonomous safety")
    if "record.valid" not in startup_source:
        raise AssertionError("startup setup no longer filters valid plugins")


def test_motion_info_uses_plugin_model_type() -> None:
    info = motion_info_from_diagnostics(
        {"filter_id": "new_plugin", "yaw_rate_radps": 0.4},
        {"id": "new_plugin", "model_type": "BICYCLE_EKF"},
    )
    if info is None:
        raise AssertionError("motion info was not extracted")
    if info.model_type != "BICYCLE_EKF":
        raise AssertionError(f"unexpected model type: {info.model_type}")


def test_filter_control_input_avoids_ground_truth_speed_yaw() -> None:
    app_source = (PROJECT_ROOT / "src" / "core" / "app.py").read_text(encoding="utf-8")
    start = app_source.index("def _feed_filter_control_input")
    end = app_source.index("    def _filter_control_timestamp", start)
    body = app_source[start:end]
    if "_latest_ground_truth_state" in body:
        raise AssertionError("_feed_filter_control_input references ground truth")
    if "_latest_estimated_state" not in body:
        raise AssertionError("_feed_filter_control_input does not use estimated state")


def run_all() -> None:
    test_angle_wrapping()
    test_ctrv_straight_motion()
    test_ctrv_turning_prediction()
    test_ctra_zero_yaw_rate_positive_accel()
    test_wrapped_yaw_update()
    test_gyro_yaw_rate_update()
    test_filter_wrappers_initialize_from_gnss_and_imu()
    test_registry_capabilities()
    test_startup_setup_does_not_hide_experimental_filters()
    test_motion_info_uses_plugin_model_type()
    test_filter_control_input_avoids_ground_truth_speed_yaw()


if __name__ == "__main__":
    run_all()
    print("motion EKF sanity checks passed")

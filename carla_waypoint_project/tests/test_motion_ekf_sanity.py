"""Small sanity checks for CTRV/CTRA EKF motion-model helpers."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.KalmanLab.filters import ca_kf, ctra_ekf, ctra_ukf, ctrv_ekf, cv_kf
from src.KalmanLab.filter_base import FilterControlInput
from src.KalmanLab.control_model import estimate_command_motion
from src.KalmanLab.registry import discover_filters
from src.evaluation.benchmark_config import SensorNoiseConfig
from src.evaluation.filter_auto_tuner import AutoTuneRequest, FilterAutoTuner
from src.evaluation.offline_replay_runner import OfflineReplayRequest, OfflineReplayRunner
from src.evaluation.sensor_noise_tune_mapper import SensorNoiseTuneMapper
from src.localization.gnss_projection import LocalGnssMeasurement


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


def test_ctra_ukf_zero_yaw_rate_positive_accel() -> None:
    state = [0.0, 0.0, 0.0, 2.0, 1.5, 0.0]
    predicted = ctra_ukf.process_model(state, dt=2.0, turn_rate_epsilon_radps=1.0e-4)
    assert_close(predicted[0], 2.0 * 2.0 + 0.5 * 1.5 * 4.0)
    assert_close(predicted[1], 0.0)
    assert_close(predicted[3], 5.0)
    assert_close(predicted[4], 1.5)


def test_ctra_ukf_core_predict_update_is_finite() -> None:
    core = ctra_ukf._CtraUkfCore(dict(ctra_ukf.TUNE))
    core.initialize((0.0, 0.0), 0.1, 4.0, 0.5, 0.05, 0.0)
    core.predict(0.1, timestamp=0.1)
    core.update_gnss_position((0.45, 0.05))
    core.update_yaw_rate(0.06)
    core.update_acceleration(0.4)
    snapshot = core.snapshot()
    if snapshot is None:
        raise AssertionError("CTRA UKF snapshot missing after predict/update")
    values = [
        snapshot.px,
        snapshot.py,
        snapshot.yaw_rad,
        snapshot.speed,
        snapshot.acceleration_mps2,
        snapshot.yaw_rate_radps,
        *core.state_vector.reshape(-1),
        *core.covariance.reshape(-1),
    ]
    if not all(math.isfinite(float(value)) for value in values):
        raise AssertionError("CTRA UKF produced non-finite state or covariance")
    if "gnss_position" not in core.nis_by_type:
        raise AssertionError("CTRA UKF did not store GNSS NIS")


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
    for module in (ctrv_ekf, ctra_ekf, ctra_ukf):
        filt = module.Filter(Projector())
        filt.process_imu(imu)
        state = filt.process_gnss(gnss)
        if state is None:
            raise AssertionError(f"{module.__name__} did not initialize")
        if state.yaw_rate_radps is None:
            raise AssertionError(f"{module.__name__} did not expose yaw-rate state")
        if state.curvature_1pm is not None and not math.isfinite(state.curvature_1pm):
            raise AssertionError(f"{module.__name__} exposed non-finite curvature")
        if module in (ctra_ekf, ctra_ukf) and state.longitudinal_accel_mps2 is None:
            raise AssertionError(f"{module.__name__} did not expose longitudinal acceleration")
        if module is ctra_ukf and state.source_filter_id != "ctra_ukf":
            raise AssertionError("CTRA UKF state did not use the plugin filter id")


def test_active_command_affects_immediate_motion_prediction() -> None:
    class Projector:
        pass

    command = FilterControlInput(
        timestamp=0.0,
        throttle=1.0,
        steer=0.0,
        brake=0.0,
        hand_brake=False,
        reverse=False,
        source="test",
        speed_mps=0.0,
        yaw_deg=0.0,
    )
    for module in (ctra_ekf, ctra_ukf, ctrv_ekf):
        active = module.Filter(Projector(), tracking_mode="active")
        disabled = module.Filter(
            Projector(),
            tune={"enable_control_input_prediction": 0.0},
            tracking_mode="active",
        )
        if module is ctrv_ekf:
            active._filter.initialize((0.0, 0.0), 0.0, 0.0, 0.0, 0.0)
            disabled._filter.initialize((0.0, 0.0), 0.0, 0.0, 0.0, 0.0)
        else:
            active._filter.initialize((0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0)
            disabled._filter.initialize((0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0)
        if not active.process_control(command):
            raise AssertionError(f"{module.__name__} rejected enabled active input")
        if disabled.process_control(command):
            raise AssertionError(f"{module.__name__} accepted disabled active input")

        active._predict_to(0.1)
        disabled._predict_to(0.1)
        active_snapshot = active._filter.snapshot()
        disabled_snapshot = disabled._filter.snapshot()
        if active_snapshot is None or disabled_snapshot is None:
            raise AssertionError(f"{module.__name__} did not produce prediction snapshots")
        if active_snapshot.px <= disabled_snapshot.px + 1.0e-6:
            raise AssertionError(
                f"{module.__name__} command did not affect position in the immediate prediction interval"
            )


def test_linear_filters_honor_disabled_active_prediction_switch() -> None:
    class Projector:
        pass

    command = FilterControlInput(
        timestamp=0.0,
        throttle=1.0,
        steer=0.0,
        brake=0.0,
        hand_brake=False,
        reverse=False,
        source="test",
    )
    for module in (ca_kf, cv_kf):
        active = module.Filter(Projector(), tracking_mode="active")
        disabled = module.Filter(
            Projector(),
            tune={"enable_control_input_prediction": 0.0},
            tracking_mode="active",
        )
        if not active.process_control(command) or disabled.process_control(command):
            raise AssertionError(f"{module.__name__} did not honor the active prediction switch")


def test_zero_command_yaw_cap_disables_steering_component() -> None:
    command = FilterControlInput(
        timestamp=0.0,
        throttle=1.0,
        steer=1.0,
        brake=0.0,
        hand_brake=False,
        reverse=False,
        source="test",
    )
    estimate = estimate_command_motion(
        command,
        speed_mps=10.0,
        yaw_deg=0.0,
        tune={
            "command_throttle_accel_gain_mps2": 3.0,
            "command_brake_decel_gain_mps2": 6.0,
            "command_max_accel_mps2": 8.0,
            "command_max_yaw_rate_dps": 0.0,
        },
        dt_s=0.1,
    )
    assert_close(estimate.yaw_rate_dps, 0.0)
    assert_close(estimate.lateral_accel_mps2, 0.0)
    assert_close(estimate.acceleration_xy[1], 0.0)


def test_ca_kf_ignores_startup_imu_acceleration_spike() -> None:
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

    filt = ca_kf.Filter(Projector(), tune={"max_valid_imu_accel_mps2": 20.0})
    filt.process_imu(
        SimpleNamespace(
            accelerometer=(100.0, 0.0, 0.0),
            gyroscope=(0.0, 0.0, 0.0),
            compass=math.pi / 2.0,
            frame=1,
            timestamp=0.0,
        )
    )
    state = filt.process_gnss(SimpleNamespace(x=1.0, y=2.0, frame=2, timestamp=0.05))
    if state is None:
        raise AssertionError("CA-KF did not initialize from GNSS")
    if state.acceleration_mps2 > 1.0e-9:
        raise AssertionError("CA-KF initialized acceleration from transient IMU")
    diagnostics = filt.get_diagnostics()
    if diagnostics.get("imu_accel_update_skipped_count", 0) < 1:
        raise AssertionError("CA-KF did not count skipped acceleration spike")
    if diagnostics.get("imu_accel_update_skipped_latest_reason") != "imu_accel_magnitude_gate":
        raise AssertionError("CA-KF skip reason missing acceleration gate")


def test_registry_capabilities() -> None:
    records = {record.filter_id: record for record in discover_filters() if record.valid}
    for filter_id in ("ctrv_ekf", "ctra_ekf", "ctra_ukf"):
        record = records.get(filter_id)
        if record is None:
            raise AssertionError(f"{filter_id} not discovered")
        if not record.benchmark_selectable:
            raise AssertionError(f"{filter_id} not benchmark selectable")
        if not record.active_tracking_supported:
            raise AssertionError(f"{filter_id} does not advertise active tracking")
        provided = set(record.provided_state_fields)
        if "yaw_rate_radps" not in provided:
            raise AssertionError(f"{filter_id} does not advertise yaw-rate state")
        if record.filter_class is None or not hasattr(record.filter_class, "process_control"):
            raise AssertionError(f"{filter_id} has no process_control")
    ctra_ukf_record = records["ctra_ukf"]
    if not ctra_ukf_record.auto_tune_enabled or not ctra_ukf_record.auto_tune_profile:
        raise AssertionError("CTRA UKF does not expose an auto-tune profile")
    tunable_keys = {str(getattr(spec, "key", "")) for spec in ctra_ukf_record.tune_specs}
    for key in ("process_position_stddev_m", "process_accel_stddev_mps2", "process_yaw_accel_stddev_radps2", "gnss_position_stddev_m"):
        if key not in tunable_keys:
            raise AssertionError(f"CTRA UKF missing tunable parameter: {key}")


def test_ctra_ukf_autotune_candidate_generation() -> None:
    records = {record.filter_id: record for record in discover_filters() if record.valid}
    record = records["ctra_ukf"]
    locked = SensorNoiseTuneMapper.locked_values(
        "ctra_ukf",
        dict(record.tune),
        SensorNoiseConfig().to_dict(),
        tuple(record.tune_specs),
    )
    if "gnss_position_stddev_m" not in locked.values:
        raise AssertionError("CTRA UKF measurement noise is not lockable from sensor-noise config")
    request = AutoTuneRequest(
        filter_id="ctra_ukf",
        sensor_log_paths=(Path("synthetic_sensor_log.csv"),),
        base_tune=dict(record.tune),
        auto_tune_profile=dict(record.auto_tune_profile or {}),
        max_trials=2,
        metadata={"random_seed": 4084},
    )
    candidates = FilterAutoTuner()._generated_candidates(request, max_trials=2, strategy="random_plus_coordinate_refinement")
    if not candidates:
        raise AssertionError("CTRA UKF auto-tuner did not generate candidates")
    if not any(candidate.tune != record.tune for candidate in candidates):
        raise AssertionError("CTRA UKF auto-tuner candidates did not change tune values")


def test_ctra_ukf_short_offline_replay_smoke(tmp_path: Path) -> None:
    log_path = tmp_path / "sensor_log.csv"
    headers = [
        "timestamp",
        "frame",
        "valid_for_metrics",
        "gnss_local_x",
        "gnss_local_y",
        "gnss_local_z",
        "gnss_timestamp",
        "gnss_frame",
        "imu_accel_x",
        "imu_accel_y",
        "imu_accel_z",
        "imu_gyro_x",
        "imu_gyro_y",
        "imu_gyro_z",
        "imu_compass",
        "imu_timestamp",
        "imu_frame",
        "ground_truth_x",
        "ground_truth_y",
        "ground_truth_z",
        "ground_truth_yaw",
        "ground_truth_speed",
        "ground_truth_vx_mps",
        "ground_truth_vy_mps",
    ]
    lines = [",".join(headers)]
    for index in range(8):
        timestamp = index * 0.1
        x = 2.0 * timestamp
        row = {
            "timestamp": timestamp,
            "frame": index,
            "valid_for_metrics": "true",
            "gnss_local_x": x,
            "gnss_local_y": 0.0,
            "gnss_local_z": 0.0,
            "gnss_timestamp": timestamp,
            "gnss_frame": index,
            "imu_accel_x": 0.0,
            "imu_accel_y": 0.0,
            "imu_accel_z": 0.0,
            "imu_gyro_x": 0.0,
            "imu_gyro_y": 0.0,
            "imu_gyro_z": 0.0,
            "imu_compass": math.pi / 2.0,
            "imu_timestamp": timestamp,
            "imu_frame": index,
            "ground_truth_x": x,
            "ground_truth_y": 0.0,
            "ground_truth_z": 0.0,
            "ground_truth_yaw": 0.0,
            "ground_truth_speed": 2.0,
            "ground_truth_vx_mps": 2.0,
            "ground_truth_vy_mps": 0.0,
        }
        lines.append(",".join(str(row[key]) for key in headers))
    log_path.write_text("\n".join(lines), encoding="utf-8")

    result = OfflineReplayRunner().run(
        OfflineReplayRequest(
            sensor_log_paths=(log_path,),
            selected_filter_ids=("ctra_ukf",),
            output_root=str(tmp_path / "out"),
            include_raw_gnss_baseline=True,
            generate_plots=False,
        )
    )
    if result.failures:
        raise AssertionError(f"CTRA UKF offline replay failed: {result.failures}")
    if result.best_filter_id != "ctra_ukf":
        raise AssertionError(f"CTRA UKF missing from offline replay result: {result.best_filter_id}")


def test_startup_setup_does_not_hide_experimental_filters() -> None:
    startup_source = (PROJECT_ROOT / "src" / "visualization" / "startup_map_selector.py").read_text(encoding="utf-8")
    if "record.valid and record.safe_for_autonomous_control" in startup_source:
        raise AssertionError("startup setup still filters by autonomous safety")
    if "record.valid" not in startup_source:
        raise AssertionError("startup setup no longer filters valid plugins")


def test_state_uses_plugin_model_type() -> None:
    state = ctra_ekf.Filter(type("Projector", (), {"project": lambda self, gnss: None})()).get_state()
    if state is not None:
        raise AssertionError("new filter instance unexpectedly had state")
    records = {record.filter_id: record for record in discover_filters() if record.valid}
    if records["ctra_ekf"].filter_info.get("model_type") != "CTRA":
        raise AssertionError("CTRA registry model type missing")


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
    test_ctra_ukf_zero_yaw_rate_positive_accel()
    test_ctra_ukf_core_predict_update_is_finite()
    test_wrapped_yaw_update()
    test_gyro_yaw_rate_update()
    test_filter_wrappers_initialize_from_gnss_and_imu()
    test_ca_kf_ignores_startup_imu_acceleration_spike()
    test_registry_capabilities()
    test_ctra_ukf_autotune_candidate_generation()
    test_startup_setup_does_not_hide_experimental_filters()
    test_state_uses_plugin_model_type()
    test_filter_control_input_avoids_ground_truth_speed_yaw()


if __name__ == "__main__":
    run_all()
    print("motion EKF sanity checks passed")

"""Single-run localization filter benchmark logging and summary metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Optional

from config.settings import BENCHMARK
from src.core.vehicle_state import VehicleState
from src.control.driving_behavior import SpeedPlan
from src.control.waypoint_tracker import TrackingStatus
from src.evaluation.consistency_metrics import (
    position_nees,
    summarize_nis_by_type,
    summarize_position_nees,
)
from src.localization.gnss_projection import GnssDiagnostics


@dataclass(frozen=True)
class FilterPerformanceSample:
    """One frame of active-filter benchmark metrics."""

    timestamp: float
    route_name: str
    phase: str
    valid_for_metrics: bool
    warmup_excluded_reason: str
    ground_truth_x: Optional[float]
    ground_truth_y: Optional[float]
    ground_truth_yaw: Optional[float]
    ground_truth_speed: Optional[float]
    filtered_x: Optional[float]
    filtered_y: Optional[float]
    filtered_yaw: Optional[float]
    filtered_speed: Optional[float]
    filtered_vx_mps: Optional[float]
    filtered_vy_mps: Optional[float]
    filtered_yaw_rate_radps: Optional[float]
    filtered_curvature_1pm: Optional[float]
    filtered_acceleration_mps2: Optional[float]
    filtered_longitudinal_accel_mps2: Optional[float]
    filtered_lateral_accel_mps2: Optional[float]
    state_source_filter_id: str
    state_model_type: str
    state_capabilities: str
    state_confidence: Optional[float]
    state_safe_for_autonomous_control: Optional[bool]
    kalman_x: Optional[float]
    kalman_y: Optional[float]
    kalman_yaw: Optional[float]
    kalman_speed: Optional[float]
    gnss_x: Optional[float]
    gnss_y: Optional[float]
    filtered_position_error_m: Optional[float]
    kalman_position_error_m: Optional[float]
    raw_gnss_error_m: Optional[float]
    cross_track_error_m: Optional[float]
    heading_error_deg: Optional[float]
    distance_to_goal_m: Optional[float]
    closest_index: int
    target_index: int
    route_completed: bool
    x_error_m: Optional[float]
    y_error_m: Optional[float]
    speed_error_mps: Optional[float]
    yaw_error_deg: Optional[float]
    curvature_score: Optional[float]
    curvature_rad_per_m: Optional[float]
    curvature_mode: Optional[str]
    nis: Optional[float]
    nis_by_type: dict[str, float]
    nis_expected_dimensions_by_type: dict[str, int]
    nees: Optional[float]
    position_nees: Optional[float]
    position_nees_diagonal_approx: Optional[float]
    position_nees_source: str
    innovation_norm: Optional[float]
    covariance_x_std_m: Optional[float]
    covariance_y_std_m: Optional[float]
    within_2sigma_x: Optional[bool]
    within_2sigma_y: Optional[bool]
    gnss_update_frame: Optional[int]
    imu_update_frame: Optional[int]


class FilterPerformanceLogger:
    """Collect one benchmark run in memory and export CSV plus JSON summary."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        benchmark_id: str = "",
        active_filter_id: str = "",
        active_filter_name: str = "Active filter",
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self._output_dir = output_dir if output_dir is not None else project_root / "logs" / "filter_tests"
        self._benchmark_id = benchmark_id
        self._active_filter_id = active_filter_id
        self._active_filter_name = active_filter_name
        self._samples: list[FilterPerformanceSample] = []
        self._route_name = ""
        self._started = False
        self._completed = False
        self._aborted = False
        self._timeout = False
        self._abort_reason: Optional[str] = None
        self._last_export_paths: Optional[tuple[Path, Path]] = None
        self._last_nis_update_counts_by_type: dict[str, int] = {}

    @property
    def samples(self) -> tuple[FilterPerformanceSample, ...]:
        return tuple(self._samples)

    @property
    def route_name(self) -> str:
        return self._route_name

    @property
    def current_position_error_m(self) -> Optional[float]:
        return self._samples[-1].filtered_position_error_m if self._samples else None

    @property
    def current_raw_gnss_error_m(self) -> Optional[float]:
        return self._samples[-1].raw_gnss_error_m if self._samples else None

    @property
    def current_cross_track_error_m(self) -> Optional[float]:
        return self._samples[-1].cross_track_error_m if self._samples else None

    @property
    def last_export_paths(self) -> Optional[tuple[Path, Path]]:
        return self._last_export_paths

    def start_route(self, route_name: str, benchmark_id: Optional[str] = None) -> None:
        self._samples = []
        self._route_name = route_name
        if benchmark_id is not None:
            self._benchmark_id = benchmark_id
        self._started = True
        self._completed = False
        self._aborted = False
        self._timeout = False
        self._abort_reason = None
        self._last_export_paths = None
        self._last_nis_update_counts_by_type = {}

    def mark_completed(self) -> None:
        self._completed = True
        self._aborted = False
        self._timeout = False
        self._abort_reason = None

    def mark_aborted(self, reason: Optional[str] = None, timeout: bool = False) -> None:
        self._completed = False
        self._aborted = True
        self._timeout = bool(timeout)
        self._abort_reason = reason

    def collect_sample(
        self,
        route_name: str,
        ground_truth_state: Optional[VehicleState],
        kalman_state: Optional[VehicleState] = None,
        gnss_diagnostics: Optional[GnssDiagnostics] = None,
        tracking: Optional[TrackingStatus] = None,
        route_completed: bool = False,
        phase: str = "driving",
        filtered_state: Optional[VehicleState] = None,
        filter_diagnostics: Optional[dict[str, object]] = None,
        speed_plan: Optional[SpeedPlan] = None,
        valid_for_metrics: Optional[bool] = None,
        warmup_excluded_reason: str = "",
    ) -> Optional[FilterPerformanceSample]:
        if not self._started or tracking is None:
            return None

        if filtered_state is None:
            filtered_state = kalman_state
        timestamp = self._sample_timestamp(ground_truth_state, filtered_state)
        filtered_error = self._position_error(filtered_state, ground_truth_state)
        raw_gnss_error = gnss_diagnostics.horizontal_error_m if gnss_diagnostics is not None else None
        x_error = self._axis_error(filtered_state, ground_truth_state, "x")
        y_error = self._axis_error(filtered_state, ground_truth_state, "y")
        speed_error = self._speed_error(filtered_state, ground_truth_state)
        yaw_error = self._yaw_error(filtered_state, ground_truth_state)
        diagnostics = filter_diagnostics or {}
        covariance_diag = diagnostics.get("covariance_diagonal")
        if covariance_diag is None and filtered_state is not None:
            covariance_diag = filtered_state.covariance_diagonal
        position_covariance = diagnostics.get("position_covariance_2x2") if isinstance(diagnostics, dict) else None
        if position_covariance is None and filtered_state is not None:
            position_covariance = filtered_state.position_covariance_2x2
        covariance_x_std = self._covariance_std(covariance_diag, 0)
        covariance_y_std = self._covariance_std(covariance_diag, 1)
        nis = self._finite_or_none(diagnostics.get("nis") if isinstance(diagnostics, dict) else None)
        nis_by_type = self._fresh_nis_by_type(diagnostics if isinstance(diagnostics, dict) else {})
        nis_expected_dimensions = self._nis_expected_dimensions(diagnostics if isinstance(diagnostics, dict) else {})
        innovation_norm = self._innovation_norm(diagnostics.get("innovation") if isinstance(diagnostics, dict) else None)
        nees_result = position_nees(
            x_error_m=x_error,
            y_error_m=y_error,
            position_covariance_2x2=position_covariance,
            covariance_diagonal=covariance_diag,
        )
        nees = self._finite_or_none(nees_result.get("nees"))
        gnss_frame = self._optional_int(diagnostics.get("last_gnss_frame") if isinstance(diagnostics, dict) else None)
        imu_frame = self._optional_int(diagnostics.get("last_imu_frame") if isinstance(diagnostics, dict) else None)
        metric_valid = bool(valid_for_metrics) if valid_for_metrics is not None else phase in ("driving", "completed")
        excluded_reason = "" if metric_valid else (warmup_excluded_reason or f"{phase}_excluded")
        sample = FilterPerformanceSample(
            timestamp=timestamp,
            route_name=route_name,
            phase=phase,
            valid_for_metrics=metric_valid,
            warmup_excluded_reason=excluded_reason,
            ground_truth_x=ground_truth_state.x if ground_truth_state is not None else None,
            ground_truth_y=ground_truth_state.y if ground_truth_state is not None else None,
            ground_truth_yaw=ground_truth_state.yaw if ground_truth_state is not None else None,
            ground_truth_speed=ground_truth_state.speed if ground_truth_state is not None else None,
            filtered_x=filtered_state.x if filtered_state is not None else None,
            filtered_y=filtered_state.y if filtered_state is not None else None,
            filtered_yaw=filtered_state.yaw if filtered_state is not None else None,
            filtered_speed=filtered_state.speed if filtered_state is not None else None,
            filtered_vx_mps=filtered_state.vx_mps if filtered_state is not None else None,
            filtered_vy_mps=filtered_state.vy_mps if filtered_state is not None else None,
            filtered_yaw_rate_radps=filtered_state.yaw_rate_radps if filtered_state is not None else None,
            filtered_curvature_1pm=filtered_state.curvature_1pm if filtered_state is not None else None,
            filtered_acceleration_mps2=filtered_state.acceleration_mps2 if filtered_state is not None else None,
            filtered_longitudinal_accel_mps2=(
                filtered_state.longitudinal_accel_mps2 if filtered_state is not None else None
            ),
            filtered_lateral_accel_mps2=filtered_state.lateral_accel_mps2 if filtered_state is not None else None,
            state_source_filter_id=filtered_state.source_filter_id if filtered_state is not None else "",
            state_model_type=filtered_state.model_type if filtered_state is not None else "",
            state_capabilities=",".join(filtered_state.capabilities()) if filtered_state is not None else "",
            state_confidence=filtered_state.confidence if filtered_state is not None else None,
            state_safe_for_autonomous_control=(
                filtered_state.safe_for_autonomous_control if filtered_state is not None else None
            ),
            kalman_x=filtered_state.x if filtered_state is not None else None,
            kalman_y=filtered_state.y if filtered_state is not None else None,
            kalman_yaw=filtered_state.yaw if filtered_state is not None else None,
            kalman_speed=filtered_state.speed if filtered_state is not None else None,
            gnss_x=gnss_diagnostics.local_x if gnss_diagnostics is not None else None,
            gnss_y=gnss_diagnostics.local_y if gnss_diagnostics is not None else None,
            filtered_position_error_m=filtered_error,
            kalman_position_error_m=filtered_error,
            raw_gnss_error_m=raw_gnss_error,
            cross_track_error_m=self._finite_or_none(tracking.cross_track_error_m),
            heading_error_deg=self._finite_or_none(tracking.heading_error_deg),
            distance_to_goal_m=self._finite_or_none(tracking.distance_to_goal_m),
            closest_index=int(tracking.closest_index),
            target_index=int(tracking.target_index),
            route_completed=route_completed,
            x_error_m=x_error,
            y_error_m=y_error,
            speed_error_mps=speed_error,
            yaw_error_deg=yaw_error,
            curvature_score=self._finite_or_none(speed_plan.curvature_score if speed_plan is not None else None),
            curvature_rad_per_m=self._finite_or_none(speed_plan.curvature_rad_per_m if speed_plan is not None else None),
            curvature_mode=speed_plan.mode if speed_plan is not None else None,
            nis=nis,
            nis_by_type=nis_by_type,
            nis_expected_dimensions_by_type=nis_expected_dimensions,
            nees=nees,
            position_nees=self._finite_or_none(nees_result.get("position_nees")),
            position_nees_diagonal_approx=self._finite_or_none(nees_result.get("position_nees_diagonal_approx")),
            position_nees_source=str(nees_result.get("position_nees_source") or "unavailable"),
            innovation_norm=innovation_norm,
            covariance_x_std_m=covariance_x_std,
            covariance_y_std_m=covariance_y_std,
            within_2sigma_x=self._within_sigma(x_error, covariance_x_std, sigma=2.0),
            within_2sigma_y=self._within_sigma(y_error, covariance_y_std, sigma=2.0),
            gnss_update_frame=gnss_frame,
            imu_update_frame=imu_frame,
        )
        self._samples.append(sample)
        return sample

    def running_rmse_m(self) -> Optional[float]:
        return self._rmse(self._finite_values(sample.filtered_position_error_m for sample in self._samples))

    def running_metrics(self, phases: Optional[tuple[str, ...]] = None) -> dict[str, Optional[float]]:
        return self._metrics_for_samples(self._samples_for_phases(phases))

    def running_driving_metrics(self) -> dict[str, Optional[float]]:
        return self.running_metrics(phases=("driving", "completed"))

    def running_sample_count(self, phases: Optional[tuple[str, ...]] = None) -> int:
        return len(self._samples_for_phases(phases))

    def build_summary(self) -> dict[str, object]:
        overall_metrics = self._metrics_for_samples(self._samples)
        eval_samples = [sample for sample in self._samples if sample.valid_for_metrics]
        eval_metrics = self._metrics_for_samples(eval_samples)
        driving_samples = eval_samples
        driving_metrics = eval_metrics
        stabilization_samples = [sample for sample in self._samples if not sample.valid_for_metrics]
        stabilization_metrics = self._metrics_for_samples(stabilization_samples)

        filtered_rmse = overall_metrics["filtered_rmse_m"]
        raw_gnss_rmse = overall_metrics["raw_gnss_rmse_m"]
        improvement_ratio = self._ratio(raw_gnss_rmse, filtered_rmse)

        eval_filtered_rmse = eval_metrics["filtered_rmse_m"]
        eval_raw_gnss_rmse = eval_metrics["raw_gnss_rmse_m"]
        eval_improvement_ratio = self._ratio(eval_raw_gnss_rmse, eval_filtered_rmse)

        completion_time = None
        if len(self._samples) >= 2:
            completion_time = self._samples[-1].timestamp - self._samples[0].timestamp
        diagnostic_notes = self._diagnostic_notes(stabilization_metrics, driving_metrics)
        warmup_excluded_s = self._duration_for_samples(stabilization_samples)

        return {
            "benchmark_id": self._benchmark_id,
            "active_filter_id": self._active_filter_id,
            "active_filter_name": self._active_filter_name,
            "route_name": self._route_name,
            "sample_count": len(self._samples),
            "valid_for_metrics_sample_count": len(eval_samples),
            "startup_transient_sample_count": len(stabilization_samples),
            "warmup_excluded_s": warmup_excluded_s,
            "route_completion_success": self._completed and not self._aborted,
            "route_aborted": self._aborted,
            "timeout": self._timeout,
            "abort_reason": self._abort_reason,
            "completion_time_s": completion_time,
            "active_filter_rmse_m": filtered_rmse,
            "filtered_rmse_m": filtered_rmse,
            "full_active_filter_rmse_m": filtered_rmse,
            "full_filtered_rmse_m": filtered_rmse,
            "full_filtered_mae_m": overall_metrics["filtered_mae_m"],
            "full_mean_position_error_m": overall_metrics["filtered_mae_m"],
            "full_filtered_max_error_m": overall_metrics["filtered_max_error_m"],
            "full_filtered_p95_error_m": overall_metrics["filtered_p95_error_m"],
            "full_filtered_p99_error_m": overall_metrics["filtered_p99_error_m"],
            "full_raw_gnss_rmse_m": raw_gnss_rmse,
            "full_raw_gnss_mae_m": overall_metrics["raw_gnss_mae_m"],
            "full_raw_gnss_max_error_m": overall_metrics["raw_gnss_max_error_m"],
            "full_raw_gnss_p95_error_m": overall_metrics["raw_gnss_p95_error_m"],
            "eval_active_filter_rmse_m": eval_filtered_rmse,
            "eval_filtered_rmse_m": eval_filtered_rmse,
            "eval_filtered_mae_m": eval_metrics["filtered_mae_m"],
            "eval_mean_position_error_m": eval_metrics["filtered_mae_m"],
            "eval_filtered_max_error_m": eval_metrics["filtered_max_error_m"],
            "eval_filtered_p95_error_m": eval_metrics["filtered_p95_error_m"],
            "eval_filtered_p99_error_m": eval_metrics["filtered_p99_error_m"],
            "eval_raw_gnss_rmse_m": eval_raw_gnss_rmse,
            "eval_raw_gnss_mae_m": eval_metrics["raw_gnss_mae_m"],
            "eval_raw_gnss_max_error_m": eval_metrics["raw_gnss_max_error_m"],
            "eval_raw_gnss_p95_error_m": eval_metrics["raw_gnss_p95_error_m"],
            "eval_filtered_improvement_ratio": eval_improvement_ratio,
            "eval_kalman_improvement_ratio": eval_improvement_ratio,
            "eval_speed_rmse_mps": eval_metrics["speed_rmse_mps"],
            "eval_yaw_rmse_deg": eval_metrics["yaw_rmse_deg"],
            "eval_mean_nis": eval_metrics["mean_nis"],
            "eval_mean_nees": eval_metrics["mean_nees"],
            "eval_legacy_mean_nis_mixed": eval_metrics["mean_nis"],
            "eval_nis_by_type_summary": eval_metrics["nis_by_type_summary"],
            "eval_position_nees_summary": eval_metrics["position_nees_summary"],
            "eval_position_nees_diagonal_approx_summary": eval_metrics["position_nees_diagonal_approx_summary"],
            "eval_position_nees_source": eval_metrics["position_nees_source"],
            "eval_mean_position_nees": eval_metrics["mean_position_nees"],
            "eval_mean_position_nees_diagonal_approx": eval_metrics["mean_position_nees_diagonal_approx"],
            "filtered_mae_m": overall_metrics["filtered_mae_m"],
            "x_rmse_m": overall_metrics["x_rmse_m"],
            "y_rmse_m": overall_metrics["y_rmse_m"],
            "speed_rmse_mps": overall_metrics["speed_rmse_mps"],
            "yaw_rmse_deg": overall_metrics["yaw_rmse_deg"],
            "filtered_max_error_m": overall_metrics["filtered_max_error_m"],
            "filtered_p95_error_m": overall_metrics["filtered_p95_error_m"],
            "kalman_rmse_m": filtered_rmse,
            "kalman_mae_m": overall_metrics["filtered_mae_m"],
            "kalman_max_error_m": overall_metrics["filtered_max_error_m"],
            "kalman_p95_error_m": overall_metrics["filtered_p95_error_m"],
            "raw_gnss_rmse_m": raw_gnss_rmse,
            "raw_gnss_mae_m": overall_metrics["raw_gnss_mae_m"],
            "raw_gnss_max_error_m": overall_metrics["raw_gnss_max_error_m"],
            "raw_gnss_p95_error_m": overall_metrics["raw_gnss_p95_error_m"],
            "active_filter_improvement_ratio": improvement_ratio,
            "filtered_improvement_ratio": improvement_ratio,
            "kalman_improvement_ratio": improvement_ratio,
            "mean_cross_track_error_m": overall_metrics["mean_cross_track_error_m"],
            "max_cross_track_error_m": overall_metrics["max_cross_track_error_m"],
            "p95_cross_track_error_m": overall_metrics["p95_cross_track_error_m"],
            "mean_heading_error_deg": overall_metrics["mean_heading_error_deg"],
            "yaw_rate_available_pct": overall_metrics["yaw_rate_available_pct"],
            "curvature_available_pct": overall_metrics["curvature_available_pct"],
            "acceleration_available_pct": overall_metrics["acceleration_available_pct"],
            "mean_abs_yaw_rate_radps": overall_metrics["mean_abs_yaw_rate_radps"],
            "mean_abs_curvature_1pm": overall_metrics["mean_abs_curvature_1pm"],
            "mean_abs_acceleration_mps2": overall_metrics["mean_abs_acceleration_mps2"],
            "mean_nis": overall_metrics["mean_nis"],
            "mean_nees": overall_metrics["mean_nees"],
            "legacy_mean_nis_mixed": overall_metrics["mean_nis"],
            "legacy_mean_nis_mixed_note": "Legacy mixed scalar NIS; prefer nis_by_type_summary.",
            "nis_by_type_summary": overall_metrics["nis_by_type_summary"],
            "position_nees_summary": overall_metrics["position_nees_summary"],
            "position_nees_diagonal_approx_summary": overall_metrics["position_nees_diagonal_approx_summary"],
            "position_nees_source": overall_metrics["position_nees_source"],
            "mean_position_nees": overall_metrics["mean_position_nees"],
            "mean_position_nees_diagonal_approx": overall_metrics["mean_position_nees_diagonal_approx"],
            "innovation_mean": overall_metrics["innovation_mean"],
            "innovation_std": overall_metrics["innovation_std"],
            "within_2sigma_x_pct": overall_metrics["within_2sigma_x_pct"],
            "within_2sigma_y_pct": overall_metrics["within_2sigma_y_pct"],
            "gnss_update_count": self._unique_count(sample.gnss_update_frame for sample in self._samples),
            "imu_update_count": self._unique_count(sample.imu_update_frame for sample in self._samples),
            "segment_metrics": self._segment_metrics(self._samples),
            "driving_sample_count": len(driving_samples),
            "driving_active_filter_rmse_m": eval_filtered_rmse,
            "driving_filtered_rmse_m": eval_filtered_rmse,
            "driving_filtered_mae_m": driving_metrics["filtered_mae_m"],
            "driving_x_rmse_m": driving_metrics["x_rmse_m"],
            "driving_y_rmse_m": driving_metrics["y_rmse_m"],
            "driving_speed_rmse_mps": driving_metrics["speed_rmse_mps"],
            "driving_yaw_rmse_deg": driving_metrics["yaw_rmse_deg"],
            "driving_filtered_max_error_m": driving_metrics["filtered_max_error_m"],
            "driving_filtered_p95_error_m": driving_metrics["filtered_p95_error_m"],
            "driving_kalman_rmse_m": eval_filtered_rmse,
            "driving_kalman_mae_m": driving_metrics["filtered_mae_m"],
            "driving_kalman_max_error_m": driving_metrics["filtered_max_error_m"],
            "driving_kalman_p95_error_m": driving_metrics["filtered_p95_error_m"],
            "driving_raw_gnss_rmse_m": eval_raw_gnss_rmse,
            "driving_raw_gnss_mae_m": driving_metrics["raw_gnss_mae_m"],
            "driving_raw_gnss_max_error_m": driving_metrics["raw_gnss_max_error_m"],
            "driving_raw_gnss_p95_error_m": driving_metrics["raw_gnss_p95_error_m"],
            "driving_active_filter_improvement_ratio": eval_improvement_ratio,
            "driving_filtered_improvement_ratio": eval_improvement_ratio,
            "driving_kalman_improvement_ratio": eval_improvement_ratio,
            "driving_mean_cross_track_error_m": driving_metrics["mean_cross_track_error_m"],
            "driving_max_cross_track_error_m": driving_metrics["max_cross_track_error_m"],
            "driving_p95_cross_track_error_m": driving_metrics["p95_cross_track_error_m"],
            "driving_mean_heading_error_deg": driving_metrics["mean_heading_error_deg"],
            "driving_yaw_rate_available_pct": driving_metrics["yaw_rate_available_pct"],
            "driving_curvature_available_pct": driving_metrics["curvature_available_pct"],
            "driving_acceleration_available_pct": driving_metrics["acceleration_available_pct"],
            "driving_mean_abs_yaw_rate_radps": driving_metrics["mean_abs_yaw_rate_radps"],
            "driving_mean_abs_curvature_1pm": driving_metrics["mean_abs_curvature_1pm"],
            "driving_mean_abs_acceleration_mps2": driving_metrics["mean_abs_acceleration_mps2"],
            "driving_mean_nis": driving_metrics["mean_nis"],
            "driving_mean_nees": driving_metrics["mean_nees"],
            "driving_legacy_mean_nis_mixed": driving_metrics["mean_nis"],
            "driving_nis_by_type_summary": driving_metrics["nis_by_type_summary"],
            "driving_position_nees_summary": driving_metrics["position_nees_summary"],
            "driving_position_nees_diagonal_approx_summary": driving_metrics["position_nees_diagonal_approx_summary"],
            "driving_position_nees_source": driving_metrics["position_nees_source"],
            "driving_mean_position_nees": driving_metrics["mean_position_nees"],
            "driving_mean_position_nees_diagonal_approx": driving_metrics["mean_position_nees_diagonal_approx"],
            "driving_segment_metrics": self._segment_metrics(driving_samples),
            "stabilization_sample_count": len(stabilization_samples),
            "stabilization_filtered_max_error_m": stabilization_metrics["filtered_max_error_m"],
            "stabilization_filtered_p95_error_m": stabilization_metrics["filtered_p95_error_m"],
            "stabilization_raw_gnss_max_error_m": stabilization_metrics["raw_gnss_max_error_m"],
            "stabilization_raw_gnss_p95_error_m": stabilization_metrics["raw_gnss_p95_error_m"],
            "diagnostic_notes": diagnostic_notes,
            "excluded_filtered_plot_sample_count": self._excluded_filtered_plot_sample_count(driving_samples),
            "excluded_kalman_plot_sample_count": self._excluded_filtered_plot_sample_count(driving_samples),
        }

    def export(self) -> tuple[Path, Path]:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        return self.export_to_files(self._output_dir / "samples.csv", self._output_dir / "summary.json")

    def export_to_folder(self, folder: Path) -> tuple[Path, Path]:
        folder.mkdir(parents=True, exist_ok=True)
        return self.export_to_files(folder / "samples.csv", folder / "summary.json")

    def export_to_files(self, csv_path: Path, json_path: Path) -> tuple[Path, Path]:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_csv(csv_path)
        json_path.write_text(json.dumps(self.build_summary(), indent=2), encoding="utf-8")
        self._last_export_paths = (csv_path, json_path)
        return csv_path, json_path

    def _write_csv(self, path: Path) -> None:
        fieldnames = list(FilterPerformanceSample.__dataclass_fields__.keys())
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for sample in self._samples:
                writer.writerow({
                    key: "" if value is None else value
                    for key, value in asdict(sample).items()
                })

    @staticmethod
    def _sample_timestamp(
        ground_truth_state: Optional[VehicleState],
        filtered_state: Optional[VehicleState],
    ) -> float:
        if ground_truth_state is not None:
            return float(ground_truth_state.timestamp)
        if filtered_state is not None:
            return float(filtered_state.timestamp)
        return 0.0

    @staticmethod
    def _position_error(state: Optional[VehicleState], ground_truth_state: Optional[VehicleState]) -> Optional[float]:
        if state is None or ground_truth_state is None:
            return None
        return math.hypot(state.x - ground_truth_state.x, state.y - ground_truth_state.y)

    @staticmethod
    def _axis_error(state: Optional[VehicleState], ground_truth_state: Optional[VehicleState], axis: str) -> Optional[float]:
        if state is None or ground_truth_state is None:
            return None
        return float(getattr(state, axis) - getattr(ground_truth_state, axis))

    @staticmethod
    def _speed_error(state: Optional[VehicleState], ground_truth_state: Optional[VehicleState]) -> Optional[float]:
        if state is None or ground_truth_state is None:
            return None
        return float(state.speed - ground_truth_state.speed)

    @staticmethod
    def _yaw_error(state: Optional[VehicleState], ground_truth_state: Optional[VehicleState]) -> Optional[float]:
        if state is None or ground_truth_state is None:
            return None
        delta = float(state.yaw - ground_truth_state.yaw)
        while delta > 180.0:
            delta -= 360.0
        while delta < -180.0:
            delta += 360.0
        return delta

    @staticmethod
    def _finite_or_none(value: Optional[float]) -> Optional[float]:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        return float(value)

    @classmethod
    def _covariance_std(cls, covariance_diag: object, index: int) -> Optional[float]:
        if not isinstance(covariance_diag, (list, tuple)) or index >= len(covariance_diag):
            return None
        value = cls._finite_or_none(covariance_diag[index])
        if value is None or value < 0.0:
            return None
        return math.sqrt(value)

    @classmethod
    def _innovation_norm(cls, innovation: object) -> Optional[float]:
        if not isinstance(innovation, (list, tuple)):
            return None
        values = cls._finite_values(innovation)
        if not values:
            return None
        return math.sqrt(sum(value * value for value in values))

    @staticmethod
    def _position_nees(
        x_error: Optional[float],
        y_error: Optional[float],
        covariance_x_std: Optional[float],
        covariance_y_std: Optional[float],
    ) -> Optional[float]:
        if x_error is None or y_error is None or covariance_x_std is None or covariance_y_std is None:
            return None
        x_var = covariance_x_std * covariance_x_std
        y_var = covariance_y_std * covariance_y_std
        if x_var <= 1.0e-9 or y_var <= 1.0e-9:
            return None
        return (x_error * x_error / x_var) + (y_error * y_error / y_var)

    def _fresh_nis_by_type(self, diagnostics: dict[str, object]) -> dict[str, float]:
        raw_values = diagnostics.get("nis_by_type")
        if not isinstance(raw_values, dict):
            return {}
        counts = self._nis_update_counts(diagnostics)
        fresh: dict[str, float] = {}
        for update_type, count in counts.items():
            if count <= self._last_nis_update_counts_by_type.get(update_type, 0):
                continue
            value = self._finite_or_none(raw_values.get(update_type))
            if value is not None:
                fresh[update_type] = value
        self._last_nis_update_counts_by_type = counts
        return fresh

    @staticmethod
    def _nis_update_counts(diagnostics: dict[str, object]) -> dict[str, int]:
        raw = diagnostics.get("nis_update_counts_by_type")
        if not isinstance(raw, dict):
            return {}
        result: dict[str, int] = {}
        for key, value in raw.items():
            try:
                result[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _nis_expected_dimensions(diagnostics: dict[str, object]) -> dict[str, int]:
        raw = diagnostics.get("nis_expected_dimensions_by_type")
        if not isinstance(raw, dict):
            return {}
        result: dict[str, int] = {}
        for key, value in raw.items():
            try:
                result[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _within_sigma(error: Optional[float], stddev: Optional[float], sigma: float) -> Optional[bool]:
        if error is None or stddev is None or stddev <= 0.0:
            return None
        return abs(error) <= sigma * stddev

    @staticmethod
    def _optional_int(value: object) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _finite_values(values: Iterable[Optional[float]]) -> list[float]:
        result = []
        for value in values:
            if not isinstance(value, (int, float)):
                continue
            if math.isfinite(float(value)):
                result.append(float(value))
        return result

    @classmethod
    def _metrics_for_samples(cls, samples: Iterable[FilterPerformanceSample]) -> dict[str, Optional[float]]:
        sample_list = list(samples)
        filtered_errors = cls._finite_values(sample.filtered_position_error_m for sample in sample_list)
        raw_gnss_errors = cls._finite_values(sample.raw_gnss_error_m for sample in sample_list)
        cross_track_errors = cls._finite_values(sample.cross_track_error_m for sample in sample_list)
        heading_errors = [
            abs(value)
            for value in cls._finite_values(sample.heading_error_deg for sample in sample_list)
        ]
        x_errors = cls._finite_values(sample.x_error_m for sample in sample_list)
        y_errors = cls._finite_values(sample.y_error_m for sample in sample_list)
        speed_errors = cls._finite_values(sample.speed_error_mps for sample in sample_list)
        yaw_errors = [abs(value) for value in cls._finite_values(sample.yaw_error_deg for sample in sample_list)]
        nis_values = cls._finite_values(sample.nis for sample in sample_list)
        consistency_rows = [
            {
                "nis_by_type": sample.nis_by_type,
                "nees": sample.nees,
                "position_nees": sample.position_nees,
                "position_nees_diagonal_approx": sample.position_nees_diagonal_approx,
            }
            for sample in sample_list
        ]
        nees_summary = summarize_position_nees(consistency_rows)
        innovation_values = cls._finite_values(sample.innovation_norm for sample in sample_list)
        yaw_rate_values = cls._finite_values(sample.filtered_yaw_rate_radps for sample in sample_list)
        curvature_values = cls._finite_values(sample.filtered_curvature_1pm for sample in sample_list)
        acceleration_values = cls._finite_values(sample.filtered_acceleration_mps2 for sample in sample_list)
        return {
            "filtered_rmse_m": cls._rmse(filtered_errors),
            "filtered_mae_m": cls._mean(filtered_errors),
            "x_rmse_m": cls._rmse(x_errors),
            "y_rmse_m": cls._rmse(y_errors),
            "speed_rmse_mps": cls._rmse(speed_errors),
            "yaw_rmse_deg": cls._rmse(yaw_errors),
            "filtered_max_error_m": max(filtered_errors) if filtered_errors else None,
            "filtered_median_error_m": cls._percentile(filtered_errors, 50.0),
            "filtered_p95_error_m": cls._percentile(filtered_errors, 95.0),
            "filtered_p99_error_m": cls._percentile(filtered_errors, 99.0),
            "raw_gnss_rmse_m": cls._rmse(raw_gnss_errors),
            "raw_gnss_mae_m": cls._mean(raw_gnss_errors),
            "raw_gnss_max_error_m": max(raw_gnss_errors) if raw_gnss_errors else None,
            "raw_gnss_median_error_m": cls._percentile(raw_gnss_errors, 50.0),
            "raw_gnss_p95_error_m": cls._percentile(raw_gnss_errors, 95.0),
            "raw_gnss_p99_error_m": cls._percentile(raw_gnss_errors, 99.0),
            "mean_cross_track_error_m": cls._mean(cross_track_errors),
            "max_cross_track_error_m": max(cross_track_errors) if cross_track_errors else None,
            "p95_cross_track_error_m": cls._percentile(cross_track_errors, 95.0),
            "mean_heading_error_deg": cls._mean(heading_errors),
            "yaw_rate_available_pct": cls._availability_percentage(
                sample.filtered_yaw_rate_radps for sample in sample_list
            ),
            "curvature_available_pct": cls._availability_percentage(
                sample.filtered_curvature_1pm for sample in sample_list
            ),
            "acceleration_available_pct": cls._availability_percentage(
                sample.filtered_acceleration_mps2 for sample in sample_list
            ),
            "mean_abs_yaw_rate_radps": cls._mean([abs(value) for value in yaw_rate_values]),
            "mean_abs_curvature_1pm": cls._mean([abs(value) for value in curvature_values]),
            "mean_abs_acceleration_mps2": cls._mean([abs(value) for value in acceleration_values]),
            "mean_nis": cls._mean(nis_values),
            "mean_nees": nees_summary["mean_nees"],
            "nis_by_type_summary": summarize_nis_by_type(consistency_rows),
            "position_nees_summary": nees_summary["position_nees_summary"],
            "position_nees_diagonal_approx_summary": nees_summary["position_nees_diagonal_approx_summary"],
            "legacy_nees_summary": nees_summary["legacy_nees_summary"],
            "position_nees_source": nees_summary["position_nees_source"],
            "mean_position_nees": nees_summary["mean_position_nees"],
            "mean_position_nees_diagonal_approx": nees_summary["mean_position_nees_diagonal_approx"],
            "position_nees_available": nees_summary["position_nees_available"],
            "position_nees_diagonal_approx_available": nees_summary["position_nees_diagonal_approx_available"],
            "innovation_mean": cls._mean(innovation_values),
            "innovation_std": cls._stddev(innovation_values),
            "within_2sigma_x_pct": cls._boolean_percentage(sample.within_2sigma_x for sample in sample_list),
            "within_2sigma_y_pct": cls._boolean_percentage(sample.within_2sigma_y for sample in sample_list),
        }

    @staticmethod
    def _diagnostic_notes(
        stabilization_metrics: dict[str, Optional[float]],
        driving_metrics: dict[str, Optional[float]],
    ) -> list[str]:
        stabilization_max = stabilization_metrics.get("filtered_max_error_m")
        driving_p95 = driving_metrics.get("filtered_p95_error_m")
        if not isinstance(stabilization_max, (int, float)) or not math.isfinite(float(stabilization_max)):
            return []
        large_relative_spike = (
            isinstance(driving_p95, (int, float))
            and math.isfinite(float(driving_p95))
            and float(driving_p95) > 0.0
            and float(stabilization_max) > 10.0 * float(driving_p95)
        )
        if float(stabilization_max) > 100.0 or large_relative_spike:
            return [
                "Large startup transient detected. Eval/driving metrics should be used for route performance comparison."
            ]
        return []

    @staticmethod
    def _duration_for_samples(samples: Iterable[FilterPerformanceSample]) -> float:
        sample_list = sorted(samples, key=lambda sample: sample.timestamp)
        if len(sample_list) < 2:
            return 0.0
        return max(0.0, float(sample_list[-1].timestamp - sample_list[0].timestamp))

    @classmethod
    def _segment_metrics(cls, samples: Iterable[FilterPerformanceSample]) -> dict[str, dict[str, Optional[float]]]:
        buckets: dict[str, list[FilterPerformanceSample]] = {}
        for sample in samples:
            segment = cls._segment_name(sample)
            buckets.setdefault(segment, []).append(sample)
        return {
            segment: {
                "sample_count": len(segment_samples),
                "position_rmse_m": cls._rmse(cls._finite_values(item.filtered_position_error_m for item in segment_samples)),
                "speed_rmse_mps": cls._rmse(cls._finite_values(item.speed_error_mps for item in segment_samples)),
                "yaw_rmse_deg": cls._rmse([abs(value) for value in cls._finite_values(item.yaw_error_deg for item in segment_samples)]),
                "mean_nis": cls._mean(cls._finite_values(item.nis for item in segment_samples)),
                "mean_nees": cls._mean(cls._finite_values(item.nees for item in segment_samples)),
                "nis_by_type_summary": summarize_nis_by_type({"nis_by_type": item.nis_by_type} for item in segment_samples),
                "position_nees_source": summarize_position_nees(
                    {
                        "nees": item.nees,
                        "position_nees": item.position_nees,
                        "position_nees_diagonal_approx": item.position_nees_diagonal_approx,
                    }
                    for item in segment_samples
                )["position_nees_source"],
            }
            for segment, segment_samples in sorted(buckets.items())
        }

    @staticmethod
    def _segment_name(sample: FilterPerformanceSample) -> str:
        mode = (sample.curvature_mode or "").lower()
        score = sample.curvature_score
        if "approach" in mode:
            return "curve_approach"
        if "curve" in mode:
            return "curve"
        if "stopping" in mode:
            return "curve_exit"
        if isinstance(score, (int, float)) and score >= 0.18:
            return "curve"
        return "straight"

    @classmethod
    def _excluded_filtered_plot_sample_count(cls, samples: Iterable[FilterPerformanceSample]) -> int:
        candidates = 0
        kept = 0
        previous_xy: Optional[tuple[float, float]] = None
        for sample in samples:
            if not cls._finite_pair(sample.filtered_x, sample.filtered_y):
                continue
            candidates += 1
            if not cls._finite_pair(sample.ground_truth_x, sample.ground_truth_y):
                continue
            if sample.filtered_position_error_m is None or not math.isfinite(sample.filtered_position_error_m):
                continue
            if sample.filtered_position_error_m > BENCHMARK.max_kalman_plot_error_m:
                continue

            current_xy = (float(sample.filtered_x), float(sample.filtered_y))
            if previous_xy is not None and math.hypot(
                current_xy[0] - previous_xy[0],
                current_xy[1] - previous_xy[1],
            ) > BENCHMARK.max_trajectory_jump_m:
                continue

            kept += 1
            previous_xy = current_xy
        return max(0, candidates - kept)

    @staticmethod
    def _finite_pair(x_value: Optional[float], y_value: Optional[float]) -> bool:
        return (
            x_value is not None
            and y_value is not None
            and math.isfinite(x_value)
            and math.isfinite(y_value)
        )

    @staticmethod
    def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        if numerator is None or denominator is None or denominator <= 0.0:
            return None
        return numerator / denominator

    @staticmethod
    def _rmse(values: list[float]) -> Optional[float]:
        if not values:
            return None
        return math.sqrt(sum(value * value for value in values) / len(values))

    @staticmethod
    def _mean(values: list[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _stddev(values: list[float]) -> Optional[float]:
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))

    @staticmethod
    def _boolean_percentage(values: Iterable[Optional[bool]]) -> Optional[float]:
        valid = [value for value in values if value is not None]
        if not valid:
            return None
        return 100.0 * sum(1 for value in valid if value) / len(valid)

    @staticmethod
    def _availability_percentage(values: Iterable[Optional[float]]) -> Optional[float]:
        values_list = list(values)
        if not values_list:
            return None
        available = 0
        for value in values_list:
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                available += 1
        return 100.0 * available / len(values_list)

    @staticmethod
    def _unique_count(values: Iterable[Optional[int]]) -> int:
        return len({value for value in values if value is not None})

    def _samples_for_phases(self, phases: Optional[tuple[str, ...]]) -> list[FilterPerformanceSample]:
        if phases is None:
            return list(self._samples)
        return [sample for sample in self._samples if sample.phase in phases]

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> Optional[float]:
        if not values:
            return None
        sorted_values = sorted(values)
        index = int(math.ceil((percentile / 100.0) * len(sorted_values))) - 1
        index = max(0, min(index, len(sorted_values) - 1))
        return sorted_values[index]

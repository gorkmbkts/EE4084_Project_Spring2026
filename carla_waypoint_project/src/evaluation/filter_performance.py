"""Single-run Kalman benchmark logging and summary metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Optional

from src.control.waypoint_tracker import TrackingStatus
from src.localization.gnss_projection import GnssDiagnostics
from src.localization.state_estimator import EgoState


@dataclass(frozen=True)
class FilterPerformanceSample:
    """One frame of Kalman benchmark metrics."""

    timestamp: float
    route_name: str
    phase: str
    ground_truth_x: Optional[float]
    ground_truth_y: Optional[float]
    ground_truth_yaw: Optional[float]
    ground_truth_speed: Optional[float]
    kalman_x: Optional[float]
    kalman_y: Optional[float]
    kalman_yaw: Optional[float]
    kalman_speed: Optional[float]
    gnss_x: Optional[float]
    gnss_y: Optional[float]
    kalman_position_error_m: Optional[float]
    raw_gnss_error_m: Optional[float]
    cross_track_error_m: Optional[float]
    heading_error_deg: Optional[float]
    distance_to_goal_m: Optional[float]
    closest_index: int
    target_index: int
    route_completed: bool


class FilterPerformanceLogger:
    """Collect one benchmark run in memory and export CSV plus JSON summary."""

    def __init__(self, output_dir: Optional[Path] = None, benchmark_id: str = "") -> None:
        project_root = Path(__file__).resolve().parents[2]
        self._output_dir = output_dir if output_dir is not None else project_root / "logs" / "filter_tests"
        self._benchmark_id = benchmark_id
        self._samples: list[FilterPerformanceSample] = []
        self._route_name = ""
        self._started = False
        self._completed = False
        self._aborted = False
        self._timeout = False
        self._abort_reason: Optional[str] = None
        self._last_export_paths: Optional[tuple[Path, Path]] = None

    @property
    def samples(self) -> tuple[FilterPerformanceSample, ...]:
        return tuple(self._samples)

    @property
    def route_name(self) -> str:
        return self._route_name

    @property
    def current_position_error_m(self) -> Optional[float]:
        return self._samples[-1].kalman_position_error_m if self._samples else None

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
        ground_truth_state: Optional[EgoState],
        kalman_state: Optional[EgoState],
        gnss_diagnostics: Optional[GnssDiagnostics],
        tracking: TrackingStatus,
        route_completed: bool,
        phase: str = "driving",
    ) -> Optional[FilterPerformanceSample]:
        if not self._started:
            return None

        timestamp = self._sample_timestamp(ground_truth_state, kalman_state)
        kalman_error = self._position_error(kalman_state, ground_truth_state)
        raw_gnss_error = gnss_diagnostics.horizontal_error_m if gnss_diagnostics is not None else None
        sample = FilterPerformanceSample(
            timestamp=timestamp,
            route_name=route_name,
            phase=phase,
            ground_truth_x=ground_truth_state.x if ground_truth_state is not None else None,
            ground_truth_y=ground_truth_state.y if ground_truth_state is not None else None,
            ground_truth_yaw=ground_truth_state.yaw if ground_truth_state is not None else None,
            ground_truth_speed=ground_truth_state.speed if ground_truth_state is not None else None,
            kalman_x=kalman_state.x if kalman_state is not None else None,
            kalman_y=kalman_state.y if kalman_state is not None else None,
            kalman_yaw=kalman_state.yaw if kalman_state is not None else None,
            kalman_speed=kalman_state.speed if kalman_state is not None else None,
            gnss_x=gnss_diagnostics.local_x if gnss_diagnostics is not None else None,
            gnss_y=gnss_diagnostics.local_y if gnss_diagnostics is not None else None,
            kalman_position_error_m=kalman_error,
            raw_gnss_error_m=raw_gnss_error,
            cross_track_error_m=self._finite_or_none(tracking.cross_track_error_m),
            heading_error_deg=self._finite_or_none(tracking.heading_error_deg),
            distance_to_goal_m=self._finite_or_none(tracking.distance_to_goal_m),
            closest_index=int(tracking.closest_index),
            target_index=int(tracking.target_index),
            route_completed=route_completed,
        )
        self._samples.append(sample)
        return sample

    def running_rmse_m(self) -> Optional[float]:
        return self._rmse(self._finite_values(sample.kalman_position_error_m for sample in self._samples))

    def build_summary(self) -> dict[str, object]:
        kalman_errors = self._finite_values(sample.kalman_position_error_m for sample in self._samples)
        raw_gnss_errors = self._finite_values(sample.raw_gnss_error_m for sample in self._samples)
        cross_track_errors = self._finite_values(sample.cross_track_error_m for sample in self._samples)
        heading_errors = [abs(value) for value in self._finite_values(sample.heading_error_deg for sample in self._samples)]

        kalman_rmse = self._rmse(kalman_errors)
        raw_gnss_rmse = self._rmse(raw_gnss_errors)
        improvement_ratio = None
        if kalman_rmse is not None and kalman_rmse > 0.0 and raw_gnss_rmse is not None:
            improvement_ratio = raw_gnss_rmse / kalman_rmse

        completion_time = None
        if len(self._samples) >= 2:
            completion_time = self._samples[-1].timestamp - self._samples[0].timestamp

        return {
            "benchmark_id": self._benchmark_id,
            "route_name": self._route_name,
            "sample_count": len(self._samples),
            "route_completion_success": self._completed and not self._aborted,
            "route_aborted": self._aborted,
            "timeout": self._timeout,
            "abort_reason": self._abort_reason,
            "completion_time_s": completion_time,
            "kalman_rmse_m": kalman_rmse,
            "kalman_mae_m": self._mean(kalman_errors),
            "kalman_max_error_m": max(kalman_errors) if kalman_errors else None,
            "kalman_p95_error_m": self._percentile(kalman_errors, 95.0),
            "raw_gnss_rmse_m": raw_gnss_rmse,
            "raw_gnss_mae_m": self._mean(raw_gnss_errors),
            "raw_gnss_max_error_m": max(raw_gnss_errors) if raw_gnss_errors else None,
            "raw_gnss_p95_error_m": self._percentile(raw_gnss_errors, 95.0),
            "kalman_improvement_ratio": improvement_ratio,
            "mean_cross_track_error_m": self._mean(cross_track_errors),
            "max_cross_track_error_m": max(cross_track_errors) if cross_track_errors else None,
            "p95_cross_track_error_m": self._percentile(cross_track_errors, 95.0),
            "mean_heading_error_deg": self._mean(heading_errors),
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
        ground_truth_state: Optional[EgoState],
        kalman_state: Optional[EgoState],
    ) -> float:
        if ground_truth_state is not None:
            return float(ground_truth_state.timestamp)
        if kalman_state is not None:
            return float(kalman_state.timestamp)
        return 0.0

    @staticmethod
    def _position_error(state: Optional[EgoState], ground_truth_state: Optional[EgoState]) -> Optional[float]:
        if state is None or ground_truth_state is None:
            return None
        return math.hypot(state.x - ground_truth_state.x, state.y - ground_truth_state.y)

    @staticmethod
    def _finite_or_none(value: Optional[float]) -> Optional[float]:
        if value is None or not math.isfinite(value):
            return None
        return float(value)

    @staticmethod
    def _finite_values(values: Iterable[Optional[float]]) -> list[float]:
        return [float(value) for value in values if value is not None and math.isfinite(value)]

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
    def _percentile(values: list[float], percentile: float) -> Optional[float]:
        if not values:
            return None
        sorted_values = sorted(values)
        index = int(math.ceil((percentile / 100.0) * len(sorted_values))) - 1
        index = max(0, min(index, len(sorted_values) - 1))
        return sorted_values[index]

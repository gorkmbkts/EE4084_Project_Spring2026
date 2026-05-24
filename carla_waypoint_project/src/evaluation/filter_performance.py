"""Localization and route-tracking performance logging."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import time
from typing import Optional

from src.control.waypoint_tracker import TrackingStatus
from src.localization.gnss_projection import GnssDiagnostics
from src.localization.state_estimator import EgoState


@dataclass(frozen=True)
class FilterPerformanceSample:
    """One frame of filter and route tracking metrics."""

    timestamp: float
    route_name: str
    ground_truth_x: Optional[float]
    ground_truth_y: Optional[float]
    estimated_x: Optional[float]
    estimated_y: Optional[float]
    gnss_x: Optional[float]
    gnss_y: Optional[float]
    position_error_m: Optional[float]
    raw_gnss_error_m: Optional[float]
    speed_mps: Optional[float]
    cross_track_error_m: Optional[float]
    heading_error_deg: Optional[float]
    closest_index: int
    target_index: int
    route_completed: bool


class FilterPerformanceLogger:
    """Collect samples in memory and export CSV plus JSON summaries."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self._output_dir = output_dir if output_dir is not None else project_root / "logs" / "filter_tests"
        self._samples: list[FilterPerformanceSample] = []
        self._route_name = ""
        self._started = False
        self._completed = False
        self._aborted = False
        self._last_export_paths: Optional[tuple[Path, Path]] = None

    @property
    def samples(self) -> tuple[FilterPerformanceSample, ...]:
        return tuple(self._samples)

    @property
    def route_name(self) -> str:
        return self._route_name

    @property
    def current_position_error_m(self) -> Optional[float]:
        return self._samples[-1].position_error_m if self._samples else None

    @property
    def current_raw_gnss_error_m(self) -> Optional[float]:
        return self._samples[-1].raw_gnss_error_m if self._samples else None

    @property
    def current_cross_track_error_m(self) -> Optional[float]:
        return self._samples[-1].cross_track_error_m if self._samples else None

    @property
    def last_export_paths(self) -> Optional[tuple[Path, Path]]:
        return self._last_export_paths

    def start_route(self, route_name: str) -> None:
        self._samples = []
        self._route_name = route_name
        self._started = True
        self._completed = False
        self._aborted = False
        self._last_export_paths = None

    def mark_completed(self) -> None:
        self._completed = True
        self._aborted = False

    def mark_aborted(self) -> None:
        self._completed = False
        self._aborted = True

    def collect_sample(
        self,
        route_name: str,
        ground_truth_state: Optional[EgoState],
        estimated_state: Optional[EgoState],
        gnss_diagnostics: Optional[GnssDiagnostics],
        tracking: TrackingStatus,
        route_completed: bool,
    ) -> Optional[FilterPerformanceSample]:
        if not self._started:
            return None

        timestamp = self._sample_timestamp(ground_truth_state, estimated_state)
        position_error = None
        if ground_truth_state is not None and estimated_state is not None:
            position_error = math.hypot(
                estimated_state.x - ground_truth_state.x,
                estimated_state.y - ground_truth_state.y,
            )

        raw_gnss_error = gnss_diagnostics.horizontal_error_m if gnss_diagnostics is not None else None
        cross_track_error = self._finite_or_none(tracking.cross_track_error_m)
        heading_error = self._finite_or_none(tracking.heading_error_deg)
        sample = FilterPerformanceSample(
            timestamp=timestamp,
            route_name=route_name,
            ground_truth_x=ground_truth_state.x if ground_truth_state is not None else None,
            ground_truth_y=ground_truth_state.y if ground_truth_state is not None else None,
            estimated_x=estimated_state.x if estimated_state is not None else None,
            estimated_y=estimated_state.y if estimated_state is not None else None,
            gnss_x=gnss_diagnostics.local_x if gnss_diagnostics is not None else None,
            gnss_y=gnss_diagnostics.local_y if gnss_diagnostics is not None else None,
            position_error_m=position_error,
            raw_gnss_error_m=raw_gnss_error,
            speed_mps=ground_truth_state.speed if ground_truth_state is not None else None,
            cross_track_error_m=cross_track_error,
            heading_error_deg=heading_error,
            closest_index=int(tracking.closest_index),
            target_index=int(tracking.target_index),
            route_completed=route_completed,
        )
        self._samples.append(sample)
        return sample

    def running_rmse_m(self) -> Optional[float]:
        values = self._finite_values(sample.position_error_m for sample in self._samples)
        if not values:
            return None
        return math.sqrt(sum(value * value for value in values) / len(values))

    def build_summary(self) -> dict[str, object]:
        position_errors = self._finite_values(sample.position_error_m for sample in self._samples)
        raw_gnss_errors = self._finite_values(sample.raw_gnss_error_m for sample in self._samples)
        cross_track_errors = self._finite_values(sample.cross_track_error_m for sample in self._samples)
        heading_errors = [abs(value) for value in self._finite_values(sample.heading_error_deg for sample in self._samples)]

        kf_rmse = self._rmse(position_errors)
        raw_rmse = self._rmse(raw_gnss_errors)
        improvement_ratio = None
        if kf_rmse is not None and kf_rmse > 0.0 and raw_rmse is not None:
            improvement_ratio = raw_rmse / kf_rmse

        completion_time = None
        if len(self._samples) >= 2:
            completion_time = self._samples[-1].timestamp - self._samples[0].timestamp

        return {
            "route_name": self._route_name,
            "sample_count": len(self._samples),
            "position_rmse_m": kf_rmse,
            "position_mae_m": self._mean(position_errors),
            "max_position_error_m": max(position_errors) if position_errors else None,
            "p95_position_error_m": self._percentile(position_errors, 95.0),
            "raw_gnss_rmse_m": raw_rmse,
            "raw_gnss_mae_m": self._mean(raw_gnss_errors),
            "kf_improvement_ratio": improvement_ratio,
            "mean_cross_track_error_m": self._mean(cross_track_errors),
            "max_cross_track_error_m": max(cross_track_errors) if cross_track_errors else None,
            "p95_cross_track_error_m": self._percentile(cross_track_errors, 95.0),
            "mean_heading_error_deg": self._mean(heading_errors),
            "route_completion_success": self._completed and not self._aborted,
            "route_aborted": self._aborted,
            "completion_time_s": completion_time,
        }

    def export(self) -> tuple[Path, Path]:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        stamp = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(now))}_{int((now % 1.0) * 1000):03d}"
        csv_path = self._output_dir / f"filter_test_samples_{stamp}.csv"
        json_path = self._output_dir / f"filter_test_summary_{stamp}.json"

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
                row = {
                    key: ("" if value is None else value)
                    for key, value in asdict(sample).items()
                }
                writer.writerow(row)

    @staticmethod
    def _sample_timestamp(
        ground_truth_state: Optional[EgoState],
        estimated_state: Optional[EgoState],
    ) -> float:
        if ground_truth_state is not None:
            return float(ground_truth_state.timestamp)
        if estimated_state is not None:
            return float(estimated_state.timestamp)
        return time.monotonic()

    @staticmethod
    def _finite_or_none(value: Optional[float]) -> Optional[float]:
        if value is None or not math.isfinite(value):
            return None
        return float(value)

    @staticmethod
    def _finite_values(values: Iterable[Optional[float]]) -> list[float]:
        result: list[float] = []
        for value in values:
            if value is not None and math.isfinite(value):
                result.append(float(value))
        return result

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

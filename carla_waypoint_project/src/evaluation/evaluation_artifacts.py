"""Filesystem helpers for offline localization recordings and evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Optional

from config.settings import BENCHMARK


OFFLINE_LOCALIZATION_EXPLANATION = (
    "Closed-loop benchmark evaluates the complete filter-controller-driving behavior system. "
    "Offline localization replay evaluates filter-only localization performance by replaying "
    "identical recorded GNSS/IMU logs through each filter."
)

OFFLINE_REPORT_NAME = "Localization Evaluation Under Identical Sensor Logs"
OFFLINE_MODE_NAME = "Offline Localization Replay"
RECORDING_DRIVER_GROUND_TRUTH_CONTROLLER = "ground_truth_controller"


@dataclass(frozen=True)
class RecordedLogInfo:
    """One route-level offline recording discovered on disk."""

    route_folder: Path
    sensor_log_path: Path
    run_folder: Path
    recording_id: str
    route_name: str
    map_name: str
    sample_count: Optional[int]
    duration_s: Optional[float]
    recording_driver: str
    sensor_noise_preset: str
    vehicle_behavior_preset: str
    created_at: str
    failure_reason: str = ""

    def label(self) -> str:
        count = self.sample_count if self.sample_count is not None else "n/a"
        duration = f"{self.duration_s:.1f}s" if isinstance(self.duration_s, (int, float)) else "n/a"
        return (
            f"{self.created_at or self.recording_id} | {self.route_name} | {self.map_name} | "
            f"{count} samples | {duration} | {self.recording_driver}"
        )


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def benchmark_root(output_root: str = BENCHMARK.output_root) -> Path:
    root = Path(output_root)
    return root if root.is_absolute() else project_root() / root


def offline_root(output_root: str = BENCHMARK.output_root) -> Path:
    return benchmark_root(output_root) / "offline_localization"


def recordings_root(output_root: str = BENCHMARK.output_root) -> Path:
    return offline_root(output_root) / "recordings"


def evaluations_root(output_root: str = BENCHMARK.output_root) -> Path:
    return offline_root(output_root) / "evaluations"


def timestamp_id(prefix: str = "") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{stamp}" if prefix else stamp


def slugify(value: object, fallback: str = "item") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return slug.strip("_") or fallback


def unique_folder(root: Path, base_name: str) -> Path:
    candidate = root / base_name
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = root / f"{base_name}_{index}"
        if not candidate.exists():
            return candidate
        index += 1


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def list_recorded_logs(output_root: str = BENCHMARK.output_root) -> list[RecordedLogInfo]:
    """Return route-level recorded sensor logs sorted newest first."""
    root = recordings_root(output_root)
    if not root.exists():
        return []

    infos: list[RecordedLogInfo] = []
    for run_folder in sorted((path for path in root.iterdir() if path.is_dir()), reverse=True):
        run_metadata = read_json(run_folder / "metadata.json")
        for route_folder in sorted(path for path in run_folder.iterdir() if path.is_dir()):
            sensor_log = route_folder / "sensor_log.csv"
            if not sensor_log.exists():
                continue
            summary = read_json(route_folder / "recording_summary.json")
            route_metadata = read_json(route_folder / "route_metadata.json")
            metadata = route_metadata or run_metadata
            route_info = metadata.get("route")
            route_name = ""
            map_name = ""
            if isinstance(route_info, dict):
                route_name = str(route_info.get("name") or "")
                map_name = str(route_info.get("map_name") or "")
            route_name = route_name or str(summary.get("route_name") or route_folder.name)
            map_name = map_name or str(summary.get("map_name") or metadata.get("map_name") or "")
            sensor_config = metadata.get("sensor_noise_config")
            behavior_config = metadata.get("vehicle_behavior_config")
            infos.append(
                RecordedLogInfo(
                    route_folder=route_folder,
                    sensor_log_path=sensor_log,
                    run_folder=run_folder,
                    recording_id=run_folder.name,
                    route_name=route_name,
                    map_name=map_name,
                    sample_count=_optional_int(summary.get("sample_count")),
                    duration_s=_optional_float(summary.get("duration_s")),
                    recording_driver=str(
                        summary.get("recording_driver")
                        or metadata.get("recording_driver")
                        or ""
                    ),
                    sensor_noise_preset=_preset_name(sensor_config),
                    vehicle_behavior_preset=_preset_name(behavior_config),
                    created_at=str(
                        summary.get("created_at")
                        or metadata.get("created_at")
                        or run_metadata.get("created_at")
                        or ""
                    ),
                    failure_reason=str(summary.get("failure_reason") or ""),
                )
            )
    return infos


def _preset_name(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("preset_name") or value.get("profile") or "")
    return ""


def _optional_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        number = float(value)
        return number if number == number else None
    return None


def _optional_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None

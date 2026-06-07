"""Offline auto-tuning for one filter over multiple recorded sensor logs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import random
from typing import Callable, Optional

from src.KalmanLab.registry import discover_filters
from src.evaluation.benchmark_config import project_commit_hash
from src.evaluation.evaluation_artifacts import (
    RecordedLogInfo,
    list_recorded_logs,
    offline_root,
    read_json,
    slugify,
    timestamp_id,
    unique_folder,
    write_json,
)
from src.evaluation.offline_replay_runner import OfflineReplayRequest, OfflineReplayRunner


@dataclass(frozen=True)
class AutoTuneRequest:
    """Configuration for one offline auto-tuning run."""

    filter_id: str
    sensor_log_paths: tuple[Path, ...]
    base_tune: dict[str, object]
    auto_tune_profile: dict[str, object]
    max_trials: int = 30
    objective_name: str = "rmse_consistency"
    output_root: str = "benchmark_results"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AutoTuneTrialResult:
    """Result for one candidate tune."""

    trial_index: int
    candidate_tune: dict[str, object]
    score: Optional[float]
    metrics: dict[str, object]
    output_folder: Optional[Path]
    failed: bool
    failure_reason: str = ""


@dataclass(frozen=True)
class AutoTuneResult:
    """Summary returned after auto-tuning finishes."""

    filter_id: str
    best_tune: dict[str, object]
    best_score: Optional[float]
    best_metrics: dict[str, object]
    selected_logs: tuple[Path, ...]
    trial_results: tuple[AutoTuneTrialResult, ...]
    output_folder: Path
    saved_config_path: Optional[Path] = None


ProgressCallback = Callable[[str, dict[str, object]], None]


class FilterAutoTuner:
    """Run a small framework-safe offline tune search for one filter."""

    def __init__(self, runner_factory: Callable[[], OfflineReplayRunner] = OfflineReplayRunner) -> None:
        self._records = {record.filter_id: record for record in discover_filters() if record.valid}
        self._runner_factory = runner_factory

    def run(
        self,
        request: AutoTuneRequest,
        progress_callback: Optional[ProgressCallback] = None,
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> AutoTuneResult:
        if request.filter_id == "raw_gnss":
            raise ValueError("Raw GNSS is a baseline and cannot be auto-tuned.")
        if not request.sensor_log_paths:
            raise ValueError("Select at least one recorded sensor log for auto tune.")
        record = self._records.get(request.filter_id)
        if record is None or record.filter_class is None:
            raise ValueError(f"Filter is not available: {request.filter_id}")
        if not record.auto_tune_enabled or not request.auto_tune_profile.get("primary"):
            raise ValueError(f"No auto-tune profile for filter: {request.filter_id}")

        max_trials = max(1, int(request.max_trials or 1))
        run_folder = unique_folder(
            _auto_tune_filter_root(request.output_root, request.filter_id),
            _run_folder_name(request.filter_id, request.sensor_log_paths, request.output_root),
        )
        run_folder.mkdir(parents=True, exist_ok=False)
        candidates = self._random_candidates(request, max_trials=max_trials)
        trial_results: list[AutoTuneTrialResult] = []
        best_tune: dict[str, object] = dict(request.base_tune)
        best_score: Optional[float] = None
        best_metrics: dict[str, object] = {}

        self._emit(progress_callback, "started", {"output_folder": str(run_folder), "trials": max_trials})
        trial_index = 1
        for candidate in candidates:
            if stop_requested is not None and stop_requested():
                self._emit(progress_callback, "stopped", {"trial_index": trial_index})
                break
            result = self._run_trial(request, run_folder, trial_index, candidate, progress_callback)
            trial_results.append(result)
            if not result.failed and result.score is not None and (best_score is None or result.score < best_score):
                best_score = result.score
                best_tune = dict(result.candidate_tune)
                best_metrics = dict(result.metrics)
                self._emit(progress_callback, "new_best", {"trial_index": trial_index, "score": best_score})
            trial_index += 1

        for candidate in self._coordinate_candidates(request, best_tune, remaining=max_trials - len(trial_results)):
            if stop_requested is not None and stop_requested():
                self._emit(progress_callback, "stopped", {"trial_index": trial_index})
                break
            result = self._run_trial(request, run_folder, trial_index, candidate, progress_callback)
            trial_results.append(result)
            if not result.failed and result.score is not None and (best_score is None or result.score < best_score):
                best_score = result.score
                best_tune = dict(result.candidate_tune)
                best_metrics = dict(result.metrics)
                self._emit(progress_callback, "new_best", {"trial_index": trial_index, "score": best_score})
            trial_index += 1

        self._write_trials_csv(run_folder / "trials.csv", trial_results)
        summary = self._summary_dict(request, record.display_name, run_folder, best_tune, best_score, best_metrics, trial_results)
        write_json(run_folder / "auto_tune_summary.json", summary)
        saved_config_path = None
        if best_score is not None:
            saved_config_path = self.save_best_tune(request, record.display_name, run_folder, best_tune, best_score, best_metrics, trial_results)
        self._emit(progress_callback, "completed", {"best_score": best_score, "saved_config_path": str(saved_config_path) if saved_config_path else ""})
        return AutoTuneResult(
            filter_id=request.filter_id,
            best_tune=best_tune,
            best_score=best_score,
            best_metrics=best_metrics,
            selected_logs=tuple(Path(path) for path in request.sensor_log_paths),
            trial_results=tuple(trial_results),
            output_folder=run_folder,
            saved_config_path=saved_config_path,
        )

    def _run_trial(
        self,
        request: AutoTuneRequest,
        run_folder: Path,
        trial_index: int,
        candidate_tune: dict[str, object],
        progress_callback: Optional[ProgressCallback],
    ) -> AutoTuneTrialResult:
        self._emit(
            progress_callback,
            "trial_started",
            {
                "trial_index": trial_index,
                "filter_id": request.filter_id,
                "candidate_tune": dict(candidate_tune),
                "log_count": len(request.sensor_log_paths),
            },
        )
        try:
            result = self._runner_factory().run(
                OfflineReplayRequest(
                    sensor_log_paths=tuple(Path(path) for path in request.sensor_log_paths),
                    selected_filter_ids=(request.filter_id,),
                    filter_tunes={request.filter_id: dict(candidate_tune)},
                    output_root=request.output_root,
                    include_raw_gnss_baseline=True,
                )
            )
            metrics = self._trial_metrics(result.output_folder, request.filter_id)
            score = objective_score(metrics, objective_name=request.objective_name, failure_count=len(result.failures))
            failed = not math.isfinite(score)
            failure_reason = "" if not failed else "Objective score was not finite."
            trial_result = AutoTuneTrialResult(
                trial_index=trial_index,
                candidate_tune=dict(candidate_tune),
                score=score,
                metrics=metrics,
                output_folder=result.output_folder,
                failed=failed,
                failure_reason=failure_reason,
            )
        except Exception as exc:
            trial_result = AutoTuneTrialResult(
                trial_index=trial_index,
                candidate_tune=dict(candidate_tune),
                score=None,
                metrics={},
                output_folder=None,
                failed=True,
                failure_reason=str(exc),
            )
        write_json(run_folder / f"trial_{trial_index:03d}.json", _trial_to_dict(trial_result))
        self._emit(
            progress_callback,
            "trial_finished",
            {
                "trial_index": trial_index,
                "score": trial_result.score,
                "failed": trial_result.failed,
                "failure_reason": trial_result.failure_reason,
            },
        )
        return trial_result

    def save_best_tune(
        self,
        request: AutoTuneRequest,
        filter_display_name: str,
        run_folder: Path,
        best_tune: dict[str, object],
        best_score: float,
        best_metrics: dict[str, object],
        trial_results: list[AutoTuneTrialResult],
    ) -> Path:
        config = self._saved_config_dict(
            request=request,
            filter_display_name=filter_display_name,
            run_folder=run_folder,
            best_tune=best_tune,
            best_score=best_score,
            best_metrics=best_metrics,
            trial_count=len(trial_results),
        )
        config_path = run_folder / "best_tune.json"
        write_json(config_path, config)
        _update_saved_config_index(request.output_root, request.filter_id, config_path, config)
        return config_path

    def _random_candidates(self, request: AutoTuneRequest, max_trials: int) -> list[dict[str, object]]:
        primary = list(request.auto_tune_profile.get("primary") or [])
        random_budget = max(1, max_trials - len(primary) * 5)
        rng = random.Random(int(request.metadata.get("random_seed", 4084)))
        candidates = [dict(request.base_tune)]
        while len(candidates) < random_budget:
            candidate = dict(request.base_tune)
            for param in primary:
                key = str(param.get("key") or "")
                if not key:
                    continue
                candidate[key] = _sample_param(rng, param, request.base_tune.get(key))
            candidates.append(candidate)
        return candidates[:max_trials]

    def _coordinate_candidates(
        self,
        request: AutoTuneRequest,
        best_tune: dict[str, object],
        remaining: int,
    ) -> list[dict[str, object]]:
        if remaining <= 0:
            return []
        candidates: list[dict[str, object]] = []
        for param in list(request.auto_tune_profile.get("primary") or []):
            key = str(param.get("key") or "")
            base_value = _optional_float(best_tune.get(key))
            if not key or base_value is None:
                continue
            for multiplier in (0.5, 0.75, 1.25, 1.5, 2.0):
                candidate = dict(best_tune)
                candidate[key] = _clamp_value(base_value * multiplier, param)
                candidates.append(candidate)
                if len(candidates) >= remaining:
                    return candidates
        return candidates

    @staticmethod
    def _trial_metrics(output_folder: Path, filter_id: str) -> dict[str, object]:
        aggregate = read_json(Path(output_folder) / "aggregate_summary.json")
        rows = aggregate.get("aggregate_rows")
        filter_rows = [row for row in rows if isinstance(row, dict) and row.get("filter_id") == filter_id] if isinstance(rows, list) else []
        eval_rmses = [_optional_float(row.get("eval_position_rmse_m") or row.get("position_rmse_m")) for row in filter_rows]
        yaw_rmses = [_optional_float(row.get("yaw_rmse_deg")) for row in filter_rows]
        divergence_counts = [_optional_float(row.get("divergence_event_count")) for row in filter_rows]
        nis_values = [_optional_float(row.get("mean_nis")) for row in filter_rows]
        nees_values = [_optional_float(row.get("mean_nees")) for row in filter_rows]
        return {
            "route_count": aggregate.get("route_count"),
            "mean_eval_position_rmse_m": _mean([value for value in eval_rmses if value is not None]),
            "mean_yaw_rmse_deg": _mean([value for value in yaw_rmses if value is not None]),
            "divergence_event_count": sum(value for value in divergence_counts if value is not None),
            "mean_nis": _mean([value for value in nis_values if value is not None]),
            "mean_nees": _mean([value for value in nees_values if value is not None]),
            "best_filter_id": aggregate.get("best_filter_id"),
            "raw_gnss_rmse_m": aggregate.get("raw_gnss_rmse_m"),
            "aggregate_summary_path": str(Path(output_folder) / "aggregate_summary.json"),
            "output_folder": str(output_folder),
        }

    def _summary_dict(
        self,
        request: AutoTuneRequest,
        filter_display_name: str,
        run_folder: Path,
        best_tune: dict[str, object],
        best_score: Optional[float],
        best_metrics: dict[str, object],
        trial_results: list[AutoTuneTrialResult],
    ) -> dict[str, object]:
        return {
            "filter_id": request.filter_id,
            "filter_display_name": filter_display_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "objective": request.objective_name,
            "score_formula": _score_formula_description(),
            "best_score": best_score,
            "best_metrics": best_metrics,
            "best_tune": dict(best_tune),
            "base_tune": dict(request.base_tune),
            "auto_tune_profile": dict(request.auto_tune_profile),
            "selected_logs": [_log_metadata(path, request.output_root) for path in request.sensor_log_paths],
            "trial_count": len(trial_results),
            "project_commit": project_commit_hash(),
            "output_folder": str(run_folder),
            "metadata": dict(request.metadata),
        }

    def _saved_config_dict(
        self,
        request: AutoTuneRequest,
        filter_display_name: str,
        run_folder: Path,
        best_tune: dict[str, object],
        best_score: float,
        best_metrics: dict[str, object],
        trial_count: int,
    ) -> dict[str, object]:
        logs = [_log_metadata(path, request.output_root) for path in request.sensor_log_paths]
        noise = noise_profile_summary(logs)
        return {
            "filter_id": request.filter_id,
            "filter_display_name": filter_display_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": "offline_auto_tune",
            "noise_profile_label": noise["label"],
            "noise_signature": noise["signature"],
            "sensor_noise_config": noise.get("representative_config") or {},
            "selected_logs": logs,
            "objective": request.objective_name,
            "score": best_score,
            "best_metrics": best_metrics,
            "best_tune": dict(best_tune),
            "base_tune": dict(request.base_tune),
            "auto_tune_profile": dict(request.auto_tune_profile),
            "trial_count": trial_count,
            "project_commit": project_commit_hash(),
            "output_folder": str(run_folder),
        }

    @staticmethod
    def _write_trials_csv(path: Path, trial_results: list[AutoTuneTrialResult]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=("trial_index", "score", "failed", "failure_reason", "candidate_tune", "metrics", "output_folder"),
            )
            writer.writeheader()
            for result in trial_results:
                writer.writerow(
                    {
                        "trial_index": result.trial_index,
                        "score": result.score,
                        "failed": result.failed,
                        "failure_reason": result.failure_reason,
                        "candidate_tune": json.dumps(result.candidate_tune, sort_keys=True),
                        "metrics": json.dumps(result.metrics, sort_keys=True),
                        "output_folder": str(result.output_folder) if result.output_folder else "",
                    }
                )

    @staticmethod
    def _emit(callback: Optional[ProgressCallback], event: str, payload: dict[str, object]) -> None:
        if callback is not None:
            callback(event, payload)


def objective_score(metrics: dict[str, object], objective_name: str = "rmse_consistency", failure_count: int = 0) -> float:
    """Compute a simple explainable score where lower is better."""
    rmse = _optional_float(metrics.get("mean_eval_position_rmse_m"))
    if rmse is None:
        return 1.0e9
    yaw = _optional_float(metrics.get("mean_yaw_rmse_deg")) or 0.0
    divergence = _optional_float(metrics.get("divergence_event_count")) or 0.0
    nis = _optional_float(metrics.get("mean_nis"))
    nees = _optional_float(metrics.get("mean_nees"))
    consistency_penalty = 0.0
    if nis is not None and nis > 9.0:
        consistency_penalty += 0.05 * (nis - 9.0)
    if nees is not None and nees > 12.0:
        consistency_penalty += 0.05 * (nees - 12.0)
    return rmse + 0.01 * yaw + 10.0 * divergence + 100.0 * max(0, failure_count) + consistency_penalty


def list_saved_tune_configs(filter_id: str, output_root: str = "benchmark_results") -> list[dict[str, object]]:
    index = read_json(_saved_config_index_path(output_root, filter_id))
    items = index.get("configs")
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, dict)]


def load_saved_tune_config(path: Path) -> dict[str, object]:
    return read_json(Path(path))


def noise_profile_summary(logs: list[dict[str, object]]) -> dict[str, object]:
    labels = {str(log.get("sensor_noise_preset") or "Custom") for log in logs}
    signatures = {json.dumps(log.get("sensor_noise_config") or {}, sort_keys=True) for log in logs}
    if not logs:
        label = "Unknown"
    elif len(labels) == 1:
        label = next(iter(labels))
    else:
        label = "Mixed Noise"
    signature = "mixed" if len(signatures) > 1 else (next(iter(signatures)) if signatures else "")
    representative_config = logs[0].get("sensor_noise_config") if logs else {}
    return {
        "label": label,
        "signature": signature,
        "representative_config": representative_config if isinstance(representative_config, dict) else {},
        "mixed": label == "Mixed Noise",
    }


def _auto_tune_filter_root(output_root: str, filter_id: str) -> Path:
    return offline_root(output_root) / "auto_tune" / slugify(filter_id, "filter")


def _saved_config_index_path(output_root: str, filter_id: str) -> Path:
    return _auto_tune_filter_root(output_root, filter_id) / "saved_tune_configs.json"


def _update_saved_config_index(output_root: str, filter_id: str, config_path: Path, config: dict[str, object]) -> None:
    index_path = _saved_config_index_path(output_root, filter_id)
    index = read_json(index_path)
    configs = index.get("configs")
    items = [dict(item) for item in configs if isinstance(item, dict)] if isinstance(configs, list) else []
    entry = {
        "path": str(config_path),
        "filter_id": filter_id,
        "created_at": config.get("created_at"),
        "noise_profile_label": config.get("noise_profile_label"),
        "noise_signature": config.get("noise_signature"),
        "score": config.get("score"),
        "mean_eval_position_rmse_m": (config.get("best_metrics") or {}).get("mean_eval_position_rmse_m")
        if isinstance(config.get("best_metrics"), dict)
        else None,
        "log_count": len(config.get("selected_logs") or []),
        "output_folder": config.get("output_folder"),
    }
    items = [item for item in items if item.get("path") != str(config_path)]
    items.insert(0, entry)
    write_json(index_path, {"configs": items})


def _run_folder_name(filter_id: str, sensor_log_paths: tuple[Path, ...], output_root: str) -> str:
    logs = [_log_metadata(path, output_root) for path in sensor_log_paths]
    noise = noise_profile_summary(logs)
    return f"{timestamp_id()}_{slugify(filter_id, 'filter')}_{slugify(noise['label'], 'noise')}"


def _log_metadata(path: Path, output_root: str) -> dict[str, object]:
    path = Path(path)
    matching = _recorded_info_by_path(output_root).get(str(path.resolve()))
    route_metadata = read_json(path.parent / "route_metadata.json")
    sensor_noise_config = route_metadata.get("sensor_noise_config")
    if matching is not None:
        return {
            "route_name": matching.route_name,
            "map_name": matching.map_name,
            "sensor_log_path": str(matching.sensor_log_path),
            "sample_count": matching.sample_count,
            "sensor_noise_preset": matching.sensor_noise_preset,
            "sensor_noise_config": sensor_noise_config if isinstance(sensor_noise_config, dict) else {},
            "recording_driver": matching.recording_driver,
            "behavior_preset": matching.vehicle_behavior_preset,
            "created_at": matching.created_at,
            "recording_id": matching.recording_id,
        }
    summary = read_json(path.parent / "recording_summary.json")
    return {
        "route_name": str(summary.get("route_name") or path.parent.name),
        "map_name": str(summary.get("map_name") or route_metadata.get("map_name") or ""),
        "sensor_log_path": str(path),
        "sample_count": summary.get("sample_count"),
        "sensor_noise_preset": _preset_name(sensor_noise_config),
        "sensor_noise_config": sensor_noise_config if isinstance(sensor_noise_config, dict) else {},
        "recording_driver": str(summary.get("recording_driver") or route_metadata.get("recording_driver") or ""),
        "behavior_preset": _preset_name(route_metadata.get("vehicle_behavior_config")),
        "created_at": str(summary.get("created_at") or route_metadata.get("created_at") or ""),
        "recording_id": path.parent.parent.name,
    }


def _recorded_info_by_path(output_root: str) -> dict[str, RecordedLogInfo]:
    result: dict[str, RecordedLogInfo] = {}
    for info in list_recorded_logs(output_root):
        try:
            result[str(info.sensor_log_path.resolve())] = info
        except OSError:
            result[str(info.sensor_log_path)] = info
    return result


def _trial_to_dict(result: AutoTuneTrialResult) -> dict[str, object]:
    return {
        "trial_index": result.trial_index,
        "candidate_tune": dict(result.candidate_tune),
        "score": result.score,
        "metrics": dict(result.metrics),
        "output_folder": str(result.output_folder) if result.output_folder else None,
        "failed": result.failed,
        "failure_reason": result.failure_reason,
    }


def _sample_param(rng: random.Random, param: dict[str, object], fallback: object) -> float:
    low = _optional_float(param.get("min"))
    high = _optional_float(param.get("max"))
    base = _optional_float(fallback)
    if low is None:
        low = max(1.0e-9, (base or 1.0) * 0.25)
    if high is None:
        high = max(low, (base or low) * 4.0)
    if str(param.get("scale") or "").lower() == "log" and low > 0.0 and high > low:
        return math.exp(rng.uniform(math.log(low), math.log(high)))
    return rng.uniform(low, high)


def _clamp_value(value: float, param: dict[str, object]) -> float:
    low = _optional_float(param.get("min"))
    high = _optional_float(param.get("max"))
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _score_formula_description() -> str:
    return (
        "score = mean_eval_position_rmse_m + 0.01 * mean_yaw_rmse_deg + "
        "10 * divergence_event_count + 100 * failures + consistency_penalty; "
        "NIS/NEES penalties are skipped when unavailable."
    )


def _preset_name(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("preset_name") or value.get("profile") or "")
    return ""


def _optional_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)

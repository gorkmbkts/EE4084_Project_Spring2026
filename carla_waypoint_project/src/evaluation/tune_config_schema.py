"""Schema v2 saved tune metadata and compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Optional

from src.evaluation.benchmark_config import project_commit_hash
from src.evaluation.evaluation_artifacts import slugify
from src.evaluation.sensor_noise_tune_mapper import noise_signature


SCHEMA_VERSION = 2
TUNER_KIND_OFFLINE = "offline_benchmark_autotuner"
TUNER_KIND_CLOSED_LOOP = "closed_loop_benchmark_autotuner"
BENCHMARK_MODE_OFFLINE = "offline_replay"
BENCHMARK_MODE_CLOSED_LOOP = "closed_loop"
TRACKING_PASSIVE = "passive"
TRACKING_ACTIVE = "active"
TUNE_SCOPE_OFFLINE = "offline_process_optimal"
TUNE_SCOPE_CLOSED_LOOP_CANDIDATE = "closed_loop_candidate"
TUNE_SCOPE_CLOSED_LOOP_VALIDATED = "closed_loop_validated"


@dataclass(frozen=True)
class TuneContext:
    filter_id: str
    benchmark_mode: str
    tracking_mode: str
    sensor_noise_signature: Optional[str] = None
    allowed_tune_scopes: tuple[str, ...] = ()
    vehicle_behavior_signature: Optional[str] = None
    actuator_realism_signature: Optional[str] = None
    include_legacy: bool = False


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    reason: str = ""


class TuneCompatibility:
    """Validate saved tune configs against the current benchmark context."""

    @staticmethod
    def check(config: dict[str, object], context: TuneContext) -> CompatibilityResult:
        if int(config.get("schema_version") or 0) != SCHEMA_VERSION:
            if context.include_legacy:
                return CompatibilityResult(True, "legacy_config_allowed_by_debug_context")
            return CompatibilityResult(False, "legacy config hidden: schema_version 2 is required")
        if str(config.get("filter_id") or "") != context.filter_id:
            return CompatibilityResult(False, "filter_id does not match selected filter")
        if str(config.get("benchmark_mode") or "") != context.benchmark_mode:
            return CompatibilityResult(False, "benchmark_mode does not match current benchmark")
        if str(config.get("tracking_mode") or "") != context.tracking_mode:
            return CompatibilityResult(False, "tracking_mode does not match current tracking mode")
        if context.allowed_tune_scopes and str(config.get("tune_scope") or "") not in context.allowed_tune_scopes:
            return CompatibilityResult(False, "tune_scope is not valid for this context")
        config_noise = str(config.get("noise_signature") or "")
        if context.sensor_noise_signature:
            if not config_noise:
                return CompatibilityResult(False, "saved tune is missing sensor noise signature")
            if config_noise != context.sensor_noise_signature:
                return CompatibilityResult(False, "sensor noise profile/signature is incompatible")
        config_behavior = str(config.get("vehicle_behavior_signature") or "")
        if context.vehicle_behavior_signature and config_behavior and config_behavior != context.vehicle_behavior_signature:
            return CompatibilityResult(False, "vehicle behavior signature is incompatible")
        config_actuator = str(config.get("actuator_realism_signature") or "")
        if context.actuator_realism_signature and config_actuator and config_actuator != context.actuator_realism_signature:
            return CompatibilityResult(False, "actuator realism signature is incompatible")
        return CompatibilityResult(True, "")


def offline_tune_context(
    filter_id: str,
    sensor_noise_config: object | None = None,
    include_legacy: bool = False,
) -> TuneContext:
    return TuneContext(
        filter_id=filter_id,
        benchmark_mode=BENCHMARK_MODE_OFFLINE,
        tracking_mode=TRACKING_PASSIVE,
        sensor_noise_signature=noise_signature(sensor_noise_config) if sensor_noise_config is not None else None,
        allowed_tune_scopes=(TUNE_SCOPE_OFFLINE,),
        include_legacy=include_legacy,
    )


def closed_loop_tune_context(
    filter_id: str,
    tracking_mode: str,
    sensor_noise_config: object | None = None,
    vehicle_behavior_config: object | None = None,
    actuator_realism_config: object | None = None,
    include_legacy: bool = False,
) -> TuneContext:
    return TuneContext(
        filter_id=filter_id,
        benchmark_mode=BENCHMARK_MODE_CLOSED_LOOP,
        tracking_mode=str(tracking_mode or TRACKING_PASSIVE),
        sensor_noise_signature=noise_signature(sensor_noise_config) if sensor_noise_config is not None else None,
        allowed_tune_scopes=(TUNE_SCOPE_CLOSED_LOOP_CANDIDATE, TUNE_SCOPE_CLOSED_LOOP_VALIDATED),
        vehicle_behavior_signature=config_signature(vehicle_behavior_config) if vehicle_behavior_config is not None else None,
        actuator_realism_signature=config_signature(actuator_realism_config) if actuator_realism_config is not None else None,
        include_legacy=include_legacy,
    )


def build_offline_schema_v2_config(
    *,
    filter_id: str,
    filter_display_name: str,
    sensor_noise_profile: str,
    noise_sig: str,
    representative_sensor_noise_config: dict[str, object],
    selected_logs: list[dict[str, object]],
    candidate_generation_strategy: str,
    optuna_available: bool,
    optuna_study_path: Optional[str],
    score: object,
    best_metrics: dict[str, object],
    best_tune: dict[str, object],
    base_tune: dict[str, object],
    locked_sensor_noise_values: dict[str, object],
    output_folder: Path,
    extra: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "filter_id": filter_id,
        "filter_display_name": filter_display_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "created_by": "carla_waypoint_project.auto_tuner",
        "tuner_kind": TUNER_KIND_OFFLINE,
        "benchmark_mode": BENCHMARK_MODE_OFFLINE,
        "tracking_mode": TRACKING_PASSIVE,
        "tune_scope": TUNE_SCOPE_OFFLINE,
        "recommended_usage": "offline_passive",
        "sensor_noise_locked_from_profile": True,
        "process_only_tune": True,
        "sensor_noise_profile": sensor_noise_profile,
        "noise_signature": noise_sig,
        "representative_sensor_noise_config": dict(representative_sensor_noise_config),
        "selected_offline_logs": selected_logs,
        "selected_logs": selected_logs,
        "candidate_generation_strategy": candidate_generation_strategy,
        "optuna_available": bool(optuna_available),
        "optuna_study_path": optuna_study_path,
        "finalist_count": None,
        "closed_loop_validation_results": None,
        "validated_in_closed_loop": False,
        "score": score,
        "best_metrics": dict(best_metrics),
        "best_tune": dict(best_tune),
        "base_tune": dict(base_tune),
        "locked_sensor_noise_values": dict(locked_sensor_noise_values),
        "project_commit": project_commit_hash(),
        "output_folder": str(output_folder),
    }
    if extra:
        data.update(extra)
    return data


def build_closed_loop_schema_v2_config(
    *,
    filter_id: str,
    filter_display_name: str,
    tracking_mode: str,
    sensor_noise_profile: str,
    noise_sig: str,
    representative_sensor_noise_config: dict[str, object],
    vehicle_behavior_profile: str,
    vehicle_behavior_config: dict[str, object],
    actuator_realism_enabled: bool,
    actuator_realism_profile: str,
    actuator_realism_config: dict[str, object],
    validation_route_name: str,
    validation_route_map: str,
    validation_route_id: str,
    selected_logs: list[dict[str, object]],
    candidate_generation_strategy: str,
    optuna_available: bool,
    optuna_study_path: Optional[str],
    finalist_count: int,
    offline_candidate_results: list[dict[str, object]],
    closed_loop_validation_results: list[dict[str, object]],
    score: object,
    best_metrics: dict[str, object],
    best_tune: dict[str, object],
    base_tune: dict[str, object],
    locked_sensor_noise_values: dict[str, object],
    output_folder: Path,
    extra: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    tracking = str(tracking_mode or TRACKING_PASSIVE)
    recommended_usage = "closed_loop_active" if tracking == TRACKING_ACTIVE else "closed_loop_passive"
    data: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "filter_id": filter_id,
        "filter_display_name": filter_display_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "created_by": "carla_waypoint_project.auto_tuner",
        "tuner_kind": TUNER_KIND_CLOSED_LOOP,
        "benchmark_mode": BENCHMARK_MODE_CLOSED_LOOP,
        "tracking_mode": tracking,
        "tune_scope": TUNE_SCOPE_CLOSED_LOOP_VALIDATED,
        "recommended_usage": recommended_usage,
        "sensor_noise_locked_from_profile": True,
        "process_only_tune": tracking != TRACKING_ACTIVE,
        "active_control_tune": tracking == TRACKING_ACTIVE,
        "sensor_noise_profile": sensor_noise_profile,
        "noise_signature": noise_sig,
        "representative_sensor_noise_config": dict(representative_sensor_noise_config),
        "vehicle_behavior_profile": vehicle_behavior_profile,
        "vehicle_behavior_signature": config_signature(vehicle_behavior_config),
        "vehicle_behavior_config": dict(vehicle_behavior_config),
        "actuator_realism_enabled": bool(actuator_realism_enabled),
        "actuator_realism_profile": actuator_realism_profile,
        "actuator_realism_signature": config_signature(actuator_realism_config),
        "actuator_realism_config": dict(actuator_realism_config),
        "validation_route_name": validation_route_name,
        "validation_route_map": validation_route_map,
        "validation_route_id": validation_route_id,
        "validation_route_signature": config_signature(
            {
                "name": validation_route_name,
                "map": validation_route_map,
                "id": validation_route_id,
            }
        ),
        "selected_offline_logs": selected_logs,
        "selected_logs": selected_logs,
        "candidate_generation_strategy": candidate_generation_strategy,
        "optuna_available": bool(optuna_available),
        "optuna_study_path": optuna_study_path,
        "finalist_count": int(finalist_count),
        "offline_candidate_results": offline_candidate_results,
        "closed_loop_validation_results": closed_loop_validation_results,
        "validated_in_closed_loop": True,
        "score": score,
        "best_metrics": dict(best_metrics),
        "best_tune": dict(best_tune),
        "base_tune": dict(base_tune),
        "locked_sensor_noise_values": dict(locked_sensor_noise_values),
        "project_commit": project_commit_hash(),
        "output_folder": str(output_folder),
    }
    if extra:
        data.update(extra)
    return data


def config_signature(config: object) -> str:
    if isinstance(config, dict):
        data = config
    else:
        to_dict = getattr(config, "to_dict", None)
        if callable(to_dict):
            try:
                result = to_dict()
            except Exception:
                result = {}
            data = result if isinstance(result, dict) else {}
        else:
            data = {}
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def noise_signature_slug(value: object) -> str:
    text = str(value or "unknown_noise")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return slugify(f"n_{digest}", "noise")

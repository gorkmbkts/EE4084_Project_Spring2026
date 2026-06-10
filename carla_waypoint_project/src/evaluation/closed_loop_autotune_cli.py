"""Non-interactive CARLA closed-loop auto-tune runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", default="ctra_ukf", dest="filter_id")
    parser.add_argument("--tracking", choices=("passive", "active"), required=True)
    parser.add_argument("--noise", choices=("Medium Noise", "High Noise"), required=True)
    parser.add_argument(
        "--actuator",
        choices=("Perfect Actuator", "Mild Realistic", "Realistic", "Delayed / Harsh Realistic"),
        required=True,
    )
    parser.add_argument("--behavior", default="Balanced")
    parser.add_argument("--route", default="mahalle")
    parser.add_argument(
        "--strategy",
        choices=("random_plus_coordinate_refinement", "optuna_tpe"),
        default="optuna_tpe",
    )
    parser.add_argument("--passive-trials", type=int, default=10)
    parser.add_argument("--active-trials", type=int, default=10)
    parser.add_argument("--joint-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4084)
    parser.add_argument("--output-root", default="benchmark_results")
    return parser


def main() -> None:
    args = _parser().parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    from src.KalmanLab.registry import discover_filters
    from src.core.app import SimulationApp
    from src.evaluation.benchmark_config import (
        ACTUATOR_REALISM_PRESETS,
        BEHAVIOR_PRESETS,
        SENSOR_NOISE_PRESETS,
        actuator_realism_from_values,
        driving_behavior_from_values,
        sensor_noise_config_from_values,
    )
    from src.evaluation.closed_loop_auto_tune import ClosedLoopAutoTuneRequest, ClosedLoopValidationRoute
    from src.evaluation.sensor_noise_tune_mapper import noise_signature
    from src.evaluation.test_route_store import TestRouteStore
    from src.evaluation.tune_config_schema import TRACKING_ACTIVE
    from src.utils.map_names import display_map_name

    records = {record.filter_id: record for record in discover_filters() if record.valid}
    record = records.get(args.filter_id)
    if record is None or not record.auto_tune_enabled or not record.auto_tune_profile:
        raise SystemExit(f"Filter has no closed-loop auto-tune profile: {args.filter_id}")

    route = next((item for item in TestRouteStore().all_routes if item.name == args.route), None)
    if route is None or not route.map_name:
        raise SystemExit(f"Saved route is unavailable or has no map: {args.route}")

    sensor_config = sensor_noise_config_from_values(
        SENSOR_NOISE_PRESETS[args.noise],
        preset_name=args.noise,
    ).to_dict()
    behavior_config = driving_behavior_from_values(
        BEHAVIOR_PRESETS[args.behavior],
        preset_name=args.behavior,
    )
    actuator_config = actuator_realism_from_values(
        ACTUATOR_REALISM_PRESETS[args.actuator],
        preset_name=args.actuator,
    )
    active_trials = max(1, args.active_trials) if args.tracking == TRACKING_ACTIVE else 0
    passive_trials = max(1, args.passive_trials)
    joint_trials = max(0, args.joint_trials)
    route_data = route.to_dict()
    route_id = f"{route.name}@{route.map_name}"
    request = ClosedLoopAutoTuneRequest(
        filter_id=args.filter_id,
        tracking_mode=args.tracking,
        validation_routes=(
            ClosedLoopValidationRoute(
                name=route.name,
                map_name=route.map_name,
                route_id=route_id,
                route_data=route_data,
            ),
        ),
        sensor_noise_config=sensor_config,
        vehicle_behavior_config=behavior_config,
        actuator_realism_config=actuator_config,
        base_tune=dict(record.tune),
        auto_tune_profile=dict(record.auto_tune_profile),
        sensor_noise_profile=args.noise,
        vehicle_behavior_profile=args.behavior,
        actuator_realism_enabled=True,
        actuator_realism_profile=args.actuator,
        trial_count=passive_trials + active_trials + joint_trials,
        passive_model_trials=passive_trials,
        active_control_trials=active_trials,
        joint_fine_tune_trials=joint_trials,
        finalist_count=1,
        strategy=args.strategy,
        output_root=args.output_root,
        random_seed=args.seed,
        keep_trial_outputs=True,
        generate_trial_plots=False,
        metadata={
            "startup_mode": "closed_loop_autotune_cli",
            "validation_route_data": route_data,
            "selected_sensor_noise_signature": noise_signature(sensor_config),
            "selected_sensor_noise_config": sensor_config,
            "direct_closed_loop_mode": True,
            "no_rendering_mode": True,
            "offscreen_mode": True,
            "route_attempt_policy": "one_attempt_per_candidate_trial",
        },
    )

    app = SimulationApp(
        requested_map_name=display_map_name(route.map_name),
        selected_map_load_name=display_map_name(route.map_name),
        closed_loop_auto_tune_request=request,
    )
    try:
        app._setup()
        result = app._closed_loop_auto_tune_result
        if result is None:
            raise RuntimeError(app._closed_loop_auto_tune_current_status)
        payload = {
            "saved_config_path": str(result.saved_config_path),
            "output_folder": str(result.output_folder),
            "tracking_mode": result.tracking_mode,
            "strategy": args.strategy,
            "best_score": result.best_score,
            "baseline_score": result.offline_auto_tune_result.baseline_score,
            "improved_over_baseline": result.offline_auto_tune_result.improved_over_baseline,
        }
        print(json.dumps(payload, indent=2), flush=True)
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()

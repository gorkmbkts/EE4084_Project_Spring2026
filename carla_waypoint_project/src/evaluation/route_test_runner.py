"""Saved-route benchmark execution and automated multi-route sequencing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import csv
from datetime import datetime
from enum import Enum
import json
import math
from pathlib import Path
import re
import time
from typing import Optional

from config.settings import BENCHMARK
from src.evaluation.benchmark_config import BenchmarkConfig, benchmark_output_root, project_commit_hash
from src.evaluation.benchmark_metadata import build_benchmark_metadata
from src.evaluation.filter_performance import FilterPerformanceLogger
from src.evaluation.test_route_store import SavedTestRoute, TestRouteStore
from src.utils.carla_import import ensure_carla_import
from src.utils.map_names import display_map_name, map_slug, maps_compatible, normalize_map_name

carla = ensure_carla_import()


class BenchmarkRunnerState(Enum):
    IDLE = "IDLE"
    LOADING_MAP = "LOADING_MAP"
    INITIALIZING_ROUTE = "INITIALIZING_ROUTE"
    RUNNING_ROUTE = "RUNNING_ROUTE"
    ROUTE_FINISHED = "ROUTE_FINISHED"
    SWITCHING_ROUTE = "SWITCHING_ROUTE"
    SWITCHING_MAP = "SWITCHING_MAP"
    TEST_FINISHED = "TEST_FINISHED"
    ERROR = "ERROR"


MAX_ROUTE_ATTEMPTS = 5


class RouteTestRunner:
    """Run one saved route or a full automated filter-evaluation test."""

    def __init__(
        self,
        world_map: "carla.Map",
        route_store: TestRouteStore,
        begin_route_callback: Callable[["carla.Waypoint", "carla.Waypoint"], None],
        reset_estimator_callback: Callable[[], None],
        plan_route_callback: Callable[["carla.Waypoint", "carla.Waypoint"], Sequence["carla.Waypoint"]],
        weather_callback: Callable[[], Optional[dict[str, object]]],
        vehicle_blueprint_callback: Callable[[], Optional[str]],
        active_filter_info_callback: Callable[[], dict[str, object]],
        active_filter_tune_callback: Callable[[], dict[str, object]],
        tracking_mode_callback: Callable[[], str] | None = None,
        active_control_used_callback: Callable[[], bool] | None = None,
        enqueue_route_plots_callback: Callable[[Path], bool] | None = None,
        enqueue_aggregate_plots_callback: Callable[[Path], bool] | None = None,
        selected_map_load_name: Optional[str] = None,
    ) -> None:
        self._world_map = world_map
        self._route_store = route_store
        self._begin_route_callback = begin_route_callback
        self._reset_estimator_callback = reset_estimator_callback
        self._plan_route_callback = plan_route_callback
        self._weather_callback = weather_callback
        self._vehicle_blueprint_callback = vehicle_blueprint_callback
        self._active_filter_info_callback = active_filter_info_callback
        self._active_filter_tune_callback = active_filter_tune_callback
        self._tracking_mode_callback = tracking_mode_callback or (lambda: "passive")
        self._active_control_used_callback = active_control_used_callback or (lambda: False)
        self._enqueue_route_plots_callback = enqueue_route_plots_callback
        self._enqueue_aggregate_plots_callback = enqueue_aggregate_plots_callback
        self._selected_map_load_name = selected_map_load_name

        self._active = False
        self._automated = False
        self._route_running = False
        self._state = BenchmarkRunnerState.IDLE
        self._routes: list[SavedTestRoute] = []
        self._current_route: Optional[SavedTestRoute] = None
        self._current_route_index = 0
        self._current_logger: Optional[FilterPerformanceLogger] = None
        self._benchmark_id: Optional[str] = None
        self._benchmark_folder: Optional[Path] = None
        self._run_folder: Optional[Path] = None
        self._config: Optional[BenchmarkConfig] = None
        self._started_monotonic: Optional[float] = None
        self._test_started_monotonic: Optional[float] = None
        self._last_exported_summary: Optional[dict[str, object]] = None
        self._route_summaries: list[dict[str, object]] = []
        self._status_text = "Benchmark idle"
        self._current_attempt = 0
        self._max_route_attempts = MAX_ROUTE_ATTEMPTS
        self._attempt_failures_by_route: dict[int, list[dict[str, object]]] = {}
        self._current_attempt_folder: Optional[Path] = None
        self._last_failure_reason = ""
        self._route_status = "idle"

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_automated(self) -> bool:
        return self._automated

    @property
    def route_running(self) -> bool:
        return self._route_running

    @property
    def state(self) -> BenchmarkRunnerState:
        return self._state

    @property
    def benchmark_id(self) -> Optional[str]:
        return self._benchmark_id or (self._config.run_id if self._config is not None else None)

    @property
    def benchmark_folder(self) -> Optional[Path]:
        if self._automated and self._run_folder is not None:
            return self._run_folder
        return self._benchmark_folder

    @property
    def current_route_folder(self) -> Optional[Path]:
        return self._benchmark_folder

    @property
    def current_logger(self) -> Optional[FilterPerformanceLogger]:
        return self._current_logger

    @property
    def current_route_name(self) -> str:
        return self._current_route.name if self._current_route is not None else ""

    @property
    def current_route_index(self) -> int:
        return self._current_route_index

    @property
    def total_routes(self) -> int:
        return len(self._routes)

    @property
    def current_attempt(self) -> int:
        return self._current_attempt

    @property
    def max_attempts(self) -> int:
        return self._max_route_attempts

    @property
    def last_failure_reason(self) -> str:
        return self._last_failure_reason

    @property
    def route_status(self) -> str:
        return self._route_status

    @property
    def status_text(self) -> str:
        return self._status_text

    @property
    def last_exported_summary(self) -> Optional[dict[str, object]]:
        return self._last_exported_summary

    @property
    def config(self) -> Optional[BenchmarkConfig]:
        return self._config

    def update_world_context(
        self,
        world_map: "carla.Map",
        route_store: TestRouteStore,
        selected_map_load_name: Optional[str] = None,
    ) -> None:
        self._world_map = world_map
        self._route_store = route_store
        self._selected_map_load_name = selected_map_load_name

    def start_selected_route(self, route: Optional[SavedTestRoute]) -> bool:
        if self._active:
            self._status_text = "Benchmark already running"
            return False
        if route is None:
            self._status_text = "Select or create a test route first"
            return False
        self._active = True
        self._automated = False
        self._routes = [route]
        self._route_summaries = []
        self._attempt_failures_by_route.clear()
        self._current_attempt = 0
        self._last_failure_reason = ""
        self._route_status = "initializing"
        started = self._start_route(route, self._route_store.current_index, route_folder=None)
        if not started:
            self._active = False
        return started

    def start_configured_benchmark(self, config: BenchmarkConfig, active_map_name: Optional[str]) -> bool:
        if self._active:
            self._status_text = "Benchmark already running"
            return False
        if not config.selected_routes:
            self._status_text = "Benchmark blocked: no selected routes"
            return False

        run_folder = _unique_folder(benchmark_output_root(config.output_root), config.run_id)
        run_folder.mkdir(parents=True, exist_ok=False)
        (run_folder / "routes").mkdir(exist_ok=True)
        (run_folder / "aggregate_plots").mkdir(exist_ok=True)
        config.run_id = run_folder.name
        config.metadata = dict(config.metadata or {})
        config.metadata["project_commit"] = project_commit_hash()
        config.save(run_folder / "config.json")

        self._active = True
        self._automated = True
        self._route_running = False
        self._state = BenchmarkRunnerState.INITIALIZING_ROUTE
        self._routes = list(config.selected_routes)
        self._current_route = None
        self._current_route_index = 0
        self._current_logger = None
        self._benchmark_id = config.run_id
        self._benchmark_folder = None
        self._run_folder = run_folder
        self._config = config
        self._started_monotonic = None
        self._test_started_monotonic = time.monotonic()
        self._last_exported_summary = None
        self._route_summaries = []
        self._attempt_failures_by_route.clear()
        self._current_attempt = 0
        self._current_attempt_folder = None
        self._last_failure_reason = ""
        self._route_status = "initializing"
        self._status_text = f"Automated benchmark ready: {len(self._routes)} route(s)"
        return self.begin_current_route(active_map_name)

    def begin_current_route(self, active_map_name: Optional[str]) -> bool:
        if not self._active or not self._automated:
            return False
        route = self._pending_route()
        if route is None:
            self._finalize_aggregate()
            return False
        if not maps_compatible(active_map_name, route.map_name):
            self._state = BenchmarkRunnerState.SWITCHING_MAP
            self._route_status = "switching_map"
            self._status_text = f"Switching map for {route.name}: {display_map_name(route.map_name)}"
            return False

        route_folder = self._route_output_folder(route, self._current_route_index)
        self._route_status = "initializing"
        started = self._start_route(route, self._current_route_index, route_folder=route_folder)
        if not started:
            self._record_route_error(route, self._status_text)
            self._advance_after_route(active_map_name)
        return started

    def needs_map_switch(self, active_map_name: Optional[str]) -> bool:
        if not self._active or not self._automated or self._route_running:
            return False
        route = self._pending_route()
        return route is not None and not maps_compatible(active_map_name, route.map_name)

    def required_map_name(self) -> Optional[str]:
        route = self._pending_route()
        return route.map_name if route is not None else None

    def stop(self, aborted: bool = True, reason: str = "Benchmark stopped") -> Optional[Path]:
        if not self._active:
            self._status_text = reason
            return None
        if self._route_running:
            self._finish_current_route(aborted=aborted, timeout=False, reason=reason)
        self._active = False
        self._route_running = False
        self._state = BenchmarkRunnerState.ERROR if aborted else BenchmarkRunnerState.TEST_FINISHED
        if self._automated:
            self._record_incomplete_pending_routes(reason)
            self._finalize_aggregate()
        self._status_text = reason
        return self.benchmark_folder

    def update(
        self,
        route_completed: bool,
        route_failed: bool = False,
        active_map_name: Optional[str] = None,
    ) -> Optional[Path]:
        if not self._active:
            return None

        if self._automated and not self._route_running:
            if self.needs_map_switch(active_map_name):
                self._state = BenchmarkRunnerState.SWITCHING_MAP
                return None
            self.begin_current_route(active_map_name)
            return None

        if not self._route_running:
            return None

        if route_failed:
            return self.fail_current_attempt(
                "Route unavailable before completion",
                simulation_time_s=None,
                active_map_name=active_map_name,
                timeout=False,
            )

        if self._timed_out():
            return self.fail_current_attempt(
                f"Route timeout after {BENCHMARK.max_pass_duration_s:.0f}s",
                simulation_time_s=None,
                active_map_name=active_map_name,
                timeout=True,
            )

        if route_completed:
            finished = self._finish_current_route(aborted=False, timeout=False, reason="Benchmark completed")
            if self._automated:
                self._advance_after_route(active_map_name)
            return finished

        return None

    def fail_current_attempt(
        self,
        reason: str,
        simulation_time_s: Optional[float] = None,
        active_map_name: Optional[str] = None,
        timeout: bool = False,
    ) -> Optional[Path]:
        """Abort the current route attempt and retry the same route when possible."""
        if not self._active or not self._route_running:
            self._status_text = reason
            self._last_failure_reason = reason
            return None

        route = self._current_route
        route_index = self._current_route_index
        attempt = max(1, self._current_attempt)
        failure_record = {
            "attempt": attempt,
            "reason": reason,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "simulation_time_s": simulation_time_s,
            "route_elapsed_s": self.elapsed_route_seconds(),
            "timeout": bool(timeout),
        }
        self._attempt_failures_by_route.setdefault(route_index, []).append(failure_record)
        self._last_failure_reason = reason
        can_retry = self._automated and attempt < self._max_route_attempts and route is not None
        final_status = "ATTEMPT_FAILED" if can_retry else "TEST_NOT_COMPLETED"
        self._route_status = "retrying" if can_retry else "failed"

        finished = self._finish_current_route(
            aborted=True,
            timeout=timeout,
            reason=reason,
            record_result=not can_retry,
            final_status=final_status,
        )
        if can_retry:
            self._state = BenchmarkRunnerState.INITIALIZING_ROUTE
            self._status_text = (
                f"Retrying {route.name}: attempt {attempt + 1}/{self._max_route_attempts} "
                f"after {reason}"
            )
            self.begin_current_route(active_map_name)
            return finished

        if self._automated:
            self._advance_after_route(active_map_name)
        return finished

    def regenerate_plots(self) -> bool:
        target = self._benchmark_folder if not self._automated else self._run_folder
        if target is None:
            self._status_text = "No benchmark output to plot"
            return False
        if self._automated:
            callback = self._enqueue_aggregate_plots_callback
        else:
            callback = self._enqueue_route_plots_callback
        if callback is not None:
            try:
                queued = callback(target)
            except Exception as exc:  # pragma: no cover - callback defensive guard.
                self._status_text = f"Plot queue failed: {exc}"
                return False
            self._status_text = f"Plots queued: {target.name}" if queued else "Plot queue unavailable"
            return queued
        self._status_text = "Plot worker unavailable"
        return False

    def elapsed_test_seconds(self) -> float:
        if self._test_started_monotonic is None:
            return 0.0
        return max(0.0, time.monotonic() - self._test_started_monotonic)

    def elapsed_route_seconds(self) -> float:
        if self._started_monotonic is None:
            return 0.0
        return max(0.0, time.monotonic() - self._started_monotonic)

    def _start_route(
        self,
        route: SavedTestRoute,
        route_index: int,
        route_folder: Optional[Path],
    ) -> bool:
        active_map_name = getattr(self._world_map, "name", None)
        if not self._route_store.route_is_compatible(route):
            self._status_text = (
                "Benchmark blocked: route map "
                f"{display_map_name(route.map_name)} is incompatible with active map "
                f"{display_map_name(active_map_name)}"
            )
            self._route_running = False
            return False

        resolved = self._route_store.resolve_route_to_waypoints(self._world_map, route)
        if resolved is None:
            self._status_text = "Failed to resolve saved test route"
            self._route_running = False
            return False

        start_waypoint, goal_waypoint = resolved
        route_waypoints = list(self._plan_route_callback(start_waypoint, goal_waypoint))
        if not route_waypoints:
            self._status_text = "Test failed: route planner returned empty route"
            self._route_running = False
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        active_filter_info = self._active_filter_info_callback()
        active_filter_tune = self._active_filter_tune_callback()
        filter_slug = _slugify(str(active_filter_info.get("id") or "filter"))
        route_slug = _slugify(route.name)
        if route_folder is None:
            benchmark_id = f"benchmark_{timestamp}_{route_slug}_{map_slug(active_map_name)}_{filter_slug}"
            route_folder = _unique_folder(_legacy_benchmark_root(), benchmark_id)
        else:
            benchmark_id = route_folder.name
        route_folder.mkdir(parents=True, exist_ok=True)
        (route_folder / "plots").mkdir(exist_ok=True)
        attempt = self._next_attempt_number(route_index)
        attempt_folder = route_folder
        if self._automated:
            attempt_folder = route_folder / f"attempt_{attempt:03d}"
            attempt_folder.mkdir(parents=True, exist_ok=True)

        sensor_config = self._config.sensor_noise_config.to_dict() if self._config is not None else None
        behavior_config = self._config.vehicle_behavior_config if self._config is not None else None
        actuator_config = self._config.actuator_realism_config if self._config is not None else None
        metadata = build_benchmark_metadata(
            benchmark_id=benchmark_id,
            timestamp=timestamp,
            route=route,
            start_waypoint=start_waypoint,
            goal_waypoint=goal_waypoint,
            route_waypoints=route_waypoints,
            map_name=active_map_name,
            selected_map_load_name=self._selected_map_load_name,
            active_map_id=normalize_map_name(active_map_name),
            weather=self._weather_callback(),
            vehicle_blueprint=self._vehicle_blueprint_callback(),
            active_filter_info=active_filter_info,
            active_filter_tune=active_filter_tune,
            tracking_mode=self._tracking_mode_callback(),
            active_control_input_used=self._active_control_used_callback(),
            sensor_noise_config=sensor_config,
            vehicle_behavior_config=behavior_config,
            actuator_realism_config=actuator_config,
            random_seed=self._config.random_seed if self._config is not None else None,
            run_id=self._config.run_id if self._config is not None else None,
            route_index=route_index + 1,
            route_count=len(self._routes),
        )
        metadata["attempt"] = attempt
        metadata["max_attempts"] = self._max_route_attempts
        _write_json(attempt_folder / "metadata.json", metadata)
        if self._automated:
            _write_json(route_folder / "metadata.json", metadata)

        logger = FilterPerformanceLogger(
            output_dir=attempt_folder,
            benchmark_id=benchmark_id,
            active_filter_id=str(active_filter_info.get("id") or ""),
            active_filter_name=str(active_filter_info.get("name") or "Active filter"),
        )
        logger.start_route(route.name, benchmark_id=benchmark_id)

        self._current_route = route
        self._current_route_index = route_index
        self._current_logger = logger
        self._benchmark_id = benchmark_id
        self._benchmark_folder = route_folder
        self._current_attempt_folder = attempt_folder
        self._current_attempt = attempt
        self._started_monotonic = time.monotonic()
        self._last_exported_summary = None
        self._active = True
        self._route_running = True
        self._state = BenchmarkRunnerState.RUNNING_ROUTE
        self._route_status = "running"
        self._status_text = f"Benchmark running: {route.name} (attempt {attempt}/{self._max_route_attempts})"

        self._reset_estimator_callback()
        self._begin_route_callback(start_waypoint, goal_waypoint)
        return True

    def _finish_current_route(
        self,
        aborted: bool,
        timeout: bool,
        reason: str,
        record_result: bool = True,
        final_status: Optional[str] = None,
    ) -> Optional[Path]:
        logger = self._current_logger
        route_folder = self._benchmark_folder
        attempt_folder = self._current_attempt_folder or route_folder
        route = self._current_route
        if logger is None or route_folder is None or route is None:
            self._route_running = False
            self._status_text = reason
            self._current_attempt = 0
            self._current_attempt_folder = None
            return route_folder

        if aborted:
            logger.mark_aborted(reason=reason, timeout=timeout)
        else:
            logger.mark_completed()

        if self._automated:
            export_folder = attempt_folder or route_folder
            csv_path = export_folder / "timeseries.csv"
            summary_path = export_folder / "route_summary.json"
            logger.export_to_files(csv_path, summary_path)
            logger.export_to_files(export_folder / "samples.csv", export_folder / "summary.json")
        else:
            _csv_path, summary_path = logger.export_to_folder(route_folder)

        summary = self._enriched_summary(logger.build_summary(), route, route_folder)
        summary["final_status"] = final_status or ("TEST_COMPLETED" if not aborted else "TEST_NOT_COMPLETED")
        if attempt_folder is not None:
            summary["attempt_folder"] = str(attempt_folder)
        _write_json(summary_path, summary)
        if self._automated:
            _write_json((attempt_folder or route_folder) / "summary.json", summary)
            if record_result:
                logger.export_to_files(route_folder / "timeseries.csv", route_folder / "route_summary.json")
                logger.export_to_files(route_folder / "samples.csv", route_folder / "summary.json")
                _write_json(route_folder / "route_summary.json", summary)
                _write_json(route_folder / "summary.json", summary)
        self._last_exported_summary = summary
        if record_result:
            self._route_summaries.append(summary)

        plot_status = ""
        if record_result and BENCHMARK.generate_plots_on_completion:
            if self._enqueue_route_plots_callback is not None:
                try:
                    queued = self._enqueue_route_plots_callback(route_folder)
                    plot_status = ": plots queued" if queued else ": plot queue unavailable"
                except Exception as exc:  # pragma: no cover - callback defensive guard.
                    plot_status = f": plot queue failed ({exc})"
            else:
                plot_status = ": plot worker unavailable"

        self._route_running = False
        self._current_route = None
        self._current_logger = None
        self._current_attempt_folder = None
        self._current_attempt = 0
        self._started_monotonic = None
        self._state = BenchmarkRunnerState.ROUTE_FINISHED
        if aborted:
            self._status_text = f"{reason}{plot_status}"
            if record_result:
                self._route_status = "failed"
        else:
            self._status_text = f"Route completed{plot_status}"
            self._route_status = "completed"
        if not self._automated:
            self._active = False
        return summary_path.parent

    def _advance_after_route(self, active_map_name: Optional[str]) -> None:
        self._current_route_index += 1
        if self._current_route_index >= len(self._routes):
            self._finalize_aggregate()
            return
        next_route = self._routes[self._current_route_index]
        if maps_compatible(active_map_name, next_route.map_name):
            self._state = BenchmarkRunnerState.SWITCHING_ROUTE
            self._route_status = "initializing"
            self._status_text = f"Starting next route: {next_route.name}"
            self.begin_current_route(active_map_name)
        else:
            self._state = BenchmarkRunnerState.SWITCHING_MAP
            self._route_status = "switching_map"
            self._status_text = f"Switching map for {next_route.name}: {display_map_name(next_route.map_name)}"

    def _finalize_aggregate(self) -> None:
        if self._run_folder is None:
            self._active = False
            self._state = BenchmarkRunnerState.TEST_FINISHED
            self._route_status = "finished"
            return
        aggregate = self._build_aggregate_summary()
        _write_json(self._run_folder / "aggregate_summary.json", aggregate)
        self._write_aggregate_csv(self._run_folder / "aggregate_summary.csv")
        plot_status = ""
        if self._enqueue_aggregate_plots_callback is not None:
            try:
                queued = self._enqueue_aggregate_plots_callback(self._run_folder)
                plot_status = ": aggregate plots queued" if queued else ": aggregate plot queue unavailable"
            except Exception as exc:  # pragma: no cover - callback defensive guard.
                aggregate["plot_error"] = str(exc)
                _write_json(self._run_folder / "aggregate_summary.json", aggregate)
                plot_status = f": aggregate plot queue failed ({exc})"
        else:
            aggregate["plot_error"] = "Plot worker unavailable"
            _write_json(self._run_folder / "aggregate_summary.json", aggregate)
            plot_status = ": aggregate plot worker unavailable"

        self._active = False
        self._route_running = False
        self._state = BenchmarkRunnerState.TEST_FINISHED
        self._route_status = "finished"
        self._status_text = f"Automated benchmark finished: {self._run_folder.name}{plot_status}"

    def _build_aggregate_summary(self) -> dict[str, object]:
        config = self._config.to_dict() if self._config is not None else {}
        successful = [summary for summary in self._route_summaries if summary.get("route_completion_success")]
        primary_rows = [_primary_metric_row(summary) for summary in self._route_summaries]
        filtered_values = _finite_row_values(primary_rows, "filtered_rmse_m")
        raw_values = _finite_row_values(primary_rows, "raw_gnss_rmse_m")
        return {
            "run_id": self._config.run_id if self._config is not None else self._benchmark_id,
            "created_at": self._config.created_at if self._config is not None else None,
            "route_count": len(self._routes),
            "completed_route_count": len(successful),
            "failed_route_count": max(0, len(self._route_summaries) - len(successful)),
            "max_route_attempts": self._max_route_attempts,
            "selected_filter": self._config.selected_filter if self._config is not None else None,
            "selected_filter_tune": config.get("selected_filter_tune"),
            "active_filter_tune": self._active_filter_tune_callback(),
            "tracking_mode": config.get("tracking_mode"),
            "active_control_input_used_by_filter": self._active_control_used_callback(),
            "sensor_noise_config": config.get("sensor_noise_config"),
                "vehicle_behavior_config": config.get("vehicle_behavior_config"),
                "actuator_realism_config": config.get("actuator_realism_config"),
            "random_seed": config.get("random_seed"),
            "project_commit": (config.get("metadata") or {}).get("project_commit") if isinstance(config.get("metadata"), dict) else None,
            "primary_metric_source": "eval_preferred",
            "mean_filtered_rmse_m": _mean(filtered_values),
            "mean_raw_gnss_rmse_m": _mean(raw_values),
            "mean_eval_filtered_rmse_m": _mean(_finite_summary_values(self._route_summaries, "eval_filtered_rmse_m")),
            "mean_eval_raw_gnss_rmse_m": _mean(_finite_summary_values(self._route_summaries, "eval_raw_gnss_rmse_m")),
            "mean_driving_filtered_rmse_m": _mean(_finite_summary_values(self._route_summaries, "driving_filtered_rmse_m")),
            "mean_driving_raw_gnss_rmse_m": _mean(_finite_summary_values(self._route_summaries, "driving_raw_gnss_rmse_m")),
            "mean_overall_filtered_rmse_m": _mean(_finite_summary_values(self._route_summaries, "filtered_rmse_m")),
            "mean_overall_raw_gnss_rmse_m": _mean(_finite_summary_values(self._route_summaries, "raw_gnss_rmse_m")),
            "mean_position_nees": _mean(_finite_row_values(primary_rows, "mean_position_nees")),
            "mean_position_nees_diagonal_approx": _mean(_finite_row_values(primary_rows, "mean_position_nees_diagonal_approx")),
            "position_nees_source": _aggregate_position_nees_source(primary_rows),
            "legacy_mean_nis_mixed": _mean(_finite_row_values(primary_rows, "legacy_mean_nis_mixed")),
            "route_summaries": self._route_summaries,
            "segment_rmse_summary": self._aggregate_segment_metrics(),
        }

    def _write_aggregate_csv(self, path: Path) -> None:
        fieldnames = [
            "route_index",
            "route_name",
            "map_name",
            "final_status",
            "route_completion_success",
            "successful_attempt",
            "attempts_used",
            "max_attempts",
            "failure_reasons",
            "primary_metric_source",
            "filtered_rmse_m",
            "raw_gnss_rmse_m",
            "improvement_percent",
            "speed_rmse_mps",
            "yaw_rmse_deg",
            "mean_nees",
            "mean_nis",
            "legacy_mean_nis_mixed",
            "mean_position_nees",
            "mean_position_nees_diagonal_approx",
            "position_nees_source",
            "eval_filtered_rmse_m",
            "eval_raw_gnss_rmse_m",
            "eval_improvement_percent",
            "valid_for_metrics_sample_count",
            "startup_transient_sample_count",
            "warmup_excluded_s",
            "driving_filtered_rmse_m",
            "driving_raw_gnss_rmse_m",
            "driving_improvement_percent",
            "driving_speed_rmse_mps",
            "driving_yaw_rmse_deg",
            "driving_mean_nees",
            "driving_mean_nis",
            "driving_legacy_mean_nis_mixed",
            "driving_mean_position_nees",
            "driving_mean_position_nees_diagonal_approx",
            "driving_position_nees_source",
            "overall_filtered_rmse_m",
            "overall_raw_gnss_rmse_m",
            "overall_improvement_percent",
            "overall_speed_rmse_mps",
            "overall_yaw_rmse_deg",
            "overall_mean_nees",
            "overall_mean_nis",
            "overall_legacy_mean_nis_mixed",
            "overall_mean_position_nees",
            "overall_mean_position_nees_diagonal_approx",
            "overall_position_nees_source",
            "completion_time_s",
            "route_folder",
            "error",
        ]
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for summary in self._route_summaries:
                metrics = _primary_metric_row(summary)
                writer.writerow(
                    {
                        "route_index": summary.get("route_index"),
                        "route_name": summary.get("route_name"),
                        "map_name": summary.get("map_name"),
                        "final_status": summary.get("final_status"),
                        "route_completion_success": summary.get("route_completion_success"),
                        "successful_attempt": summary.get("successful_attempt"),
                        "attempts_used": summary.get("attempts_used"),
                        "max_attempts": summary.get("max_attempts"),
                        "failure_reasons": _failure_reasons_text(summary.get("failed_attempts")),
                        "primary_metric_source": metrics.get("primary_metric_source"),
                        "filtered_rmse_m": metrics.get("filtered_rmse_m"),
                        "raw_gnss_rmse_m": metrics.get("raw_gnss_rmse_m"),
                        "improvement_percent": metrics.get("improvement_percent"),
                        "speed_rmse_mps": metrics.get("speed_rmse_mps"),
                        "yaw_rmse_deg": metrics.get("yaw_rmse_deg"),
                        "mean_nees": metrics.get("mean_nees"),
                        "mean_nis": metrics.get("mean_nis"),
                        "legacy_mean_nis_mixed": metrics.get("legacy_mean_nis_mixed"),
                        "mean_position_nees": metrics.get("mean_position_nees"),
                        "mean_position_nees_diagonal_approx": metrics.get("mean_position_nees_diagonal_approx"),
                        "position_nees_source": metrics.get("position_nees_source"),
                        "eval_filtered_rmse_m": metrics.get("eval_filtered_rmse_m"),
                        "eval_raw_gnss_rmse_m": metrics.get("eval_raw_gnss_rmse_m"),
                        "eval_improvement_percent": metrics.get("eval_improvement_percent"),
                        "valid_for_metrics_sample_count": summary.get("valid_for_metrics_sample_count"),
                        "startup_transient_sample_count": summary.get("startup_transient_sample_count"),
                        "warmup_excluded_s": summary.get("warmup_excluded_s"),
                        "driving_filtered_rmse_m": metrics.get("driving_filtered_rmse_m"),
                        "driving_raw_gnss_rmse_m": metrics.get("driving_raw_gnss_rmse_m"),
                        "driving_improvement_percent": metrics.get("driving_improvement_percent"),
                        "driving_speed_rmse_mps": metrics.get("driving_speed_rmse_mps"),
                        "driving_yaw_rmse_deg": metrics.get("driving_yaw_rmse_deg"),
                        "driving_mean_nees": metrics.get("driving_mean_nees"),
                        "driving_mean_nis": metrics.get("driving_mean_nis"),
                        "driving_legacy_mean_nis_mixed": metrics.get("driving_legacy_mean_nis_mixed"),
                        "driving_mean_position_nees": metrics.get("driving_mean_position_nees"),
                        "driving_mean_position_nees_diagonal_approx": metrics.get("driving_mean_position_nees_diagonal_approx"),
                        "driving_position_nees_source": metrics.get("driving_position_nees_source"),
                        "overall_filtered_rmse_m": metrics.get("overall_filtered_rmse_m"),
                        "overall_raw_gnss_rmse_m": metrics.get("overall_raw_gnss_rmse_m"),
                        "overall_improvement_percent": metrics.get("overall_improvement_percent"),
                        "overall_speed_rmse_mps": metrics.get("overall_speed_rmse_mps"),
                        "overall_yaw_rmse_deg": metrics.get("overall_yaw_rmse_deg"),
                        "overall_mean_nees": metrics.get("overall_mean_nees"),
                        "overall_mean_nis": metrics.get("overall_mean_nis"),
                        "overall_legacy_mean_nis_mixed": metrics.get("overall_legacy_mean_nis_mixed"),
                        "overall_mean_position_nees": metrics.get("overall_mean_position_nees"),
                        "overall_mean_position_nees_diagonal_approx": metrics.get("overall_mean_position_nees_diagonal_approx"),
                        "overall_position_nees_source": metrics.get("overall_position_nees_source"),
                        "completion_time_s": summary.get("completion_time_s"),
                        "route_folder": summary.get("route_folder"),
                        "error": summary.get("error"),
                    }
                )

    def _aggregate_segment_metrics(self) -> dict[str, dict[str, object]]:
        buckets: dict[str, list[float]] = {}
        for summary in self._route_summaries:
            segments = summary.get("driving_segment_metrics") or summary.get("segment_metrics")
            if not isinstance(segments, dict):
                continue
            for segment, metrics in segments.items():
                if not isinstance(metrics, dict):
                    continue
                value = _optional_float(metrics.get("position_rmse_m"))
                if value is not None:
                    buckets.setdefault(str(segment), []).append(value)
        return {
            segment: {
                "route_count": len(values),
                "position_rmse_m": _mean(values),
            }
            for segment, values in buckets.items()
        }

    def _record_route_error(self, route: SavedTestRoute, reason: str) -> None:
        route_folder = self._route_output_folder(route, self._current_route_index)
        route_folder.mkdir(parents=True, exist_ok=True)
        failure_record = {
            "attempt": max(1, self._next_attempt_number(self._current_route_index)),
            "reason": reason,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "simulation_time_s": None,
            "route_elapsed_s": None,
            "timeout": False,
        }
        self._attempt_failures_by_route.setdefault(self._current_route_index, []).append(failure_record)
        self._last_failure_reason = reason
        summary = self._enriched_summary(
            {
                "benchmark_id": route_folder.name,
                "route_name": route.name,
                "route_completion_success": False,
                "route_aborted": True,
                "abort_reason": reason,
                "error": reason,
                "sample_count": 0,
                "final_status": "TEST_NOT_COMPLETED",
            },
            route,
            route_folder,
        )
        _write_json(route_folder / "route_summary.json", summary)
        self._route_summaries.append(summary)
        self._last_exported_summary = summary

    def _record_incomplete_pending_routes(self, reason: str) -> None:
        if not self._automated:
            return
        for index in range(self._current_route_index, len(self._routes)):
            route = self._routes[index]
            if any(summary.get("route_index") == index + 1 for summary in self._route_summaries):
                continue
            self._current_route_index = index
            self._record_route_error(route, reason)

    def _enriched_summary(
        self,
        summary: dict[str, object],
        route: SavedTestRoute,
        route_folder: Path,
    ) -> dict[str, object]:
        enriched = dict(summary)
        active_map_name = getattr(self._world_map, "name", None)
        config = self._config
        failed_attempts = list(self._attempt_failures_by_route.get(self._current_route_index, []))
        attempts_used = max(self._current_attempt, len(failed_attempts))
        if attempts_used <= 0 and enriched.get("route_completion_success") is not None:
            attempts_used = 1
        final_status = enriched.get("final_status")
        if final_status is None:
            final_status = "TEST_COMPLETED" if enriched.get("route_completion_success") else "TEST_NOT_COMPLETED"
        route_success = bool(enriched.get("route_completion_success"))
        enriched.update(
            {
                "run_id": config.run_id if config is not None else self._benchmark_id,
                "route_index": self._current_route_index + 1,
                "route_count": len(self._routes),
                "route_name": route.name,
                "map_name": active_map_name,
                "route_map_name": route.map_name,
                "selected_filter": config.selected_filter if config is not None else None,
                "selected_filter_tune": config.selected_filter_tune if config is not None else None,
                "active_filter_tune": self._active_filter_tune_callback(),
                "tracking_mode": config.tracking_mode if config is not None else self._tracking_mode_callback(),
                "active_control_input_used_by_filter": self._active_control_used_callback(),
                "sensor_noise_config": config.sensor_noise_config.to_dict() if config is not None else None,
                "vehicle_behavior_config": config.vehicle_behavior_config if config is not None else None,
                "actuator_realism_config": config.actuator_realism_config if config is not None else None,
                "random_seed": config.random_seed if config is not None else None,
                "route_folder": str(route_folder),
                "final_status": final_status,
                "successful_attempt": attempts_used if route_success else None,
                "attempts_used": attempts_used,
                "max_attempts": self._max_route_attempts,
                "failed_attempts": failed_attempts,
                "last_failure_reason": failed_attempts[-1]["reason"] if failed_attempts else None,
            }
        )
        primary_metrics = _primary_metric_row(enriched)
        enriched["improvement_percent"] = primary_metrics.get("improvement_percent")
        enriched["primary_metric_source"] = primary_metrics.get("primary_metric_source")
        return enriched

    def _pending_route(self) -> Optional[SavedTestRoute]:
        if self._current_route_index < 0 or self._current_route_index >= len(self._routes):
            return None
        return self._routes[self._current_route_index]

    def _compact_route_output_enabled(self) -> bool:
        config = self._config
        metadata = config.metadata if config is not None and isinstance(config.metadata, dict) else {}
        return bool(metadata.get("compact_route_output"))

    def _route_output_folder(self, route: SavedTestRoute, route_index: int) -> Path:
        if self._compact_route_output_enabled():
            folder_name = f"r{route_index + 1:03d}"
        else:
            folder_name = f"route_{route_index + 1:03d}_{_slugify(route.name)}"
        if self._run_folder is None:
            return _legacy_benchmark_root() / folder_name
        return self._run_folder / "routes" / folder_name

    def _next_attempt_number(self, route_index: int) -> int:
        return len(self._attempt_failures_by_route.get(route_index, [])) + 1

    def _timed_out(self) -> bool:
        if self._started_monotonic is None:
            return False
        elapsed = time.monotonic() - self._started_monotonic
        return elapsed >= BENCHMARK.max_pass_duration_s


def _legacy_benchmark_root() -> Path:
    root = Path(BENCHMARK.output_root)
    if root.is_absolute():
        return root
    project_root = Path(__file__).resolve().parents[2]
    return project_root / root


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "route"


def _unique_folder(root: Path, base_name: str) -> Path:
    candidate = root / base_name
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = root / f"{base_name}_{index}"
        if not candidate.exists():
            return candidate
        index += 1


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _optional_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _improvement_percent(raw_rmse: Optional[float], filtered_rmse: Optional[float]) -> Optional[float]:
    if raw_rmse is None or raw_rmse <= 0.0 or filtered_rmse is None:
        return None
    return 100.0 * (raw_rmse - filtered_rmse) / raw_rmse


def _prefer_driving_metric(summary: dict[str, object], driving_key: str, overall_key: str) -> Optional[float]:
    driving = _optional_float(summary.get(driving_key))
    if driving is not None:
        return driving
    return _optional_float(summary.get(overall_key))


def _prefer_eval_metric(
    summary: dict[str, object],
    eval_key: str,
    driving_key: str,
    overall_key: str,
) -> Optional[float]:
    eval_value = _optional_float(summary.get(eval_key))
    if eval_value is not None:
        return eval_value
    return _prefer_driving_metric(summary, driving_key, overall_key)


def _prefer_consistency_source(summary: dict[str, object], eval_key: str, driving_key: str, overall_key: str) -> str:
    for key in (eval_key, driving_key, overall_key):
        value = str(summary.get(key) or "")
        if value:
            return value
    return ""


def _primary_metric_row(summary: dict[str, object]) -> dict[str, Optional[float] | str]:
    eval_filtered = _optional_float(summary.get("eval_filtered_rmse_m"))
    eval_raw = _optional_float(summary.get("eval_raw_gnss_rmse_m"))
    driving_filtered = _optional_float(summary.get("driving_filtered_rmse_m"))
    driving_raw = _optional_float(summary.get("driving_raw_gnss_rmse_m"))
    overall_filtered = _optional_float(summary.get("filtered_rmse_m"))
    overall_raw = _optional_float(summary.get("raw_gnss_rmse_m"))

    primary_filtered = eval_filtered if eval_filtered is not None else driving_filtered if driving_filtered is not None else overall_filtered
    primary_raw = eval_raw if eval_raw is not None else driving_raw if driving_raw is not None else overall_raw
    if eval_filtered is not None or eval_raw is not None:
        primary_source = "eval"
    elif driving_filtered is not None or driving_raw is not None:
        primary_source = "driving"
    else:
        primary_source = "overall_fallback"

    return {
        "primary_metric_source": primary_source,
        "filtered_rmse_m": primary_filtered,
        "raw_gnss_rmse_m": primary_raw,
        "improvement_percent": _improvement_percent(primary_raw, primary_filtered),
        "speed_rmse_mps": _prefer_eval_metric(summary, "eval_speed_rmse_mps", "driving_speed_rmse_mps", "speed_rmse_mps"),
        "yaw_rmse_deg": _prefer_eval_metric(summary, "eval_yaw_rmse_deg", "driving_yaw_rmse_deg", "yaw_rmse_deg"),
        "mean_nees": _prefer_eval_metric(summary, "eval_mean_nees", "driving_mean_nees", "mean_nees"),
        "mean_nis": _prefer_eval_metric(summary, "eval_mean_nis", "driving_mean_nis", "mean_nis"),
        "legacy_mean_nis_mixed": _prefer_eval_metric(summary, "eval_legacy_mean_nis_mixed", "driving_legacy_mean_nis_mixed", "legacy_mean_nis_mixed"),
        "mean_position_nees": _prefer_eval_metric(summary, "eval_mean_position_nees", "driving_mean_position_nees", "mean_position_nees"),
        "mean_position_nees_diagonal_approx": _prefer_eval_metric(
            summary,
            "eval_mean_position_nees_diagonal_approx",
            "driving_mean_position_nees_diagonal_approx",
            "mean_position_nees_diagonal_approx",
        ),
        "position_nees_source": _prefer_consistency_source(summary, "eval_position_nees_source", "driving_position_nees_source", "position_nees_source"),
        "eval_filtered_rmse_m": eval_filtered,
        "eval_raw_gnss_rmse_m": eval_raw,
        "eval_improvement_percent": _improvement_percent(eval_raw, eval_filtered),
        "driving_filtered_rmse_m": driving_filtered,
        "driving_raw_gnss_rmse_m": driving_raw,
        "driving_improvement_percent": _improvement_percent(driving_raw, driving_filtered),
        "driving_speed_rmse_mps": _optional_float(summary.get("driving_speed_rmse_mps")),
        "driving_yaw_rmse_deg": _optional_float(summary.get("driving_yaw_rmse_deg")),
        "driving_mean_nees": _optional_float(summary.get("driving_mean_nees")),
        "driving_mean_nis": _optional_float(summary.get("driving_mean_nis")),
        "driving_legacy_mean_nis_mixed": _optional_float(summary.get("driving_legacy_mean_nis_mixed")),
        "driving_mean_position_nees": _optional_float(summary.get("driving_mean_position_nees")),
        "driving_mean_position_nees_diagonal_approx": _optional_float(summary.get("driving_mean_position_nees_diagonal_approx")),
        "driving_position_nees_source": str(summary.get("driving_position_nees_source") or ""),
        "overall_filtered_rmse_m": overall_filtered,
        "overall_raw_gnss_rmse_m": overall_raw,
        "overall_improvement_percent": _improvement_percent(overall_raw, overall_filtered),
        "overall_speed_rmse_mps": _optional_float(summary.get("speed_rmse_mps")),
        "overall_yaw_rmse_deg": _optional_float(summary.get("yaw_rmse_deg")),
        "overall_mean_nees": _optional_float(summary.get("mean_nees")),
        "overall_mean_nis": _optional_float(summary.get("mean_nis")),
        "overall_legacy_mean_nis_mixed": _optional_float(summary.get("legacy_mean_nis_mixed")),
        "overall_mean_position_nees": _optional_float(summary.get("mean_position_nees")),
        "overall_mean_position_nees_diagonal_approx": _optional_float(summary.get("mean_position_nees_diagonal_approx")),
        "overall_position_nees_source": str(summary.get("position_nees_source") or ""),
    }


def _failure_reasons_text(value: object) -> str:
    if not isinstance(value, list):
        return ""
    reasons = []
    for item in value:
        if isinstance(item, dict):
            attempt = item.get("attempt")
            reason = item.get("reason")
            if reason:
                reasons.append(f"{attempt}: {reason}" if attempt is not None else str(reason))
    return " | ".join(reasons)


def _finite_summary_values(summaries: Sequence[dict[str, object]], key: str) -> list[float]:
    values = []
    for summary in summaries:
        value = _optional_float(summary.get(key))
        if value is not None:
            values.append(value)
    return values


def _finite_row_values(rows: Sequence[dict[str, object]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = _optional_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _aggregate_position_nees_source(rows: Sequence[dict[str, object]]) -> str:
    sources = {str(row.get("position_nees_source") or "") for row in rows if row.get("position_nees_source")}
    if "full_2x2" in sources:
        return "full_2x2"
    if "diagonal_approx" in sources:
        return "diagonal_approx"
    return "unavailable"


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)

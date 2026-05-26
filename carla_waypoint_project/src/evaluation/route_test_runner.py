"""Single-run Kalman benchmark execution for saved routes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Optional

from config.settings import BENCHMARK
from src.evaluation.benchmark_metadata import build_benchmark_metadata
from src.evaluation.filter_performance import FilterPerformanceLogger
from src.evaluation.test_route_store import SavedTestRoute, TestRouteStore
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class RouteTestRunner:
    """Resolve one saved route and run a single Kalman-controlled benchmark."""

    def __init__(
        self,
        world_map: "carla.Map",
        route_store: TestRouteStore,
        begin_route_callback: Callable[["carla.Waypoint", "carla.Waypoint"], None],
        reset_estimator_callback: Callable[[], None],
        plan_route_callback: Callable[["carla.Waypoint", "carla.Waypoint"], Sequence["carla.Waypoint"]],
        weather_callback: Callable[[], Optional[dict[str, object]]],
        vehicle_blueprint_callback: Callable[[], Optional[str]],
    ) -> None:
        self._world_map = world_map
        self._route_store = route_store
        self._begin_route_callback = begin_route_callback
        self._reset_estimator_callback = reset_estimator_callback
        self._plan_route_callback = plan_route_callback
        self._weather_callback = weather_callback
        self._vehicle_blueprint_callback = vehicle_blueprint_callback

        self._active = False
        self._current_route: Optional[SavedTestRoute] = None
        self._current_route_index = 0
        self._current_logger: Optional[FilterPerformanceLogger] = None
        self._benchmark_id: Optional[str] = None
        self._benchmark_folder: Optional[Path] = None
        self._started_monotonic: Optional[float] = None
        self._last_exported_summary: Optional[dict[str, object]] = None
        self._status_text = "Benchmark idle"

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def benchmark_id(self) -> Optional[str]:
        return self._benchmark_id

    @property
    def benchmark_folder(self) -> Optional[Path]:
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
    def status_text(self) -> str:
        return self._status_text

    @property
    def last_exported_summary(self) -> Optional[dict[str, object]]:
        return self._last_exported_summary

    def start_selected_route(self, route: Optional[SavedTestRoute]) -> bool:
        if self._active:
            self._status_text = "Benchmark already running"
            return False
        if route is None:
            self._status_text = "Select or create a test route first"
            return False
        return self._start_route(route, self._route_store.current_index)

    def stop(self, aborted: bool = True, reason: str = "Benchmark stopped") -> Optional[Path]:
        if not self._active:
            self._status_text = reason
            return None
        return self._finish(aborted=aborted, timeout=False, reason=reason)

    def update(self, route_completed: bool, route_failed: bool = False) -> Optional[Path]:
        if not self._active:
            return None

        if route_failed:
            return self._finish(aborted=True, timeout=False, reason="Benchmark aborted: route unavailable")

        if self._timed_out():
            return self._finish(aborted=True, timeout=True, reason="Benchmark timeout")

        if route_completed:
            return self._finish(aborted=False, timeout=False, reason="Benchmark completed: plots saved")

        return None

    def regenerate_plots(self) -> bool:
        if self._benchmark_folder is None:
            self._status_text = "No benchmark output to plot"
            return False
        try:
            from src.evaluation.benchmark_plotter import generate_benchmark_plots

            generate_benchmark_plots(self._benchmark_folder)
        except Exception as exc:  # pragma: no cover - matplotlib and filesystem dependent.
            self._status_text = f"Plot generation failed: {exc}"
            return False
        self._status_text = f"Plots saved: {self._benchmark_folder.name}"
        return True

    def _start_route(self, route: SavedTestRoute, route_index: int) -> bool:
        resolved = self._route_store.resolve_route_to_waypoints(self._world_map, route)
        if resolved is None:
            self._status_text = "Failed to resolve saved test route"
            return False

        start_waypoint, goal_waypoint = resolved
        route_waypoints = list(self._plan_route_callback(start_waypoint, goal_waypoint))
        if not route_waypoints:
            self._status_text = "Test failed: route planner returned empty route"
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        route_slug = _slugify(route.name)
        benchmark_id = f"benchmark_{timestamp}_{route_slug}"
        benchmark_folder = _unique_benchmark_folder(_benchmark_root(), benchmark_id)
        benchmark_id = benchmark_folder.name
        benchmark_folder.mkdir(parents=True, exist_ok=False)
        (benchmark_folder / "plots").mkdir(exist_ok=True)

        metadata = build_benchmark_metadata(
            benchmark_id=benchmark_id,
            timestamp=timestamp,
            route=route,
            start_waypoint=start_waypoint,
            goal_waypoint=goal_waypoint,
            route_waypoints=route_waypoints,
            map_name=getattr(self._world_map, "name", route.map_name),
            weather=self._weather_callback(),
            vehicle_blueprint=self._vehicle_blueprint_callback(),
        )
        _write_json(benchmark_folder / "metadata.json", metadata)

        logger = FilterPerformanceLogger(output_dir=benchmark_folder, benchmark_id=benchmark_id)
        logger.start_route(route.name, benchmark_id=benchmark_id)

        self._current_route = route
        self._current_route_index = route_index
        self._current_logger = logger
        self._benchmark_id = benchmark_id
        self._benchmark_folder = benchmark_folder
        self._started_monotonic = time.monotonic()
        self._last_exported_summary = None
        self._active = True
        self._status_text = f"Benchmark running: {route.name}"

        self._reset_estimator_callback()
        self._begin_route_callback(start_waypoint, goal_waypoint)
        return True

    def _finish(self, aborted: bool, timeout: bool, reason: str) -> Optional[Path]:
        logger = self._current_logger
        benchmark_folder = self._benchmark_folder
        if logger is None or benchmark_folder is None:
            self._active = False
            self._status_text = reason
            return benchmark_folder

        if aborted:
            logger.mark_aborted(reason=reason, timeout=timeout)
        else:
            logger.mark_completed()
        _csv_path, summary_path = logger.export_to_folder(benchmark_folder)
        self._last_exported_summary = logger.build_summary()

        plot_status = ""
        if BENCHMARK.generate_plots_on_completion:
            try:
                from src.evaluation.benchmark_plotter import generate_benchmark_plots

                generate_benchmark_plots(benchmark_folder)
                plot_status = ": plots saved"
            except Exception as exc:  # pragma: no cover - matplotlib and filesystem dependent.
                plot_status = f": plot generation failed ({exc})"

        self._active = False
        self._current_route = None
        self._current_logger = None
        self._started_monotonic = None
        if aborted:
            self._status_text = f"{reason}{plot_status}"
        else:
            self._status_text = f"Benchmark completed{plot_status}"
        return summary_path.parent

    def _timed_out(self) -> bool:
        if self._started_monotonic is None:
            return False
        elapsed = time.monotonic() - self._started_monotonic
        return elapsed >= BENCHMARK.max_pass_duration_s


def _benchmark_root() -> Path:
    root = Path(BENCHMARK.output_root)
    if root.is_absolute():
        return root
    project_root = Path(__file__).resolve().parents[2]
    return project_root / root


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "route"


def _unique_benchmark_folder(root: Path, base_name: str) -> Path:
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

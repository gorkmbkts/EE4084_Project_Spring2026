"""State manager for saved-route performance test execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Optional

from src.evaluation.filter_performance import FilterPerformanceLogger
from src.evaluation.test_route_store import SavedTestRoute, TestRouteStore
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class RouteTestRunner:
    """Resolve saved routes and drive them through the app's route pipeline."""

    def __init__(
        self,
        world_map: "carla.Map",
        route_store: TestRouteStore,
        performance_logger: FilterPerformanceLogger,
        begin_route_callback: Callable[["carla.Waypoint", "carla.Waypoint"], None],
        reset_estimator_callback: Callable[[], None],
    ) -> None:
        self._world_map = world_map
        self._route_store = route_store
        self._performance_logger = performance_logger
        self._begin_route_callback = begin_route_callback
        self._reset_estimator_callback = reset_estimator_callback
        self._active = False
        self._current_route: Optional[SavedTestRoute] = None
        self._current_route_index = 0
        self._queued_routes: list[SavedTestRoute] = []
        self._run_all = False
        self._status_text = "Test idle"

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def current_route_name(self) -> str:
        return self._current_route.name if self._current_route is not None else ""

    @property
    def current_route_index(self) -> int:
        return self._current_route_index

    @property
    def status_text(self) -> str:
        return self._status_text

    def start_selected_route(self, route: Optional[SavedTestRoute]) -> bool:
        self._run_all = False
        self._queued_routes = []
        if route is None:
            self._status_text = "Select or create a test route first"
            return False
        return self._start_route(route, self._route_store.current_index)

    def start_all_routes(self, routes: Sequence[SavedTestRoute]) -> bool:
        if not routes:
            self._status_text = "Select or create a test route first"
            return False
        self._run_all = True
        self._queued_routes = list(routes)
        return self._start_route(self._queued_routes[0], 0)

    def stop(self, aborted: bool = True, reason: str = "Test stopped") -> Optional[tuple[object, object]]:
        if not self._active:
            self._status_text = reason
            return None

        if aborted:
            self._performance_logger.mark_aborted()
        else:
            self._performance_logger.mark_completed()
        paths = self._performance_logger.export()
        self._active = False
        self._current_route = None
        self._queued_routes = []
        self._run_all = False
        self._status_text = reason
        return paths

    def update(self, route_completed: bool, route_failed: bool = False) -> Optional[tuple[object, object]]:
        if not self._active:
            return None

        if route_failed:
            return self.stop(aborted=True, reason="Test failed: route unavailable")

        if not route_completed:
            return None

        self._performance_logger.mark_completed()
        paths = self._performance_logger.export()
        completed_name = self.current_route_name

        if self._run_all and self._queued_routes and self._current_route_index + 1 < len(self._queued_routes):
            next_index = self._current_route_index + 1
            next_route = self._queued_routes[next_index]
            if self._start_route(next_route, next_index):
                self._status_text = f"Completed {completed_name}, started {next_route.name}"
            return paths

        self._active = False
        self._current_route = None
        self._queued_routes = []
        self._run_all = False
        self._status_text = f"Test completed: {completed_name}"
        return paths

    def _start_route(self, route: SavedTestRoute, route_index: int) -> bool:
        resolved = self._route_store.resolve_route_to_waypoints(self._world_map, route)
        if resolved is None:
            self._status_text = "Failed to resolve saved test route"
            return False

        start_waypoint, goal_waypoint = resolved
        self._current_route = route
        self._current_route_index = route_index
        self._active = True
        self._status_text = f"Test running: {route.name}"
        self._performance_logger.start_route(route.name)
        self._reset_estimator_callback()
        self._begin_route_callback(start_waypoint, goal_waypoint)
        return True

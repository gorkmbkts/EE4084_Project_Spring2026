"""Route progress tracking and target waypoint selection."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
from typing import List, Optional, Sequence

from config.settings import WAYPOINT_TRACKER
from src.core.vehicle_state import VehicleState
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


@dataclass(frozen=True)
class TrackingStatus:
    """Current route tracking state for control and visualization."""

    target_waypoint: Optional["carla.Waypoint"]
    closest_index: int
    target_index: int
    cross_track_error_m: float
    distance_to_goal_m: float
    heading_error_deg: Optional[float]
    completed: bool
    route_size: int = 0
    search_start_index: int = 0
    search_end_index: int = 0


class WaypointTracker:
    """Track progress along a global route without jumping backward."""

    def __init__(
        self,
        lookahead_base_m: float = WAYPOINT_TRACKER.lookahead_base_m,
        lookahead_gain_s: float = WAYPOINT_TRACKER.lookahead_gain_s,
        search_backtrack_count: int = WAYPOINT_TRACKER.search_backtrack_count,
        search_forward_count: int = WAYPOINT_TRACKER.search_forward_count,
        max_closest_index_advance_per_update: int = WAYPOINT_TRACKER.max_closest_index_advance_per_update,
        max_target_index_ahead_of_closest: int = WAYPOINT_TRACKER.max_target_index_ahead_of_closest,
        completion_distance_m: float = WAYPOINT_TRACKER.completion_distance_m,
        completion_index_window: int = WAYPOINT_TRACKER.completion_index_window,
    ) -> None:
        self._lookahead_base_m = lookahead_base_m
        self._lookahead_gain_s = lookahead_gain_s
        self._search_backtrack_count = search_backtrack_count
        self._search_forward_count = search_forward_count
        self._max_closest_index_advance_per_update = max(1, int(max_closest_index_advance_per_update))
        self._max_target_index_ahead_of_closest = max(1, int(max_target_index_ahead_of_closest))
        self._completion_distance_m = completion_distance_m
        self._completion_index_window = completion_index_window
        self._route: List["carla.Waypoint"] = []
        self._closest_index = 0
        self._target_index = 0
        self._completed = False
        self._current_target: Optional["carla.Waypoint"] = None
        self._route_distances_m: List[float] = []
        self._cross_track_error_m = float("inf")
        self._distance_to_goal_m = float("inf")
        self._heading_error_deg: Optional[float] = None
        self._search_start_index = 0
        self._search_end_index = 0

    @property
    def closest_index(self) -> int:
        return self._closest_index

    @property
    def target_index(self) -> int:
        return self._target_index

    @property
    def completed(self) -> bool:
        return self._completed

    def set_route(self, route: Sequence["carla.Waypoint"]) -> None:
        """Set a new global route and reset tracking progress."""
        self._route = list(route)
        self._route_distances_m = self._compute_route_distances(self._route)
        self._closest_index = 0
        self._target_index = 0
        self._completed = False
        self._current_target = self._route[0] if self._route else None
        self._cross_track_error_m = float("inf")
        self._distance_to_goal_m = float("inf")
        self._heading_error_deg = None
        self._search_start_index = 0
        initial_window_end = min(len(self._route), self._search_forward_count + 1)
        self._search_end_index = max(0, initial_window_end - 1)

    def clear_route(self) -> None:
        """Clear route and tracking state."""
        self.set_route([])

    def update(self, state: VehicleState) -> TrackingStatus:
        """Update closest and lookahead target waypoints for the ego state."""
        if not self._route:
            self._completed = False
            self._current_target = None
            self._cross_track_error_m = float("inf")
            self._distance_to_goal_m = float("inf")
            self._heading_error_deg = None
            return self._status()

        self._closest_index, self._cross_track_error_m = self._find_closest_index(state)
        candidate_target_index = self._find_target_index(state)
        self._target_index = self._bounded_target_index(candidate_target_index)
        self._current_target = self._route[self._target_index]
        self._distance_to_goal_m = self._compute_distance_to_goal(state)
        self._heading_error_deg = self._compute_heading_error(state, self._current_target)

        if self._completed or self._is_route_completed():
            self._completed = True
            self._target_index = len(self._route) - 1
            self._current_target = self._route[-1]
            self._heading_error_deg = self._compute_heading_error(state, self._current_target)
        if self._completed:
            self._distance_to_goal_m = self._compute_distance_to_goal(state)
        return self._status()

    def get_target(self) -> Optional["carla.Waypoint"]:
        """Return current target waypoint."""
        return self._current_target

    def get_preview_waypoints(self, max_count: int = 60) -> List["carla.Waypoint"]:
        """Return a forward slice of the route for camera overlay rendering."""
        if not self._route:
            return []
        end_index = min(len(self._route), self._closest_index + max_count)
        return self._route[self._closest_index:end_index]

    def _find_closest_index(self, state: VehicleState) -> tuple[int, float]:
        best_index = self._closest_index
        best_distance = float("inf")
        search_start, search_end = self._search_window()
        self._search_start_index = search_start
        self._search_end_index = max(search_start, search_end - 1)

        for index in range(search_start, search_end):
            location = self._route[index].transform.location
            distance = math.hypot(location.x - state.x, location.y - state.y)
            if distance < best_distance:
                best_distance = distance
                best_index = index

        # Route progress is monotonic. A nearby future road can move progress
        # forward only if it lies inside the bounded search window; a later
        # localization correction cannot move progress backward.
        max_progress_index = min(
            len(self._route) - 1,
            self._closest_index + self._max_closest_index_advance_per_update,
        )
        progress_index = min(max(self._closest_index, best_index), max_progress_index)
        if progress_index != best_index:
            location = self._route[progress_index].transform.location
            best_distance = math.hypot(location.x - state.x, location.y - state.y)

        return progress_index, best_distance

    def _find_target_index(self, state: VehicleState) -> int:
        lookahead_m = max(
            self._lookahead_base_m,
            self._lookahead_base_m + self._lookahead_gain_s * state.speed,
        )

        if not self._route_distances_m:
            return 0

        start_index = min(self._closest_index, len(self._route_distances_m) - 1)
        target_distance_m = self._route_distances_m[start_index] + lookahead_m
        target_index = bisect_left(self._route_distances_m, target_distance_m, lo=start_index)
        return min(target_index, len(self._route) - 1)

    def _bounded_target_index(self, candidate_target_index: int) -> int:
        if not self._route:
            return 0

        if self._closest_index >= len(self._route) - 1:
            return len(self._route) - 1

        first_forward_index = self._closest_index + 1
        farthest_allowed_index = min(
            len(self._route) - 1,
            self._closest_index + self._max_target_index_ahead_of_closest,
        )
        target_index = max(first_forward_index, int(candidate_target_index))
        return min(target_index, farthest_allowed_index)

    def _search_window(self) -> tuple[int, int]:
        if not self._route:
            return 0, 0

        search_start = max(0, self._closest_index - self._search_backtrack_count)
        search_end = min(len(self._route), self._closest_index + self._search_forward_count + 1)
        if search_end <= search_start:
            search_end = min(len(self._route), search_start + 1)
        return search_start, search_end

    def _compute_distance_to_goal(self, state: VehicleState) -> float:
        goal_location = self._route[-1].transform.location
        return math.hypot(goal_location.x - state.x, goal_location.y - state.y)

    def _is_route_completed(self) -> bool:
        near_goal = self._distance_to_goal_m <= self._completion_distance_m
        end_window_start = max(0, len(self._route) - self._completion_index_window)
        near_end_index = self._closest_index >= end_window_start
        target_at_end = self._target_index >= len(self._route) - 1
        return near_goal and (near_end_index or target_at_end)

    @staticmethod
    def _compute_heading_error(
        state: VehicleState,
        target_waypoint: Optional["carla.Waypoint"],
    ) -> Optional[float]:
        if target_waypoint is None:
            return None

        target_location = target_waypoint.transform.location
        dx = target_location.x - state.x
        dy = target_location.y - state.y
        if math.hypot(dx, dy) < 0.1:
            desired_yaw = float(target_waypoint.transform.rotation.yaw)
        else:
            desired_yaw = math.degrees(math.atan2(dy, dx))
        return WaypointTracker._normalize_angle_deg(desired_yaw - state.yaw)

    @staticmethod
    def _normalize_angle_deg(angle_deg: float) -> float:
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return angle_deg

    @staticmethod
    def _compute_route_distances(route: Sequence["carla.Waypoint"]) -> List[float]:
        distances = [0.0]
        for previous, current in zip(route, route[1:]):
            step = previous.transform.location.distance(current.transform.location)
            distances.append(distances[-1] + float(step))
        return distances if route else []

    def _status(self) -> TrackingStatus:
        return TrackingStatus(
            target_waypoint=self._current_target,
            closest_index=self._closest_index,
            target_index=self._target_index,
            cross_track_error_m=self._cross_track_error_m,
            distance_to_goal_m=self._distance_to_goal_m,
            heading_error_deg=self._heading_error_deg,
            completed=self._completed,
            route_size=len(self._route),
            search_start_index=self._search_start_index,
            search_end_index=self._search_end_index,
        )

"""Route progress tracking and target waypoint selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence

from config.settings import WAYPOINT_TRACKER
from src.localization.state_estimator import EgoState
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


class WaypointTracker:
    """Track progress along a global route without jumping backward."""

    def __init__(
        self,
        lookahead_base_m: float = WAYPOINT_TRACKER.lookahead_base_m,
        lookahead_gain_s: float = WAYPOINT_TRACKER.lookahead_gain_s,
        search_backtrack_count: int = WAYPOINT_TRACKER.search_backtrack_count,
        search_forward_count: int = WAYPOINT_TRACKER.search_forward_count,
        completion_distance_m: float = WAYPOINT_TRACKER.completion_distance_m,
        completion_index_window: int = WAYPOINT_TRACKER.completion_index_window,
    ) -> None:
        self._lookahead_base_m = lookahead_base_m
        self._lookahead_gain_s = lookahead_gain_s
        self._search_backtrack_count = search_backtrack_count
        self._search_forward_count = search_forward_count
        self._completion_distance_m = completion_distance_m
        self._completion_index_window = completion_index_window
        self._route: List["carla.Waypoint"] = []
        self._closest_index = 0
        self._target_index = 0
        self._completed = False
        self._current_target: Optional["carla.Waypoint"] = None
        self._cross_track_error_m = float("inf")
        self._distance_to_goal_m = float("inf")
        self._heading_error_deg: Optional[float] = None

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
        self._closest_index = 0
        self._target_index = 0
        self._completed = False
        self._current_target = self._route[0] if self._route else None
        self._cross_track_error_m = float("inf")
        self._distance_to_goal_m = float("inf")
        self._heading_error_deg = None

    def clear_route(self) -> None:
        """Clear route and tracking state."""
        self.set_route([])

    def update(self, state: EgoState) -> TrackingStatus:
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
        self._target_index = max(self._target_index, candidate_target_index)
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

    def _find_closest_index(self, state: EgoState) -> tuple[int, float]:
        if self._closest_index == 0:
            search_start = 0
        else:
            search_start = max(0, self._closest_index - self._search_backtrack_count)
        search_end = min(len(self._route), self._closest_index + self._search_forward_count)
        if search_start >= search_end:
            search_start = 0
            search_end = len(self._route)

        best_index = self._closest_index
        best_distance = float("inf")
        for index in range(search_start, search_end):
            location = self._route[index].transform.location
            distance = math.hypot(location.x - state.x, location.y - state.y)
            if distance < best_distance:
                best_distance = distance
                best_index = index

        return best_index, best_distance

    def _find_target_index(self, state: EgoState) -> int:
        lookahead_m = max(
            self._lookahead_base_m,
            self._lookahead_base_m + self._lookahead_gain_s * state.speed,
        )

        accumulated = 0.0
        index = self._closest_index
        while index < len(self._route) - 1 and accumulated < lookahead_m:
            current = self._route[index].transform.location
            following = self._route[index + 1].transform.location
            accumulated += current.distance(following)
            index += 1
        return index

    def _compute_distance_to_goal(self, state: EgoState) -> float:
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
        state: EgoState,
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

    def _status(self) -> TrackingStatus:
        return TrackingStatus(
            target_waypoint=self._current_target,
            closest_index=self._closest_index,
            target_index=self._target_index,
            cross_track_error_m=self._cross_track_error_m,
            distance_to_goal_m=self._distance_to_goal_m,
            heading_error_deg=self._heading_error_deg,
            completed=self._completed,
        )

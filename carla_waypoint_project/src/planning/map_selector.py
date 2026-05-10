"""Mouse-driven A/B route endpoint selection on the CARLA road network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


@dataclass(frozen=True)
class RouteEndpoints:
    """Selected route endpoints after snapping to drivable waypoints."""

    start: "carla.Waypoint"
    goal: "carla.Waypoint"


class MapSelector:
    """Store route endpoint selections as snapped CARLA waypoints."""

    def __init__(self, world_map: "carla.Map") -> None:
        self._world_map = world_map
        self._start: Optional["carla.Waypoint"] = None
        self._goal: Optional["carla.Waypoint"] = None

    @property
    def start(self) -> Optional["carla.Waypoint"]:
        return self._start

    @property
    def goal(self) -> Optional["carla.Waypoint"]:
        return self._goal

    @property
    def endpoints(self) -> Optional[RouteEndpoints]:
        if self._start is None or self._goal is None:
            return None
        return RouteEndpoints(start=self._start, goal=self._goal)

    def reset(self) -> None:
        """Clear A and B selections."""
        self._start = None
        self._goal = None

    def select_world_location(self, location: "carla.Location") -> Optional["carla.Waypoint"]:
        """Snap a clicked world location and store it as A or B.

        If both endpoints already exist, the next click starts a new selection
        cycle and becomes A.
        """
        waypoint = self._world_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return None

        if self._start is None or (self._start is not None and self._goal is not None):
            self._start = waypoint
            self._goal = None
        else:
            self._goal = waypoint

        return waypoint

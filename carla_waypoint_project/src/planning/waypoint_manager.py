"""Waypoint generation utilities based on CARLA map and ego pose."""

from __future__ import annotations

from typing import List

from config.settings import WAYPOINT
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class WaypointManager:
    """Generate future waypoints ahead of the current ego vehicle pose."""

    def __init__(
        self,
        world_map: "carla.Map",
        waypoint_count: int = WAYPOINT.count,
        step_distance_m: float = WAYPOINT.step_distance_m,
    ) -> None:
        self._world_map = world_map
        self._waypoint_count = waypoint_count
        self._step_distance_m = step_distance_m

    def get_current_waypoint(self, vehicle: "carla.Vehicle") -> "carla.Waypoint":
        """Return lane-projected waypoint at current vehicle location."""
        vehicle_loc = vehicle.get_location()
        return self._world_map.get_waypoint(
            vehicle_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

    def get_future_waypoints(self, vehicle: "carla.Vehicle") -> List["carla.Waypoint"]:
        """Return a short forward waypoint list (~30 points by default)."""
        current_wp = self.get_current_waypoint(vehicle)
        waypoints = [current_wp]

        for _ in range(self._waypoint_count - 1):
            next_wps = current_wp.next(self._step_distance_m)
            if not next_wps:
                break
            current_wp = next_wps[0]
            waypoints.append(current_wp)
    
        return waypoints


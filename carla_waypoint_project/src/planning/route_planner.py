"""Global route planning from snapped CARLA waypoints."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple

from config.settings import ROUTE_PLANNER
from src.utils.carla_import import ensure_carla_agents_import, ensure_carla_import

carla = ensure_carla_import()

NodeKey = Tuple[float, float, float]


@dataclass(frozen=True)
class _FallbackEdge:
    start_key: NodeKey
    end_key: NodeKey
    path: List["carla.Waypoint"]
    length_m: float


class RoutePlanner:
    """Plan an ordered drivable waypoint route between two snapped endpoints."""

    def __init__(
        self,
        world_map: "carla.Map",
        sampling_resolution_m: float = ROUTE_PLANNER.sampling_resolution_m,
    ) -> None:
        self._world_map = world_map
        self._sampling_resolution_m = sampling_resolution_m
        self._route: List["carla.Waypoint"] = []
        self._global_route_planner = None
        self._fallback_graph: Optional[Dict[NodeKey, List[_FallbackEdge]]] = None
        self._fallback_edges: Optional[List[_FallbackEdge]] = None
        self._planner_error: Optional[str] = None
        self._initialize_global_route_planner()

    @property
    def planner_error(self) -> Optional[str]:
        return self._planner_error

    def snap_to_driving_waypoint(self, location: "carla.Location") -> Optional["carla.Waypoint"]:
        """Project a world location onto the nearest drivable waypoint."""
        return self._world_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

    def set_destination(self, start: "carla.Location", goal: "carla.Location") -> None:
        """Compatibility wrapper that accepts raw locations and plans a route."""
        start_wp = self.snap_to_driving_waypoint(start)
        goal_wp = self.snap_to_driving_waypoint(goal)
        if start_wp is None or goal_wp is None:
            self._route = []
            return
        self.generate_route(start_wp, goal_wp)

    def generate_route(self, start: "carla.Waypoint", goal: "carla.Waypoint") -> List["carla.Waypoint"]:
        """Generate and store a stable waypoint route from A to B."""
        route = self._generate_with_global_route_planner(start, goal)
        if not route:
            route = self._generate_with_fallback_graph(start, goal)

        self._route = self._densify_route(self._deduplicate_route(route))
        return self.get_route()

    def clear_route(self) -> None:
        """Clear the stored global route."""
        self._route = []

    def get_route(self) -> List["carla.Waypoint"]:
        """Return a copy of the currently stored route."""
        return list(self._route)

    def _initialize_global_route_planner(self) -> None:
        try:
            module = ensure_carla_agents_import("agents.navigation.global_route_planner")
            planner_cls = module.GlobalRoutePlanner
            self._global_route_planner = planner_cls(self._world_map, self._sampling_resolution_m)
            self._planner_error = None
        except Exception as exc:  # pragma: no cover - depends on local CARLA PythonAPI.
            self._global_route_planner = None
            self._planner_error = f"CARLA GlobalRoutePlanner unavailable: {exc}"

    def _generate_with_global_route_planner(
        self,
        start: "carla.Waypoint",
        goal: "carla.Waypoint",
    ) -> List["carla.Waypoint"]:
        if self._global_route_planner is None:
            return []

        try:
            trace = self._global_route_planner.trace_route(
                start.transform.location,
                goal.transform.location,
            )
        except Exception as exc:
            self._planner_error = f"GlobalRoutePlanner failed: {exc}"
            return []

        route = [waypoint for waypoint, _road_option in trace]
        if not route:
            return []

        if route[0].transform.location.distance(start.transform.location) > self._sampling_resolution_m:
            route.insert(0, start)
        if route[-1].transform.location.distance(goal.transform.location) > self._sampling_resolution_m:
            route.append(goal)
        self._planner_error = None
        return route

    def _generate_with_fallback_graph(
        self,
        start: "carla.Waypoint",
        goal: "carla.Waypoint",
    ) -> List["carla.Waypoint"]:
        self._build_fallback_graph_if_needed()
        assert self._fallback_edges is not None

        start_edge, start_index = self._find_nearest_fallback_edge(start)
        goal_edge, goal_index = self._find_nearest_fallback_edge(goal)
        if start_edge is None or goal_edge is None:
            self._planner_error = "Fallback planner could not localize A or B on the topology graph."
            return []

        if start_edge == goal_edge and start_index <= goal_index:
            self._planner_error = None
            return [start] + start_edge.path[start_index : goal_index + 1] + [goal]

        middle_edges = self._shortest_fallback_edges(start_edge.end_key, goal_edge.start_key)
        if middle_edges is None:
            self._planner_error = "Fallback planner could not connect A to B."
            return []

        route = [start]
        route.extend(start_edge.path[start_index:])
        for edge in middle_edges:
            route.extend(edge.path[1:])
        route.extend(goal_edge.path[: goal_index + 1])
        route.append(goal)
        self._planner_error = None
        return route

    def _build_fallback_graph_if_needed(self) -> None:
        if self._fallback_graph is not None and self._fallback_edges is not None:
            return

        graph: Dict[NodeKey, List[_FallbackEdge]] = {}
        edges: List[_FallbackEdge] = []
        for entry_wp, exit_wp in self._world_map.get_topology():
            path = self._sample_segment(entry_wp, exit_wp)
            if len(path) < 2:
                continue

            start_key = self._node_key(path[0].transform.location)
            end_key = self._node_key(path[-1].transform.location)
            edge = _FallbackEdge(
                start_key=start_key,
                end_key=end_key,
                path=path,
                length_m=self._path_length(path),
            )
            graph.setdefault(start_key, []).append(edge)
            graph.setdefault(end_key, [])
            edges.append(edge)

        self._fallback_graph = graph
        self._fallback_edges = edges

    def _sample_segment(
        self,
        entry_wp: "carla.Waypoint",
        exit_wp: "carla.Waypoint",
    ) -> List["carla.Waypoint"]:
        path = [entry_wp]
        current_wp = entry_wp
        exit_location = exit_wp.transform.location
        max_steps = 3000

        for _ in range(max_steps):
            if current_wp.transform.location.distance(exit_location) <= self._sampling_resolution_m:
                break
            next_wps = current_wp.next(self._sampling_resolution_m)
            if not next_wps:
                break
            current_wp = next_wps[0]
            path.append(current_wp)

        if path[-1].transform.location.distance(exit_location) > 0.25:
            path.append(exit_wp)
        return path

    def _find_nearest_fallback_edge(
        self,
        waypoint: "carla.Waypoint",
    ) -> Tuple[Optional[_FallbackEdge], int]:
        assert self._fallback_edges is not None
        best_edge: Optional[_FallbackEdge] = None
        best_index = 0
        best_distance = float("inf")

        for edge in self._fallback_edges:
            lane_match = any(
                candidate.road_id == waypoint.road_id
                and candidate.section_id == waypoint.section_id
                and candidate.lane_id == waypoint.lane_id
                for candidate in (edge.path[0], edge.path[-1])
            )
            if not lane_match and best_distance < ROUTE_PLANNER.snap_search_radius_m:
                continue

            for index, candidate in enumerate(edge.path):
                distance = candidate.transform.location.distance(waypoint.transform.location)
                if lane_match:
                    distance *= 0.5
                if distance < best_distance:
                    best_distance = distance
                    best_edge = edge
                    best_index = index

        return best_edge, best_index

    def _shortest_fallback_edges(
        self,
        start_key: NodeKey,
        goal_key: NodeKey,
    ) -> Optional[List[_FallbackEdge]]:
        assert self._fallback_graph is not None
        queue: List[Tuple[float, NodeKey]] = [(0.0, start_key)]
        distances: Dict[NodeKey, float] = {start_key: 0.0}
        previous: Dict[NodeKey, Tuple[NodeKey, _FallbackEdge]] = {}

        while queue:
            current_distance, current_key = heapq.heappop(queue)
            if current_key == goal_key:
                break
            if current_distance > distances.get(current_key, float("inf")):
                continue

            for edge in self._fallback_graph.get(current_key, []):
                next_distance = current_distance + edge.length_m
                if next_distance >= distances.get(edge.end_key, float("inf")):
                    continue
                distances[edge.end_key] = next_distance
                previous[edge.end_key] = (current_key, edge)
                heapq.heappush(queue, (next_distance, edge.end_key))

        if goal_key not in distances:
            return None

        edges_reversed: List[_FallbackEdge] = []
        current_key = goal_key
        while current_key != start_key:
            prev_key, edge = previous[current_key]
            edges_reversed.append(edge)
            current_key = prev_key

        return list(reversed(edges_reversed))

    @staticmethod
    def _node_key(location: "carla.Location") -> NodeKey:
        return (round(location.x, 1), round(location.y, 1), round(location.z, 1))

    @staticmethod
    def _path_length(path: Sequence["carla.Waypoint"]) -> float:
        total = 0.0
        for first, second in zip(path, path[1:]):
            total += first.transform.location.distance(second.transform.location)
        return total

    def _densify_route(self, route: Sequence["carla.Waypoint"]) -> List["carla.Waypoint"]:
        if len(route) < 2:
            return list(route)

        dense_route: List["carla.Waypoint"] = [route[0]]
        max_step_m = max(0.5, self._sampling_resolution_m * 1.25)
        for next_trace_wp in route[1:]:
            current_wp = dense_route[-1]
            distance_to_next = current_wp.transform.location.distance(next_trace_wp.transform.location)
            if distance_to_next <= max_step_m:
                dense_route.append(next_trace_wp)
                continue

            max_steps = max(1, int(math.ceil(distance_to_next / self._sampling_resolution_m)) + 20)
            for _ in range(max_steps):
                current_location = current_wp.transform.location
                if current_location.distance(next_trace_wp.transform.location) <= max_step_m:
                    break

                next_candidates = current_wp.next(self._sampling_resolution_m)
                if not next_candidates:
                    break

                current_wp = self._choose_next_waypoint_toward(next_candidates, next_trace_wp)
                dense_route.append(current_wp)

            if dense_route[-1].transform.location.distance(next_trace_wp.transform.location) > 0.2:
                dense_route.append(next_trace_wp)

        return self._deduplicate_route(dense_route)

    @staticmethod
    def _deduplicate_route(route: Sequence["carla.Waypoint"]) -> List["carla.Waypoint"]:
        deduped: List["carla.Waypoint"] = []
        for waypoint in route:
            if deduped and waypoint.transform.location.distance(deduped[-1].transform.location) < 0.2:
                continue
            deduped.append(waypoint)
        return deduped

    @staticmethod
    def _choose_next_waypoint_toward(
        candidates: Sequence["carla.Waypoint"],
        target: "carla.Waypoint",
    ) -> "carla.Waypoint":
        def score(candidate: "carla.Waypoint") -> float:
            distance = candidate.transform.location.distance(target.transform.location)
            if (
                candidate.road_id == target.road_id
                and candidate.section_id == target.section_id
                and candidate.lane_id == target.lane_id
            ):
                distance *= 0.5
            return float(distance)

        return min(candidates, key=score)

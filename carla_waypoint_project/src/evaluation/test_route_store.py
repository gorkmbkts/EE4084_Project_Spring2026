"""JSON-backed saved test route storage."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

from src.planning.map_selector import RouteEndpoints
from src.utils.carla_import import ensure_carla_import
from src.utils.map_names import display_map_name, maps_compatible, normalize_map_name

carla = ensure_carla_import()


@dataclass(frozen=True)
class RoutePoint:
    """Serializable CARLA world coordinate."""

    x: float
    y: float
    z: float

    @classmethod
    def from_location(cls, location: "carla.Location") -> "RoutePoint":
        return cls(x=float(location.x), y=float(location.y), z=float(location.z))

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RoutePoint":
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data.get("z", 0.0)),
        )

    def to_location(self) -> "carla.Location":
        return carla.Location(x=self.x, y=self.y, z=self.z)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass(frozen=True)
class SavedTestRoute:
    """Serializable test route endpoints."""

    name: str
    start: RoutePoint
    goal: RoutePoint
    map_name: Optional[str] = None
    created_from: str = "2d_map"

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
        fallback_map_name: Optional[str] = None,
    ) -> "SavedTestRoute":
        return cls(
            name=str(data["name"]),
            start=RoutePoint.from_dict(data["start"]),  # type: ignore[arg-type]
            goal=RoutePoint.from_dict(data["goal"]),  # type: ignore[arg-type]
            map_name=str(data.get("map_name") or fallback_map_name) if (data.get("map_name") or fallback_map_name) else None,
            created_from=str(data.get("created_from") or "2d_map"),
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "name": self.name,
            "start": self.start.to_dict(),
            "goal": self.goal.to_dict(),
            "created_from": self.created_from,
        }
        if self.map_name:
            data["map_name"] = self.map_name
        return data


class TestRouteStore:
    """Load, save, select, and resolve saved A/B test routes."""

    def __init__(
        self,
        path: Optional[Path] = None,
        map_name: Optional[str] = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self._path = path if path is not None else project_root / "config" / "test_routes.json"
        self._map_name = map_name
        self._file_map_name: Optional[str] = None
        self._all_routes: list[SavedTestRoute] = []
        self._routes: list[SavedTestRoute] = []
        self._current_index = 0
        self.load_routes()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def routes(self) -> tuple[SavedTestRoute, ...]:
        return tuple(self._routes)

    @property
    def all_routes(self) -> tuple[SavedTestRoute, ...]:
        return tuple(self._all_routes)

    @property
    def active_map_name(self) -> Optional[str]:
        return self._map_name

    @property
    def active_map_id(self) -> Optional[str]:
        return normalize_map_name(self._map_name)

    @property
    def current_index(self) -> int:
        return self._current_index

    def route_count(self) -> int:
        return len(self._routes)

    def all_route_count(self) -> int:
        return len(self._all_routes)

    def other_map_route_count(self) -> int:
        return max(0, len(self._all_routes) - len(self._routes))

    def has_routes(self) -> bool:
        return bool(self._routes)

    def route_is_compatible(self, route: SavedTestRoute) -> bool:
        return maps_compatible(self._map_name, route.map_name)

    def active_map_display_name(self) -> str:
        return display_map_name(self._map_name)

    def load_routes(self) -> tuple[SavedTestRoute, ...]:
        if not self._path.exists():
            self._all_routes = []
            self._routes = []
            self._current_index = 0
            return self.routes

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._all_routes = []
            self._routes = []
            self._current_index = 0
            return self.routes

        top_level_map_name = data.get("map_name") if isinstance(data, dict) else None
        self._file_map_name = str(top_level_map_name) if top_level_map_name else None
        if self._map_name is None and top_level_map_name:
            self._map_name = str(top_level_map_name)
        fallback_route_map_name = str(top_level_map_name) if top_level_map_name else self._map_name

        loaded: list[SavedTestRoute] = []
        raw_routes = data.get("routes", []) if isinstance(data, dict) else []
        if isinstance(raw_routes, list):
            for raw_route in raw_routes:
                if not isinstance(raw_route, dict):
                    continue
                try:
                    loaded.append(SavedTestRoute.from_dict(raw_route, fallback_route_map_name))
                except (KeyError, TypeError, ValueError):
                    continue

        self._all_routes = loaded
        self._routes = self._compatible_routes(loaded)
        self._current_index = min(self._current_index, max(0, len(self._routes) - 1))
        return self.routes

    def save_routes(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "map_name": self._file_map_name or self._map_name,
            "routes": [route.to_dict() for route in self._all_routes],
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add_route_from_endpoints(
        self,
        name: Optional[str],
        endpoints: RouteEndpoints,
    ) -> SavedTestRoute:
        route_name = name or self.next_route_name()
        route = SavedTestRoute(
            name=route_name,
            start=RoutePoint.from_location(endpoints.start.transform.location),
            goal=RoutePoint.from_location(endpoints.goal.transform.location),
            map_name=self._map_name,
            created_from="2d_map",
        )
        self._all_routes.append(route)
        self._routes = self._compatible_routes(self._all_routes)
        self._current_index = max(0, len(self._routes) - 1)
        self.save_routes()
        return route

    def delete_route(self, index: int) -> bool:
        if index < 0 or index >= len(self._routes):
            return False
        route = self._routes[index]
        try:
            self._all_routes.remove(route)
        except ValueError:
            return False
        self._routes = self._compatible_routes(self._all_routes)
        self._current_index = min(self._current_index, max(0, len(self._routes) - 1))
        self.save_routes()
        return True

    def get_current_route(self) -> Optional[SavedTestRoute]:
        if not self._routes:
            return None
        self._current_index = min(self._current_index, len(self._routes) - 1)
        return self._routes[self._current_index]

    def next_route(self) -> Optional[SavedTestRoute]:
        if not self._routes:
            return None
        self._current_index = (self._current_index + 1) % len(self._routes)
        return self._routes[self._current_index]

    def previous_route(self) -> Optional[SavedTestRoute]:
        if not self._routes:
            return None
        self._current_index = (self._current_index - 1) % len(self._routes)
        return self._routes[self._current_index]

    def next_route_name(self) -> str:
        existing = {route.name for route in self._all_routes}
        index = 1
        while True:
            candidate = f"test_route_{index:03d}"
            if candidate not in existing:
                return candidate
            index += 1

    def _compatible_routes(self, routes: list[SavedTestRoute]) -> list[SavedTestRoute]:
        if self._map_name is None:
            return list(routes)
        return [route for route in routes if self.route_is_compatible(route)]

    @staticmethod
    def resolve_route_to_waypoints(
        world_map: "carla.Map",
        saved_route: SavedTestRoute,
    ) -> Optional[tuple["carla.Waypoint", "carla.Waypoint"]]:
        try:
            start_waypoint = world_map.get_waypoint(
                saved_route.start.to_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            goal_waypoint = world_map.get_waypoint(
                saved_route.goal.to_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except RuntimeError:
            return None

        if start_waypoint is None or goal_waypoint is None:
            return None
        return start_waypoint, goal_waypoint

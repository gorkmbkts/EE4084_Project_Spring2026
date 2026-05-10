"""CARLA client and world access management."""

from __future__ import annotations

from config.settings import CARLA
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class CarlaClientManager:
    """Own CARLA client connection and expose world-level objects."""

    def __init__(
        self,
        host: str = CARLA.host,
        port: int = CARLA.port,
        timeout_seconds: float = CARLA.timeout_seconds,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout_seconds = timeout_seconds

        self._client = None
        self._world = None
        self._world_map = None
        self._blueprint_library = None

    def connect(self) -> None:
        """Connect to CARLA and cache world objects."""
        self._client = carla.Client(self._host, self._port)
        self._client.set_timeout(self._timeout_seconds)

        self._world = self._client.get_world()
        self._world_map = self._world.get_map()
        self._blueprint_library = self._world.get_blueprint_library()

    @property
    def client(self) -> "carla.Client":
        if self._client is None:
            raise RuntimeError("CARLA client is not connected. Call connect() first.")
        return self._client

    @property
    def world(self) -> "carla.World":
        if self._world is None:
            raise RuntimeError("CARLA world is not available. Call connect() first.")
        return self._world

    @property
    def world_map(self) -> "carla.Map":
        if self._world_map is None:
            raise RuntimeError("CARLA map is not available. Call connect() first.")
        return self._world_map

    @property
    def blueprint_library(self) -> "carla.BlueprintLibrary":
        if self._blueprint_library is None:
            raise RuntimeError("Blueprint library is not available. Call connect() first.")
        return self._blueprint_library


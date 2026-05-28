"""CARLA client and world access management."""

from __future__ import annotations

import time
from typing import Optional

from config.settings import CARLA, SIMULATION
from src.utils.map_names import display_map_name, maps_compatible
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class CarlaConnectionError(RuntimeError):
    """Raised when the CARLA simulator cannot provide a world."""


class CarlaClientManager:
    """Own CARLA client connection and expose world-level objects."""

    def __init__(
        self,
        host: str = CARLA.host,
        port: int = CARLA.port,
        timeout_seconds: float = CARLA.timeout_seconds,
        connection_attempts: int = CARLA.connection_attempts,
        retry_delay_seconds: float = CARLA.retry_delay_seconds,
        requested_map_name: Optional[str] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout_seconds = timeout_seconds
        self._connection_attempts = max(1, int(connection_attempts))
        self._retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self._requested_map_name = requested_map_name

        self._client = None
        self._world = None
        self._world_map = None
        self._blueprint_library = None
        self._previous_world_settings: Optional["carla.WorldSettings"] = None
        self._sync_enabled = False
        self._sync_error: Optional[str] = None

    def connect(self) -> None:
        """Connect to CARLA, cache world objects, and apply simulation settings."""
        self._client = carla.Client(self._host, self._port)
        self._client.set_timeout(self._timeout_seconds)

        if self._requested_map_name:
            self._world = self._load_requested_world(self._requested_map_name)
        else:
            self._world = self._get_world_with_retries()
        try:
            if self._world is None:
                self._world = self._get_world_with_retries()
            self._world_map = self._world.get_map()
            self._blueprint_library = self._world.get_blueprint_library()
        except RuntimeError as exc:
            raise CarlaConnectionError(f"Connected to CARLA but failed to read world metadata: {exc}") from exc

        self.enable_synchronous_mode(
            enabled=SIMULATION.synchronous_mode,
            fixed_delta_seconds=SIMULATION.fixed_delta_seconds,
        )

    def load_world(self, map_name: str) -> None:
        """Load another CARLA world and refresh cached map/blueprint handles."""
        if not map_name:
            raise CarlaConnectionError("Cannot load an empty CARLA map name.")
        load_name = self.resolve_map_load_name(map_name)
        self.restore_world_settings()
        try:
            original_timeout = self._timeout_seconds
            self.client.set_timeout(max(float(original_timeout), 60.0))
            self._world = self.client.load_world(load_name)
            if self._world is None:
                self._world = self._get_world_with_retries()
            self._world_map = self._world.get_map()
            self._blueprint_library = self._world.get_blueprint_library()
        except RuntimeError as exc:
            raise CarlaConnectionError(
                f"Failed to load CARLA map {display_map_name(map_name)} via {load_name}: {exc}"
            ) from exc
        finally:
            try:
                self.client.set_timeout(self._timeout_seconds)
            except RuntimeError:
                pass

        self.enable_synchronous_mode(
            enabled=SIMULATION.synchronous_mode,
            fixed_delta_seconds=SIMULATION.fixed_delta_seconds,
        )

    def resolve_map_load_name(self, map_name: str) -> str:
        """Resolve saved route map metadata to a CARLA load_world map name."""
        try:
            available_maps = [str(item) for item in self.client.get_available_maps() or []]
        except RuntimeError:
            available_maps = []
        for candidate in available_maps:
            if maps_compatible(candidate, map_name):
                return candidate
        return display_map_name(map_name)

    def _load_requested_world(self, requested_map_name: str) -> "carla.World":
        try:
            world = self.client.load_world(requested_map_name)
        except RuntimeError as exc:
            raise CarlaConnectionError(
                f"Failed to load CARLA map {display_map_name(requested_map_name)}: {exc}"
            ) from exc
        if world is None:
            return self._get_world_with_retries()
        return world

    def _get_world_with_retries(self) -> "carla.World":
        last_error: Optional[RuntimeError] = None
        for attempt in range(1, self._connection_attempts + 1):
            try:
                return self.client.get_world()
            except RuntimeError as exc:
                last_error = exc
                if attempt < self._connection_attempts and self._retry_delay_seconds > 0.0:
                    time.sleep(self._retry_delay_seconds)

        detail = str(last_error) if last_error is not None else "unknown error"
        raise CarlaConnectionError(
            "Could not connect to a responsive CARLA simulator at "
            f"{self._host}:{self._port} after {self._connection_attempts} attempt(s). "
            "Start CARLA first and wait until the map is fully loaded. If CARLA is already open, "
            "close duplicate CARLA windows/processes so only one simulator owns the RPC port. "
            f"Last CARLA error: {detail}"
        ) from last_error

    def enable_synchronous_mode(self, enabled: bool, fixed_delta_seconds: float) -> None:
        """Apply CARLA synchronous/fixed-step settings when requested.

        Failure is recorded and leaves the app in an async fallback mode so the
        pygame UI can still run and report the problem.
        """
        if self._world is None or not enabled:
            self._sync_enabled = False
            return

        try:
            self._previous_world_settings = self._world.get_settings()
            settings = self._world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = float(fixed_delta_seconds)
            self._world.apply_settings(settings)
            self._sync_enabled = True
            self._sync_error = None
        except RuntimeError as exc:
            self._sync_enabled = False
            self._sync_error = f"Synchronous mode unavailable: {exc}"

    def tick(self) -> Optional[int]:
        """Advance one CARLA simulation frame when synchronous mode is active."""
        if not self._sync_enabled:
            return None

        try:
            return int(self.world.tick())
        except RuntimeError as exc:
            self._sync_enabled = False
            self._sync_error = f"CARLA sync tick failed: {exc}"
            return None

    def restore_world_settings(self) -> None:
        """Restore CARLA world settings captured before enabling sync mode."""
        if self._world is None or self._previous_world_settings is None:
            return

        try:
            self._world.apply_settings(self._previous_world_settings)
        except RuntimeError:
            pass
        finally:
            self._previous_world_settings = None
            self._sync_enabled = False

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

    @property
    def sync_enabled(self) -> bool:
        return self._sync_enabled

    @property
    def fixed_delta_seconds(self) -> float:
        return SIMULATION.fixed_delta_seconds

    @property
    def sync_status(self) -> str:
        if self._sync_enabled:
            return f"Sync ON dt={SIMULATION.fixed_delta_seconds:.3f}s"
        if self._sync_error:
            return self._sync_error
        return "Sync OFF"

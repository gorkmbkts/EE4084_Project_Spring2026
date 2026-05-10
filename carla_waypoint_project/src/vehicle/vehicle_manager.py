"""Vehicle spawning and lifecycle management."""

from __future__ import annotations

import random
from typing import Optional

from config.settings import VEHICLE
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class VehicleManager:
    """Spawn and own the primary ego vehicle actor."""

    def __init__(
        self,
        world: "carla.World",
        world_map: "carla.Map",
        blueprint_library: "carla.BlueprintLibrary",
    ) -> None:
        self._world = world
        self._world_map = world_map
        self._blueprint_library = blueprint_library
        self._vehicle: Optional["carla.Vehicle"] = None

    @property
    def vehicle(self) -> "carla.Vehicle":
        if self._vehicle is None:
            raise RuntimeError("Vehicle has not been spawned yet.")
        return self._vehicle

    def _select_vehicle_blueprint(self) -> "carla.ActorBlueprint":
        primary_candidates = self._blueprint_library.filter(VEHICLE.primary_blueprint_filter)
        if primary_candidates:
            return random.choice(primary_candidates)

        fallback_candidates = self._blueprint_library.filter(VEHICLE.fallback_blueprint_filter)
        if not fallback_candidates:
            raise RuntimeError("No vehicle blueprints available in this map/session.")
        return random.choice(fallback_candidates)

    def spawn_vehicle(self, spawn_point_index: int | None = VEHICLE.spawn_point_index) -> "carla.Vehicle":
        """Spawn an ego vehicle and ensure autopilot stays disabled."""
        vehicle_bp = self._select_vehicle_blueprint()
        spawn_points = self._world_map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found in CARLA map.")

        if spawn_point_index is not None:
            if spawn_point_index < 0 or spawn_point_index >= len(spawn_points):
                raise ValueError(
                    f"spawn_point_index={spawn_point_index} out of range [0, {len(spawn_points) - 1}]"
                )
            vehicle = self._world.try_spawn_actor(vehicle_bp, spawn_points[spawn_point_index])
            if vehicle is None:
                raise RuntimeError(f"Failed to spawn vehicle at index {spawn_point_index}.")
            vehicle.set_autopilot(False)
            self._vehicle = vehicle
            return vehicle

        shuffled = list(spawn_points)
        random.shuffle(shuffled)
        for spawn_point in shuffled:
            vehicle = self._world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle is not None:
                vehicle.set_autopilot(False)
                self._vehicle = vehicle
                return vehicle

        raise RuntimeError("Failed to spawn vehicle at any spawn point.")

    def teleport_to_waypoint(
        self,
        waypoint: "carla.Waypoint",
        z_offset_m: float = VEHICLE.teleport_z_offset_m,
    ) -> None:
        """Move the ego vehicle to a drivable waypoint and stop its motion."""
        vehicle = self.vehicle
        vehicle.set_autopilot(False)
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))

        waypoint_transform = waypoint.transform
        teleport_transform = carla.Transform(
            carla.Location(
                x=waypoint_transform.location.x,
                y=waypoint_transform.location.y,
                z=waypoint_transform.location.z + z_offset_m,
            ),
            carla.Rotation(
                pitch=waypoint_transform.rotation.pitch,
                yaw=waypoint_transform.rotation.yaw,
                roll=waypoint_transform.rotation.roll,
            ),
        )

        vehicle.set_transform(teleport_transform)
        self._reset_vehicle_motion(vehicle)
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))

    @staticmethod
    def _reset_vehicle_motion(vehicle: "carla.Vehicle") -> None:
        """Clear residual velocity after teleporting."""
        zero_vector = carla.Vector3D(0.0, 0.0, 0.0)
        try:
            vehicle.set_target_velocity(zero_vector)
            vehicle.set_target_angular_velocity(zero_vector)
        except (AttributeError, RuntimeError):
            pass

    def destroy(self) -> None:
        """Destroy vehicle actor if it exists."""
        if self._vehicle is not None:
            self._vehicle.destroy()
            self._vehicle = None

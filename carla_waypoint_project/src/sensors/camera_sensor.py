"""RGB camera sensor wrapper."""

from __future__ import annotations

from threading import Lock
from typing import Optional

import numpy as np
import pygame

from config.settings import CAMERA
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class CameraSensor:
    """Manage a CARLA RGB camera actor and its latest pygame frame."""

    def __init__(
        self,
        world: "carla.World",
        blueprint_library: "carla.BlueprintLibrary",
        attach_to: "carla.Actor",
    ) -> None:
        self._world = world
        self._blueprint_library = blueprint_library
        self._attach_to = attach_to

        self._actor: Optional["carla.Sensor"] = None
        self._latest_surface: Optional[pygame.Surface] = None
        self._surface_lock = Lock()

    @property
    def actor(self) -> "carla.Sensor":
        if self._actor is None:
            raise RuntimeError("Camera actor is not spawned yet.")
        return self._actor

    def spawn(self) -> None:
        """Spawn the RGB camera and start listening for image frames."""
        camera_bp = self._blueprint_library.find(CAMERA.blueprint_id)
        camera_bp.set_attribute("image_size_x", str(CAMERA.image_width))
        camera_bp.set_attribute("image_size_y", str(CAMERA.image_height))
        camera_bp.set_attribute("fov", str(CAMERA.fov_deg))

        camera_transform = carla.Transform(
            carla.Location(
                x=CAMERA.relative_x,
                y=CAMERA.relative_y,
                z=CAMERA.relative_z,
            ),
            carla.Rotation(
                pitch=CAMERA.relative_pitch,
                yaw=CAMERA.relative_yaw,
                roll=CAMERA.relative_roll,
            ),
        )

        self._actor = self._world.spawn_actor(camera_bp, camera_transform, attach_to=self._attach_to)
        self._actor.listen(self._on_image)

    def _on_image(self, image: "carla.Image") -> None:
        """Camera callback that updates the latest pygame surface."""
        surface = self._image_to_surface(image)
        with self._surface_lock:
            self._latest_surface = surface

    @staticmethod
    def _image_to_surface(image: "carla.Image") -> pygame.Surface:
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))
        img = img[:, :, :3][:, :, ::-1]
        return pygame.surfarray.make_surface(img.swapaxes(0, 1))

    def get_latest_surface(self) -> Optional[pygame.Surface]:
        """Return the latest received camera frame."""
        with self._surface_lock:
            return self._latest_surface

    def destroy(self) -> None:
        """Stop and destroy camera actor safely."""
        if self._actor is not None:
            self._actor.stop()
            self._actor.destroy()
            self._actor = None


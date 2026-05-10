"""Manual keyboard control for the ego vehicle."""

from __future__ import annotations

import pygame

from config.settings import MANUAL_CONTROL
from src.utils.carla_import import ensure_carla_import

carla = ensure_carla_import()


class ManualController:
    """Translate pygame keyboard state into ``carla.VehicleControl``."""

    def __init__(self, vehicle: "carla.Vehicle") -> None:
        self._vehicle = vehicle

    def process_events(self) -> bool:
        """Process window-level events. Returns False when app should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def _build_control(self) -> "carla.VehicleControl":
        keys = pygame.key.get_pressed()

        throttle = MANUAL_CONTROL.throttle if keys[pygame.K_w] else 0.0
        brake = MANUAL_CONTROL.brake if keys[pygame.K_s] else 0.0

        steer = 0.0
        if keys[pygame.K_a]:
            steer -= MANUAL_CONTROL.steer
        if keys[pygame.K_d]:
            steer += MANUAL_CONTROL.steer

        hand_brake = bool(keys[pygame.K_SPACE])

        return carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=hand_brake,
            reverse=False,
            manual_gear_shift=False,
        )

    def apply_control(self) -> None:
        """Apply one manual-control step to the vehicle."""
        control = self._build_control()
        self._vehicle.apply_control(control)


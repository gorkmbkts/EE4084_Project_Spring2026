"""pygame display wrapper for camera + overlay rendering."""

from __future__ import annotations

from typing import Optional

import pygame

from config.settings import DISPLAY


class PygameDisplay:
    """Own the pygame window and present rendered frames."""

    def __init__(
        self,
        width: int = DISPLAY.width,
        height: int = DISPLAY.height,
        title: str = DISPLAY.title,
    ) -> None:
        pygame.init()
        self._surface = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self._clear_color = DISPLAY.clear_color

    @property
    def surface(self) -> pygame.Surface:
        return self._surface

    def begin_frame(self, camera_surface: Optional[pygame.Surface]) -> None:
        """Blit current camera frame (or clear if frame is not ready)."""
        if camera_surface is None:
            self._surface.fill(self._clear_color)
        else:
            self._surface.blit(camera_surface, (0, 0))

    def end_frame(self) -> None:
        """Present the composed frame to the screen."""
        pygame.display.flip()

    def shutdown(self) -> None:
        """Close pygame resources."""
        pygame.quit()


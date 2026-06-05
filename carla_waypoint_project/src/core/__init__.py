"""Core application orchestration and CARLA connection logic."""

__all__ = ["SimulationApp", "CarlaClientManager"]


def __getattr__(name: str) -> object:
    if name == "SimulationApp":
        from .app import SimulationApp

        return SimulationApp
    if name == "CarlaClientManager":
        from .carla_client import CarlaClientManager

        return CarlaClientManager
    raise AttributeError(name)

"""Entry point for the modular CARLA waypoint visualization project."""

import sys

from src.core.carla_client import CarlaConnectionError
from src.core.app import SimulationApp


def main() -> None:
    app = SimulationApp()
    try:
        app.run()
    except CarlaConnectionError as exc:
        print(f"CARLA connection error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

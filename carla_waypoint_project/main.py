"""Entry point for the modular CARLA waypoint visualization project."""

from src.core.app import SimulationApp


def main() -> None:
    app = SimulationApp()
    app.run()


if __name__ == "__main__":
    main()


"""Entry point for the modular CARLA waypoint visualization project."""

import sys

from src.core.carla_server import (
    CarlaExecutableNotFound,
    CarlaLaunchError,
    CarlaServerManager,
    CarlaServerTimeout,
    CarlaStartupCancelled,
)
from src.visualization.startup_map_selector import StartupMapSelector


def main() -> None:
    selector = StartupMapSelector()
    server_manager = None
    try:
        server_manager = CarlaServerManager()
        client = server_manager.ensure_server_running(
            status_callback=lambda status, detail: selector.show_status(
                status,
                detail,
                executable_path=server_manager.executable_path,
            )
        )
        selection = selector.choose_map(
            client,
            executable_path=server_manager.executable_path,
        )
        if selection is None:
            return

        from src.core.carla_client import CarlaConnectionError
        from src.core.app import SimulationApp

        app = SimulationApp(
            selected_map_load_name=selection.selected_map_load_name,
            existing_display_surface=selector.surface,
        )
        try:
            app.run()
        except CarlaConnectionError as exc:
            selector.wait_for_error_ack(f"CARLA connection error: {exc}")
            print(f"CARLA connection error: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
    except (CarlaExecutableNotFound, CarlaLaunchError, CarlaServerTimeout) as exc:
        selector.wait_for_error_ack(str(exc))
        print(f"CARLA startup error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except ModuleNotFoundError as exc:
        if exc.name != "carla":
            raise
        message = (
            "CARLA Python API not found. Place CARLA_0.9.16 near the project "
            "or install the CARLA Python package for this environment."
        )
        selector.wait_for_error_ack(message)
        print(message, file=sys.stderr)
        raise SystemExit(1) from None
    except CarlaStartupCancelled:
        return
    finally:
        if server_manager is not None:
            server_manager.shutdown_if_owned()


if __name__ == "__main__":
    main()

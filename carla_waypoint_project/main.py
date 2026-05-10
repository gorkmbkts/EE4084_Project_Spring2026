"""Entry point for the modular CARLA waypoint visualization project."""

import sys
import subprocess
import time

from src.core.carla_client import CarlaConnectionError
from src.core.app import SimulationApp


CARLA_EXE_PATH = r"C:\Users\gorke\Desktop\lectures\EE4084 Kalman and Bayesian Filters\EE4084_Project_Spring2026\CARLA_0.9.16\CarlaUE4.exe"


def start_carla_exe() -> subprocess.Popen[bytes]:
    process = subprocess.Popen([CARLA_EXE_PATH])
    while process.poll() is not None:
        time.sleep(0.5)
    return process

def main() -> None:
    
    app = SimulationApp()
    try:
        app.run()
    except CarlaConnectionError as exc:
        print(f"CARLA connection error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    start_carla_exe()
    
    main()

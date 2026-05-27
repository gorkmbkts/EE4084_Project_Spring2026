"""CARLA process discovery, launch, and RPC readiness management."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import subprocess
import sys
import time
from typing import Optional

from config.settings import CARLA
from src.utils.carla_import import ensure_carla_import


StartupStatusCallback = Callable[[str, Optional[str]], bool | None]


class CarlaExecutableNotFound(RuntimeError):
    """Raised when the local CARLA executable cannot be found."""


class CarlaLaunchError(RuntimeError):
    """Raised when CARLA cannot be launched or exits during startup."""


class CarlaServerTimeout(RuntimeError):
    """Raised when CARLA RPC does not become responsive in time."""


class CarlaStartupCancelled(RuntimeError):
    """Raised when the user cancels the startup flow."""


class CarlaServerManager:
    """Ensure one CARLA server is reachable, optionally launching the local exe."""

    def __init__(
        self,
        host: str = CARLA.host,
        port: int = CARLA.port,
        timeout_seconds: float = CARLA.timeout_seconds,
        launch_timeout_seconds: float = CARLA.launch_timeout_seconds,
        auto_launch: bool = CARLA.auto_launch,
        executable_path: str | None = CARLA.executable_path,
        startup_quality_level: str | None = CARLA.startup_quality_level,
        shutdown_launched_server_on_exit: bool = CARLA.shutdown_launched_server_on_exit,
    ) -> None:
        self._host = host
        self._port = int(port)
        self._timeout_seconds = float(timeout_seconds)
        self._launch_timeout_seconds = float(launch_timeout_seconds)
        self._auto_launch = bool(auto_launch)
        self._configured_executable_path = executable_path
        self._startup_quality_level = startup_quality_level
        self._shutdown_launched_server_on_exit = bool(shutdown_launched_server_on_exit)

        self._project_root = Path(__file__).resolve().parents[2]
        self._process: Optional[subprocess.Popen[object]] = None
        self._launched_by_app = False
        self._waiting_for_existing_process = False
        self._executable_path: Optional[Path] = None
        self._carla = ensure_carla_import()

    @property
    def executable_path(self) -> Optional[Path]:
        return self._executable_path

    @property
    def launched_by_app(self) -> bool:
        return self._launched_by_app

    def ensure_server_running(self, status_callback: StartupStatusCallback | None = None):
        """Return a responsive CARLA client, launching the server if configured."""
        self._executable_path = self.find_executable()
        self._notify(status_callback, "Not running", "Checking localhost CARLA RPC.")
        existing_client = self._try_connect(timeout_seconds=min(3.0, self._timeout_seconds))
        if existing_client is not None:
            self._notify(status_callback, "Connected", "Using already-running CARLA RPC server.")
            return existing_client

        if self._carla_process_running():
            self._waiting_for_existing_process = True
            self._notify(
                status_callback,
                "Waiting for RPC",
                "Detected an existing CarlaUE4.exe process; waiting instead of launching another.",
            )
            return self.wait_until_ready(status_callback=status_callback)

        if not self._auto_launch:
            raise CarlaServerTimeout(
                f"Could not connect to CARLA at {self._host}:{self._port}; auto_launch is disabled."
            )

        if self._executable_path is None:
            raise CarlaExecutableNotFound(
                "CARLA executable not found. Set CARLA.executable_path in config/settings.py "
                "or place CARLA_0.9.16 near the project."
            )

        self._notify(
            status_callback,
            "Launching CARLA",
            f"Starting {self._executable_path}",
        )
        self.launch()
        return self.wait_until_ready(status_callback=status_callback)

    def find_executable(self) -> Optional[Path]:
        """Find CarlaUE4.exe using explicit settings first, then local candidates."""
        if self._configured_executable_path:
            explicit = Path(self._configured_executable_path).expanduser()
            if explicit.exists() and explicit.is_file():
                return explicit.resolve()
            return None

        candidates = [
            self._project_root / "CARLA_0.9.16" / "CarlaUE4.exe",
            self._project_root.parent / "CARLA_0.9.16" / "CarlaUE4.exe",
            self._project_root / "CARLA_0.9.16" / "WindowsNoEditor" / "CarlaUE4.exe",
            self._project_root.parent / "CARLA_0.9.16" / "WindowsNoEditor" / "CarlaUE4.exe",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        return None

    def launch(self) -> None:
        """Launch CARLA once and remember that this app owns the process."""
        if self._process is not None and self._process.poll() is None:
            return
        if self._executable_path is None:
            self._executable_path = self.find_executable()
        if self._executable_path is None:
            raise CarlaExecutableNotFound(
                "CARLA executable not found. Set CARLA.executable_path in config/settings.py "
                "or place CARLA_0.9.16 near the project."
            )

        args = [str(self._executable_path), f"-carla-rpc-port={self._port}"]
        if self._startup_quality_level:
            args.append(f"-quality-level={self._startup_quality_level}")

        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        try:
            self._process = subprocess.Popen(
                args,
                cwd=str(self._executable_path.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise CarlaLaunchError(f"Failed to launch CARLA executable: {exc}") from exc

        self._launched_by_app = True
        self._waiting_for_existing_process = False

    def wait_until_ready(self, status_callback: StartupStatusCallback | None = None):
        """Poll until CARLA RPC is responsive or startup times out."""
        deadline = time.monotonic() + max(1.0, self._launch_timeout_seconds)
        last_error: Optional[str] = None
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise CarlaLaunchError(
                    f"CARLA exited before RPC became available (exit code {self._process.returncode})."
                )
            if (
                self._process is None
                and self._waiting_for_existing_process
                and not self._carla_process_running()
            ):
                raise CarlaLaunchError("CARLA closed before RPC became available.")

            remaining = max(0.0, deadline - time.monotonic())
            detail = f"Waiting for RPC at {self._host}:{self._port} ({remaining:.0f}s left)."
            self._notify(status_callback, "Waiting for RPC", detail)

            try:
                client = self._try_connect(timeout_seconds=min(3.0, self._timeout_seconds))
            except Exception as exc:  # pragma: no cover - defensive around CARLA client variants.
                last_error = str(exc)
                client = None
            if client is not None:
                self._waiting_for_existing_process = False
                self._notify(status_callback, "Connected", "CARLA RPC is ready.")
                return client

            time.sleep(1.0)

        detail = f" Last CARLA error: {last_error}" if last_error else ""
        raise CarlaServerTimeout(
            f"Timed out waiting for CARLA RPC at {self._host}:{self._port} "
            f"after {self._launch_timeout_seconds:.0f} seconds.{detail}"
        )

    def shutdown_if_owned(self) -> None:
        """Terminate only the CARLA process this app launched, if configured."""
        if not self._shutdown_launched_server_on_exit:
            return
        if not self._launched_by_app or self._process is None:
            return
        if self._process.poll() is not None:
            return

        self._process.terminate()
        try:
            self._process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5.0)

    def _try_connect(self, timeout_seconds: float):
        client = self._carla.Client(self._host, self._port)
        client.set_timeout(float(timeout_seconds))
        try:
            client.get_world()
        except Exception:
            return None
        return client

    @staticmethod
    def _carla_process_running() -> bool:
        try:
            if sys.platform == "win32":
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq CarlaUE4.exe", "/FO", "CSV", "/NH"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                return "CarlaUE4.exe" in result.stdout

            result = subprocess.run(
                ["pgrep", "-f", "CarlaUE4"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _notify(
        status_callback: StartupStatusCallback | None,
        status: str,
        detail: Optional[str] = None,
    ) -> None:
        if status_callback is None:
            return
        keep_running = status_callback(status, detail)
        if keep_running is False:
            raise CarlaStartupCancelled("Startup cancelled")

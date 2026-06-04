"""Serialized background plot generation for benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
import traceback
from typing import Literal, Optional

PlotJobKind = Literal["route", "aggregate"]


@dataclass(frozen=True)
class BenchmarkPlotJobStatus:
    """Snapshot of the background benchmark plot queue."""

    pending_jobs: int
    running_job: Optional[str]
    completed_jobs: int
    failed_jobs: int
    latest_error: Optional[str]

    @property
    def running_jobs(self) -> int:
        return 1 if self.running_job else 0


@dataclass(frozen=True)
class _PlotJob:
    kind: PlotJobKind
    folder: Path


_STOP = object()


class BenchmarkPlotJobWorker:
    """Run matplotlib benchmark plot jobs off the pygame/CARLA loop.

    The worker accepts only filesystem paths. It deliberately imports the
    plotting module inside the worker thread so matplotlib work stays isolated
    from the simulation frame.
    """

    def __init__(self) -> None:
        self._queue: Queue[_PlotJob | object] = Queue()
        self._lock = Lock()
        self._thread: Optional[Thread] = None
        self._accepting_jobs = True
        self._pending_jobs = 0
        self._running_job: Optional[_PlotJob] = None
        self._completed_jobs = 0
        self._failed_jobs = 0
        self._latest_error: Optional[str] = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._accepting_jobs = True
            self._thread = Thread(target=self._run, name="BenchmarkPlotJobWorker", daemon=True)
            self._thread.start()

    def enqueue_route_plots(self, route_folder: Path) -> bool:
        return self._enqueue(_PlotJob(kind="route", folder=Path(route_folder)))

    def enqueue_aggregate_plots(self, run_folder: Path) -> bool:
        return self._enqueue(_PlotJob(kind="aggregate", folder=Path(run_folder)))

    def poll_status(self) -> BenchmarkPlotJobStatus:
        with self._lock:
            running_job = self._running_label(self._running_job)
            return BenchmarkPlotJobStatus(
                pending_jobs=self._pending_jobs,
                running_job=running_job,
                completed_jobs=self._completed_jobs,
                failed_jobs=self._failed_jobs,
                latest_error=self._latest_error,
            )

    def status_text(self) -> str:
        status = self.poll_status()
        return (
            "Plot jobs: "
            f"pending {status.pending_jobs}, "
            f"running {status.running_jobs}, "
            f"done {status.completed_jobs}, "
            f"failed {status.failed_jobs}"
        )

    def shutdown(self, wait: bool = False, timeout_s: float = 2.0) -> None:
        with self._lock:
            self._accepting_jobs = False
            thread = self._thread
            self._pending_jobs = 0

        self._drain_pending_jobs()
        if thread is not None and thread.is_alive():
            self._queue.put(_STOP)
            if wait:
                thread.join(timeout=max(0.0, float(timeout_s)))

    def _enqueue(self, job: _PlotJob) -> bool:
        with self._lock:
            if not self._accepting_jobs:
                return False
            self._pending_jobs += 1
        self.start()
        self._queue.put(job)
        return True

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                job = item
                if not isinstance(job, _PlotJob):
                    return
                with self._lock:
                    self._pending_jobs = max(0, self._pending_jobs - 1)
                    self._running_job = job
                try:
                    self._run_job(job)
                except Exception as exc:  # pragma: no cover - plotting/filesystem dependent.
                    self._record_failure(job, exc, traceback.format_exc())
                else:
                    with self._lock:
                        self._completed_jobs += 1
                finally:
                    with self._lock:
                        self._running_job = None
            finally:
                self._queue.task_done()

    @staticmethod
    def _run_job(job: _PlotJob) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        if job.kind == "route":
            from src.evaluation.benchmark_plotter import generate_benchmark_plots

            generate_benchmark_plots(job.folder)
            return
        if job.kind == "aggregate":
            from src.evaluation.benchmark_plotter import generate_aggregate_benchmark_plots

            generate_aggregate_benchmark_plots(job.folder)
            return
        raise ValueError(f"Unsupported plot job kind: {job.kind}")

    def _record_failure(self, job: _PlotJob, exc: Exception, formatted_traceback: str) -> None:
        latest_error = f"{job.kind} plots failed for {job.folder.name}: {exc}"
        with self._lock:
            self._failed_jobs += 1
            self._latest_error = self._shorten(latest_error, max_length=180)
        self._append_error_log(job, latest_error, formatted_traceback)
        self._record_plot_error(job, latest_error)
        print(f"[benchmark plots] {latest_error}\n{formatted_traceback}", flush=True)

    @staticmethod
    def _append_error_log(job: _PlotJob, latest_error: str, formatted_traceback: str) -> None:
        try:
            job.folder.mkdir(parents=True, exist_ok=True)
            with (job.folder / "plot_generation_errors.log").open("a", encoding="utf-8") as log_file:
                log_file.write(f"[{datetime.now().isoformat(timespec='seconds')}] {latest_error}\n")
                log_file.write(formatted_traceback)
                if not formatted_traceback.endswith("\n"):
                    log_file.write("\n")
                log_file.write("\n")
        except Exception:
            pass

    def _drain_pending_jobs(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                return
            try:
                if item is _STOP:
                    self._queue.put(_STOP)
                    return
            finally:
                self._queue.task_done()

    @staticmethod
    def _record_plot_error(job: _PlotJob, latest_error: str) -> None:
        if job.kind == "aggregate":
            paths = (job.folder / "aggregate_summary.json",)
        else:
            paths = (job.folder / "route_summary.json", job.folder / "summary.json")
        for path in paths:
            try:
                if not path.exists():
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                data["plot_error"] = latest_error
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                continue

    @staticmethod
    def _running_label(job: Optional[_PlotJob]) -> Optional[str]:
        if job is None:
            return None
        return f"{job.kind}: {job.folder.name}"

    @staticmethod
    def _shorten(text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        return text[: max(0, max_length - 3)].rstrip() + "..."

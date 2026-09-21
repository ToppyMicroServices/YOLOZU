from __future__ import annotations

from collections.abc import Mapping
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..manifest_resources import workspace_root


_JOB_ID_RE = re.compile(r"job_[A-Za-z0-9_-]{1,64}\Z")


@dataclass
class _JobState:
    job_id: str
    name: str
    status: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    future: Future | None = None
    cancel_event: threading.Event | None = None
    timer: threading.Timer | None = None
    on_done: Callable[[str], None] | None = None


class JobManager:
    def __init__(self, max_workers: int = 2, storage_dir: str | Path | None = None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="yolozu-mcp-job")
        self._lock = threading.Lock()
        self._jobs: dict[str, _JobState] = {}
        self._closed = False
        if storage_dir is None:
            storage_dir = workspace_root() / "runs" / "mcp_jobs"
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    def _job_file(self, job_id: str) -> Path:
        return self._storage_dir / f"{job_id}.json"

    def _serialize(self, job: _JobState) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "name": job.name,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "result": job.result,
            "error": job.error,
        }

    def _persist(self, job: _JobState) -> None:
        path = self._job_file(job.job_id)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as handle:
                json.dump(
                    self._serialize(job),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _load_from_disk(self) -> None:
        for path in sorted(self._storage_dir.glob("job_*.json")):
            try:
                if path.is_symlink() or path.stat().st_size > 32 * 1024 * 1024:
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                job_id = payload.get("job_id")
                if job_id != path.stem or _JOB_ID_RE.fullmatch(str(job_id)) is None:
                    continue
                state = _JobState(
                    job_id=str(job_id),
                    name=str(payload.get("name") or "unknown"),
                    status=str(payload.get("status") or "unknown"),
                    created_at=float(payload.get("created_at") or 0.0),
                    started_at=payload.get("started_at"),
                    finished_at=payload.get("finished_at"),
                    result=payload.get("result"),
                    error=payload.get("error"),
                )
                if state.status in ("queued", "running"):
                    state.status = "unknown"
                self._jobs[state.job_id] = state
            except (json.JSONDecodeError, OSError, UnicodeDecodeError, TypeError, ValueError):
                continue

    def purge_terminal_before(self, cutoff: float) -> int:
        """Remove terminal job records older than an explicit Unix cutoff."""

        if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
            raise ValueError("cutoff must be a Unix timestamp")
        removed = 0
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.status not in {"completed", "failed", "cancelled", "timed_out", "unknown"}:
                    continue
                finished = job.finished_at if job.finished_at is not None else job.created_at
                if finished >= float(cutoff):
                    continue
                path = self._job_file(job_id)
                if path.is_symlink():
                    continue
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                self._jobs.pop(job_id, None)
                removed += 1
        return removed

    def submit(
        self, name: str, fn: Callable[[], dict[str, Any]], *,
        on_done: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        state = _JobState(job_id=job_id, name=name, status="queued", created_at=time.time())
        state.cancel_event = cancel_event
        state.on_done = on_done

        def _run() -> None:
            try:
                with self._lock:
                    state.status = "running"
                    state.started_at = time.time()
                    self._persist(state)
                result = fn()
                failed = False
                failure_error: str | None = None
                if isinstance(result, Mapping):
                    result_ok = result.get("ok")
                    exit_code = result.get("exit_code")
                    failed = result_ok is False or (
                        isinstance(exit_code, int)
                        and not isinstance(exit_code, bool)
                        and exit_code != 0
                    )
                    if failed:
                        raw_error = result.get("error")
                        if isinstance(raw_error, Mapping):
                            raw_error = (
                                raw_error.get("message")
                                or raw_error.get("code")
                            )
                        failure_error = str(
                            raw_error
                            or result.get("summary")
                            or f"{name} returned an unsuccessful result"
                        )
                with self._lock:
                    state.status = "failed" if failed else "completed"
                    if isinstance(result, Mapping) and result.get("outcome") in {"cancelled", "timed_out"}:
                        state.status = result["outcome"]
                    state.result = result
                    state.error = failure_error
                    state.finished_at = time.time()
            except BaseException as exc:
                # This is a background job boundary: even SystemExit/MemoryError
                # must release capacity. Do not retain an exception traceback in
                # the Future (it can retain model weights and decoded images).
                with self._lock:
                    state.status = "failed"
                    state.error = f"{type(exc).__name__}: {exc}"[:1024]
                    state.finished_at = time.time()
            finally:
                try:
                    with self._lock:
                        try:
                            self._persist(state)
                        except Exception as exc:
                            state.error = f"job state persistence failed: {type(exc).__name__}"
                            state.status = "failed"
                finally:
                    if on_done is not None:
                        on_done(job_id)
                    state.on_done = None

        def forget_future(_future: Future) -> None:
            with self._lock:
                state.future = None
                state.cancel_event = None
                if state.timer is not None:
                    state.timer.cancel()
                    state.timer = None

        def expire() -> None:
            result = self.cancel(job_id)
            if result and result.get("cancelled"):
                with self._lock:
                    state.status = "timed_out"
                    state.result = {"ok": False, "exit_code": 1, "outcome": "timed_out"}
                    try:
                        self._persist(state)
                    except OSError:
                        state.error = "timed out; job state persistence failed"

        with self._lock:
            if self._closed:
                raise RuntimeError("job manager is closed")
            self._persist(state)
            try:
                self._jobs[job_id] = state
                if deadline is not None:
                    state.timer = threading.Timer(max(0, deadline - time.monotonic()), expire)
                    state.timer.daemon = True
                    state.timer.start()
                future = self._executor.submit(_run)
                state.future = future
            except BaseException:
                if state.timer is not None:
                    state.timer.cancel()
                self._jobs.pop(job_id, None)
                self._job_file(job_id).unlink(missing_ok=True)
                raise
        future.add_done_callback(forget_future)
        return job_id

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "job_id": job.job_id,
                    "name": job.name,
                    "status": job.status,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                }
                for job in self._jobs.values()
            ]

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "job_id": job.job_id,
                "name": job.name,
                "status": job.status,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "error": job.error,
                "result": job.result,
            }

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in ("completed", "failed", "cancelled", "timed_out"):
                return {"job_id": job_id, "cancelled": False, "reason": f"already_{job.status}"}
            # Future.cancel invokes callbacks synchronously. The manager lock is
            # released before doing so to avoid re-entering forget_future.
            future = job.future
            if job.cancel_event is not None:
                job.cancel_event.set()
        if future and future.cancel():
            with self._lock:
                job.status = "cancelled"
                job.finished_at = time.time()
                try:
                    self._persist(job)
                except OSError:
                    job.error = "cancelled; job state persistence failed"
                on_done = job.on_done
                job.on_done = None
            if on_done is not None:
                on_done(job_id)
            return {"job_id": job_id, "cancelled": True}
        with self._lock:
            if job.cancel_event is not None:
                return {"job_id": job_id, "cancelled": False, "reason": "cancellation_requested"}
            return {"job_id": job_id, "cancelled": False, "reason": "running"}

    def shutdown(self, *, timeout: float | None = None) -> None:
        """Reject new work, cancel queued work, and signal owned running jobs."""
        with self._lock:
            self._closed = True
            ids = list(self._jobs)
            futures = [job.future for job in self._jobs.values() if job.future is not None]
        for job_id in ids:
            self.cancel(job_id)
        self._executor.shutdown(wait=False, cancel_futures=True)
        if timeout is not None and futures:
            _done, pending = wait([future for future in futures if not future.done()], timeout=timeout)
            if pending:
                raise RuntimeError("job shutdown exceeded its cleanup deadline")

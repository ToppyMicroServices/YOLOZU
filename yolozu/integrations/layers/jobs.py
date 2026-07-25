from __future__ import annotations

from collections.abc import Mapping
import json
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..manifest_resources import workspace_root


_JOB_FAILURE_ERRORS = (
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    KeyError,
    LookupError,
    AssertionError,
)


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


class JobManager:
    def __init__(self, max_workers: int = 2, storage_dir: str | Path | None = None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="yolozu-mcp-job")
        self._lock = threading.Lock()
        self._jobs: dict[str, _JobState] = {}
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
        self._job_file(job.job_id).write_text(json.dumps(self._serialize(job), ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_from_disk(self) -> None:
        for path in sorted(self._storage_dir.glob("job_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                state = _JobState(
                    job_id=str(payload.get("job_id") or path.stem),
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

    def submit(self, name: str, fn: Callable[[], dict[str, Any]]) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        state = _JobState(job_id=job_id, name=name, status="queued", created_at=time.time())
        self._persist(state)

        def _run() -> dict[str, Any]:
            with self._lock:
                state.status = "running"
                state.started_at = time.time()
                self._persist(state)
            try:
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
                    state.result = result
                    state.error = failure_error
                    state.finished_at = time.time()
                    self._persist(state)
                return result
            except _JOB_FAILURE_ERRORS as exc:
                with self._lock:
                    state.status = "failed"
                    state.error = str(exc)
                    state.finished_at = time.time()
                    self._persist(state)
                raise

        with self._lock:
            self._jobs[job_id] = state
            state.future = self._executor.submit(_run)
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
            if job.future and job.future.cancel():
                job.status = "cancelled"
                job.finished_at = time.time()
                self._persist(job)
                return {"job_id": job_id, "cancelled": True}
            if job.status in ("completed", "failed", "cancelled"):
                return {"job_id": job_id, "cancelled": False, "reason": f"already_{job.status}"}
            return {"job_id": job_id, "cancelled": False, "reason": "running"}

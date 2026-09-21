"""Supervise one code-owned image pipeline, including its selection preflight."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from typing import Any

from yolozu.process_lifetime import ProcessGuard, stop_and_reap


MAX_RESULT_BYTES = 2 * 1024 * 1024


def execute_image_job(request: dict[str, Any]) -> dict[str, Any]:
    from yolozu.adaptive.processing import ProcessingError, process_images
    from yolozu.adaptive.recommendation import (
        RecommendationError,
        recommend_image_pipeline,
    )

    try:
        recommendation = recommend_image_pipeline(
            request["job_spec"],
            request["input"],
            workspace_root=request["workspace"],
        )
        decision = recommendation.get("decision")
        if not isinstance(decision, dict):
            raise ValueError("qualified selection returned an invalid decision")
        if decision.get("status") != "selected":
            return {
                "ok": True,
                "exit_code": 0,
                "outcome": "abstained",
                "decision": decision,
            }
        processed = process_images(
            request["job_spec"],
            decision,
            request["input"],
            request["output"],
            workspace_root=request["workspace"],
            dry_run=not request["execute"],
        )
        return {
            "ok": bool(processed.get("ok")),
            "exit_code": int(processed.get("exit_code", 0)),
            "outcome": (
                ("completed" if processed.get("executed") else "ready")
                if processed.get("ok")
                else "rejected"
            ),
            "decision": decision,
            "executed": bool(processed.get("executed")),
            "_output_token": request["output_token"]
            if processed.get("ok") and processed.get("executed")
            else None,
        }
    except (RecommendationError, ProcessingError) as exc:
        return {
            "ok": False,
            "exit_code": 1,
            "outcome": "rejected",
            "error": {"code": exc.code, "message": exc.public_message},
        }


def _worker(connection: Any, request: dict[str, Any]) -> None:
    try:
        os.setsid()
        result = execute_image_job(request)
        data = json.dumps(result).encode("utf-8")
        if len(data) > MAX_RESULT_BYTES:
            raise ValueError("image job result exceeds its bound")
        connection.send_bytes(data)
    except BaseException:
        try:
            connection.send_bytes(
                b'{"ok":false,"exit_code":1,"outcome":"failed","error":"image job failed"}'
            )
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


def run_image_job(
    request: dict[str, Any], cancel: threading.Event, deadline: float
) -> dict[str, Any]:
    """Bound queue-to-completion time; cancellation does not kill a host thread."""

    def stopped(outcome: str) -> dict[str, Any]:
        return {"ok": False, "exit_code": 1, "outcome": outcome}

    if time.monotonic() >= deadline:
        return stopped("timed_out")
    if cancel.is_set():
        return stopped("cancelled")
    if os.name != "posix":
        raise RuntimeError("image job execution requires POSIX process ownership")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(child, request), daemon=False)
    guard = None
    started = False
    try:
        process.start()
        started = True
        child.close()
        guard = ProcessGuard(process.pid, deadline)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return stopped("timed_out")
            if cancel.is_set():
                return stopped("cancelled")
            if parent.poll(min(0.05, remaining)):
                try:
                    data = parent.recv_bytes(MAX_RESULT_BYTES)
                except (EOFError, OSError):
                    return stopped(
                        "timed_out" if time.monotonic() >= deadline else "failed"
                    )
                result = json.loads(data)
                if not isinstance(result, dict):
                    raise ValueError("invalid image job result")
                return result
            if not process.is_alive():
                return stopped("failed")
    finally:
        child.close()
        parent.close()
        reaped = not started
        try:
            if started:
                stop_and_reap(process)
                reaped = not process.is_alive()
                if not reaped:
                    raise RuntimeError("image job process could not be reaped")
                process.close()
        finally:
            if guard is not None:
                guard.close(disarm=reaped)

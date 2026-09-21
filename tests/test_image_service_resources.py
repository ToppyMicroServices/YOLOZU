from __future__ import annotations

import base64
import gc
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import weakref

from yolozu.adaptive.qualification import _ForkedRunnerSession
from yolozu.integrations.image_job_execution import run_image_job
from yolozu.integrations.image_service import (
    ImageService,
    ImageServiceError,
    _safe_json_read,
)
from yolozu.integrations.layers.jobs import JobManager
from tests.test_image_service import _png_bytes
from tests.test_image_service_plugin import client


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _wait(predicate, timeout=5):
    until = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= until:
            raise AssertionError("bounded test condition did not become true")
        time.sleep(0.01)


def _slow_worker(connection, request):
    os.setsid()
    Path(request["pid_file"]).write_text(str(os.getpid()))
    time.sleep(20)  # Additional safety bound if a regression breaks ownership.
    connection.close()


def _slow_service_worker(connection, request):
    _slow_worker(
        connection, {"pid_file": str(Path(request["workspace"]) / "worker.pid")}
    )


class _SlowCloseRunner:
    runner_id = "synthetic-resource-test"
    runner_version = "1"

    def close(self):
        time.sleep(20)


def _runner_owner(connection):
    os.setsid()
    session = _ForkedRunnerSession(
        factory=_SlowCloseRunner,
        bundle=None,
        environment=None,
        artifacts=None,
        inputs=[],
        labels=(),
        outer_deadline_ns=time.monotonic_ns() + 20_000_000_000,
    )
    connection.send((session._process.pid, session._guard._process.pid))
    try:
        time.sleep(20)
    finally:
        session.close(0.1)
        connection.close()


def _job_owner(connection, pid_file):
    os.setsid()
    connection.send(os.getpid())
    connection.close()
    with patch("yolozu.integrations.image_job_execution._worker", _slow_worker):
        run_image_job({"pid_file": pid_file}, threading.Event(), time.monotonic() + 20)


class TestJobCleanup(unittest.TestCase):
    def test_asset_stat_failure_does_not_keep_reference(self):
        with tempfile.TemporaryDirectory() as root:
            service = ImageService(workspace=root)
            asset = "asset_" + "a" * 24
            service._job_assets["job_fixture"] = (asset, "run_" + "b" * 24)
            service._active_assets[asset] = 1
            with patch.object(Path, "is_dir", side_effect=PermissionError("injected")):
                service._release_job("job_fixture")
            self.assertEqual(service._active_assets, {})
            self.assertEqual(service._job_assets, {})
            service.close()

    def manager(self, root):
        manager = JobManager(max_workers=1, storage_dir=Path(root) / "jobs")
        self.addCleanup(lambda: manager.shutdown(timeout=5))
        return manager

    def finished(self, manager, job):
        _wait(lambda: manager._jobs[job].future is None)

    def test_all_worker_exceptions_release_tracebacks_and_capacity(self):
        class Payload:
            pass

        for exception in (ZeroDivisionError, MemoryError, SystemExit):
            with (
                self.subTest(exception=exception),
                tempfile.TemporaryDirectory() as root,
            ):
                manager = self.manager(root)
                references = []
                released = threading.Event()

                def fail():
                    payload = Payload()
                    payload.data = bytearray(256 * 1024)
                    references.append(weakref.ref(payload))
                    raise exception("injected")

                job = manager.submit("failure", fail, on_done=lambda _: released.set())
                self.finished(manager, job)
                self.assertTrue(released.is_set())
                self.assertEqual(manager.status(job)["status"], "failed")
                gc.collect()
                self.assertIsNone(references[0]())
                self.assertEqual(manager.purge_terminal_before(time.time() + 1), 1)

    def test_start_persistence_failure_releases_asset_even_without_body(self):
        with tempfile.TemporaryDirectory() as root:
            service = ImageService(workspace=root)
            try:
                asset = service.put_asset(
                    content_base64=base64.b64encode(_png_bytes()).decode(),
                    media_type="image/png",
                )["asset"]["asset_id"]
                manager = service._job_manager()
                persist = manager._persist

                def failing(state):
                    if state.status == "running":
                        raise OSError("injected storage failure")
                    persist(state)

                with (
                    patch.object(manager, "_persist", side_effect=failing),
                    patch("yolozu.integrations.image_service.run_image_job") as body,
                ):
                    job = service.submit_job(asset_id=asset, fixed_classes=["cat"])[
                        "job"
                    ]["job_id"]
                    self.finished(manager, job)
                    body.assert_not_called()
                self.assertEqual(manager.status(job)["status"], "failed")
                self.assertEqual(service._active_assets, {})
                self.assertEqual(service._job_assets, {})
            finally:
                service.close()

    def test_queue_deadline_expires_before_worker_is_available(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.manager(root)
            release = threading.Event()
            manager.submit("blocker", lambda: (release.wait(3), {})[1])
            done = threading.Event()
            try:
                with patch("builtins.print") as unused:
                    job = manager.submit(
                        "queued",
                        lambda: unused(),
                        on_done=lambda _: done.set(),
                        deadline=time.monotonic() + 0.05,
                    )
                    _wait(lambda: manager.status(job)["status"] == "timed_out")
                    self.assertTrue(done.is_set())
                    unused.assert_not_called()
            finally:
                release.set()
                manager.shutdown(timeout=5)

    def test_rejected_submit_leaves_no_record(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.manager(root)
            with patch.object(
                manager._executor, "submit", side_effect=RuntimeError("injected")
            ):
                with self.assertRaises(RuntimeError):
                    manager.submit("rejected", lambda: {})
            self.assertEqual(manager.list(), [])
            self.assertEqual(list((Path(root) / "jobs").glob("job_*.json")), [])
            manager.shutdown()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                manager.submit("closed", lambda: {})
            self.assertEqual(manager.list(), [])


@unittest.skipUnless(os.name == "posix", "POSIX ownership")
class TestResourceOwnership(unittest.TestCase):
    def test_interrupted_output_siblings_expire_only_after_job_release(self):
        with tempfile.TemporaryDirectory() as root:
            service = ImageService(workspace=root, retention_seconds=300)
            try:
                service._ensure_storage()
                token = "run_" + "a" * 24
                stage = service.outputs_root / (f".{token}.stage." + "b" * 32)
                marker = (
                    service.outputs_root / f".{token}.yolozu-output-transaction.json"
                )
                unowned = service.outputs_root / ".run_unowned.stage.other"
                stage.mkdir()
                marker.write_text("{}")
                unowned.mkdir()
                old = time.time() - 301
                for path in (stage, marker, unowned):
                    os.utime(path, (old, old))
                service._job_assets["job_fixture"] = ("asset_" + "a" * 24, token)
                self.assertEqual(service.purge_expired()["outputs"], 0)
                service._job_assets.clear()
                self.assertEqual(service.purge_expired()["outputs"], 2)
                self.assertTrue(unowned.exists())
            finally:
                service.close()

    def test_probe_setup_failure_reaps_process_and_guard(self):
        from yolozu.adaptive import environment

        spec = next(
            item
            for item in environment._LIMIT_TEST_SPECS
            if item.probe_id == "limit_test_timeout"
        )
        processes = []
        popen = subprocess.Popen

        def record(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            return process

        with (
            patch.object(subprocess, "Popen", side_effect=record),
            patch.object(
                environment.selectors,
                "DefaultSelector",
                side_effect=OSError("injected"),
            ),
        ):
            with self.assertRaises(OSError):
                environment._run_bounded_probe(spec, 1)
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.poll() is not None for process in processes))

    def test_service_running_cancel_and_shutdown_release_assets_without_deadlock(self):
        for mode in ("cancel", "close"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                service = ImageService(workspace=root)
                try:
                    service.start_maintenance()
                    asset = service.put_asset(
                        content_base64=base64.b64encode(_png_bytes()).decode(),
                        media_type="image/png",
                    )["asset"]["asset_id"]
                    with patch(
                        "yolozu.integrations.image_job_execution._worker",
                        _slow_service_worker,
                    ):
                        job = service.submit_job(asset_id=asset, fixed_classes=["cat"])[
                            "job"
                        ]["job_id"]
                        pid_file = Path(root) / "worker.pid"
                        _wait(pid_file.exists)
                        pid = int(pid_file.read_text())
                        if mode == "cancel":
                            result = service.cancel_job(job_id=job)
                            self.assertEqual(
                                result["job"]["reason"], "cancellation_requested"
                            )
                        else:
                            service.close()
                        _wait(lambda: service._jobs._jobs[job].future is None)
                    self.assertEqual(service._jobs.status(job)["status"], "cancelled")
                    self.assertFalse(_alive(pid))
                    self.assertEqual(service._active_assets, {})
                    self.assertEqual(service._job_assets, {})
                finally:
                    service.close()

    def test_repeated_service_lifecycle_leaves_no_extra_threads(self):
        baseline = {thread.ident for thread in threading.enumerate()}
        for _ in range(10):
            with tempfile.TemporaryDirectory() as root:
                service = ImageService(workspace=root)
                try:
                    service.start_maintenance()
                    asset = service.put_asset(
                        content_base64=base64.b64encode(_png_bytes()).decode(),
                        media_type="image/png",
                    )["asset"]["asset_id"]
                    with patch(
                        "yolozu.integrations.image_service.run_image_job",
                        return_value={"ok": True, "outcome": "abstained"},
                    ):
                        job = service.submit_job(asset_id=asset, fixed_classes=["cat"])[
                            "job"
                        ]["job_id"]
                        _wait(lambda: service._jobs._jobs[job].future is None)
                finally:
                    service.close()
        _wait(
            lambda: (
                not any(
                    thread.ident not in baseline for thread in threading.enumerate()
                )
            )
        )

    def test_bad_file_types_do_not_leak_descriptors(self):
        with tempfile.TemporaryDirectory() as root:
            before = len(os.listdir("/dev/fd"))
            for _ in range(30):
                for read in (client.read_image, _safe_json_read):
                    with self.assertRaises(IsADirectoryError):
                        read(Path(root))
            gc.collect()
            self.assertEqual(len(os.listdir("/dev/fd")), before)

    def test_tenant_lock_rejects_other_instance_and_process_then_releases(self):
        with tempfile.TemporaryDirectory() as root:
            first, second = ImageService(workspace=root), ImageService(workspace=root)
            try:
                first._ensure_storage()
                with self.assertRaisesRegex(ImageServiceError, "tenant_in_use"):
                    second.purge_expired()
                code = "from yolozu.integrations.image_service import ImageService, ImageServiceError; import sys\ns=ImageService(workspace=sys.argv[1])\ntry: s._ensure_storage()\nexcept ImageServiceError as e: print(e.code)\nelse: s.close(); raise SystemExit(1)"
                result = subprocess.run(
                    [sys.executable, "-I", "-c", code, root],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "tenant_in_use")
                first.close()
                second._ensure_storage()
                second.close()
                with self.assertRaisesRegex(ImageServiceError, "service_closed"):
                    second.start_maintenance()
            finally:
                first.close()
                second.close()

    def test_spawned_running_job_cancel_and_deadline_reap_process(self):
        for mode in ("cancel", "timeout"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                pid_file = Path(root) / "pid"
                cancel = threading.Event()
                output = []

                def run():
                    output.append(
                        run_image_job(
                            {"pid_file": str(pid_file)},
                            cancel,
                            time.monotonic() + (1.5 if mode == "timeout" else 10),
                        )
                    )

                with patch(
                    "yolozu.integrations.image_job_execution._worker", _slow_worker
                ):
                    thread = threading.Thread(target=run)
                    thread.start()
                    try:
                        _wait(pid_file.exists)
                        pid = int(pid_file.read_text())
                        if mode == "cancel":
                            cancel.set()
                        thread.join(5)
                        self.assertFalse(thread.is_alive())
                        self.assertEqual(
                            output[0]["outcome"],
                            "cancelled" if mode == "cancel" else "timed_out",
                        )
                        self.assertFalse(_alive(pid))
                    finally:
                        cancel.set()
                        thread.join(5)

    def test_close_timeout_closes_runner_pipe_and_process_handles(self):
        before = len(os.listdir("/dev/fd"))
        for _ in range(3):
            session = _ForkedRunnerSession(
                factory=_SlowCloseRunner,
                bundle=None,
                environment=None,
                artifacts=None,
                inputs=[],
                labels=(),
                outer_deadline_ns=time.monotonic_ns() + 5_000_000_000,
            )
            pid = session._process.pid
            with self.assertRaises(ValueError):
                session.close(0.01)
            self.assertTrue(session._connection.closed)
            self.assertFalse(_alive(pid))
            session.close(0.01)
        gc.collect()
        self.assertEqual(len(os.listdir("/dev/fd")), before)

    def test_watchdog_timeout_reaps_guard_without_requiring_later_close(self):
        session = _ForkedRunnerSession(
            factory=_SlowCloseRunner,
            bundle=None,
            environment=None,
            artifacts=None,
            inputs=[],
            labels=(),
            outer_deadline_ns=time.monotonic_ns() + 5_000_000_000,
        )
        try:
            session._deadline = time.monotonic_ns() - 1
            with self.assertRaises(ValueError):
                session._remaining_seconds(1)
            self.assertFalse(session._process.is_alive())
            self.assertTrue(session._connection.closed)
            self.assertIsNotNone(session._guard._process.poll())
        finally:
            session.close(0.1)

    def test_owner_death_reaps_detached_runner_and_guard(self):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        owner = context.Process(target=_runner_owner, args=(child,))
        pids = []
        try:
            owner.start()
            child.close()
            self.assertTrue(parent.poll(5))
            pids = parent.recv()
            os.killpg(owner.pid, signal.SIGKILL)
            owner.join(3)
            for pid in pids:
                _wait(lambda: not _alive(pid))
        finally:
            for pid in pids:
                if _alive(pid):
                    os.kill(pid, signal.SIGKILL)
            if owner.is_alive():
                owner.kill()
            owner.join(3)
            owner.close()
            parent.close()
            child.close()

    def test_owner_death_reaps_whole_image_job(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as root:
            pid_file = Path(root) / "pid"
            parent, child = context.Pipe(duplex=False)
            owner = context.Process(target=_job_owner, args=(child, str(pid_file)))
            pid = None
            try:
                owner.start()
                child.close()
                self.assertTrue(parent.poll(5))
                self.assertEqual(parent.recv(), owner.pid)
                _wait(pid_file.exists)
                pid = int(pid_file.read_text())
                os.killpg(owner.pid, signal.SIGKILL)
                owner.join(3)
                _wait(lambda: not _alive(pid))
            finally:
                if pid is not None and _alive(pid):
                    os.kill(pid, signal.SIGKILL)
                if owner.is_alive():
                    owner.kill()
                owner.join(3)
                owner.close()
                parent.close()
                child.close()

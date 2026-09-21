"""POSIX process-group ownership that survives abrupt owner termination."""

from __future__ import annotations

import os
import signal
import subprocess
import sys


# A separate interpreter remains runnable even if native code holds the owner's
# GIL. Its only input is a lifetime pipe, never a user-supplied command.
_GUARD_PROGRAM = """
import os, select, signal, sys, time
pid = int(sys.argv[1])
deadline = float(sys.argv[2])
ready, _, _ = select.select([0], [], [], max(0, deadline - time.monotonic()))
if ready and os.read(0, 1) == b'x':
    sys.exit(0)
for sig in (signal.SIGTERM, signal.SIGKILL):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            # The owned group and its leader have already exited.
            pass
    if sig == signal.SIGTERM:
        time.sleep(0.2)
"""
_WRITERS: set[int] = set()


def _after_fork() -> None:
    # A later fork must not keep another process's ownership pipe alive.
    for descriptor in _WRITERS:
        os.close(descriptor)
    _WRITERS.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


class ProcessGuard:
    """Kill an owned process group on pipe EOF or an absolute deadline."""

    def __init__(self, pid: int, deadline: float) -> None:
        if os.name != "posix":
            raise RuntimeError("process ownership requires POSIX")
        self._owner = os.getpid()
        read_fd, self._writer = os.pipe()
        _WRITERS.add(self._writer)
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-I", "-c", _GUARD_PROGRAM, str(pid), str(deadline)],
                stdin=read_fd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
                env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
            )
        except BaseException:
            _WRITERS.discard(self._writer)
            os.close(self._writer)
            self._writer = -1
            raise
        finally:
            os.close(read_fd)

    def close(self, *, disarm: bool = True) -> None:
        """Disarm only after the owned process/group has been reaped."""
        if self._owner != os.getpid() or self._writer < 0:
            return
        try:
            try:
                if disarm:
                    os.write(self._writer, b"x")
            except BrokenPipeError:
                # The guard has already exited; it still needs to be reaped.
                pass
        finally:
            _WRITERS.discard(self._writer)
            os.close(self._writer)
            self._writer = -1
        try:
            self._process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=1)


def kill_group(pid: int) -> None:
    """Stop a code-owned group, including descendants of a dead group leader."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            # Neither the owned process group nor its leader remains.
            pass


def stop_and_reap(process, *, timeout: float = 2) -> None:
    """Reap a multiprocessing child when its guard and owner race to stop it."""
    try:
        kill_group(process.pid)
    except PermissionError:
        # Our macOS deadline test can race the guard's signal here. Reap the
        # child before retrying; a genuine denial still propagates. Never treat
        # EPERM itself as successful termination.
        process.join(timeout=0.3)
        kill_group(process.pid)
    process.join(timeout=timeout)
    if process.is_alive():
        raise RuntimeError("owned process could not be reaped")

"""Import-stable subprocess workers for Algorithm Scout parser tests."""

from __future__ import annotations

import os
import resource
import signal
import time
from pathlib import Path

from yolozu.adaptive.algorithm_scout import (
    _apply_pdf_resource_limits,
    _disable_child_process_creation,
)


def sleep_pdf_worker(connection, body, temp_dir, limits) -> None:
    del connection, body, temp_dir, limits
    os.setsid()
    time.sleep(5)


def large_ipc_pdf_worker(connection, body, temp_dir, limits) -> None:
    del body, temp_dir, limits
    os.setsid()
    connection.send_bytes(b"xx")
    connection.close()


def rss_pdf_worker(connection, body, temp_dir, limits) -> None:
    del connection, body, temp_dir, limits
    os.setsid()
    allocation = bytearray(2 * 1024 * 1024)
    allocation[0] = 1
    time.sleep(5)


def cpu_pdf_worker(connection, body, temp_dir, limits) -> None:
    del connection, body, temp_dir
    os.setsid()
    _apply_pdf_resource_limits(limits)
    while True:
        pass


def pid_pdf_worker(connection, body, temp_dir, limits) -> None:
    del body, temp_dir, limits
    os.setsid()
    _disable_child_process_creation()
    try:
        os.fork()
    except PermissionError:
        connection.send_bytes(b'{"code":"pdf_pid_limit","ok":false}')
    connection.close()


def temp_pdf_worker(connection, body, temp_dir, limits) -> None:
    del body
    os.setsid()
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits.pdf_temp_bytes, limits.pdf_temp_bytes))
    try:
        with (Path(temp_dir) / "overflow").open("wb") as output:
            output.write(b"xx")
            output.flush()
    except OSError:
        connection.send_bytes(b'{"code":"pdf_temp_limit","ok":false}')
    connection.close()

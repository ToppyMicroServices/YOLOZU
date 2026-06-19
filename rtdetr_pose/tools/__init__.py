"""Command-line wrappers for the RT-DETR pose reference trainer.

Some workflows put ``rtdetr_pose/`` early on ``sys.path``. In that layout this
directory can be discovered as a top-level ``tools`` package before the repo
root ``tools/`` directory. When that happens, delegate submodule imports back
to the repo root tool directory so imports such as ``from tools import release``
keep resolving to the repository tools.
"""

from __future__ import annotations

from pathlib import Path

if __name__ == "tools":
    __path__ = [str(Path(__file__).resolve().parents[2] / "tools")]  # type: ignore[name-defined]

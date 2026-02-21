from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    # When running tests under `rtdetr_pose/`, pytest may add `.../rtdetr_pose`
    # to sys.path without adding the repo root, which breaks sibling imports
    # like `import yolozu`.
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

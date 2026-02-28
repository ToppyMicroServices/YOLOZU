"""3-D geometry, camera intrinsics, rotations, and constraints.

Backward-compatible re-exports: ``from yolozu.geometry import X`` continues
to work for all symbols previously in the flat ``yolozu.geometry`` module.
"""

# Re-export the original geometry module's public API at package level.
from yolozu.geometry.geometry import *  # noqa: F401,F403

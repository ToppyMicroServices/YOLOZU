"""RT-DETR pose reference-trainer package shim.

This repository stores the implementation under `rtdetr_pose/rtdetr_pose/`.
When installed as a wheel, that nested layout would normally expose modules as
`rtdetr_pose.rtdetr_pose.*`, which breaks imports like `rtdetr_pose.dataset`.

To keep the public import surface stable (`import rtdetr_pose.*`), this shim
extends the package search path to include the nested implementation directory.
"""

from __future__ import annotations

import os
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_base = os.path.dirname(__file__)
_impl = os.path.join(_base, "rtdetr_pose")
if _impl not in __path__:  # type: ignore[name-defined]
    __path__.append(_impl)  # type: ignore[name-defined]

#!/usr/bin/env python3
"""Generate backward-compatibility shim files at old yolozu/<module>.py locations."""
import pathlib

SHIMS = {}

TEMPLATE = '''\
"""Backward-compatibility shim \u2014 canonical location: ``yolozu.{new_path}``."""

# Re-export everything so ``from yolozu.{old_name} import X`` keeps working.
from yolozu.{new_path} import *  # noqa: F401,F403
'''

root = pathlib.Path("yolozu")
created = 0
for old_name, new_path in SHIMS.items():
    path = root / f"{old_name}.py"
    path.write_text(TEMPLATE.format(old_name=old_name, new_path=new_path))
    created += 1

print(f"Created {created} shim files")

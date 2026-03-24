#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_legacy_main(filename: str):
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / 'tools' / filename
    spec = importlib.util.spec_from_file_location(f'_legacy_{target.stem}', target)
    if spec is None or spec.loader is None:
        raise SystemExit(f'failed to load legacy tool: {target}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, 'main')


def main(argv=None):
    return _load_legacy_main('support_ultralytics_detr.py')(argv)


if __name__ == '__main__':
    raise SystemExit(main())

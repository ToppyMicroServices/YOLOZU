"""Record source-checkout unittest results; this is not installed-wheel evidence."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "unittest.log").exists() or (output / "tests.json").exists():
        parser.error(f"output directory already contains test evidence: {output}")
    metadata_command = [
        sys.executable,
        "-I",
        "-c",
        "import importlib.metadata as m, json; "
        "print(json.dumps({name: m.version(name) for name in "
        "['yolozu', 'Pillow', 'numpy', 'PyYAML', 'pycocotools']}))",
    ]
    environment_versions = json.loads(
        subprocess.check_output(metadata_command, text=True)
    )
    import yolozu.api

    source_module = Path(yolozu.api.__file__).resolve().relative_to(root)
    start = time.perf_counter()
    with (output / "unittest.log").open("w", encoding="utf-8") as log:
        original_out, original_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = log
        try:
            suite = unittest.defaultTestLoader.discover(str(root / "tests"))
            result = unittest.TextTestRunner(stream=log, verbosity=1).run(suite)
        finally:
            sys.stdout, sys.stderr = original_out, original_err
    payload = {
        "command": [sys.executable, *sys.argv],
        "scope": "Source-checkout regression tests. The existing .venv has stale distribution metadata; this run does not verify an installed candidate wheel or published package.",
        "python_version": sys.version.split()[0],
        "source_module": str(source_module),
        "environment_yolozu_distribution_version": environment_versions["yolozu"],
        "environment_pillow_distribution_version": environment_versions["Pillow"],
        "environment_distribution_versions": environment_versions,
        "environment_metadata_command": metadata_command,
        "suite": "tests",
        "tests_run": result.testsRun,
        "failures": [str(case) for case, _ in result.failures],
        "errors": [str(case) for case, _ in result.errors],
        "skipped": [{"test": str(case), "reason": reason} for case, reason in result.skipped],
        "seconds": round(time.perf_counter() - start, 3),
        "ok": result.wasSuccessful(),
    }
    (output / "tests.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

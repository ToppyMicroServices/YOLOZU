#!/usr/bin/env python3
"""Verify an installed candidate wheel using a portable labeled sample."""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


_PROBE = """
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import yolozu
distribution = metadata.distribution('yolozu')
direct = json.loads(distribution.read_text('direct_url.json') or '{}')
versions = {}
for name in ('yolozu', 'numpy', 'Pillow', 'PyYAML', 'pycocotools', 'typing_extensions'):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({
    'version': distribution.version, 'module_version': yolozu.__version__,
    'module_file': str(Path(yolozu.__file__).resolve()),
    'distribution_module_file': str(Path(distribution.locate_file('yolozu/__init__.py')).resolve()),
    'editable': bool(direct.get('dir_info', {}).get('editable', False)),
    'python': platform.python_version(), 'system': platform.system(),
    'machine': platform.machine(), 'dependencies': versions,
}))
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): _sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def _verify_wheel_identity(wheel: Path, probe: dict) -> dict:
    if probe["editable"]:
        raise ValueError("candidate must be installed non-editably")
    module_file = Path(probe["module_file"]).resolve()
    if module_file != Path(probe["distribution_module_file"]).resolve():
        raise ValueError("imported module does not match installed distribution metadata")
    package_parent = module_file.parent.parent
    checked: dict[str, str] = {}
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("expected one wheel METADATA file")
        metadata = email.message_from_bytes(archive.read(metadata_names[0]))
        if metadata["Name"].lower() != "yolozu":
            raise ValueError("wheel must contain YOLOZU")
        if not (metadata["Version"] == probe["version"] == probe["module_version"]):
            raise ValueError("wheel, installed metadata, and module versions differ")
        for name in sorted(archive.namelist()):
            if not name.startswith(("yolozu/", "rtdetr_pose/")) or name.endswith("/"):
                continue
            if ".." in Path(name).parts:
                raise ValueError("wheel contains an invalid package path")
            expected = hashlib.sha256(archive.read(name)).hexdigest()
            installed = package_parent / name
            if not installed.is_file() or _sha256(installed) != expected:
                raise ValueError(f"installed package differs from candidate wheel: {name}")
            checked[name] = expected
    if "yolozu/__init__.py" not in checked:
        raise ValueError("wheel has no YOLOZU package")
    return {"filename": wheel.name, "sha256": _sha256(wheel), "version": probe["version"],
            "installed_package_files_verified": len(checked),
            "package_content_sha256": hashlib.sha256(
                json.dumps(checked, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()}


def _check_sample_layout(sample: Path) -> dict:
    for name in ("data.yaml", "predictions.json", "sample_manifest.json"):
        if not (sample / name).is_file():
            raise ValueError(f"sample is missing {name}")
    if not list(sample.glob("*.png")):
        raise ValueError("sample is missing its preview PNG")
    counts = {}
    class_ids = set()
    for split in ("train", "val"):
        images = [path for path in (sample / "images" / split).iterdir() if path.is_file()]
        labels = sorted((sample / "labels" / split).glob("*.txt"))
        if len(images) != 4 or len(labels) != 4:
            raise ValueError(f"sample split {split} must contain four images and four labels")
        if {path.stem for path in images} != {path.stem for path in labels}:
            raise ValueError(f"sample split {split} has mismatched image/label stems")
        counts[split] = {"images": len(images), "label_files": len(labels)}
        for label in labels:
            for line in label.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    class_ids.add(int(line.split()[0]))
    if class_ids != {0, 1}:
        raise ValueError("sample must exercise both classes 0 and 1")
    return {"splits": counts, "class_ids": sorted(class_ids)}


def _check_metrics(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("dry_run") is not False:
        raise ValueError("compatibility requires actual COCO evaluation, not dry-run")
    metrics = report.get("metrics") or {}
    for name in ("map50", "map50_95", "map75"):
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"COCO metric {name} is not a finite number")
        if not math.isclose(value, 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"perfect sample COCO metric {name} differs from 1.0: {value}")
    return metrics


def run_check(wheel: Path, output: Path, *, source_revision: str | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    report = {"kind": "yolozu_sample_compatibility", "schema_version": 1,
              "source_revision": source_revision, "steps": [], "ok": False}

    def run(name: str, args: list[str], cwd: Path) -> str:
        start = time.perf_counter()
        result = subprocess.run([sys.executable, "-I", *args], cwd=cwd, capture_output=True,
                                text=True, timeout=120, check=False)
        (logs / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        report["steps"].append({"name": name, "exit_code": result.returncode,
                                "seconds": round(time.perf_counter() - start, 6),
                                "log": f"logs/{name}.log"})
        if result.returncode:
            raise RuntimeError(f"{name} failed with exit code {result.returncode}; see logs/{name}.log")
        return result.stdout

    try:
        with tempfile.TemporaryDirectory(prefix="yolozu-sample-compat-cwd-") as temporary:
            cwd = Path(temporary)
            probe = json.loads(run("installed-identity", ["-c", _PROBE], cwd))
            report["wheel"] = _verify_wheel_identity(wheel, probe)
            report["runtime"] = {key: probe[key] for key in ("python", "system", "machine", "dependencies")}
            report["noneditable_candidate_verified"] = True
            run("pip-check", ["-m", "pip", "check"], cwd)
            sample = output / "sample"
            run("generate-sample", ["-m", "yolozu", "demo", "dataset", "--run-dir", str(sample), "--seed", "0"], cwd)
            report["sample"] = _check_sample_layout(sample)
            before = _snapshot(sample)

            def evaluate(name: str, directory: Path) -> dict:
                for split in ("train", "val"):
                    run(f"{name}-validate-{split}", ["-m", "yolozu", "validate", "dataset", str(directory),
                        "--split", split, "--strict"], cwd)
                run(f"{name}-validate-yaml", ["-m", "yolozu", "validate", "dataset", str(directory / "data.yaml"),
                    "--split", "val", "--strict"], cwd)
                run(f"{name}-validate-predictions", ["-m", "yolozu", "validate", "predictions",
                    str(directory / "predictions.json"), "--strict"], cwd)
                evaluation = output / f"{name}-coco.json"
                run(f"{name}-coco", ["-m", "yolozu", "eval-coco", "--dataset", str(directory), "--split", "val",
                    "--predictions", str(directory / "predictions.json"), "--output", str(evaluation)], cwd)
                return _check_metrics(evaluation)

            original_metrics = evaluate("original", sample)
            relocated = output / "relocated sample with spaces"
            shutil.copytree(sample, relocated)
            moved_metrics = evaluate("relocated", relocated)
            if original_metrics != moved_metrics:
                raise ValueError("COCO metrics changed after relocating the sample")
            if _snapshot(sample) != before or _snapshot(relocated) != before:
                raise ValueError("sample files changed during validation, copying, or evaluation")
            report.update({"metrics": original_metrics, "relocated_metrics": moved_metrics,
                           "sample_files_sha256": before, "source_files_unchanged": True,
                           "relocation_metrics_identical": True, "ok": True})
    except Exception as exc:
        report["error"] = str(exc)
    (output / "compatibility-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path, help="Exact candidate wheel already installed in this interpreter.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for report, logs, and sample copies.")
    parser.add_argument("--source-revision", help="Source Git revision used to build the candidate wheel.")
    args = parser.parse_args(argv)
    wheel, output = args.wheel.resolve(), args.output_dir.resolve()
    if not wheel.is_file():
        parser.error(f"wheel not found: {wheel}")
    if output.exists():
        parser.error(f"output directory already exists: {output}")
    report = run_check(wheel, output, source_revision=args.source_revision)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

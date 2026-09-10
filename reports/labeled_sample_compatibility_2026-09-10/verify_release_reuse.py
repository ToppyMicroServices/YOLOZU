"""Check one generated labeled sample with installed public YOLOZU releases."""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for samples, commands, logs, and report.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    audit = Path(__file__).parent
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root))
    from yolozu.demos.labeled_dataset import generate_labeled_sample
    spec = importlib.util.spec_from_file_location("sample_compatibility_audit", root / "tools/ci/check_sample_compatibility.py")
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)

    sample = output / "shared labeled sample"
    generate_labeled_sample(run_dir=sample, seed=0)
    before = checks._snapshot(sample)
    predictions = json.loads((sample / "predictions.json").read_text())
    if not all(not Path(entry["image"]).is_absolute() for entry in predictions["predictions"]):
        raise RuntimeError("sample predictions contain absolute paths")
    relocated = output / "relocated labeled sample with spaces"
    shutil.copytree(sample, relocated)
    environments = {
        "4.6.0": audit / "legacy-4.6-venv/bin/python",
        "4.7.0": root / "reports/product_audit_2026-09-09/public-install-network/venv/bin/python",
    }
    report = {"kind": "public_release_labeled_sample_reuse", "schema_version": 1,
              "scope": "Same synthetic labeled sample, public 4.6.0 and 4.7.0, macOS arm64 Python3.14; not universal compatibility.",
              "generator_source_sha256": checks._sha256(root / "yolozu/demos/labeled_dataset.py"),
              "sample_manifest_sha256": before["sample_manifest.json"], "source_files_sha256": before,
              "relative_prediction_paths": True, "releases": [], "ok": False}
    with tempfile.TemporaryDirectory(prefix="yolozu-public-release-reuse-") as temporary:
        cwd = Path(temporary)
        if root in cwd.parents:
            raise RuntimeError("execution directory must be outside the repository")
        for version, interpreter in environments.items():
            release_output = output / version
            logs = release_output / "logs"
            logs.mkdir(parents=True)
            result = {"expected_version": version, "environment": interpreter.parent.parent.relative_to(root).as_posix(),
                      "cwd": "temporary directory outside repository", "steps": [], "ok": False}

            def run(name, arguments):
                command = [str(interpreter), "-I", *arguments]
                started = time.perf_counter()
                proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=120)
                (logs / f"{name}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
                result["steps"].append({"name": name, "argv": [arg.replace(str(root), "<repository>") for arg in command],
                    "exit_code": proc.returncode, "seconds": round(time.perf_counter() - started, 6),
                    "log": f"{version}/logs/{name}.log"})
                if proc.returncode:
                    raise RuntimeError(f"{name}: exit {proc.returncode}")
                return proc.stdout

            try:
                probe = json.loads(run("installed-identity", ["-c", checks._PROBE]))
                if probe["version"] != version:
                    raise RuntimeError("wrong installed version")
                wheel = audit / "public-wheels" / f"yolozu-{version}-py3-none-any.whl"
                result["wheel"] = checks._verify_wheel_identity(wheel, probe)
                result["runtime"] = {key: probe[key] for key in ("python", "system", "machine", "dependencies")}
                result["module_file_in_environment"] = Path(probe["module_file"]).relative_to(interpreter.parent.parent).as_posix()
                run("pip-check", ["-m", "pip", "check"])
                for name, command in (("dataset-help", ["validate", "dataset"]),
                                      ("predictions-help", ["validate", "predictions"]),
                                      ("eval-help", ["eval-coco"])):
                    run(name, ["-m", "yolozu", *command, "--help"])
                metrics = {}
                for name, dataset in (("original", sample), ("relocated", relocated)):
                    for split in ("train", "val"):
                        run(f"{name}-dataset-{split}", ["-m", "yolozu", "validate", "dataset", str(dataset), "--split", split, "--strict"])
                    run(f"{name}-yaml", ["-m", "yolozu", "validate", "dataset", str(dataset / "data.yaml"), "--split", "val", "--strict"])
                    run(f"{name}-predictions", ["-m", "yolozu", "validate", "predictions", str(dataset / "predictions.json"), "--strict"])
                    evaluation = release_output / f"{name}-coco.json"
                    run(f"{name}-coco", ["-m", "yolozu", "eval-coco", "--dataset", str(dataset), "--split", "val",
                        "--predictions", str(dataset / "predictions.json"), "--output", str(evaluation)])
                    metrics[name] = checks._check_metrics(evaluation)
                if metrics["original"] != metrics["relocated"]:
                    raise RuntimeError("relocation changed COCO metrics")
                result.update({"metrics": metrics, "ok": True})
            except Exception as exc:
                result["error"] = str(exc)
            report["releases"].append(result)
    report["source_files_unchanged"] = checks._snapshot(sample) == before and checks._snapshot(relocated) == before
    report["cross_release_metrics_identical"] = all(item["ok"] for item in report["releases"]) and (
        report["releases"][0]["metrics"] == report["releases"][1]["metrics"])
    report["ok"] = report["source_files_unchanged"] and report["cross_release_metrics_identical"]
    (output / "release-reuse-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "report": str(output / "release-reuse-report.json"),
                      "releases": [{key: value for key, value in item.items() if key in ("expected_version", "ok", "error", "metrics", "runtime", "wheel")}
                                   for item in report["releases"]]}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

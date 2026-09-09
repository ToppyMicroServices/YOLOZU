"""Execute first-user routes from the installed candidate without checkout assets."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New audit output directory.")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import yolozu
    from yolozu.cli_entry import GUIDE_ROUTES
    location = Path(yolozu.__file__).resolve()
    if Path(sys.prefix).resolve() not in location.parents:
        raise RuntimeError(f"Expected an installed package in this environment: {location}")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    report = {"installed_version": importlib.metadata.version("yolozu"),
              "kind": "installed_candidate_journey", "python": sys.version,
              "verification_python_optimize": sys.flags.optimize,
              "installed_outside_checkout": True, "steps": [], "ok": False}

    def run(name: str, command: list[str], cwd: Path):
        start = time.perf_counter()
        result = subprocess.run([sys.executable, "-m", "yolozu", *command],
                                cwd=cwd, env=environment, capture_output=True,
                                text=True, timeout=90)
        (output / f"{name}.log").write_text(result.stdout + result.stderr)
        report["steps"].append({"name": name, "argv": command, "exit_code": result.returncode,
                                "seconds": round(time.perf_counter() - start, 6)})
        if result.returncode != 0:
            raise RuntimeError(f"{name} exited {result.returncode}: {result.stderr}")

    try:
        for goal in ("evaluate", "export", "debug"):
            work = output / goal
            work.mkdir()
            for index, line in enumerate(GUIDE_ROUTES[goal]["commands"]):
                command = shlex.split(line)
                if not command or command[0] != "yolozu":
                    raise ValueError(f"Unexpected guide command: {line}")
                run(f"{goal}-{index}", command[1:], work)
            for item in GUIDE_ROUTES[goal]["outputs"]:
                if item.startswith("reports/") and " " not in item:
                    if not (work / item).is_file():
                        raise RuntimeError(f"Missing promised output: {item}")
        from PIL import Image
        for filename in (
            output / "export/reports/doctor_proof/toy_dataset/images/val2017/proof_0001.png",
            output / "export/reports/predict_overlays/000000_proof_0001.png",
        ):
            with Image.open(filename) as image:
                image.verify()
            with Image.open(filename) as image:
                image.load()
        run("real-coco", ["eval-coco", "--dataset", "reports/doctor_proof/toy_dataset",
                         "--split", "val2017", "--predictions", "reports/doctor_proof/known_predictions.json",
                         "--output", "reports/official-coco.json"], output / "evaluate")
        evaluation = json.loads((output / "evaluate/reports/official-coco.json").read_text())
        if evaluation["dry_run"] or not (evaluation["metrics"]["map50_95"] > 0.999):
            raise RuntimeError("Toy official COCO evaluation did not match the expected result")
        report["official_coco_metrics"] = evaluation["metrics"]
        run("synthetic-demo", ["demo", "instance-seg", "--background", "synthetic",
                              "--inference", "none", "--run-dir", "demo"], output)
        report["ok"] = True
    except Exception as exc:
        report["error"] = str(exc)
    # Bind evidence to the installed candidate's actual source bytes.
    package = location.parent
    report["source_sha256"] = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in ("cli_entry.py", "core/doctor.py", "datasets/dataset_validator.py", "eval/simple_map.py")
    }
    (output / "installed-journey.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

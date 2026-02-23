#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run_ok(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"expected success: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def _run_fail(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode == 0:
        raise SystemExit(f"expected failure: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run schema compatibility gate checks against validator CLIs.")
    parser.add_argument("--output-dir", default="reports/ci_schema_gate", help="Directory to write temporary fixtures.")
    args = parser.parse_args()

    repo = Path.cwd()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pred_v1 = out / "pred_v1.json"
    pred_v1.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "predictions": [
                    {
                        "image": "000001.jpg",
                        "detections": [
                            {
                                "class_id": 0,
                                "score": 0.9,
                                "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    pred_v2 = out / "pred_v2.json"
    pred_v2.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "predictions": [{"image": "000001.jpg", "detections": []}],
            }
        ),
        encoding="utf-8",
    )

    seg_v1 = out / "seg_v1.json"
    seg_v1.write_text(json.dumps({"schema_version": 1, "predictions": [{"id": "0001", "mask": "m.png"}]}), encoding="utf-8")

    seg_v2 = out / "seg_v2.json"
    seg_v2.write_text(json.dumps({"schema_version": 2, "predictions": [{"id": "0001", "mask": "m.png"}]}), encoding="utf-8")

    inst_v1 = out / "inst_v1.json"
    inst_v1.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "predictions": [{"image": "0001.jpg", "instances": [{"class_id": 0, "score": 0.9, "mask": "m.png"}]}],
            }
        ),
        encoding="utf-8",
    )

    inst_v2 = out / "inst_v2.json"
    inst_v2.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "predictions": [{"image": "0001.jpg", "instances": [{"class_id": 0, "score": 0.9, "mask": "m.png"}]}],
            }
        ),
        encoding="utf-8",
    )

    _run_ok([sys.executable, "-m", "yolozu.cli", "validate", "predictions", str(pred_v1), "--strict"], repo)
    _run_ok([sys.executable, "-m", "yolozu.cli", "validate", "seg", str(seg_v1)], repo)
    _run_ok([sys.executable, "-m", "yolozu.cli", "validate", "instance-seg", str(inst_v1)], repo)

    _run_fail([sys.executable, "-m", "yolozu.cli", "validate", "predictions", str(pred_v2), "--strict"], repo)
    _run_fail([sys.executable, "-m", "yolozu.cli", "validate", "seg", str(seg_v2)], repo)
    _run_fail([sys.executable, "-m", "yolozu.cli", "validate", "instance-seg", str(inst_v2)], repo)

    print("schema compatibility gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

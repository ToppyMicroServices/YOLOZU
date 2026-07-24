#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "rtdetr_pose"))

from yolozu.run_record import build_run_record


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return repo_root / p


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="export_trt",
        description="Canonical PyTorch → ONNX → TensorRT export route (rtdetr_pose model).",
    )
    p.add_argument("--config", default="rtdetr_pose/configs/base.json", help="rtdetr_pose config path.")
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional checkpoint path; full compatibility is required by default "
            "and --skip-onnx is not allowed."
        ),
    )
    p.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help=(
            "Explicitly allow transfer/diagnostic loading of only name-and-shape "
            "matches and record status=partial in ONNX metadata."
        ),
    )
    p.add_argument("--device", default="cpu", help="Torch device for ONNX export (default: cpu).")
    p.add_argument("--image-size", type=int, default=320, help="Dummy square input size for ONNX export.")
    p.add_argument("--onnx", default="models/model.onnx", help="Where to write the ONNX model.")
    p.add_argument(
        "--onnx-meta",
        default=None,
        help="Where to write ONNX export metadata JSON (default: <onnx>.meta.json).",
    )
    p.add_argument("--opset", type=int, default=18, help="ONNX opset version (default: 18).")
    p.add_argument("--input-name", default="images", help="ONNX input tensor name (default: images).")
    p.add_argument("--dynamic-hw", action="store_true", help="Export ONNX with dynamic height/width axes.")

    p.add_argument("--engine", default="engines/model_fp16.plan", help="Where to write the TensorRT engine.")
    p.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp16", help="Engine precision.")
    p.add_argument("--min-shape", default=None, help="Min shape (e.g. 1x3x320x320). Default: derived from --image-size.")
    p.add_argument("--opt-shape", default=None, help="Opt shape. Default: derived from --image-size.")
    p.add_argument("--max-shape", default=None, help="Max shape. Default: derived from --image-size.")
    p.add_argument("--workspace", type=int, default=4096, help="Workspace size in MiB (default: 4096).")
    p.add_argument("--timing-cache", default="engines/timing.cache", help="Timing cache path.")
    p.add_argument("--trtexec", default="trtexec", help="Path to trtexec binary (default: trtexec).")
    p.add_argument("--extra-args", default=None, help="Extra trtexec args (quoted string).")
    p.add_argument(
        "--engine-meta",
        default=None,
        help="Where to write engine build metadata JSON (default: <engine>.meta.json).",
    )

    p.add_argument("--skip-onnx", action="store_true", help="Skip PyTorch→ONNX export and use existing --onnx.")
    p.add_argument("--skip-engine", action="store_true", help="Skip ONNX→TRT engine build.")
    p.add_argument("--dry-run", action="store_true", help="Print commands and write meta without executing heavy steps.")
    return p.parse_args(argv)


def _default_shape(image_size: int) -> str:
    s = max(1, int(image_size))
    return f"1x3x{s}x{s}"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if bool(args.skip_onnx) and args.checkpoint:
        raise SystemExit(
            "--checkpoint cannot be used with --skip-onnx because the existing ONNX artifact is not rebuilt"
        )
    if bool(args.allow_partial_checkpoint):
        if not args.checkpoint:
            raise SystemExit("--allow-partial-checkpoint requires --checkpoint")

    onnx_path = _resolve(str(args.onnx))
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_meta_path = (
        _resolve(str(args.onnx_meta))
        if args.onnx_meta
        else onnx_path.with_suffix(onnx_path.suffix + ".meta.json")
    )
    onnx_meta_path.parent.mkdir(parents=True, exist_ok=True)

    engine_path = _resolve(str(args.engine))
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_meta_path = (
        _resolve(str(args.engine_meta))
        if args.engine_meta
        else engine_path.with_suffix(engine_path.suffix + ".meta.json")
    )
    engine_meta_path.parent.mkdir(parents=True, exist_ok=True)

    if not bool(args.dry_run):
        if not bool(args.skip_onnx):
            onnx_path.unlink(missing_ok=True)
            onnx_meta_path.unlink(missing_ok=True)
        if not bool(args.skip_engine):
            engine_path.unlink(missing_ok=True)
            engine_meta_path.unlink(missing_ok=True)

    min_shape = str(args.min_shape) if args.min_shape else _default_shape(int(args.image_size))
    opt_shape = str(args.opt_shape) if args.opt_shape else _default_shape(int(args.image_size))
    max_shape = str(args.max_shape) if args.max_shape else _default_shape(int(args.image_size))

    run_record = build_run_record(repo_root=repo_root, argv=sys.argv, args=vars(args))

    onnx_export_report: dict[str, Any] = {
        "enabled": not bool(args.skip_onnx),
        "skipped": bool(args.skip_onnx),
        "dry_run": bool(args.dry_run),
        "opset": int(args.opset),
        "dynamic_hw": bool(args.dynamic_hw),
        "input_name": str(args.input_name),
        "dummy_input": {"shape": [1, 3, int(args.image_size), int(args.image_size)], "dtype": "float32"},
        "checkpoint": None if not args.checkpoint else str(_resolve(str(args.checkpoint))),
        "checkpoint_report": None,
    }

    if not args.skip_onnx and not args.dry_run:
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise SystemExit("torch is required for PyTorch→ONNX export") from exc

        from rtdetr_pose.config import load_config
        from rtdetr_pose.export import export_onnx
        from rtdetr_pose.factory import build_model

        cfg = load_config(_resolve(str(args.config)))
        model = build_model(cfg.model).eval()

        if args.checkpoint:
            ckpt_path = _resolve(str(args.checkpoint))
            if not ckpt_path.exists():
                raise SystemExit(f"checkpoint not found: {ckpt_path}")
            from yolozu.inference.checkpoint_compatibility import (
                load_checkpoint_compatible,
            )

            onnx_export_report["checkpoint_report"] = load_checkpoint_compatible(
                model,
                ckpt_path,
                config_identity=_resolve(str(args.config)),
                allow_partial=bool(args.allow_partial_checkpoint),
            )

        device = str(args.device)
        model.to(device)
        dummy = torch.zeros((1, 3, int(args.image_size), int(args.image_size)), dtype=torch.float32, device=device)
        export_onnx(
            model,
            dummy,
            str(onnx_path),
            opset_version=int(args.opset),
            input_name=str(args.input_name),
            dynamic_hw=bool(args.dynamic_hw),
        )
        onnx_export_report["onnx_sha256"] = _sha256(onnx_path)
        onnx_export_report["exported"] = True
    else:
        onnx_export_report["exported"] = False
        if onnx_path.exists():
            onnx_export_report["onnx_sha256"] = _sha256(onnx_path)

    onnx_meta = {
        "timestamp_utc": _now_utc(),
        "onnx": str(onnx_path),
        "report": onnx_export_report,
        "run_record": run_record,
    }
    onnx_meta_path.write_text(json.dumps(onnx_meta, indent=2, sort_keys=True), encoding="utf-8")

    if args.skip_engine:
        print(onnx_path)
        return 0

    if not onnx_path.exists():
        if args.dry_run:
            print("warning: --onnx does not exist; skipping engine build in --dry-run", file=sys.stderr)
            print(onnx_path)
            return 0
        raise SystemExit(f"onnx not found: {onnx_path}")

    cmd = [
        sys.executable,
        str(repo_root / "tools" / "build_trt_engine.py"),
        "--onnx",
        str(onnx_path),
        "--engine",
        str(engine_path),
        "--precision",
        str(args.precision),
        "--input-name",
        str(args.input_name),
        "--min-shape",
        str(min_shape),
        "--opt-shape",
        str(opt_shape),
        "--max-shape",
        str(max_shape),
        "--workspace",
        str(int(args.workspace)),
        "--timing-cache",
        str(_resolve(str(args.timing_cache))),
        "--trtexec",
        str(args.trtexec),
        "--meta-output",
        str(engine_meta_path),
    ]
    if args.extra_args:
        cmd.extend(["--extra-args", str(args.extra_args)])
    if args.dry_run:
        cmd.append("--dry-run")

    print(shlex.join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True)
    print(engine_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

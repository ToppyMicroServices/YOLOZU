import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="Path to ONNX model.")
    p.add_argument("--engine", default="engines/model.plan", help="Where to write the TensorRT engine.")
    p.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp16", help="Engine precision.")
    p.add_argument("--input-name", default="images", help="ONNX input tensor name.")
    p.add_argument("--min-shape", default="1x3x640x640", help="Min input shape (e.g., 1x3x640x640).")
    p.add_argument("--opt-shape", default="1x3x640x640", help="Opt input shape.")
    p.add_argument("--max-shape", default="1x3x640x640", help="Max input shape.")
    p.add_argument("--workspace", type=int, default=4096, help="Workspace size in MiB.")
    p.add_argument("--timing-cache", default="engines/timing.cache", help="Timing cache path.")
    p.add_argument("--calib-cache", default=None, help="INT8 calibration cache path (required for int8).")
    p.add_argument("--calib-dataset", default=None, help="Optional dataset root to build a calibration image list.")
    p.add_argument("--calib-split", default=None, help="Dataset split for calibration list (default: auto).")
    p.add_argument("--calib-images", type=int, default=100, help="Number of images for calibration list.")
    p.add_argument(
        "--calib-list-output",
        default="reports/calib_images.txt",
        help="Where to write calibration image list (if --calib-dataset is set).",
    )
    p.add_argument("--trtexec", default="trtexec", help="Path to trtexec binary.")
    p.add_argument(
        "--builder",
        choices=("auto", "trtexec", "python"),
        default="auto",
        help="Engine builder backend (default: auto). 'python' uses TensorRT Python API when trtexec is unavailable.",
    )
    p.add_argument("--extra-args", default=None, help="Extra trtexec args (quoted string).")
    p.add_argument("--meta-output", default="reports/trt_engine_meta.json", help="Where to write build metadata JSON.")
    p.add_argument("--dry-run", action="store_true", help="Print command and write meta without running trtexec.")
    return p.parse_args(argv)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_input_name(network: object, requested: str) -> str:
    """Best-effort input-name resolution helper.

    This is primarily for compatibility with older TensorRT-python based builders
    and for unit tests; the current implementation uses trtexec.
    """

    if not hasattr(network, "num_inputs"):
        return requested
    try:
        count = int(getattr(network, "num_inputs"))
        inputs = [getattr(network.get_input(i), "name", None) for i in range(count)]
        inputs = [str(name) for name in inputs if name]
    except Exception:
        return requested
    if requested in inputs:
        return requested
    if inputs:
        return inputs[0]
    return requested


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _write_calib_list(dataset_root: str, *, split: str | None, limit: int, output_path: Path) -> list[str]:
    manifest = build_manifest(dataset_root, split=split)
    records = manifest["images"][: max(0, int(limit))]
    paths = [str(r["image"]) for r in records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(paths) + ("\n" if paths else ""))
    return paths


def _parse_shape(value: str) -> tuple[int, ...]:
    raw = str(value).replace(",", "x").lower()
    parts = [p.strip() for p in raw.split("x") if p.strip()]
    if not parts:
        raise ValueError(f"invalid shape: {value!r}")
    dims = tuple(int(p) for p in parts)
    if any(d <= 0 for d in dims):
        raise ValueError(f"invalid shape: {value!r}")
    return dims


def _build_command(args: argparse.Namespace, *, onnx_path: Path, engine_path: Path, timing_cache: Path) -> list[str]:
    cmd = [
        args.trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={args.input_name}:{args.min_shape}",
        f"--optShapes={args.input_name}:{args.opt_shape}",
        f"--maxShapes={args.input_name}:{args.max_shape}",
        f"--workspace={int(args.workspace)}",
        f"--timingCacheFile={timing_cache}",
    ]
    if args.precision == "fp16":
        cmd.append("--fp16")
    elif args.precision == "int8":
        cmd.append("--int8")
        if args.calib_cache:
            cmd.append(f"--calib={_resolve_path(args.calib_cache)}")
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def _trtexec_available(trtexec: str) -> bool:
    if not trtexec:
        return False
    p = Path(trtexec)
    if p.is_absolute() or "/" in str(trtexec):
        return p.exists()
    return shutil.which(str(trtexec)) is not None


def _git_head() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
        return out.decode("utf-8").strip() or None
    except Exception:
        return None


def _run_capture(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, cwd=repo_root, stderr=subprocess.STDOUT)
    except Exception:
        return None
    try:
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _parse_cuda_version(nvidia_smi_text: str) -> str | None:
    m = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", str(nvidia_smi_text))
    return None if not m else m.group(1)


def _compute_cap_to_sm(compute_cap: str) -> str | None:
    m = re.match(r"^\s*(\d+)\.(\d+)\s*$", str(compute_cap))
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)}"


def _nvidia_smi_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    raw = _run_capture(["nvidia-smi"])
    if raw:
        info["raw"] = raw
        info["cuda_version"] = _parse_cuda_version(raw)

    query = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    if query:
        for line in query.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            gpus.append(
                {
                    "name": parts[0],
                    "uuid": parts[1],
                    "compute_cap": parts[2],
                    "sm": _compute_cap_to_sm(parts[2]),
                    "driver_version": parts[3],
                }
            )
    info["gpus"] = gpus
    return info


def _trtexec_version(trtexec: str) -> str | None:
    return _run_capture([trtexec, "--version"])


def _tensorrt_py_version() -> str | None:
    try:
        import tensorrt  # type: ignore
    except Exception:
        return None
    v = getattr(tensorrt, "__version__", None)
    return None if v is None else str(v)


def _tensorrt_py_available() -> bool:
    return _tensorrt_py_version() is not None


def _build_engine_python(
    *,
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    input_name: str,
    min_shape: tuple[int, ...],
    opt_shape: tuple[int, ...],
    max_shape: tuple[int, ...],
    workspace_mib: int,
) -> dict[str, Any]:
    try:
        import tensorrt as trt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tensorrt python package is required for --builder python") from exc

    if str(precision) == "int8":
        raise RuntimeError("--builder python does not support --precision int8 yet (use --builder trtexec)")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        errors: list[str] = []
        for i in range(int(parser.num_errors)):
            try:
                errors.append(str(parser.get_error(i)))
            except Exception:
                errors.append("<unknown parser error>")
        raise RuntimeError("failed to parse ONNX with TensorRT OnnxParser:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mib) * 1024 * 1024)

    if str(precision) == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    input_names = [network.get_input(i).name for i in range(int(network.num_inputs))]
    if not input_names:
        raise RuntimeError("TensorRT network has no inputs")
    # TensorRT does not reliably validate the input name on set_shape() across versions.
    # Apply the same profile to all network inputs (typical ONNX models here have 1 input).
    for name in input_names:
        profile.set_shape(str(name), min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")

    engine_path.write_bytes(bytes(serialized))
    return {
        "builder": "python",
        "tensorrt_py": getattr(trt, "__version__", None),
        "explicit_batch": True,
        "network_inputs": [{"name": str(name)} for name in input_names],
        "profile": {"min": list(min_shape), "opt": list(opt_shape), "max": list(max_shape)},
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    onnx_path = _resolve_path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f"onnx not found: {onnx_path}")

    engine_path = _resolve_path(args.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    timing_cache = _resolve_path(args.timing_cache)
    timing_cache.parent.mkdir(parents=True, exist_ok=True)

    calib_list_path = None
    calib_list = None
    if args.calib_dataset:
        calib_list_path = _resolve_path(args.calib_list_output)
        calib_list = _write_calib_list(
            args.calib_dataset,
            split=args.calib_split,
            limit=args.calib_images,
            output_path=calib_list_path,
        )

    cmd = _build_command(args, onnx_path=onnx_path, engine_path=engine_path, timing_cache=timing_cache)

    meta: dict[str, Any] = {
        "timestamp": _now_utc(),
        "git_head": _git_head(),
        "onnx": str(onnx_path),
        "onnx_sha256": _sha256(onnx_path),
        "engine": str(engine_path),
        "precision": args.precision,
        "input_name": args.input_name,
        "shapes": {"min": args.min_shape, "opt": args.opt_shape, "max": args.max_shape},
        "workspace_mib": int(args.workspace),
        "timing_cache": str(timing_cache),
        "calib_cache": None if not args.calib_cache else str(_resolve_path(args.calib_cache)),
        "calib_list": None if calib_list_path is None else str(calib_list_path),
        "calib_images": None if calib_list is None else len(calib_list),
        "builder": str(args.builder),
        "command": cmd,
        "command_str": shlex.join(cmd),
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        },
        "nvidia": _nvidia_smi_info(),
        "tensorrt": {
            "trtexec": str(args.trtexec),
            "trtexec_version": _trtexec_version(str(args.trtexec)),
            "tensorrt_py": _tensorrt_py_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": sys.version,
    }

    meta_path = _resolve_path(args.meta_output)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    if args.precision == "int8" and not args.calib_cache:
        print("warning: --precision int8 without --calib-cache; trtexec may refuse to build", file=sys.stderr)

    if args.dry_run:
        print(meta["command_str"])
        return 0

    builder_mode = str(args.builder or "auto").lower()
    if builder_mode not in ("auto", "trtexec", "python"):
        raise SystemExit(f"unknown --builder: {args.builder}")

    effective = builder_mode
    if builder_mode == "auto":
        if _trtexec_available(str(args.trtexec)):
            effective = "trtexec"
        elif _tensorrt_py_available():
            effective = "python"
        else:
            meta["builder_effective"] = "none"
            meta["error"] = (
                "no TensorRT builder available: neither trtexec nor Python TensorRT detected. "
                "Install TensorRT (trtexec) or `tensorrt` Python package, or rerun with --dry-run."
            )
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
            raise SystemExit(meta["error"])
    meta["builder_effective"] = effective

    if effective == "trtexec":
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            raise SystemExit(f"trtexec not found: {args.trtexec} (try --builder python)") from exc
    else:
        if not _tensorrt_py_available():
            meta["error"] = "Python TensorRT package not available (install `tensorrt` or use --builder trtexec)."
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
            raise SystemExit(meta["error"])
        try:
            python_report = _build_engine_python(
                onnx_path=onnx_path,
                engine_path=engine_path,
                precision=str(args.precision),
                input_name=str(args.input_name),
                min_shape=_parse_shape(str(args.min_shape)),
                opt_shape=_parse_shape(str(args.opt_shape)),
                max_shape=_parse_shape(str(args.max_shape)),
                workspace_mib=int(args.workspace),
            )
        except Exception as exc:
            raise SystemExit(f"TensorRT python build failed: {exc}") from exc
        meta["python_builder"] = python_report

    if engine_path.exists():
        meta["engine_sha256"] = _sha256(engine_path)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

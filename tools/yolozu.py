#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.tta.presets import apply_ttt_preset_args  # noqa: E402

DEFAULT_PREDICTIONS_PATH = "reports/predictions.json"


def _manifest_path() -> Path:
    return repo_root / "tools" / "manifest.json"


def _load_tool_manifest() -> dict[str, Any]:
    p = _manifest_path()
    return json.loads(p.read_text(encoding="utf-8"))


def _registry_payload(*, tool: dict[str, Any] | None = None) -> dict[str, Any]:
    obj = _load_tool_manifest()
    if tool is not None:
        return {
            "kind": "yolozu_tool_spec",
            "schema_version": 1,
            "timestamp": _now_utc(),
            "repo": obj.get("repo"),
            "contracts": obj.get("contracts"),
            "tool": tool,
        }
    return {
        "kind": "yolozu_tool_registry",
        "schema_version": 1,
        "timestamp": _now_utc(),
        "repo": obj.get("repo"),
        "contracts": obj.get("contracts"),
        "tools": obj.get("tools") or [],
    }


def _find_flag_value(argv: list[str], flag: str) -> str | None:
    # Very small argv parser: --flag value (no equals form)
    for i in range(len(argv) - 1):
        if argv[i] == flag:
            return argv[i + 1]
    return None


def _find_flag_value_any(argv: list[str], flag: str) -> str | None:
    # Supports: --flag value  OR  --flag=value
    v = _find_flag_value(argv, flag)
    if v is not None:
        return v
    prefix = flag + "="
    for tok in argv:
        if tok.startswith(prefix):
            return tok[len(prefix) :]
    return None


def _extract_forwarded_flags(argv: list[str]) -> set[str]:
    flags: set[str] = set()
    for tok in argv:
        if tok == "--":
            continue
        if tok == "-h":
            flags.add("-h")
            continue
        if tok.startswith("--"):
            flag = tok.split("=", 1)[0]
            flags.add(flag)
    return flags


def _is_repo_relative_path_like(value: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if value.startswith("/"):
        return False
    parts = Path(value).parts
    if ".." in parts:
        return False
    return True


def _within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _parse_contract_validator_cmd(template: str, *, path: str) -> list[str] | None:
    if not isinstance(template, str) or not template.strip():
        return None
    # Remove optional tokens written like [--strict]
    cleaned = " ".join(tok for tok in template.split() if not (tok.startswith("[") and tok.endswith("]")))
    if "<path>" not in cleaned:
        return None
    try:
        tokens = shlex.split(cleaned)
    except Exception:
        tokens = cleaned.split()
    out: list[str] = []
    for tok in tokens:
        out.append(path if tok == "<path>" else tok)
    return out


def _registry_validate(_: argparse.Namespace) -> int:
    script = repo_root / "tools" / "validate_tool_manifest.py"
    out = _subprocess_or_die([sys.executable, str(script)])
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    return 0


def _registry_list(args: argparse.Namespace) -> int:
    obj = _load_tool_manifest()
    tools = list(obj.get("tools") or [])
    tags = getattr(args, "tag", None) or []
    contracts = getattr(args, "contract", None) or []

    def _tool_matches(t: dict[str, Any]) -> bool:
        if tags:
            tt = set(t.get("tags") or [])
            if not all(tag in tt for tag in tags):
                return False
        if contracts:
            c = t.get("contracts") or {}
            cons = set(c.get("consumes") or [])
            prod = set(c.get("produces") or [])
            have = cons | prod
            if not all(cid in have for cid in contracts):
                return False
        return True

    tools = [t for t in tools if isinstance(t, dict) and _tool_matches(t)]

    if getattr(args, "json", False):
        payload = _registry_payload()
        payload["tools"] = tools
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    for t in tools:
        tid = t.get("id")
        summary = t.get("summary")
        runner = t.get("runner")
        entrypoint = t.get("entrypoint")
        print(f"- {tid}: {summary} ({runner} {entrypoint})")
    return 0


def _registry_show(args: argparse.Namespace) -> int:
    tool_id = str(getattr(args, "id"))
    obj = _load_tool_manifest()
    tools = [t for t in (obj.get("tools") or []) if isinstance(t, dict) and t.get("id") == tool_id]
    if not tools:
        raise SystemExit(f"unknown tool id: {tool_id}")
    tool = tools[0]

    if getattr(args, "json", False):
        payload = _registry_payload(tool=tool)
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    print(json.dumps(tool, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def _registry_run(args: argparse.Namespace) -> int:
    tool_id = str(getattr(args, "id"))
    forwarded = getattr(args, "forward_args", None)
    forward_args: list[str] = [str(x) for x in forwarded] if isinstance(forwarded, list) else []
    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]

    manifest = _load_tool_manifest()
    tools = [t for t in (manifest.get("tools") or []) if isinstance(t, dict) and t.get("id") == tool_id]
    if not tools:
        raise SystemExit(f"unknown tool id: {tool_id}")
    tool = tools[0]

    requires = tool.get("requires") or {}
    platform_spec = tool.get("platform") or {}
    needs_network = bool(requires.get("network"))
    gpu_required = bool(platform_spec.get("gpu_required"))
    if needs_network and not bool(getattr(args, "allow_network", False)):
        raise SystemExit("tool requires network access; rerun with --allow-network")
    if gpu_required and not bool(getattr(args, "allow_gpu", False)):
        raise SystemExit("tool requires GPU; rerun with --allow-gpu")

    allowed_write_roots = list(getattr(args, "allow_write_root", None) or ["reports"])
    allow_unsafe_paths = bool(getattr(args, "allow_unsafe_paths", False))
    dry_run = bool(getattr(args, "dry_run", False))
    allow_undeclared_effects = bool(getattr(args, "allow_undeclared_effects", False))
    allow_unknown_flags_cli = bool(getattr(args, "allow_unknown_flags", False))

    # Enforce stable flag surface (no hidden write flags).
    declared_flags: set[str] = set()
    for item in (tool.get("inputs") or []):
        if isinstance(item, dict) and isinstance(item.get("flag"), str) and item.get("flag"):
            declared_flags.add(str(item["flag"]))

    effects = tool.get("effects")
    if effects is None:
        if not allow_undeclared_effects:
            raise SystemExit(
                "tool has no declarative effects metadata (tool.effects). "
                "Add effects to tools/manifest.json or rerun with --allow-undeclared-effects."
            )
        effects = {}
    if not isinstance(effects, dict):
        raise SystemExit("invalid tool.effects (expected object)")

    allow_unknown_flags_tool = bool(effects.get("allow_unknown_flags", False))
    allow_unknown_flags = bool(allow_unknown_flags_tool or allow_unknown_flags_cli)

    forwarded_flags = _extract_forwarded_flags(forward_args)
    always_ok = {"-h", "--help"}
    unknown = sorted(f for f in forwarded_flags if f not in always_ok and f not in declared_flags)
    if unknown and not allow_unknown_flags:
        raise SystemExit(
            "unknown forwarded flags (not declared in tools/manifest.json inputs):\n"
            + "\n".join(f"- {f}" for f in unknown)
            + "\nUse --allow-unknown-flags to bypass (not recommended for agents)."
        )

    # Construct base command
    runner = tool.get("runner")
    entrypoint = tool.get("entrypoint")
    if runner not in {"python3", "bash"}:
        raise SystemExit("unsupported runner")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise SystemExit("missing entrypoint")
    if entrypoint.startswith("/") or ".." in Path(entrypoint).parts:
        raise SystemExit("invalid entrypoint path")
    entry_path = (repo_root / entrypoint)
    if not entry_path.exists():
        raise SystemExit(f"entrypoint not found: {entrypoint}")

    cmd: list[str] = ["python3", str(entry_path)] if runner == "python3" else ["bash", str(entry_path)]
    cmd.extend(forward_args)

    # Safety: enforce declared write effects (no heuristics).
    roots: list[Path] = []
    for r in allowed_write_roots:
        if not isinstance(r, str) or not r:
            continue
        if r.startswith("/") or ".." in Path(r).parts:
            raise SystemExit(f"invalid --allow-write-root: {r}")
        roots.append(repo_root / r)

    def _check_write_path(src: str, value: str) -> None:
        if not isinstance(value, str) or not value:
            return
        # Convention: allow '-' to mean stdout (no filesystem write).
        if value.strip() == "-":
            return
        if (value.startswith("/") or ".." in Path(value).parts) and not allow_unsafe_paths:
            raise SystemExit(f"unsafe path blocked ({src}): {value} (use --allow-unsafe-paths to override)")

        # Only enforce containment for repo-relative outputs.
        if _is_repo_relative_path_like(value):
            resolved = (repo_root / value)
            if roots and not any(_within(r, resolved) for r in roots):
                roots_str = ", ".join(str(Path(r).relative_to(repo_root)) for r in roots)
                raise SystemExit(
                    f"write path blocked ({src}): {value} is outside allowed roots: {roots_str} "
                    "(use --allow-write-root to add a root)"
                )

    # fixed writes
    fixed = effects.get("fixed_writes")
    if fixed is not None:
        if not isinstance(fixed, list):
            raise SystemExit("invalid tool.effects.fixed_writes (expected list)")
        for j, fw in enumerate(fixed):
            if not isinstance(fw, dict):
                raise SystemExit(f"invalid tool.effects.fixed_writes[{j}] (expected object)")
            path = fw.get("path")
            if isinstance(path, str) and path:
                _check_write_path(f"fixed_writes[{j}]", path)

    # flag-based writes
    writes = effects.get("writes")
    if writes is not None:
        if not isinstance(writes, list):
            raise SystemExit("invalid tool.effects.writes (expected list)")
        for j, w in enumerate(writes):
            if not isinstance(w, dict):
                raise SystemExit(f"invalid tool.effects.writes[{j}] (expected object)")
            flag = w.get("flag")
            if not isinstance(flag, str) or not flag.startswith("--"):
                raise SystemExit(f"invalid tool.effects.writes[{j}].flag")

            value = _find_flag_value_any(forward_args, flag)
            if value is None:
                # fallback to manifest-declared default value for this flag
                default_value: str | None = None
                for item in (tool.get("inputs") or []):
                    if isinstance(item, dict) and item.get("flag") == flag:
                        d = item.get("default")
                        if isinstance(d, str) and d:
                            default_value = d
                        break
                if default_value is None:
                    # Tool may have an internal default; rely on fixed_writes for safety.
                    continue
                if "<" in default_value or ">" in default_value:
                    raise SystemExit(
                        f"write effect default contains placeholder; pass an explicit value for {flag}: {default_value}"
                    )
                value = default_value

            _check_write_path(flag, value)

    if dry_run:
        print("DRY_RUN:")
        print(" ".join(shlex.quote(x) for x in cmd))
        return 0

    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    # Post-run: best-effort contract validation of produced artifacts.
    contracts_registry = manifest.get("contracts") or {}
    produces = (tool.get("contracts") or {}).get("produces") or []
    contract_outputs = tool.get("contract_outputs") or {}

    for contract_id in produces:
        if not isinstance(contract_id, str) or not contract_id:
            continue
        spec = contracts_registry.get(contract_id) if isinstance(contracts_registry, dict) else None
        if not isinstance(spec, dict):
            continue
        validator_tpl = spec.get("validator")
        if not isinstance(validator_tpl, str) or not validator_tpl.strip():
            continue

        output_name = contract_outputs.get(contract_id) if isinstance(contract_outputs, dict) else None
        output_default: str | None = None
        if isinstance(output_name, str) and output_name:
            for out in (tool.get("outputs") or []):
                if isinstance(out, dict) and out.get("name") == output_name:
                    d = out.get("default")
                    if isinstance(d, str) and d:
                        output_default = d
                    break

        # Contract output path must be discoverable without heuristics.
        out_path = _find_flag_value_any(forward_args, "--output") or output_default
        if not out_path:
            continue

        vcmd = _parse_contract_validator_cmd(validator_tpl, path=out_path)
        if not vcmd:
            continue
        subprocess.run(vcmd, cwd=str(repo_root), check=True)

    return 0


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run_capture(cmd: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd or repo_root), stderr=subprocess.STDOUT)
    except Exception:
        return None
    try:
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _git_head() -> str | None:
    return _run_capture(["git", "rev-parse", "HEAD"])


def _git_is_dirty() -> bool | None:
    try:
        unstaged = subprocess.run(["git", "diff", "--quiet"], cwd=str(repo_root), check=False).returncode != 0
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(repo_root), check=False).returncode != 0
        return bool(unstaged or staged)
    except Exception:
        return None


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: str | Path) -> str | None:
    try:
        p = Path(path)
        return _sha256_bytes(p.read_bytes())
    except Exception:
        return None


def _sha256_json(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _sha256_bytes(data)


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version  # py3.8+

        return version(name)
    except Exception:
        return None


def _gather_gpu_info() -> dict[str, Any]:
    gpu: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": None,
        "nvidia_smi_list": None,
    }

    smi = _run_capture(["nvidia-smi", "-L"])
    if smi:
        gpu["nvidia_smi"] = smi
        gpu["nvidia_smi_list"] = [line.strip() for line in smi.splitlines() if line.strip()]

    # torch (optional)
    try:
        import torch  # type: ignore

        torch_info: dict[str, Any] = {
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if torch_info["cuda_available"]:
            torch_info["device_count"] = int(torch.cuda.device_count())
            devices = []
            for i in range(int(torch.cuda.device_count())):
                name = None
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:
                    name = None
                cap = None
                try:
                    cap = torch.cuda.get_device_capability(i)
                except Exception:
                    cap = None
                devices.append({"index": int(i), "name": name, "capability": cap})
            torch_info["devices"] = devices
        gpu["torch"] = torch_info
    except Exception:
        gpu["torch"] = None

    # onnxruntime providers (optional)
    try:
        import onnxruntime as ort  # type: ignore

        gpu["onnxruntime_providers"] = list(getattr(ort, "get_available_providers")())
        gpu["onnxruntime_version"] = getattr(ort, "__version__", None)
    except Exception:
        gpu["onnxruntime_providers"] = None
        gpu["onnxruntime_version"] = None

    return gpu


def _gather_env_info() -> dict[str, Any]:
    return {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "packages": {
            "torch": _pkg_version("torch"),
            "onnxruntime": _pkg_version("onnxruntime"),
            "tensorrt": _pkg_version("tensorrt"),
            "numpy": _pkg_version("numpy"),
            "Pillow": _pkg_version("Pillow"),
        },
    }


def _base_run_meta(*, seed: int | None, notes: str | None, config_fingerprint: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": _now_utc(),
        "seed": seed,
        "notes": notes,
        "config_hash": _sha256_json(config_fingerprint),
        "git": {"head": _git_head(), "dirty": _git_is_dirty()},
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "gpu": _gather_gpu_info(),
        "env": _gather_env_info(),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def _ensure_wrapper(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "predictions" in payload:
        preds = payload.get("predictions")
        meta = payload.get("meta")
        if isinstance(preds, list) and (meta is None or isinstance(meta, dict)):
            return {"predictions": preds, "meta": dict(meta or {})}
    if isinstance(payload, list):
        return {"predictions": payload, "meta": {}}
    if isinstance(payload, dict):
        # Legacy mapping format.
        preds = [{"image": str(k), "detections": v if isinstance(v, list) else []} for k, v in payload.items()]
        return {"predictions": preds, "meta": {}}
    raise ValueError("unsupported predictions payload")


def _subprocess_or_die(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.stderr and proc.stderr.strip():
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_root / p


def _output_config_hash(path: Path) -> str | None:
    try:
        payload = _ensure_wrapper(_load_json(path))
    except Exception:
        return None
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return None
    run = meta.get("run")
    if not isinstance(run, dict):
        return None
    got = run.get("config_hash")
    return got if isinstance(got, str) and got else None


def _ensure_output_matches(path: Path, *, expected_config_hash: str) -> None:
    got = _output_config_hash(path)
    if got is None:
        raise SystemExit(f"output exists but missing meta.run.config_hash: {path} (use --force to overwrite)")
    if got != expected_config_hash:
        raise SystemExit(
            "output exists but does not match current config_hash:\n"
            f"  path: {path}\n"
            f"  expected: {expected_config_hash}\n"
            f"  got: {got}\n"
            "Use --force to overwrite, or choose a different --output/--run-dir/--cache-dir."
        )


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def _parse_common_export_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--backend",
        choices=("dummy", "torch", "onnxrt", "trt", "yolox", "opencv-dnn", "opencv-dnn-rtdetr", "opencv-dnn-yolo"),
        default="dummy",
        help="Inference backend (default: dummy).",
    )
    p.add_argument("--dataset", default=None, help="YOLO-format dataset root (defaults to data/coco128).")
    p.add_argument("--split", default=None, help="Dataset split under images/ and labels/ (default: auto).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    p.add_argument("--output", default=DEFAULT_PREDICTIONS_PATH, help="Predictions JSON output path.")
    p.add_argument(
        "--run-dir",
        default=None,
        help="Optional run directory. When set and --output is default, writes <run-dir>/predictions.json.",
    )
    p.add_argument(
        "--cache",
        action="store_true",
        help="Enable fingerprinted run cache. When set and --output is default, writes into --cache-dir/<config_hash>/predictions.json.",
    )
    p.add_argument("--cache-dir", default="runs/yolozu_runs", help="Cache root directory (default: runs/yolozu_runs).")
    p.add_argument("--notes", default=None, help="Notes to store in meta.run.")
    p.add_argument("--seed", type=int, default=None, help="Optional seed to store in meta.run.")
    p.add_argument("--force", action="store_true", help="Overwrite outputs if they exist.")
    p.add_argument("--dry-run", action="store_true", help="Backend dry-run when supported (onnxrt/trt).")
    p.add_argument("--strict", action="store_true", help="Enable strict predictions validation when backend supports it.")

    # Torch backend (rtdetr_pose adapter).
    p.add_argument("--config", default="rtdetr_pose/configs/base.json", help="Torch config path (rtdetr_pose).")
    p.add_argument("--checkpoint", default=None, help="Torch checkpoint path (optional).")
    p.add_argument("--device", default="cpu", help="Torch device (default: cpu).")
    p.add_argument("--image-size", type=int, nargs="+", default=None, help="Torch image size (one or two ints).")
    p.add_argument("--score-threshold", type=float, default=0.3, help="Torch score threshold (default: 0.3).")
    p.add_argument("--max-detections", type=int, default=50, help="Torch max detections (default: 50).")
    p.add_argument("--lora-r", type=int, default=0, help="Enable LoRA by setting rank r>0 (default: 0 disables).")
    p.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha scaling (default: r).")
    p.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout on inputs (default: 0.0).")
    p.add_argument(
        "--lora-target",
        default="head",
        choices=("head", "all_linear", "all_conv1x1", "all_linear_conv1x1"),
        help="Where to apply LoRA (default: head).",
    )
    p.add_argument(
        "--lora-freeze-base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze base weights and train LoRA params only (default: false).",
    )
    p.add_argument(
        "--lora-train-bias",
        choices=("none", "all"),
        default="none",
        help="If LoRA is enabled, optionally train biases too (default: none).",
    )
    p.add_argument("--tta", action="store_true", help="Enable TTA post-transform on predictions.")
    p.add_argument("--tta-seed", type=int, default=None, help="Seed for TTA randomness.")
    p.add_argument("--tta-flip-prob", type=float, default=0.5, help="Flip probability for TTA.")
    p.add_argument("--tta-norm-only", action="store_true", help="Update only normalized bbox values for TTA.")
    p.add_argument("--tta-log-out", default=None, help="Optional path to write TTA log JSON.")
    p.add_argument("--ttt", action="store_true", help="Enable test-time training (TTT) before inference.")
    p.add_argument(
        "--ttt-preset",
        choices=("safe", "adapter_only", "mim_safe", "cotta_safe", "eata_safe", "sar_safe"),
        default=None,
        help="Recommended TTT presets that override core knobs and fill safety guards unless explicitly set.",
    )
    p.add_argument("--ttt-method", choices=("tent", "mim", "cotta", "eata", "sar"), default="tent", help="TTT method (default: tent).")
    p.add_argument(
        "--ttt-reset",
        choices=("stream", "sample"),
        default="stream",
        help="TTT reset policy: stream keeps adapted weights; sample resets per image (default: stream).",
    )
    p.add_argument("--ttt-steps", type=int, default=1, help="Total TTT steps to run (default: 1).")
    p.add_argument("--ttt-batch-size", type=int, default=1, help="TTT batch size (default: 1).")
    p.add_argument("--ttt-lr", type=float, default=1e-4, help="TTT learning rate (default: 1e-4).")
    p.add_argument(
        "--ttt-stop-on-non-finite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop TTT if loss/grad/update norms become non-finite (default: true).",
    )
    p.add_argument(
        "--ttt-rollback-on-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rollback last TTT step when a guard triggers (default: true).",
    )
    p.add_argument("--ttt-max-grad-norm", type=float, default=None, help="Optional grad clipping norm (default: none).")
    p.add_argument(
        "--ttt-max-update-norm",
        type=float,
        default=None,
        help="Stop if per-step weight update L2 norm exceeds this (default: none).",
    )
    p.add_argument(
        "--ttt-max-total-update-norm",
        type=float,
        default=None,
        help="Stop if total drift from initial weights exceeds this (default: none).",
    )
    p.add_argument(
        "--ttt-max-loss-ratio",
        type=float,
        default=None,
        help="Stop if loss exceeds (initial_loss * ratio) (default: none).",
    )
    p.add_argument(
        "--ttt-max-loss-increase",
        type=float,
        default=None,
        help="Stop if loss exceeds (initial_loss + delta) (default: none).",
    )
    p.add_argument(
        "--ttt-update-filter",
        choices=("all", "norm_only", "adapter_only", "lora_only", "lora_norm_only"),
        default="all",
        help="Which parameters to update during TTT (default: all).",
    )
    p.add_argument(
        "--ttt-include",
        action="append",
        default=None,
        help="Only update parameters whose name contains this substring (repeatable).",
    )
    p.add_argument(
        "--ttt-exclude",
        action="append",
        default=None,
        help="Exclude parameters whose name contains this substring (repeatable).",
    )
    p.add_argument("--ttt-max-batches", type=int, default=1, help="Cap number of distinct batches used for TTT.")
    p.add_argument("--ttt-seed", type=int, default=None, help="Optional RNG seed for TTT.")
    p.add_argument("--ttt-mask-prob", type=float, default=0.6, help="MIM mask probability (default: 0.6).")
    p.add_argument("--ttt-patch-size", type=int, default=16, help="MIM patch size (default: 16).")
    p.add_argument("--ttt-mask-value", type=float, default=0.0, help="MIM mask fill value (default: 0.0).")
    p.add_argument("--ttt-cotta-ema-momentum", type=float, default=0.999, help="CoTTA EMA momentum (default: 0.999).")
    p.add_argument(
        "--ttt-cotta-augmentations",
        action="append",
        default=None,
        help="CoTTA augmentation branch name (repeatable, e.g. identity/hflip).",
    )
    p.add_argument(
        "--ttt-cotta-aggregation",
        choices=("confidence_weighted_mean", "mean"),
        default="confidence_weighted_mean",
        help="CoTTA logits aggregation mode (default: confidence_weighted_mean).",
    )
    p.add_argument("--ttt-cotta-restore-prob", type=float, default=0.01, help="CoTTA stochastic restore probability (default: 0.01).")
    p.add_argument("--ttt-cotta-restore-interval", type=int, default=1, help="CoTTA restore cadence in steps (default: 1).")
    p.add_argument("--ttt-eata-conf-min", type=float, default=0.2, help="EATA minimum confidence threshold (default: 0.2).")
    p.add_argument("--ttt-eata-entropy-min", type=float, default=0.05, help="EATA minimum entropy threshold (default: 0.05).")
    p.add_argument("--ttt-eata-entropy-max", type=float, default=3.0, help="EATA maximum entropy threshold (default: 3.0).")
    p.add_argument("--ttt-eata-min-valid-dets", type=int, default=1, help="EATA minimum valid detections per sample (default: 1).")
    p.add_argument("--ttt-eata-anchor-lambda", type=float, default=1e-3, help="EATA anchor regularization weight (default: 1e-3).")
    p.add_argument("--ttt-eata-selected-ratio-min", type=float, default=0.0, help="EATA minimum selected-sample ratio per step (default: 0.0).")
    p.add_argument("--ttt-eata-max-skip-streak", type=int, default=3, help="EATA max consecutive skipped steps before stop (default: 3).")
    p.add_argument("--ttt-sar-rho", type=float, default=0.05, help="SAR perturbation radius rho (default: 0.05).")
    p.add_argument(
        "--ttt-sar-adaptive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use adaptive SAR perturbation scaling by parameter magnitude (default: false).",
    )
    p.add_argument("--ttt-sar-first-step-scale", type=float, default=1.0, help="SAR first-step scaling factor (default: 1.0).")
    p.add_argument("--ttt-log-out", default=None, help="Optional path to write TTT log JSON.")

    # ONNXRuntime/TensorRT backend (YOLO26 exporters).
    p.add_argument("--model", default=None, help="Model path (.onnx for onnxrt, .plan for trt).")
    p.add_argument("--exp", default=None, help="YOLOX exp file path for --backend yolox.")
    p.add_argument("--weights", default=None, help="YOLOX checkpoint path for --backend yolox.")
    p.add_argument("--input-name", default="images", help="Input tensor/binding name (default: images).")
    p.add_argument("--combined-output", default="output0", help="Combined output name (default: output0).")
    p.add_argument(
        "--boxes-scale",
        choices=("abs", "norm"),
        default="abs",
        help="Combined boxes scale (default: abs).",
    )
    p.add_argument("--min-score", type=float, default=0.0, help="Score threshold (default: 0.0).")
    p.add_argument("--topk", type=int, default=300, help="Top-K per image (default: 300).")

    # OpenCV DNN RT-DETR backend.
    p.add_argument("--onnx", default=None, help="ONNX model path for --backend opencv-dnn-rtdetr (alias: --model).")
    p.add_argument("--imgsz", type=int, default=640, help="Input size for opencv-dnn-rtdetr backend (default: 640).")
    p.add_argument(
        "--score-thr",
        type=float,
        default=0.01,
        help="Score threshold for opencv-dnn-rtdetr backend (default: 0.01).",
    )
    p.add_argument(
        "--keep-aspect",
        action="store_true",
        help="Keep aspect ratio via letterbox in opencv-dnn-rtdetr backend (default: False).",
    )
    p.add_argument(
        "--dnn-backend",
        default="opencv",
        help="OpenCV DNN backend for opencv-dnn-rtdetr (e.g. opencv, cuda) (default: opencv).",
    )
    p.add_argument(
        "--dnn-target",
        default="cpu",
        help="OpenCV DNN target for opencv-dnn-rtdetr (e.g. cpu, cuda, cuda_fp16) (default: cpu).",
    )
    # OpenCV DNN YOLO backend (reuses imgsz/score-thr).
    p.add_argument(
        "--nms-iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold for opencv-dnn-yolo backend (default: 0.45).",
    )
    p.add_argument("--agnostic-nms", action="store_true", help="Class-agnostic NMS for opencv-dnn-yolo backend.")
    p.add_argument(
        "--raw-format",
        choices=("yolo_84", "yolo_85_obj"),
        default="yolo_84",
        help="Raw head layout for opencv-dnn-yolo backend (default: yolo_84).",
    )
    p.add_argument(
        "--decode",
        choices=("auto", "yolo_84", "yolo_85_obj", "rtdetr"),
        default="auto",
        help="Unified OpenCV decode selection for --backend opencv-dnn (default: auto).",
    )
    p.add_argument(
        "--preprocess",
        choices=("yolo_letterbox_640", "rtdetr_resize_640", "rtdetr_letterbox_640"),
        default=None,
        help="Unified OpenCV preprocess preset for --backend opencv-dnn.",
    )
    p.add_argument(
        "--dump-io",
        default=None,
        help="Optional IO probe dump path for OpenCV backends (input/output tensor names/shapes/dtypes).",
    )


def _export_with_backend(
    args: argparse.Namespace,
    *,
    dataset_override: str | None = None,
    dataset_meta: str | None = None,
) -> Path:
    dataset = dataset_override or (args.dataset if args.dataset else str(repo_root / "data" / "coco128"))
    dataset_fp = dataset_meta or dataset

    backend = str(args.backend)

    adapter = None
    config_fp: dict[str, Any]

    if backend in ("dummy", "torch"):
        adapter = "dummy" if backend == "dummy" else "rtdetr_pose"
        lora_enabled = bool(backend == "torch" and int(args.lora_r) > 0)
        tta_enabled = bool(args.tta)
        ttt_enabled = bool(args.ttt)
        if ttt_enabled:
            apply_ttt_preset_args(args)
        config_fp = {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "adapter": adapter,
            "config": str(args.config) if backend == "torch" else None,
            "config_sha256": _sha256_file(repo_root / str(args.config)) if backend == "torch" else None,
            "checkpoint": str(args.checkpoint) if backend == "torch" else None,
            "checkpoint_sha256": _sha256_file(args.checkpoint) if backend == "torch" and args.checkpoint else None,
            "device": str(args.device) if backend == "torch" else None,
            "image_size": list(args.image_size) if backend == "torch" and args.image_size else None,
            "score_threshold": float(args.score_threshold) if backend == "torch" else None,
            "max_detections": int(args.max_detections) if backend == "torch" else None,
            "lora": {
                "enabled": lora_enabled,
                "r": int(args.lora_r) if lora_enabled else 0,
                "alpha": float(args.lora_alpha) if lora_enabled and args.lora_alpha is not None else None,
                "dropout": float(args.lora_dropout) if lora_enabled else None,
                "target": str(args.lora_target) if lora_enabled else None,
                "freeze_base": bool(args.lora_freeze_base) if lora_enabled else None,
                "train_bias": str(args.lora_train_bias) if lora_enabled else None,
            },
            "tta": {
                "enabled": tta_enabled,
                "seed": args.tta_seed if tta_enabled else None,
                "flip_prob": float(args.tta_flip_prob) if tta_enabled else None,
                "norm_only": bool(args.tta_norm_only) if tta_enabled else None,
            },
            "ttt": {
                "enabled": ttt_enabled,
                "preset": args.ttt_preset if ttt_enabled else None,
                "method": str(args.ttt_method) if ttt_enabled else None,
                "reset": str(args.ttt_reset) if ttt_enabled else None,
                "steps": int(args.ttt_steps) if ttt_enabled else None,
                "batch_size": int(args.ttt_batch_size) if ttt_enabled else None,
                "lr": float(args.ttt_lr) if ttt_enabled else None,
                "stop_on_non_finite": bool(args.ttt_stop_on_non_finite) if ttt_enabled else None,
                "rollback_on_stop": bool(args.ttt_rollback_on_stop) if ttt_enabled else None,
                "max_grad_norm": float(args.ttt_max_grad_norm) if ttt_enabled and args.ttt_max_grad_norm is not None else None,
                "max_update_norm": float(args.ttt_max_update_norm) if ttt_enabled and args.ttt_max_update_norm is not None else None,
                "max_total_update_norm": (
                    float(args.ttt_max_total_update_norm)
                    if ttt_enabled and args.ttt_max_total_update_norm is not None
                    else None
                ),
                "max_loss_ratio": float(args.ttt_max_loss_ratio) if ttt_enabled and args.ttt_max_loss_ratio is not None else None,
                "max_loss_increase": (
                    float(args.ttt_max_loss_increase)
                    if ttt_enabled and args.ttt_max_loss_increase is not None
                    else None
                ),
                "update_filter": str(args.ttt_update_filter) if ttt_enabled else None,
                "include": list(args.ttt_include) if ttt_enabled and args.ttt_include else None,
                "exclude": list(args.ttt_exclude) if ttt_enabled and args.ttt_exclude else None,
                "max_batches": int(args.ttt_max_batches) if ttt_enabled else None,
                "seed": args.ttt_seed if ttt_enabled else None,
                "mim": {
                    "mask_prob": float(args.ttt_mask_prob) if ttt_enabled else None,
                    "patch_size": int(args.ttt_patch_size) if ttt_enabled else None,
                    "mask_value": float(args.ttt_mask_value) if ttt_enabled else None,
                },
                "cotta": {
                    "ema_momentum": float(args.ttt_cotta_ema_momentum) if ttt_enabled else None,
                    "augmentations": list(args.ttt_cotta_augmentations) if ttt_enabled and args.ttt_cotta_augmentations else None,
                    "aggregation": str(args.ttt_cotta_aggregation) if ttt_enabled else None,
                    "restore_prob": float(args.ttt_cotta_restore_prob) if ttt_enabled else None,
                    "restore_interval": int(args.ttt_cotta_restore_interval) if ttt_enabled else None,
                },
                "eata": {
                    "conf_min": float(args.ttt_eata_conf_min) if ttt_enabled else None,
                    "entropy_min": float(args.ttt_eata_entropy_min) if ttt_enabled else None,
                    "entropy_max": float(args.ttt_eata_entropy_max) if ttt_enabled else None,
                    "min_valid_dets": int(args.ttt_eata_min_valid_dets) if ttt_enabled else None,
                    "anchor_lambda": float(args.ttt_eata_anchor_lambda) if ttt_enabled else None,
                    "selected_ratio_min": float(args.ttt_eata_selected_ratio_min) if ttt_enabled else None,
                    "max_skip_streak": int(args.ttt_eata_max_skip_streak) if ttt_enabled else None,
                },
                "sar": {
                    "rho": float(args.ttt_sar_rho) if ttt_enabled else None,
                    "adaptive": bool(args.ttt_sar_adaptive) if ttt_enabled else None,
                    "first_step_scale": float(args.ttt_sar_first_step_scale) if ttt_enabled else None,
                },
            },
        }
    elif backend == "onnxrt":
        if args.tta or args.ttt or int(args.lora_r) > 0:
            raise SystemExit("--tta/--ttt/--lora-* are only supported for --backend dummy/torch")
        model = args.model
        if not model:
            raise SystemExit("--model is required for --backend onnxrt")
        config_fp = {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "model": str(model),
            "model_sha256": _sha256_file(model),
            "input_name": str(args.input_name),
            "combined_output": str(args.combined_output),
            "boxes_scale": str(args.boxes_scale),
            "min_score": float(args.min_score),
            "topk": int(args.topk),
            "dry_run": bool(args.dry_run),
        }
    elif backend == "trt":
        if args.tta or args.ttt or int(args.lora_r) > 0:
            raise SystemExit("--tta/--ttt/--lora-* are only supported for --backend dummy/torch")
        model = args.model
        if not model:
            raise SystemExit("--model is required for --backend trt")
        config_fp = {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "engine": str(model),
            "engine_sha256": _sha256_file(model),
            "input_name": str(args.input_name),
            "combined_output": str(args.combined_output),
            "boxes_scale": str(args.boxes_scale),
            "min_score": float(args.min_score),
            "topk": int(args.topk),
            "dry_run": bool(args.dry_run),
        }
    elif backend == "yolox":
        config_fp = {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "exp": str(args.exp) if args.exp else None,
            "weights": str(args.weights) if args.weights else None,
            "weights_sha256": _sha256_file(args.weights) if args.weights else None,
            "device": str(args.device),
            "imgsz": int(args.imgsz),
            "score_thr": float(args.score_thr),
            "nms_thr": float(args.nms_iou),
            "topk": int(args.topk),
            "dry_run": bool(args.dry_run),
        }
    elif backend == "opencv-dnn":
        onnx_model = args.onnx or args.model
        if not onnx_model:
            raise SystemExit("--onnx (or --model) is required for --backend opencv-dnn")
        config_fp = {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "onnx": str(onnx_model),
            "onnx_sha256": _sha256_file(onnx_model),
            "imgsz": int(args.imgsz),
            "score_thr": float(args.score_thr),
            "nms_iou": float(args.nms_iou),
            "topk": int(args.topk),
            "decode": str(args.decode),
            "preprocess": str(args.preprocess) if args.preprocess else None,
            "dnn_backend": str(args.dnn_backend),
            "dnn_target": str(args.dnn_target),
            "dump_io": str(args.dump_io) if args.dump_io else None,
            "dry_run": bool(args.dry_run),
        }
    elif backend == "opencv-dnn-rtdetr":
        onnx_model = args.onnx or args.model
        if not onnx_model:
            raise SystemExit("--onnx (or --model) is required for --backend opencv-dnn-rtdetr")
        config_fp = {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "onnx": str(onnx_model),
            "onnx_sha256": _sha256_file(onnx_model),
            "imgsz": int(args.imgsz),
            "score_thr": float(args.score_thr),
            "keep_aspect": bool(args.keep_aspect),
            "dnn_backend": str(args.dnn_backend),
            "dnn_target": str(args.dnn_target),
            "topk": int(args.topk),
            "dump_io": str(args.dump_io) if args.dump_io else None,
        }
    elif backend == "opencv-dnn-yolo":
        onnx_model = args.onnx or args.model
        if not onnx_model:
            raise SystemExit("--onnx (or --model) is required for --backend opencv-dnn-yolo")
        config_fp = {
            "backend": backend,
            "dataset": str(dataset_fp),
            "split": args.split,
            "max_images": args.max_images,
            "onnx": str(onnx_model),
            "onnx_sha256": _sha256_file(onnx_model),
            "imgsz": int(args.imgsz),
            "score_thr": float(args.score_thr),
            "nms_iou": float(args.nms_iou),
            "agnostic_nms": bool(args.agnostic_nms),
            "raw_format": str(args.raw_format),
            "dump_io": str(args.dump_io) if args.dump_io else None,
        }
    else:
        raise SystemExit(f"unknown backend: {backend}")

    config_hash = _sha256_json(config_fp)

    out_path = _resolve_path(args.output)

    run_dir = None
    if args.run_dir and args.output == DEFAULT_PREDICTIONS_PATH:
        run_dir = _resolve_path(args.run_dir)
        out_path = run_dir / "predictions.json"

    cache_out = None
    if args.cache:
        cache_out = _resolve_path(args.cache_dir) / config_hash / "predictions.json"
        if args.output == DEFAULT_PREDICTIONS_PATH and not args.run_dir:
            out_path = cache_out

    if out_path.exists() and not args.force:
        _ensure_output_matches(out_path, expected_config_hash=config_hash)
        return out_path

    if cache_out is not None and cache_out.exists() and not args.force:
        _ensure_output_matches(cache_out, expected_config_hash=config_hash)
        if cache_out != out_path:
            _copy_file(cache_out, out_path)
        return out_path

    if backend in ("dummy", "torch"):
        if adapter is None:
            raise SystemExit("internal error: missing adapter")
        cmd = [
            sys.executable,
            "tools/export_predictions.py",
            "--adapter",
            adapter,
            "--dataset",
            str(dataset),
            "--output",
            str(out_path),
            "--wrap",
        ]
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])

        if args.tta:
            cmd.append("--tta")
        if args.tta_seed is not None:
            cmd.extend(["--tta-seed", str(int(args.tta_seed))])
        cmd.extend(["--tta-flip-prob", str(float(args.tta_flip_prob))])
        if args.tta_norm_only:
            cmd.append("--tta-norm-only")
        if args.tta_log_out:
            cmd.extend(["--tta-log-out", str(args.tta_log_out)])

        if args.ttt:
            cmd.append("--ttt")
            if args.ttt_preset:
                cmd.extend(["--ttt-preset", str(args.ttt_preset)])
            cmd.extend(["--ttt-method", str(args.ttt_method)])
            cmd.extend(["--ttt-reset", str(args.ttt_reset)])
            cmd.extend(["--ttt-steps", str(int(args.ttt_steps))])
            cmd.extend(["--ttt-batch-size", str(int(args.ttt_batch_size))])
            cmd.extend(["--ttt-lr", str(float(args.ttt_lr))])
            cmd.append(
                "--ttt-stop-on-non-finite" if bool(args.ttt_stop_on_non_finite) else "--no-ttt-stop-on-non-finite"
            )
            cmd.append(
                "--ttt-rollback-on-stop" if bool(args.ttt_rollback_on_stop) else "--no-ttt-rollback-on-stop"
            )
            if args.ttt_max_grad_norm is not None:
                cmd.extend(["--ttt-max-grad-norm", str(float(args.ttt_max_grad_norm))])
            if args.ttt_max_update_norm is not None:
                cmd.extend(["--ttt-max-update-norm", str(float(args.ttt_max_update_norm))])
            if args.ttt_max_total_update_norm is not None:
                cmd.extend(["--ttt-max-total-update-norm", str(float(args.ttt_max_total_update_norm))])
            if args.ttt_max_loss_ratio is not None:
                cmd.extend(["--ttt-max-loss-ratio", str(float(args.ttt_max_loss_ratio))])
            if args.ttt_max_loss_increase is not None:
                cmd.extend(["--ttt-max-loss-increase", str(float(args.ttt_max_loss_increase))])
            cmd.extend(["--ttt-update-filter", str(args.ttt_update_filter)])
            if args.ttt_include:
                for inc in args.ttt_include:
                    cmd.extend(["--ttt-include", str(inc)])
            if args.ttt_exclude:
                for exc in args.ttt_exclude:
                    cmd.extend(["--ttt-exclude", str(exc)])
            cmd.extend(["--ttt-max-batches", str(int(args.ttt_max_batches))])
            if args.ttt_seed is not None:
                cmd.extend(["--ttt-seed", str(int(args.ttt_seed))])
            cmd.extend(["--ttt-mask-prob", str(float(args.ttt_mask_prob))])
            cmd.extend(["--ttt-patch-size", str(int(args.ttt_patch_size))])
            cmd.extend(["--ttt-mask-value", str(float(args.ttt_mask_value))])
            cmd.extend(["--ttt-cotta-ema-momentum", str(float(args.ttt_cotta_ema_momentum))])
            if args.ttt_cotta_augmentations:
                for aug in args.ttt_cotta_augmentations:
                    cmd.extend(["--ttt-cotta-augmentations", str(aug)])
            cmd.extend(["--ttt-cotta-aggregation", str(args.ttt_cotta_aggregation)])
            cmd.extend(["--ttt-cotta-restore-prob", str(float(args.ttt_cotta_restore_prob))])
            cmd.extend(["--ttt-cotta-restore-interval", str(int(args.ttt_cotta_restore_interval))])
            cmd.extend(["--ttt-eata-conf-min", str(float(args.ttt_eata_conf_min))])
            cmd.extend(["--ttt-eata-entropy-min", str(float(args.ttt_eata_entropy_min))])
            cmd.extend(["--ttt-eata-entropy-max", str(float(args.ttt_eata_entropy_max))])
            cmd.extend(["--ttt-eata-min-valid-dets", str(int(args.ttt_eata_min_valid_dets))])
            cmd.extend(["--ttt-eata-anchor-lambda", str(float(args.ttt_eata_anchor_lambda))])
            cmd.extend(["--ttt-eata-selected-ratio-min", str(float(args.ttt_eata_selected_ratio_min))])
            cmd.extend(["--ttt-eata-max-skip-streak", str(int(args.ttt_eata_max_skip_streak))])
            cmd.extend(["--ttt-sar-rho", str(float(args.ttt_sar_rho))])
            cmd.append("--ttt-sar-adaptive" if bool(args.ttt_sar_adaptive) else "--no-ttt-sar-adaptive")
            cmd.extend(["--ttt-sar-first-step-scale", str(float(args.ttt_sar_first_step_scale))])
            if args.ttt_log_out:
                cmd.extend(["--ttt-log-out", str(args.ttt_log_out)])

        if backend == "torch":
            cmd.extend(
                [
                    "--config",
                    str(args.config),
                    "--device",
                    str(args.device),
                    "--score-threshold",
                    str(float(args.score_threshold)),
                    "--max-detections",
                    str(int(args.max_detections)),
                    "--lora-r",
                    str(int(args.lora_r)),
                    "--lora-dropout",
                    str(float(args.lora_dropout)),
                    "--lora-target",
                    str(args.lora_target),
                    "--lora-train-bias",
                    str(args.lora_train_bias),
                ]
            )
            if args.checkpoint:
                cmd.extend(["--checkpoint", str(args.checkpoint)])
            if args.image_size:
                cmd.extend(["--image-size", *[str(int(x)) for x in args.image_size]])
            if args.lora_alpha is not None:
                cmd.extend(["--lora-alpha", str(float(args.lora_alpha))])
            cmd.append("--lora-freeze-base" if bool(args.lora_freeze_base) else "--no-lora-freeze-base")

        _subprocess_or_die(cmd)
    elif backend == "onnxrt":
        cmd = [
            sys.executable,
            "tools/export_predictions_onnxrt.py",
            "--dataset",
            str(dataset),
            "--onnx",
            str(args.model),
            "--input-name",
            str(args.input_name),
            "--combined-output",
            str(args.combined_output),
            "--boxes-scale",
            str(args.boxes_scale),
            "--min-score",
            str(float(args.min_score)),
            "--topk",
            str(int(args.topk)),
            "--output",
            str(out_path),
            "--wrap",
        ]
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])
        if args.dry_run:
            cmd.append("--dry-run")
        _subprocess_or_die(cmd)
    elif backend == "trt":
        cmd = [
            sys.executable,
            "tools/export_predictions_trt.py",
            "--dataset",
            str(dataset),
            "--engine",
            str(args.model),
            "--input-name",
            str(args.input_name),
            "--combined-output",
            str(args.combined_output),
            "--boxes-scale",
            str(args.boxes_scale),
            "--min-score",
            str(float(args.min_score)),
            "--topk",
            str(int(args.topk)),
            "--output",
            str(out_path),
            "--wrap",
        ]
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])
        if args.dry_run:
            cmd.append("--dry-run")
        _subprocess_or_die(cmd)
    elif backend == "yolox":
        cmd = [
            sys.executable,
            "tools/export_predictions_yolox.py",
            "--dataset",
            str(dataset),
            "--imgsz",
            str(int(args.imgsz)),
            "--score-thr",
            str(float(args.score_thr)),
            "--nms-thr",
            str(float(args.nms_iou)),
            "--topk",
            str(int(args.topk)),
            "--device",
            str(args.device),
            "--output",
            str(out_path),
        ]
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])
        if args.exp:
            cmd.extend(["--exp", str(args.exp)])
        if args.weights:
            cmd.extend(["--weights", str(args.weights)])
        if args.dry_run:
            cmd.append("--dry-run")
        if args.strict:
            cmd.append("--strict")
        _subprocess_or_die(cmd)
    elif backend == "opencv-dnn":
        onnx_model = args.onnx or args.model
        if not onnx_model:
            raise SystemExit("--onnx (or --model) is required for --backend opencv-dnn")
        cmd = [
            sys.executable,
            "tools/export_predictions_opencv_dnn_unified.py",
            "--dataset",
            str(dataset),
            "--onnx",
            str(onnx_model),
            "--imgsz",
            str(int(args.imgsz)),
            "--score-thr",
            str(float(args.score_thr)),
            "--nms-iou",
            str(float(args.nms_iou)),
            "--topk",
            str(int(args.topk)),
            "--decode",
            str(args.decode),
            "--dnn-backend",
            str(args.dnn_backend),
            "--dnn-target",
            str(args.dnn_target),
            "--output",
            str(out_path),
        ]
        if args.preprocess:
            cmd.extend(["--preprocess", str(args.preprocess)])
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.strict:
            cmd.append("--strict")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.dump_io:
            cmd.extend(["--dump-io", str(args.dump_io)])
        _subprocess_or_die(cmd)
    elif backend == "opencv-dnn-rtdetr":
        onnx_model = args.onnx or args.model
        if not onnx_model:
            raise SystemExit("--onnx (or --model) is required for --backend opencv-dnn-rtdetr")
        cmd = [
            sys.executable,
            "tools/export_predictions_opencv_dnn_rtdetr.py",
            "--dataset",
            str(dataset),
            "--onnx",
            str(onnx_model),
            "--imgsz",
            str(int(args.imgsz)),
            "--score-thr",
            str(float(args.score_thr)),
            "--output",
            str(out_path),
        ]
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])
        if args.keep_aspect:
            cmd.append("--keep-aspect")
        if args.dnn_backend:
            cmd.extend(["--dnn-backend", str(args.dnn_backend)])
        if args.dnn_target:
            cmd.extend(["--dnn-target", str(args.dnn_target)])
        if args.topk is not None:
            cmd.extend(["--topk", str(int(args.topk))])
        if args.dump_io:
            cmd.extend(["--dump-io", str(args.dump_io)])
        if args.dry_run:
            cmd.append("--dry-run")
        if args.strict:
            cmd.append("--strict")
        _subprocess_or_die(cmd)
    elif backend == "opencv-dnn-yolo":
        onnx_model = args.onnx or args.model
        if not onnx_model:
            raise SystemExit("--onnx (or --model) is required for --backend opencv-dnn-yolo")
        cmd = [
            sys.executable,
            "tools/export_predictions_opencv_dnn.py",
            "--dataset",
            str(dataset),
            "--onnx",
            str(onnx_model),
            "--input-size",
            str(int(args.imgsz)),
            "--min-score",
            str(float(args.score_thr)),
            "--output",
            str(out_path),
        ]
        if args.split:
            cmd.extend(["--split", str(args.split)])
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])
        if args.nms_iou is not None:
            cmd.extend(["--nms-iou", str(float(args.nms_iou))])
        if args.agnostic_nms:
            cmd.append("--agnostic-nms")
        if args.raw_format:
            cmd.extend(["--raw-format", str(args.raw_format)])
        if args.dump_io:
            cmd.extend(["--dump-io", str(args.dump_io)])
        if args.dry_run:
            cmd.append("--dry-run")
        if args.strict:
            cmd.append("--strict")
        _subprocess_or_die(cmd)
    else:  # pragma: no cover
        raise SystemExit(f"unknown backend: {backend}")

    payload = _ensure_wrapper(_load_json(out_path))
    payload["meta"]["run"] = _base_run_meta(seed=args.seed, notes=args.notes, config_fingerprint=config_fp)
    _write_json(out_path, payload)

    if cache_out is not None and cache_out != out_path:
        _copy_file(out_path, cache_out)

    meta_dir = cache_out.parent if cache_out is not None else run_dir
    if meta_dir is not None:
        _write_json(meta_dir / "run_config.json", {"config_hash": config_hash, "config_fingerprint": config_fp})

    return out_path


def _doctor(args: argparse.Namespace) -> int:
    from yolozu.doctor import write_doctor_report

    return int(write_doctor_report(output=str(args.output), cwd=repo_root))


def _sweep(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "tools/hpo_sweep.py",
        "--config",
        str(args.config),
    ]
    if args.resume:
        cmd.append("--resume")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.max_runs is not None:
        cmd.extend(["--max-runs", str(int(args.max_runs))])
    out = _subprocess_or_die(cmd)
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    return 0


def _continual_train(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "rtdetr_pose/tools/train_continual.py", "--config", str(args.config)]
    if args.run_dir:
        cmd.extend(["--run-dir", str(args.run_dir)])
    if args.replay_size is not None:
        cmd.extend(["--replay-size", str(int(args.replay_size))])
    if args.replay_fraction is not None:
        cmd.extend(["--replay-fraction", str(float(args.replay_fraction))])
    if args.replay_per_task_cap is not None:
        cmd.extend(["--replay-per-task-cap", str(int(args.replay_per_task_cap))])
    out = _subprocess_or_die(cmd)
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    return 0


def _continual_eval(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "tools/eval_continual.py", "--run-json", str(args.run_json)]
    if args.device:
        cmd.extend(["--device", str(args.device)])
    if args.image_size is not None:
        cmd.extend(["--image-size", str(int(args.image_size))])
    if args.max_images is not None:
        cmd.extend(["--max-images", str(int(args.max_images))])
    if args.metric:
        cmd.extend(["--metric", str(args.metric)])
    if args.metric_key:
        cmd.extend(["--metric-key", str(args.metric_key)])
    if args.output:
        cmd.extend(["--output", str(args.output)])
    if args.html:
        cmd.extend(["--html", str(args.html)])
    if args.force:
        cmd.append("--force")

    # Pose-specific args (safe to forward; eval_continual validates per-metric).
    if args.iou_threshold is not None:
        cmd.extend(["--iou-threshold", str(float(args.iou_threshold))])
    if args.min_score is not None:
        cmd.extend(["--min-score", str(float(args.min_score))])
    if args.success_rot_deg is not None:
        cmd.extend(["--success-rot-deg", str(float(args.success_rot_deg))])
    if args.success_trans is not None:
        cmd.extend(["--success-trans", str(float(args.success_trans))])
    if args.keep_per_image is not None:
        cmd.extend(["--keep-per-image", str(int(args.keep_per_image))])

    out = _subprocess_or_die(cmd)
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    return 0


def _iter_images(input_dir: Path, *, patterns: Iterable[str]) -> list[Path]:
    images: list[Path] = []
    for pat in patterns:
        images.extend(sorted(input_dir.glob(pat)))
    # De-dup while preserving order.
    seen: set[str] = set()
    out: list[Path] = []
    for p in images:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _render_overlays(
    payload: dict[str, Any],
    *,
    overlays_dir: Path,
    max_images: int | None,
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Pillow is required for overlays: {exc}") from exc

    overlays_dir.mkdir(parents=True, exist_ok=True)

    preds = payload.get("predictions")
    if not isinstance(preds, list):
        raise SystemExit("invalid predictions payload: missing predictions[]")

    written = 0
    index: list[dict[str, Any]] = []

    for entry in preds:
        if max_images is not None and written >= int(max_images):
            break
        if not isinstance(entry, dict):
            continue
        image_path = entry.get("image")
        if not isinstance(image_path, str) or not image_path:
            continue

        dets = entry.get("detections") or []
        if not isinstance(dets, list):
            dets = []

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        draw = ImageDraw.Draw(img)
        w, h = img.size
        for det in dets:
            if not isinstance(det, dict):
                continue
            bbox = det.get("bbox")
            if not isinstance(bbox, dict):
                continue
            try:
                cx = float(bbox.get("cx"))
                cy = float(bbox.get("cy"))
                bw = float(bbox.get("w"))
                bh = float(bbox.get("h"))
            except Exception:
                continue
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

            kps_raw = det.get("keypoints")
            if kps_raw is not None:
                try:
                    from yolozu.keypoints import keypoints_to_pixels, normalize_keypoints

                    kps = normalize_keypoints(kps_raw, where="detection.keypoints")
                    pts = keypoints_to_pixels(kps, width=int(w), height=int(h))
                    r = 3
                    for px, py, v in pts:
                        if v is not None:
                            try:
                                if float(v) <= 0.0:
                                    continue
                            except Exception:
                                pass
                        draw.ellipse([px - r, py - r, px + r, py + r], outline=(0, 0, 255), width=2)
                except Exception:
                    pass

        out_name = f"{written:06d}_{Path(image_path).name}"
        out_path = overlays_dir / out_name
        img.save(out_path)
        index.append(
            {
                "image": image_path,
                "overlay": str(out_path),
                "detections": int(len(dets)),
            }
        )
        written += 1

    return {"overlays_dir": str(overlays_dir), "count": int(written), "items": index}


def _write_html_report(
    *,
    html_path: Path,
    overlays_index: dict[str, Any],
    title: str,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    items = overlays_index.get("items") if isinstance(overlays_index, dict) else None
    if not isinstance(items, list):
        items = []

    # Use relative paths for portability.
    def rel(p: str) -> str:
        try:
            return str(Path(p).relative_to(html_path.parent))
        except Exception:
            return str(p)

    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8" />',
        f"  <title>{title}</title>",
        "  <style>",
        "    body{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:16px;}",
        "    .grid{display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap:16px;}",
        "    .card{border:1px solid #ddd; border-radius:8px; padding:8px;}",
        "    img{max-width:100%; height:auto; border-radius:6px;}",
        "    .meta{color:#666; font-size:12px; overflow-wrap:anywhere;}",
        "  </style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        f"<p class='meta'>Generated: {_now_utc()}</p>",
        "<div class='grid'>",
    ]

    for it in items:
        if not isinstance(it, dict):
            continue
        overlay = it.get("overlay")
        image = it.get("image")
        dets = it.get("detections")
        if not isinstance(overlay, str) or not overlay:
            continue
        lines.extend(
            [
                "<div class='card'>",
                f"  <img src='{rel(overlay)}' />",
                f"  <div class='meta'>image: {image}</div>",
                f"  <div class='meta'>detections: {dets}</div>",
                "</div>",
            ]
        )

    lines.extend(["</div>", "</body>", "</html>"])
    html_path.write_text("\n".join(lines), encoding="utf-8")


def _predict_images(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = repo_root / input_dir
    if not input_dir.exists():
        raise SystemExit(f"input dir not found: {input_dir}")

    patterns = args.glob if args.glob else ["*.jpg", "*.jpeg", "*.png"]
    images = _iter_images(input_dir, patterns=patterns)
    if args.max_images is not None:
        images = images[: int(args.max_images)]
    if not images:
        raise SystemExit(f"no images matched under: {input_dir}")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    overlays_dir = Path(args.overlays_dir)
    if not overlays_dir.is_absolute():
        overlays_dir = repo_root / overlays_dir

    html_path = None
    if args.html:
        html_path = Path(args.html)
        if not html_path.is_absolute():
            html_path = repo_root / html_path

    with tempfile.TemporaryDirectory(prefix="yolozu_predict_images_") as td:
        tmp_root = Path(td)
        split = "train2017"
        images_dir = tmp_root / "images" / split
        labels_dir = tmp_root / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        mapping: dict[str, str] = {}
        for idx, src in enumerate(images):
            dst = images_dir / f"{idx:06d}_{src.name}"
            try:
                os.symlink(str(src.resolve()), str(dst))
            except Exception:
                # Fallback to copy if symlinks are not permitted.
                dst.write_bytes(src.read_bytes())
            mapping[str(dst)] = str(src.resolve())

        export_args = argparse.Namespace(**vars(args))
        export_args.dataset = str(tmp_root)
        export_args.split = split
        export_args.output = str(out_path)
        export_path = _export_with_backend(
            export_args,
            dataset_override=str(tmp_root),
            dataset_meta=str(input_dir),
        )

        payload = _ensure_wrapper(_load_json(export_path))
        # Rewrite image paths back to the original source paths for portability.
        for entry in payload.get("predictions", []):
            if not isinstance(entry, dict):
                continue
            img = entry.get("image")
            if isinstance(img, str) and img in mapping:
                entry["image"] = mapping[img]
        _write_json(out_path, payload)

    overlays_index = _render_overlays(payload, overlays_dir=overlays_dir, max_images=args.max_images)
    if html_path is not None:
        _write_html_report(html_path=html_path, overlays_index=overlays_index, title=str(args.title))
        print(html_path)
    else:
        print(out_path)
    return 0


def _prepare_keypoints_dataset(args: argparse.Namespace) -> int:
    script = repo_root / "tools" / "prepare_keypoints_dataset.py"
    cmd: list[str] = [
        sys.executable,
        str(script),
        "--source",
        str(args.source),
        "--out",
        str(args.out),
        "--format",
        str(args.format),
    ]

    if bool(getattr(args, "list_formats", False)):
        cmd.append("--list-formats")

    if args.split:
        cmd.extend(["--split", str(args.split)])
    if args.num_keypoints is not None:
        cmd.extend(["--num-keypoints", str(int(args.num_keypoints))])
    if args.keypoint_names:
        cmd.extend(["--keypoint-names", str(args.keypoint_names)])

    if args.annotations:
        cmd.extend(["--annotations", str(args.annotations)])
    if args.images_dir:
        cmd.extend(["--images-dir", str(args.images_dir)])
    if args.out_split:
        cmd.extend(["--out-split", str(args.out_split)])
    if args.min_kps is not None:
        cmd.extend(["--min-kps", str(int(args.min_kps))])
    if args.max_images is not None:
        cmd.extend(["--max-images", str(int(args.max_images))])
    if bool(args.link_images):
        cmd.append("--link-images")
    if args.category_id is not None:
        cmd.extend(["--category-id", str(int(args.category_id))])
    if args.category_name:
        cmd.extend(["--category-name", str(args.category_name)])
    if args.class_id is not None:
        cmd.extend(["--class-id", str(int(args.class_id))])
    if args.cvat_images_dir:
        cmd.extend(["--cvat-images-dir", str(args.cvat_images_dir)])

    out = _subprocess_or_die(cmd)
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    return 0


def _eval_keypoints(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "tools/eval_keypoints.py",
        "--dataset",
        str(args.dataset),
        "--predictions",
        str(args.predictions),
        "--output",
        str(args.output),
        "--iou-threshold",
        str(float(args.iou_threshold)),
        "--pck-threshold",
        str(float(args.pck_threshold)),
        "--min-score",
        str(float(args.min_score)),
        "--per-image-limit",
        str(int(args.per_image_limit)),
        "--max-overlays",
        str(int(args.max_overlays)),
        "--overlay-sort",
        str(args.overlay_sort),
        "--overlay-max-size",
        str(int(args.overlay_max_size)),
        "--kp-radius",
        str(int(args.kp_radius)),
    ]
    if args.split is not None:
        cmd.extend(["--split", str(args.split)])
    if args.max_images is not None:
        cmd.extend(["--max-images", str(int(args.max_images))])
    if args.html is not None:
        cmd.extend(["--html", str(args.html)])
    if args.title is not None:
        cmd.extend(["--title", str(args.title)])
    if args.overlays_dir is not None:
        cmd.extend(["--overlays-dir", str(args.overlays_dir)])
    if bool(args.kp_line):
        cmd.append("--kp-line")
    if bool(getattr(args, "oks", False)):
        cmd.append("--oks")
    oks_sigmas = getattr(args, "oks_sigmas", None)
    if oks_sigmas:
        cmd.extend(["--oks-sigmas", str(oks_sigmas)])
    oks_sigmas_file = getattr(args, "oks_sigmas_file", None)
    if oks_sigmas_file:
        cmd.extend(["--oks-sigmas-file", str(oks_sigmas_file)])
    oks_max_dets = getattr(args, "oks_max_dets", None)
    if oks_max_dets is not None:
        cmd.extend(["--oks-max-dets", str(int(oks_max_dets))])

    out = _subprocess_or_die(cmd)
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    return 0


def _eval_instance_seg(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "tools/eval_instance_segmentation.py",
        "--dataset",
        str(args.dataset),
        "--predictions",
        str(args.predictions),
        "--output",
        str(args.output),
        "--min-score",
        str(float(args.min_score)),
        "--diag-iou",
        str(float(args.diag_iou)),
        "--per-image-limit",
        str(int(args.per_image_limit)),
    ]

    if args.split is not None:
        cmd.extend(["--split", str(args.split)])
    if args.pred_root is not None:
        cmd.extend(["--pred-root", str(args.pred_root)])
    if args.classes is not None:
        cmd.extend(["--classes", str(args.classes)])
    if args.html is not None:
        cmd.extend(["--html", str(args.html)])
    if args.title is not None:
        cmd.extend(["--title", str(args.title)])
    if args.overlays_dir is not None:
        cmd.extend(["--overlays-dir", str(args.overlays_dir)])

    cmd.extend(["--max-overlays", str(int(args.max_overlays))])
    cmd.extend(["--overlay-sort", str(args.overlay_sort)])
    cmd.extend(["--overlay-max-size", str(int(args.overlay_max_size))])
    cmd.extend(["--overlay-alpha", str(float(args.overlay_alpha))])

    if args.max_images is not None:
        cmd.extend(["--max-images", str(int(args.max_images))])
    if args.allow_rgb_masks:
        cmd.append("--allow-rgb-masks")

    out = _subprocess_or_die(cmd)
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    return 0


def _passthrough_pkg_cli(args: argparse.Namespace) -> int:
    from yolozu.cli import main as pkg_main

    cmd = str(getattr(args, "_pkg_cmd"))
    forwarded = getattr(args, "forward_args", None)
    argv = [cmd]
    if isinstance(forwarded, list):
        argv.extend(str(token) for token in forwarded)
    return int(pkg_main(argv))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="yolozu",
        description="YOLOZU unified CLI (P0/P1/P2 building blocks).",
        epilog=(
            "© 2026 ToppyMicroServices OÜ\n"
            "Legal address: Karamelli tn 2, 11317 Tallinn, Harju County, Estonia\n"
            "Registry code: 16551297\n"
            "Contact: develop@toppymicros.com"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_doctor = sub.add_parser("doctor", help="Print environment diagnostics as JSON.")
    p_doctor.add_argument("--output", default="reports/doctor.json", help="Output JSON path.")
    p_doctor.set_defaults(_fn=_doctor)

    p_sweep = sub.add_parser("sweep", help="Run a parameter sweep (wrapper around tools/hpo_sweep.py).")
    p_sweep.add_argument("--config", required=True, help="Path to sweep config JSON.")
    p_sweep.add_argument("--resume", action="store_true", help="Skip runs already present in results jsonl.")
    p_sweep.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    p_sweep.add_argument("--max-runs", type=int, default=None, help="Optional cap for number of runs.")
    p_sweep.set_defaults(_fn=_sweep)

    p_ct = sub.add_parser("continual-train", help="Run continual fine-tuning for rtdetr_pose.")
    p_ct.add_argument("--config", required=True, help="YAML/JSON continual learning config.")
    p_ct.add_argument("--run-dir", default=None, help="Optional run directory (default: runs/continual/<stamp>_rtdetr_pose).")
    p_ct.add_argument("--replay-size", type=int, default=None, help="Override continual.replay_size (0 disables replay).")
    p_ct.add_argument("--replay-fraction", type=float, default=None, help="Override continual.replay_fraction.")
    p_ct.add_argument("--replay-per-task-cap", type=int, default=None, help="Override continual.replay_per_task_cap.")
    p_ct.set_defaults(_fn=_continual_train)

    p_ce = sub.add_parser("continual-eval", help="Evaluate a continual run (simple mAP proxy or pose metrics).")
    p_ce.add_argument("--run-json", required=True, help="Path to runs/.../continual_run.json produced by train_continual.py.")
    p_ce.add_argument("--device", default="cpu", help="Torch device for export (default: cpu).")
    p_ce.add_argument("--image-size", type=int, default=320, help="Adapter image size (square, default: 320).")
    p_ce.add_argument("--max-images", type=int, default=None, help="Optional cap for export/eval.")
    p_ce.add_argument("--metric", choices=("simple_map", "pose"), default="simple_map", help="Metric backend (default: simple_map).")
    p_ce.add_argument("--metric-key", default=None, help="Metric key for CL summaries (default depends on --metric).")
    p_ce.add_argument("--iou-threshold", type=float, default=None, help="Pose matching IoU threshold (default: 0.5).")
    p_ce.add_argument("--min-score", type=float, default=None, help="Pose eval min score (default: 0.0).")
    p_ce.add_argument("--success-rot-deg", type=float, default=None, help="Pose success rotation threshold in degrees (default: 15).")
    p_ce.add_argument("--success-trans", type=float, default=None, help="Pose success translation threshold in meters (default: 0.1).")
    p_ce.add_argument("--keep-per-image", type=int, default=None, help="Keep N per-image summaries (default: 0).")
    p_ce.add_argument("--output", default=None, help="Output JSON path (default: <run_dir>/continual_eval.json).")
    p_ce.add_argument("--html", default=None, help="Optional HTML report path (default: <run_dir>/continual_eval.html).")
    p_ce.add_argument("--force", action="store_true", help="Overwrite existing prediction/eval outputs.")
    p_ce.set_defaults(_fn=_continual_eval)

    p_export = sub.add_parser("export", help="Export predictions JSON via a selected backend.")
    _parse_common_export_args(p_export)
    p_export.set_defaults(_fn=lambda a: (print(_export_with_backend(a)), 0)[1])

    p_pi = sub.add_parser("predict-images", help="Run inference on a folder of images and write overlays/HTML.")
    _parse_common_export_args(p_pi)
    p_pi.add_argument("--input-dir", required=True, help="Folder containing images.")
    p_pi.add_argument("--glob", action="append", default=None, help="Glob pattern under --input-dir (repeatable).")
    p_pi.add_argument("--overlays-dir", default="reports/overlays", help="Directory to write overlay images.")
    p_pi.add_argument("--html", default="reports/predict_images.html", help="Optional HTML report output path.")
    p_pi.add_argument("--title", default="YOLOZU predict-images report", help="HTML title.")
    p_pi.set_defaults(_fn=_predict_images)

    p_pkd = sub.add_parser(
        "prepare-keypoints-dataset",
        help="Prepare keypoints dataset in one command (auto-detect YOLO Pose or COCO keypoints).",
    )
    p_pkd.add_argument("--source", required=True, help="Source path (YOLO Pose root or COCO root).")
    p_pkd.add_argument("--out", required=True, help="Output dataset root.")
    p_pkd.add_argument(
        "--format",
        default="auto",
        help="Input format (supported: auto, yolo_pose, coco, cvat_xml; use --list-formats for matrix).",
    )
    p_pkd.add_argument(
        "--list-formats",
        action="store_true",
        help="Print supported/unsupported formats and conversion routes.",
    )
    p_pkd.add_argument("--split", default=None, help="Split name for YOLO Pose mode (default: auto-detect).")
    p_pkd.add_argument("--num-keypoints", type=int, default=None, help="Optional num_keypoints metadata (YOLO Pose mode).")
    p_pkd.add_argument("--keypoint-names", default=None, help="Optional comma-separated keypoint names (YOLO Pose mode).")
    p_pkd.add_argument("--annotations", default="annotations/person_keypoints_val2017.json", help="COCO mode annotations path.")
    p_pkd.add_argument("--images-dir", default="val2017", help="COCO mode images dir under source root.")
    p_pkd.add_argument("--out-split", default="val2017", help="COCO mode output split (default: val2017).")
    p_pkd.add_argument("--min-kps", type=int, default=1, help="COCO mode minimum labeled keypoints (default: 1).")
    p_pkd.add_argument("--max-images", type=int, default=None, help="COCO mode optional image cap.")
    p_pkd.add_argument("--link-images", action="store_true", help="COCO mode: symlink images into output.")
    p_pkd.add_argument("--category-id", type=int, default=None, help="COCO mode target category id.")
    p_pkd.add_argument("--category-name", default=None, help="COCO mode target category name.")
    p_pkd.add_argument("--class-id", type=int, default=0, help="COCO mode class_id to emit (default: 0).")
    p_pkd.add_argument("--cvat-images-dir", default=None, help="CVAT XML mode images root override.")
    p_pkd.set_defaults(_fn=_prepare_keypoints_dataset)

    p_kp = sub.add_parser("eval-keypoints", help="Evaluate keypoint predictions (PCK + optional OKS mAP) and write a report.")
    p_kp.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    p_kp.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p_kp.add_argument("--predictions", required=True, help="Predictions JSON (detections may include keypoints).")
    p_kp.add_argument("--output", default="reports/keypoints_eval.json", help="Output JSON report path.")
    p_kp.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching (default: 0.5).")
    p_kp.add_argument("--pck-threshold", type=float, default=0.1, help="PCK threshold (default: 0.1).")
    p_kp.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold (default: 0.0).")
    p_kp.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images to evaluate.")
    p_kp.add_argument("--per-image-limit", type=int, default=100, help="Per-image rows stored in report/HTML (default: 100).")
    p_kp.add_argument("--html", default=None, help="Optional HTML report path.")
    p_kp.add_argument("--title", default="YOLOZU keypoints eval report", help="HTML title.")
    p_kp.add_argument("--overlays-dir", default=None, help="Optional directory to write overlay images for HTML.")
    p_kp.add_argument("--max-overlays", type=int, default=0, help="Max overlays to render (default: 0).")
    p_kp.add_argument(
        "--overlay-sort",
        choices=("worst", "best", "first"),
        default="worst",
        help="How to select overlay samples (default: worst).",
    )
    p_kp.add_argument("--overlay-max-size", type=int, default=768, help="Max size (max(H,W)) for overlay images (default: 768).")
    p_kp.add_argument("--kp-radius", type=int, default=3, help="Keypoint marker radius (default: 3).")
    p_kp.add_argument("--kp-line", action="store_true", help="Draw gt→pred keypoint error lines.")
    p_kp.add_argument("--oks", action="store_true", help="Also compute COCO OKS mAP (requires pycocotools).")
    p_kp.add_argument("--oks-sigmas", default=None, help="OKS sigmas: 'coco17' or comma-separated floats (len=K).")
    p_kp.add_argument("--oks-sigmas-file", default=None, help="JSON file containing list[float] sigmas (len=K).")
    p_kp.add_argument("--oks-max-dets", type=int, default=20, help="COCOeval maxDets for keypoints (default: 20).")
    p_kp.set_defaults(_fn=_eval_keypoints)

    p_is = sub.add_parser("eval-instance-seg", help="Evaluate instance segmentation predictions (PNG masks) and write a report.")
    p_is.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    p_is.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p_is.add_argument("--predictions", required=True, help="Instance segmentation predictions JSON.")
    p_is.add_argument("--pred-root", default=None, help="Optional root to resolve relative prediction mask paths.")
    p_is.add_argument("--classes", default=None, help="Optional classes.txt/classes.json for class_id→name.")
    p_is.add_argument("--output", default="reports/instance_seg_eval.json", help="Output JSON report path.")
    p_is.add_argument("--html", default=None, help="Optional HTML report path.")
    p_is.add_argument("--title", default="YOLOZU instance segmentation eval report", help="HTML title.")
    p_is.add_argument("--overlays-dir", default=None, help="Optional directory to write overlay images for HTML.")
    p_is.add_argument("--max-overlays", type=int, default=0, help="Max overlays to render (default: 0).")
    p_is.add_argument(
        "--overlay-sort",
        choices=("worst", "best", "first"),
        default="worst",
        help="How to select overlay samples (default: worst).",
    )
    p_is.add_argument("--overlay-max-size", type=int, default=768, help="Max size (max(H,W)) for overlay images (default: 768).")
    p_is.add_argument("--overlay-alpha", type=float, default=0.5, help="Mask overlay alpha (default: 0.5).")
    p_is.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold for predictions (default: 0.0).")
    p_is.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images to evaluate.")
    p_is.add_argument("--diag-iou", type=float, default=0.5, help="IoU threshold used for per-image diagnostics/overlay selection (default: 0.5).")
    p_is.add_argument("--per-image-limit", type=int, default=100, help="How many per-image rows to store in the report/meta and HTML (default: 100).")
    p_is.add_argument(
        "--allow-rgb-masks",
        action="store_true",
        help="Allow 3-channel masks (uses channel 0; intended for grayscale stored as RGB).",
    )
    p_is.set_defaults(_fn=_eval_instance_seg)

    p_cal = sub.add_parser("calibrate", help="Delegate to yolozu package CLI calibrate command.")
    p_cal.add_argument("forward_args", nargs=argparse.REMAINDER, help="Arguments forwarded to `yolozu calibrate`.")
    p_cal.set_defaults(_fn=_passthrough_pkg_cli, _pkg_cmd="calibrate")

    p_elt = sub.add_parser("eval-long-tail", help="Delegate to yolozu package CLI eval-long-tail command.")
    p_elt.add_argument("forward_args", nargs=argparse.REMAINDER, help="Arguments forwarded to `yolozu eval-long-tail`.")
    p_elt.set_defaults(_fn=_passthrough_pkg_cli, _pkg_cmd="eval-long-tail")

    p_ltr = sub.add_parser("long-tail-recipe", help="Delegate to yolozu package CLI long-tail-recipe command.")
    p_ltr.add_argument("forward_args", nargs=argparse.REMAINDER, help="Arguments forwarded to `yolozu long-tail-recipe`.")
    p_ltr.set_defaults(_fn=_passthrough_pkg_cli, _pkg_cmd="long-tail-recipe")

    p_reg = sub.add_parser("registry", help="AI-first tool registry: list/show/validate/run tools from tools/manifest.json.")
    reg = p_reg.add_subparsers(dest="registry_cmd", required=True)

    p_reg_validate = reg.add_parser("validate", help="Validate tools/manifest.json references.")
    p_reg_validate.set_defaults(_fn=_registry_validate)

    p_reg_list = reg.add_parser("list", help="List tools in the manifest (text or JSON).")
    p_reg_list.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    p_reg_list.add_argument("--tag", action="append", default=None, help="Filter by tag (repeatable, AND).")
    p_reg_list.add_argument("--contract", action="append", default=None, help="Filter by contract id (repeatable, AND).")
    p_reg_list.set_defaults(_fn=_registry_list)

    p_reg_show = reg.add_parser("show", help="Show a single tool spec (text or JSON).")
    p_reg_show.add_argument("id", help="Tool id from the manifest.")
    p_reg_show.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    p_reg_show.set_defaults(_fn=_registry_show)

    p_reg_run = reg.add_parser("run", help="Safely run a tool by id with allowlisted side effects.")
    p_reg_run.add_argument("id", help="Tool id from the manifest.")
    p_reg_run.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing.")
    p_reg_run.add_argument("--allow-network", action="store_true", help="Allow tools that require network access.")
    p_reg_run.add_argument("--allow-gpu", action="store_true", help="Allow tools that require GPU.")
    p_reg_run.add_argument(
        "--allow-write-root",
        action="append",
        default=None,
        help="Allow writing under this repo-relative root (repeatable). Default: reports",
    )
    p_reg_run.add_argument("--allow-unsafe-paths", action="store_true", help="Allow absolute paths or '..' segments.")
    p_reg_run.add_argument(
        "--allow-unknown-flags",
        action="store_true",
        help="Allow forwarding flags not declared in tool.inputs (not recommended for agents).",
    )
    p_reg_run.add_argument(
        "--allow-undeclared-effects",
        action="store_true",
        help="Allow running tools without tool.effects declarations (not recommended for agents).",
    )
    p_reg_run.add_argument("forward_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the tool entrypoint.")
    p_reg_run.set_defaults(_fn=_registry_run)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    if raw_argv and raw_argv[0] in {"calibrate", "eval-long-tail", "long-tail-recipe"}:
        from yolozu.cli import main as pkg_main

        return int(pkg_main(raw_argv))

    args = _parse_args(raw_argv)
    fn = getattr(args, "_fn", None)
    if fn is None:
        raise SystemExit("missing handler")
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())

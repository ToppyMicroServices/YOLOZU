#!/usr/bin/env python3
"""Legacy compatibility wrapper around the canonical ``yolozu`` CLI.

The supported top-level entrypoint is ``yolozu`` (or ``python3 -m yolozu``).
This repo-local script remains for backwards compatibility in existing checkouts
and forwards overlapping commands while still exposing a few repo-only helper
wrappers.
"""

from __future__ import annotations

import argparse
import json
import logging
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

from yolozu.inference.export_orchestrator import (  # noqa: E402
    ensure_wrapper as _ensure_wrapper,
    export_with_backend as _run_export_with_backend,
    load_json as _load_json,
    parse_common_export_args as _parse_common_export_args,
    sha256_json as _sha256_json,
    write_json as _write_json,
)

logger = logging.getLogger(__name__)


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


def _subprocess_or_die(cmd: list[str]) -> str:
    if len(cmd) >= 2:
        candidate = Path(str(cmd[1]))
        if candidate.suffix == ".py":
            script_path = candidate if candidate.is_absolute() else (repo_root / candidate)
            if not script_path.is_file():
                raise SystemExit(f"required script not found: {candidate}")
    proc = subprocess.run(cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.stderr and proc.stderr.strip():
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def _export_with_backend(
    args: argparse.Namespace,
    *,
    dataset_override: str | None = None,
    dataset_meta: str | None = None,
) -> Path:
    return _run_export_with_backend(
        args,
        subprocess_or_die=_subprocess_or_die,
        base_run_meta=_base_run_meta,
        dataset_override=dataset_override,
        dataset_meta=dataset_meta,
    )


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


def _continual_decide(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "tools/continual_decide.py", "--eval-json", str(args.eval_json)]
    if args.curation_json:
        cmd.extend(["--curation-json", str(args.curation_json)])
    if args.run_json:
        cmd.extend(["--run-json", str(args.run_json)])
    if args.max_forgetting is not None:
        cmd.extend(["--max-forgetting", str(float(args.max_forgetting))])
    if args.min_avg_acc is not None:
        cmd.extend(["--min-avg-acc", str(float(args.min_avg_acc))])
    if args.min_new_task_score is not None:
        cmd.extend(["--min-new-task-score", str(float(args.min_new_task_score))])
    if args.min_old_task_final is not None:
        cmd.extend(["--min-old-task-final", str(float(args.min_old_task_final))])
    if args.min_reviewed_labels is not None:
        cmd.extend(["--min-reviewed-labels", str(int(args.min_reviewed_labels))])
    if args.min_highconf_pseudo_labels is not None:
        cmd.extend(["--min-highconf-pseudo-labels", str(int(args.min_highconf_pseudo_labels))])
    if args.min_total_curated_examples is not None:
        cmd.extend(["--min-total-curated-examples", str(int(args.min_total_curated_examples))])
    if args.max_candidate_share is not None:
        cmd.extend(["--max-candidate-share", str(float(args.max_candidate_share))])
    if args.ttt_active:
        cmd.append("--ttt-active")
    if args.allow_ttt_active_promotion:
        cmd.append("--allow-ttt-active-promotion")
    if args.output:
        cmd.extend(["--output", str(args.output)])
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
                            except Exception as exc:
                                logger.debug("keypoint visibility coercion skipped: %s", exc, exc_info=True)
                        draw.ellipse([px - r, py - r, px + r, py + r], outline=(0, 0, 255), width=2)
                except Exception as exc:
                    logger.debug("keypoint overlay rendering skipped: %s", exc, exc_info=True)

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

    patterns = args.glob if args.glob else ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.gif"]
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

    forwarded = getattr(args, "forward_args", None)
    prefix = getattr(args, "_pkg_argv", None)
    if isinstance(prefix, list) and prefix:
        argv = [str(token) for token in prefix]
    else:
        cmd = str(getattr(args, "_pkg_cmd"))
        argv = [cmd]
    if isinstance(forwarded, list):
        argv.extend(str(token) for token in forwarded)
    return int(pkg_main(argv))


def _passthrough_list_models(args: argparse.Namespace) -> int:
    from yolozu.cli import main as pkg_main

    argv = ["list", "models"]
    if getattr(args, "registry", None):
        argv.extend(["--registry", str(args.registry)])
    if bool(getattr(args, "json", False)):
        argv.append("--json")
    return int(pkg_main(argv))


def _completion(args: argparse.Namespace) -> int:
    from yolozu.cli import main as pkg_main

    argv = ["completion", "--shell", str(args.shell), "--command", str(args.command), "--output", str(args.output)]
    return int(pkg_main(argv))


def _release(_: argparse.Namespace) -> int:
    cmd = ["bash", "release.sh"]
    out = _subprocess_or_die(cmd)
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    return 0


def _support_ultralytics_detr(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "tools/support_external_training.py"]
    forwarded = getattr(args, "forward_args", None)
    if isinstance(forwarded, list):
        cmd.extend(str(x) for x in forwarded)
    out = _subprocess_or_die(cmd)
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    return 0


def _support_external_training(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "tools/support_external_training.py"]
    forwarded = getattr(args, "forward_args", None)
    if isinstance(forwarded, list):
        cmd.extend(str(x) for x in forwarded)
    out = _subprocess_or_die(cmd)
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="yolozu",
        description="Legacy compatibility wrapper around the canonical `yolozu` CLI.",
        epilog=(
            "© 2026 ToppyMicroServices OÜ\n"
            "Legal address: Karamelli tn 2, 11317 Tallinn, Harju County, Estonia\n"
            "Registry code: 16551297\n"
            "Contact: develop@toppymicros.com"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_doctor = sub.add_parser("doctor", aliases=["dr"], help="Print environment diagnostics as JSON.")
    p_doctor.add_argument("-o", "--output", default="reports/doctor.json", help="Output JSON path.")
    p_doctor.set_defaults(_fn=_doctor)

    p_sweep = sub.add_parser("sweep", aliases=["sw"], help="Run a parameter sweep (wrapper around tools/hpo_sweep.py).")
    p_sweep.add_argument("-c", "--config", required=True, help="Path to sweep config JSON.")
    p_sweep.add_argument("-r", "--resume", action="store_true", help="Skip runs already present in results jsonl.")
    p_sweep.add_argument("-n", "--dry-run", action="store_true", help="Print commands without executing.")
    p_sweep.add_argument("-m", "--max-runs", type=int, default=None, help="Optional cap for number of runs.")
    p_sweep.set_defaults(_fn=_sweep)

    p_ct = sub.add_parser("continual-train", aliases=["ct"], help="Run continual fine-tuning for rtdetr_pose.")
    p_ct.add_argument("-c", "--config", required=True, help="YAML/JSON continual learning config.")
    p_ct.add_argument("-r", "--run-dir", default=None, help="Optional run directory (default: runs/continual/<stamp>_rtdetr_pose).")
    p_ct.add_argument("--replay-size", type=int, default=None, help="Override continual.replay_size (0 disables replay).")
    p_ct.add_argument("--replay-fraction", type=float, default=None, help="Override continual.replay_fraction.")
    p_ct.add_argument("--replay-per-task-cap", type=int, default=None, help="Override continual.replay_per_task_cap.")
    p_ct.set_defaults(_fn=_continual_train)

    p_ce = sub.add_parser("continual-eval", aliases=["ce"], help="Evaluate a continual run (simple mAP proxy or pose metrics).")
    p_ce.add_argument("-r", "--run-json", required=True, help="Path to runs/.../continual_run.json produced by train_continual.py.")
    p_ce.add_argument("-d", "--device", default="cpu", help="Torch device for export (default: cpu).")
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

    p_cd = sub.add_parser("continual-decide", aliases=["cd"], help="Decide whether a continual-learning candidate should be promoted, reviewed, or held.")
    p_cd.add_argument("--eval-json", required=True, help="Path to continual_eval.json produced by tools/eval_continual.py.")
    p_cd.add_argument("--curation-json", default=None, help="Optional curation summary JSON with reviewed/pseudo-label counts.")
    p_cd.add_argument("--run-json", default=None, help="Optional continual_run.json path for provenance.")
    p_cd.add_argument("--max-forgetting", type=float, default=0.05, help="Hard gate: forgetting must be <= this value.")
    p_cd.add_argument("--min-avg-acc", type=float, default=None, help="Optional hard gate: avg_acc must be >= this value.")
    p_cd.add_argument("--min-new-task-score", type=float, default=None, help="Optional hard gate: newest task score must be >= this value.")
    p_cd.add_argument("--min-old-task-final", type=float, default=None, help="Optional hard gate: minimum retained score on previous tasks must be >= this value.")
    p_cd.add_argument("--min-reviewed-labels", type=int, default=0, help="Soft gate: reviewed label count should be >= this value.")
    p_cd.add_argument("--min-highconf-pseudo-labels", type=int, default=0, help="Soft gate: trusted pseudo-label count should be >= this value.")
    p_cd.add_argument("--min-total-curated-examples", type=int, default=0, help="Soft gate: reviewed + trusted pseudo-label count should be >= this value.")
    p_cd.add_argument("--max-candidate-share", type=float, default=None, help="Soft gate: candidate_images / samples_total should be <= this value.")
    p_cd.add_argument("--ttt-active", action="store_true", help="Mark that TTT was active; by default this forces review.")
    p_cd.add_argument("--allow-ttt-active-promotion", action="store_true", help="Allow automatic promotion even when --ttt-active is set.")
    p_cd.add_argument("--output", default=None, help="Output JSON path (default: sibling of --eval-json named continual_promotion_decision.json).")
    p_cd.set_defaults(_fn=_continual_decide)

    p_export = sub.add_parser("export", help="Export predictions JSON via a selected backend.")
    _parse_common_export_args(p_export)
    p_export.set_defaults(_fn=lambda a: (print(_export_with_backend(a)), 0)[1])

    p_pi = sub.add_parser("predict-images", aliases=["pi"], help="Run inference on a folder of images and write overlays/HTML.")
    _parse_common_export_args(p_pi)
    p_pi.add_argument("-i", "--input-dir", required=True, help="Folder containing images.")
    p_pi.add_argument("--glob", action="append", default=None, help="Glob pattern under --input-dir (repeatable).")
    p_pi.add_argument("-v", "--overlays-dir", default="reports/overlays", help="Directory to write overlay images.")
    p_pi.add_argument("-H", "--html", default="reports/predict_images.html", help="Optional HTML report output path.")
    p_pi.add_argument("--title", default="YOLOZU predict-images report", help="HTML title.")
    p_pi.set_defaults(_fn=_predict_images)

    p_pkd = sub.add_parser(
        "prepare-keypoints-dataset",
        aliases=["pkd"],
        help="Prepare keypoints dataset in one command (auto-detect YOLO Pose or COCO keypoints).",
    )
    p_pkd.add_argument("-s", "--source", required=True, help="Source path (YOLO Pose root or COCO root).")
    p_pkd.add_argument("-o", "--out", required=True, help="Output dataset root.")
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

    p_kp = sub.add_parser("eval-keypoints", aliases=["ek"], help="Evaluate keypoint predictions (PCK + optional OKS mAP) and write a report.")
    p_kp.add_argument("-d", "--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    p_kp.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p_kp.add_argument("-p", "--predictions", required=True, help="Predictions JSON (detections may include keypoints).")
    p_kp.add_argument("-o", "--output", default="reports/keypoints_eval.json", help="Output JSON report path.")
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

    p_is = sub.add_parser("eval-instance-seg", aliases=["eis"], help="Evaluate instance segmentation predictions (PNG masks) and write a report.")
    p_is.add_argument("-d", "--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    p_is.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p_is.add_argument("-p", "--predictions", required=True, help="Instance segmentation predictions JSON.")
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

    p_demo = sub.add_parser("demo", help="Delegate to yolozu package CLI demo command.")
    p_demo.add_argument("forward_args", nargs=argparse.REMAINDER, help="Arguments forwarded to `yolozu demo`.")
    p_demo.set_defaults(_fn=_passthrough_pkg_cli, _pkg_cmd="demo")

    p_export_dataset = sub.add_parser("export-dataset", help="Delegate to yolozu package CLI export-dataset command.")
    p_export_dataset.add_argument("forward_args", nargs=argparse.REMAINDER, help="Arguments forwarded to `yolozu export-dataset`.")
    p_export_dataset.set_defaults(_fn=_passthrough_pkg_cli, _pkg_cmd="export-dataset")

    p_completion = sub.add_parser("completion", aliases=["comp"], help="Print shell completion script (bash/zsh).")
    p_completion.add_argument("-s", "--shell", choices=("bash", "zsh"), default="bash", help="Target shell (default: bash).")
    p_completion.add_argument("-c", "--command", default="yolozu", help="Command name to bind completion to (default: yolozu).")
    p_completion.add_argument("-o", "--output", default="-", help="Output path (default: stdout).")
    p_completion.set_defaults(_fn=_completion)

    p_release = sub.add_parser("release", help="Run single-command release automation (tag + GitHub/PyPI/Zenodo flow).")
    p_release.set_defaults(_fn=_release)

    p_support_ext = sub.add_parser(
        "support-external-training",
        aliases=["set", "external-training"],
        help="External training support wrapper (Apache-2.0 YOLOX lane + optional bridges).",
    )
    p_support_ext.add_argument(
        "forward_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to tools/support_external_training.py.",
    )
    p_support_ext.set_defaults(_fn=_support_external_training)

    p_support_ud = sub.add_parser(
        "support-ultralytics-detr",
        aliases=["sud", "ud"],
        help="Legacy alias for the external training support wrapper.",
    )
    p_support_ud.add_argument(
        "forward_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to tools/support_external_training.py.",
    )
    p_support_ud.set_defaults(_fn=_support_ultralytics_detr)

    p_fetch = sub.add_parser("fetch", help="Delegate to yolozu package CLI fetch command.")
    p_fetch.add_argument("forward_args", nargs=argparse.REMAINDER, help="Arguments forwarded to `yolozu fetch`.")
    p_fetch.set_defaults(_fn=_passthrough_pkg_cli, _pkg_cmd="fetch")

    p_list = sub.add_parser("list", help="Delegate to yolozu package CLI list command.")
    list_sub = p_list.add_subparsers(dest="list_cmd", required=True)
    p_list_models = list_sub.add_parser("models", help="Delegate to `yolozu list models`.")
    p_list_models.add_argument("--registry", default=None, help="Optional registry JSON path override.")
    p_list_models.add_argument("-j", "--json", action="store_true", help="Emit JSON output.")
    p_list_models.set_defaults(_fn=_passthrough_list_models)

    p_reg = sub.add_parser("registry", help="AI-first tool registry: list/show/validate/run tools from tools/manifest.json.")
    reg = p_reg.add_subparsers(dest="registry_cmd", required=True)

    p_reg_validate = reg.add_parser("validate", help="Validate tools/manifest.json references.")
    p_reg_validate.set_defaults(_fn=_registry_validate)

    p_reg_list = reg.add_parser("list", help="List tools in the manifest (text or JSON).")
    p_reg_list.add_argument("-j", "--json", action="store_true", help="Emit machine-readable JSON.")
    p_reg_list.add_argument("--tag", action="append", default=None, help="Filter by tag (repeatable, AND).")
    p_reg_list.add_argument("--contract", action="append", default=None, help="Filter by contract id (repeatable, AND).")
    p_reg_list.set_defaults(_fn=_registry_list)

    p_reg_show = reg.add_parser("show", help="Show a single tool spec (text or JSON).")
    p_reg_show.add_argument("id", help="Tool id from the manifest.")
    p_reg_show.add_argument("-j", "--json", action="store_true", help="Emit machine-readable JSON.")
    p_reg_show.set_defaults(_fn=_registry_show)

    p_reg_run = reg.add_parser("run", help="Safely run a tool by id with allowlisted side effects.")
    p_reg_run.add_argument("id", help="Tool id from the manifest.")
    p_reg_run.add_argument("-n", "--dry-run", action="store_true", help="Print the resolved command without executing.")
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
    if raw_argv and raw_argv[0] in {"calibrate", "eval-long-tail", "long-tail-recipe", "export-dataset"}:
        from yolozu.cli import main as pkg_main

        return int(pkg_main(raw_argv))

    args = _parse_args(raw_argv)
    fn = getattr(args, "_fn", None)
    if fn is None:
        raise SystemExit("missing handler")
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""YOLO/DETR integration support tool.

Provides three-layer support helpers:
1) trainer/runner,
2) repo integration wrappers,
3) export/deploy (ONNX + optional TensorRT handoff).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.integrations.min_adapter import (  # noqa: E402
    canonicalize_predictions_file,
    layered_support_matrix,
    resolve_internal_dataset,
    write_ultralytics_data_yaml,
)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _run(cmd: list[str], *, cwd: Path = repo_root) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)


_PRESETS: dict[str, dict[str, str]] = {
    "smoke": {
        "from_format": "internal",
        "dataset": "data/smoke",
        "split": "val",
        "ultralytics_model": "yolo11n.pt",
        "hf_model_id": "facebook/detr-resnet-50",
        "provider": "ultralytics",
        "onnx_output": "models/yolo11n.onnx",
    },
    "coco128": {
        "from_format": "auto",
        "dataset": "data/coco128",
        "split": "val",
        "ultralytics_model": "yolo11n.pt",
        "hf_model_id": "facebook/detr-resnet-50",
        "provider": "ultralytics",
        "onnx_output": "models/yolo11n_coco128.onnx",
    },
}


def _default_preset_name() -> str:
    raw = str(os.environ.get("YOLOZU_CLI_PRESET", "smoke") or "smoke").strip().lower()
    if not raw:
        return "smoke"
    return raw


def _resolve_preset(name: str | None) -> tuple[str, dict[str, str]]:
    key = str(name or _default_preset_name()).strip().lower()
    if key in {"none", "off", "custom"}:
        return "none", {}
    if key not in _PRESETS:
        allowed = ", ".join(sorted(list(_PRESETS.keys()) + ["none"]))
        raise SystemExit(f"unknown --preset '{key}'. expected one of: {allowed}")
    return key, dict(_PRESETS[key])


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return value
            continue
        return value
    return None


def _resolve_value(
    cli_value: Any,
    *,
    env: str | None = None,
    preset: dict[str, str] | None = None,
    preset_key: str | None = None,
    fallback: Any = None,
) -> Any:
    env_value = None
    if env:
        env_value = os.environ.get(env)
    preset_value = None
    if preset and preset_key:
        preset_value = preset.get(preset_key)
    return _first_present(cli_value, env_value, preset_value, fallback)


def _require_nonempty(value: Any, *, message: str) -> str:
    out = _resolve_value(value)
    if out is None:
        raise SystemExit(message)
    s = str(out).strip()
    if not s:
        raise SystemExit(message)
    return s


def _preset_choices() -> tuple[str, ...]:
    return tuple(sorted(list(_PRESETS.keys()) + ["none"]))


def _cmd_layers(args: argparse.Namespace) -> int:
    matrix = layered_support_matrix()
    if bool(args.json):
        print(json.dumps(matrix, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    for layer, desc in (matrix.get("layers") or {}).items():
        print(f"[{layer}] {desc}")
    providers = matrix.get("providers") or {}
    for provider, spec in providers.items():
        print(f"\n{provider}:")
        for layer in ("trainer_runner", "repo_impl", "export_deploy"):
            print(f"  - {layer}: {spec.get(layer)}")
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    preset_name, preset = _resolve_preset(getattr(args, "preset", None))
    source_format = str(
        _resolve_value(
            args.from_format,
            env="YOLOZU_FROM_FORMAT",
            preset=preset,
            preset_key="from_format",
            fallback="auto",
        )
    )
    dataset = _resolve_value(
        args.dataset,
        env="YOLOZU_DATASET",
        preset=preset,
        preset_key="dataset",
    )
    split = _resolve_value(
        args.split,
        env="YOLOZU_SPLIT",
        preset=preset,
        preset_key="split",
    )
    output = _resolve_value(args.output, fallback="runs/support_ultralytics_detr/dataset")
    report_out = _resolve_value(args.report, fallback="reports/support_ultralytics_detr.dataset.json")
    resolution = resolve_internal_dataset(
        source_format=source_format,
        dataset=str(dataset) if dataset else None,
        split=str(split) if split else None,
        output=str(output),
        instances_json=str(args.instances_json) if args.instances_json else None,
        images_dir=str(args.images_dir) if args.images_dir else None,
        force=bool(args.force),
    )
    report = {
        "task": "dataset_convert",
        "timestamp": _now_utc(),
        "ok": True,
        "source_format": resolution.source_format,
        "dataset_root": str(resolution.dataset_root),
        "split": resolution.split,
        "preset": preset_name,
        "dataset_wrapper": str(resolution.dataset_wrapper) if resolution.dataset_wrapper else None,
        "notes": list(resolution.notes),
        "layers": {
            "trainer_runner": "dataset conversion pre-step",
            "repo_impl": "resolve_internal_dataset",
            "export_deploy": "compatible with ONNX/TensorRT export wrappers",
        },
    }
    report_path = Path(str(report_out)).resolve()
    _write_json(report_path, report)
    print(str(report_path))
    return 0


def _ultralytics_train_template(
    *,
    model: str,
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    name: str,
) -> str:
    return (
        "yolo train "
        f"model={model} data={data_yaml} epochs={int(epochs)} imgsz={int(imgsz)} "
        f"batch={int(batch)} device={device} project={project} name={name}"
    )


def _cmd_train_ultralytics(args: argparse.Namespace) -> int:
    preset_name, preset = _resolve_preset(getattr(args, "preset", None))
    model_name = _require_nonempty(
        _resolve_value(
            args.model,
            env="YOLOZU_MODEL",
            preset=preset,
            preset_key="ultralytics_model",
        ),
        message="model is required (set --model/-m, YOLOZU_MODEL, or --preset).",
    )
    dataset_input = _require_nonempty(
        _resolve_value(
            args.dataset,
            env="YOLOZU_DATASET",
            preset=preset,
            preset_key="dataset",
        ),
        message="dataset is required (set --dataset/-d, YOLOZU_DATASET, or --preset).",
    )
    source_format = str(
        _resolve_value(
            args.from_format,
            env="YOLOZU_FROM_FORMAT",
            preset=preset,
            preset_key="from_format",
            fallback="auto",
        )
    )
    split = str(
        _resolve_value(
            args.split,
            env="YOLOZU_SPLIT",
            preset=preset,
            preset_key="split",
            fallback="train",
        )
    )
    work_dir = Path(str(_resolve_value(args.work_dir, fallback="runs/support_ultralytics_detr/ultralytics"))).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    resolution = resolve_internal_dataset(
        source_format=source_format,
        dataset=dataset_input,
        split=split,
        output=work_dir / "dataset",
        instances_json=str(args.instances_json) if args.instances_json else None,
        images_dir=str(args.images_dir) if args.images_dir else None,
        force=bool(args.force),
    )

    dataset_path = Path(dataset_input).resolve()
    if dataset_path.is_file() and dataset_path.suffix.lower() in (".yaml", ".yml") and resolution.source_format == "ultralytics":
        data_yaml = dataset_path
    else:
        data_yaml = write_ultralytics_data_yaml(
            dataset_root=resolution.dataset_root,
            split=resolution.split,
            output=work_dir / "ultralytics_data.yaml",
        )

    template = _ultralytics_train_template(
        model=model_name,
        data_yaml=data_yaml,
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        project=str(args.project),
        name=str(args.name),
    )

    training_executed = False
    run_dir: str | None = None
    runtime_error: str | None = None

    if not bool(args.dry_run):
        try:
            from ultralytics import YOLO  # type: ignore

            model = YOLO(model_name)
            result = model.train(
                data=str(data_yaml),
                epochs=int(args.epochs),
                imgsz=int(args.imgsz),
                batch=int(args.batch),
                device=str(args.device),
                project=str(args.project),
                name=str(args.name),
                workers=int(args.workers),
            )
            run_dir = str(getattr(result, "save_dir", "") or "")
            training_executed = True
        except Exception as exc:
            runtime_error = str(exc)

    ok = bool(args.dry_run) or training_executed
    report = {
        "task": "train_ultralytics",
        "timestamp": _now_utc(),
        "ok": ok,
        "dry_run": bool(args.dry_run),
        "model": model_name,
        "dataset_root": str(resolution.dataset_root),
        "split": resolution.split,
        "preset": preset_name,
        "data_yaml": str(data_yaml),
        "template_train_command": template,
        "template_predict_normalize_command": (
            "python3 tools/support_ultralytics_detr.py predict-normalize "
            f"--ultralytics-model {model_name} --dataset {resolution.dataset_root} "
            f"--split {resolution.split} --output reports/ultralytics_predictions.normalized.json "
            "--report reports/ultralytics_predict_normalize_report.json"
        ),
        "training_executed": training_executed,
        "run_dir": run_dir,
        "runtime_error": runtime_error,
        "layers": {
            "trainer_runner": "ultralytics.YOLO.train",
            "repo_impl": "support_ultralytics_detr train-ultralytics",
            "export_deploy": "support_ultralytics_detr export-onnx --provider ultralytics",
        },
    }
    report_path = Path(str(_resolve_value(args.output, fallback="reports/support_ultralytics_detr.train_ultralytics.json"))).resolve()
    _write_json(report_path, report)
    print(str(report_path))
    return 0 if ok else 1


def _cmd_train_hf_detr(args: argparse.Namespace) -> int:
    preset_name, preset = _resolve_preset(getattr(args, "preset", None))
    model_id = _require_nonempty(
        _resolve_value(
            args.model_id,
            env="YOLOZU_HF_MODEL_ID",
            preset=preset,
            preset_key="hf_model_id",
            fallback="facebook/detr-resnet-50",
        ),
        message="model id resolution failed for HF DETR entry.",
    )
    dataset_input = _require_nonempty(
        _resolve_value(
            args.dataset,
            env="YOLOZU_DATASET",
            preset=preset,
            preset_key="dataset",
        ),
        message="dataset is required (set --dataset/-d, YOLOZU_DATASET, or --preset).",
    )
    source_format = str(
        _resolve_value(
            args.from_format,
            env="YOLOZU_FROM_FORMAT",
            preset=preset,
            preset_key="from_format",
            fallback="auto",
        )
    )
    split = str(
        _resolve_value(
            args.split,
            env="YOLOZU_SPLIT",
            preset=preset,
            preset_key="split",
            fallback="train",
        )
    )
    work_dir = Path(str(_resolve_value(args.work_dir, fallback="runs/support_ultralytics_detr/hf_detr"))).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    resolution = resolve_internal_dataset(
        source_format=source_format,
        dataset=dataset_input,
        split=split,
        output=work_dir / "dataset",
        instances_json=str(args.instances_json) if args.instances_json else None,
        images_dir=str(args.images_dir) if args.images_dir else None,
        force=bool(args.force),
    )

    train_script = str(args.train_script).strip() if args.train_script else ""
    if train_script:
        command = [
            str(args.python),
            train_script,
            "--model-id",
            model_id,
            "--dataset",
            str(resolution.dataset_root),
            "--split",
            str(resolution.split),
            "--epochs",
            str(int(args.epochs)),
            "--batch-size",
            str(int(args.batch_size)),
            "--learning-rate",
            str(float(args.learning_rate)),
            "--max-steps",
            str(int(args.max_steps)),
        ]
    else:
        command = [
            "accelerate",
            "launch",
            "<hf_detr_train_script.py>",
            "--model-id",
            model_id,
            "--dataset",
            str(resolution.dataset_root),
            "--split",
            str(resolution.split),
            "--epochs",
            str(int(args.epochs)),
            "--batch-size",
            str(int(args.batch_size)),
            "--learning-rate",
            str(float(args.learning_rate)),
            "--max-steps",
            str(int(args.max_steps)),
        ]

    training_executed = False
    runtime_error: str | None = None
    proc_info: dict[str, Any] | None = None

    if not bool(args.dry_run):
        if train_script:
            proc = _run(command, cwd=repo_root)
            proc_info = {
                "returncode": int(proc.returncode),
                "stdout_tail": str(proc.stdout or "").splitlines()[-20:],
                "stderr_tail": str(proc.stderr or "").splitlines()[-20:],
            }
            training_executed = proc.returncode == 0
            if proc.returncode != 0:
                runtime_error = f"external HF train script failed ({proc.returncode})"
        else:
            runtime_error = (
                "HF DETR entry requires --train-script for non-dry execution. "
                "Use --dry-run to get template command or provide a script."
            )

    ok = bool(args.dry_run) or training_executed
    report = {
        "task": "train_hf_detr",
        "timestamp": _now_utc(),
        "ok": ok,
        "dry_run": bool(args.dry_run),
        "model_id": model_id,
        "dataset_root": str(resolution.dataset_root),
        "split": resolution.split,
        "preset": preset_name,
        "train_script": train_script or None,
        "template_train_command": " ".join(command),
        "training_executed": training_executed,
        "runtime_error": runtime_error,
        "process": proc_info,
        "layers": {
            "trainer_runner": "transformers/accelerate entry script",
            "repo_impl": "support_ultralytics_detr train-hf-detr",
            "export_deploy": "support_ultralytics_detr export-onnx --provider hf_detr",
        },
    }
    report_path = Path(str(_resolve_value(args.output, fallback="reports/support_ultralytics_detr.train_hf_detr.json"))).resolve()
    _write_json(report_path, report)
    print(str(report_path))
    return 0 if ok else 1


def _cmd_export_onnx(args: argparse.Namespace) -> int:
    preset_name, preset = _resolve_preset(getattr(args, "preset", None))
    provider = str(
        _resolve_value(
            args.provider,
            env="YOLOZU_PROVIDER",
            preset=preset,
            preset_key="provider",
            fallback="ultralytics",
        )
    )
    default_model_key = "ultralytics_model" if provider == "ultralytics" else "hf_model_id"
    model_name = _require_nonempty(
        _resolve_value(
            args.model,
            env="YOLOZU_MODEL",
            preset=preset,
            preset_key=default_model_key,
        ),
        message="model is required for export (set --model/-m, YOLOZU_MODEL, or --preset).",
    )
    output_path = _resolve_value(
        args.output,
        env="YOLOZU_ONNX_OUTPUT",
        preset=preset,
        preset_key="onnx_output",
        fallback="models/exported.onnx",
    )
    out_path = Path(str(output_path)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_written: str | None = None
    runtime_error: str | None = None
    trt_info: dict[str, Any] | None = None
    template = ""

    if provider == "ultralytics":
        template = (
            "yolo export "
            f"model={model_name} format=onnx imgsz={int(args.imgsz)} opset={int(args.opset)} "
            f"dynamic={bool(args.dynamic)} half={bool(args.half)}"
        )
        if not bool(args.dry_run):
            try:
                from ultralytics import YOLO  # type: ignore

                model = YOLO(model_name)
                exported = model.export(
                    format="onnx",
                    imgsz=int(args.imgsz),
                    opset=int(args.opset),
                    dynamic=bool(args.dynamic),
                    half=bool(args.half),
                    simplify=bool(args.simplify),
                    device=str(args.device),
                )
                exported_path = Path(str(exported)).resolve() if exported else None
                if exported_path is None or not exported_path.exists():
                    raise RuntimeError("Ultralytics export returned no output path")
                if exported_path != out_path:
                    out_path.write_bytes(exported_path.read_bytes())
                onnx_written = str(out_path)
            except Exception as exc:
                runtime_error = str(exc)
    elif provider == "hf_detr":
        template = (
            f"{args.python} - <<'PY'  # export {model_name} to ONNX\n"
            "# uses transformers AutoModelForObjectDetection + torch.onnx.export\nPY"
        )
        if not bool(args.dry_run):
            try:
                import torch  # type: ignore
                from transformers import AutoModelForObjectDetection  # type: ignore

                model = AutoModelForObjectDetection.from_pretrained(model_name)
                model.eval()

                class _Wrap(torch.nn.Module):
                    def __init__(self, inner: Any):
                        super().__init__()
                        self.inner = inner

                    def forward(self, pixel_values: Any):
                        out = self.inner(pixel_values=pixel_values)
                        return out.logits, out.pred_boxes

                wrapped = _Wrap(model)
                dummy = torch.randn(1, 3, int(args.imgsz), int(args.imgsz))
                dynamic_axes = None
                if bool(args.dynamic):
                    dynamic_axes = {
                        "pixel_values": {0: "batch", 2: "height", 3: "width"},
                        "logits": {0: "batch"},
                        "pred_boxes": {0: "batch"},
                    }
                torch.onnx.export(
                    wrapped,
                    (dummy,),
                    str(out_path),
                    input_names=["pixel_values"],
                    output_names=["logits", "pred_boxes"],
                    dynamic_axes=dynamic_axes,
                    opset_version=int(args.opset),
                )
                onnx_written = str(out_path)
            except Exception as exc:
                runtime_error = str(exc)
    else:
        raise SystemExit("--provider must be ultralytics|hf_detr")

    if args.trt_engine:
        trt_cmd = [
            str(args.python),
            str(repo_root / "tools" / "build_trt_engine.py"),
            "--onnx",
            str(out_path),
            "--engine",
            str(Path(str(args.trt_engine)).resolve()),
            "--precision",
            str(args.trt_precision),
        ]
        if bool(args.dry_run):
            trt_info = {"dry_run": True, "cmd": trt_cmd}
        elif runtime_error is None:
            proc = _run(trt_cmd, cwd=repo_root)
            trt_info = {
                "dry_run": False,
                "cmd": trt_cmd,
                "returncode": int(proc.returncode),
                "stdout_tail": str(proc.stdout or "").splitlines()[-20:],
                "stderr_tail": str(proc.stderr or "").splitlines()[-20:],
            }
            if proc.returncode != 0:
                runtime_error = f"TensorRT engine build failed ({proc.returncode})"

    ok = bool(args.dry_run) or (runtime_error is None and onnx_written is not None)
    report = {
        "task": "export_onnx",
        "timestamp": _now_utc(),
        "ok": ok,
        "dry_run": bool(args.dry_run),
        "provider": provider,
        "model": model_name,
        "preset": preset_name,
        "output": str(out_path),
        "template_export_command": template,
        "onnx_written": onnx_written,
        "runtime_error": runtime_error,
        "trt": trt_info,
        "layers": {
            "trainer_runner": "N/A",
            "repo_impl": "support_ultralytics_detr export-onnx",
            "export_deploy": "ONNX + optional TensorRT bridge",
        },
    }
    report_path = Path(str(_resolve_value(args.report, fallback="reports/support_ultralytics_detr.export_onnx.json"))).resolve()
    _write_json(report_path, report)
    print(str(report_path))
    return 0 if ok else 1


def _cmd_predict_normalize(args: argparse.Namespace) -> int:
    preset_name, preset = _resolve_preset(getattr(args, "preset", None))
    input_value = _resolve_value(args.input)
    input_path = Path(str(input_value)).resolve() if input_value else None
    output_path = Path(str(_resolve_value(args.output, fallback="reports/predictions.normalized.json"))).resolve()
    report_out = Path(str(_resolve_value(args.report, fallback="reports/support_ultralytics_detr.predict_normalize.json"))).resolve()
    ultralytics_model = _resolve_value(
        args.ultralytics_model,
        env="YOLOZU_MODEL",
        preset=preset,
        preset_key="ultralytics_model",
    )
    dataset_input = _resolve_value(
        args.dataset,
        env="YOLOZU_DATASET",
        preset=preset,
        preset_key="dataset",
    )
    split = str(
        _resolve_value(
            args.split,
            env="YOLOZU_SPLIT",
            preset=preset,
            preset_key="split",
            fallback="val",
        )
    )
    work_dir = Path(str(_resolve_value(args.work_dir, fallback="runs/support_ultralytics_detr/predict"))).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    export_report: dict[str, Any] | None = None

    if input_path is None and ultralytics_model:
        if not dataset_input:
            raise SystemExit("--dataset is required when using --ultralytics-model")
        raw_export = work_dir / "ultralytics_raw_predictions.json"
        cmd = [
            str(args.python),
            str(repo_root / "tools" / "export_predictions_ultralytics.py"),
            "--model",
            str(ultralytics_model),
            "--dataset",
            str(dataset_input),
            "--split",
            split,
            "--output",
            str(raw_export),
        ]
        if bool(args.ultralytics_dry_run):
            cmd.append("--dry-run")
        proc = _run(cmd, cwd=repo_root)
        export_report = {
            "cmd": cmd,
            "returncode": int(proc.returncode),
            "stdout_tail": str(proc.stdout or "").splitlines()[-20:],
            "stderr_tail": str(proc.stderr or "").splitlines()[-20:],
        }
        if proc.returncode != 0:
            report = {
                "task": "predict_normalize",
                "timestamp": _now_utc(),
                "ok": False,
                "error": f"ultralytics export failed ({proc.returncode})",
                "export": export_report,
            }
            _write_json(report_out, report)
            print(str(report_out))
            return 1
        input_path = raw_export

    if input_path is None:
        raise SystemExit("provide --input or --ultralytics-model with --dataset")

    canonical = canonicalize_predictions_file(
        input_path=input_path,
        output_path=output_path,
        strict=bool(args.strict),
        classes_json=(str(args.classes) if args.classes else None),
        assume_class_id_is_category_id=bool(args.assume_class_id_is_category_id),
    )
    report = {
        "task": "predict_normalize",
        "timestamp": _now_utc(),
        "ok": True,
        "strict": bool(args.strict),
        "preset": preset_name,
        "input": str(input_path),
        "output": str(output_path),
        "canonicalization": canonical,
        "export": export_report,
        "layers": {
            "trainer_runner": "N/A",
            "repo_impl": "canonicalize_predictions_file",
            "export_deploy": "normalized predictions usable for ONNX/TensorRT parity/eval",
        },
    }
    _write_json(report_out, report)
    print(str(report_out))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    preset_choices = _preset_choices()
    format_choices = ("auto", "internal", "ultralytics", "coco", "coco_instances")
    preset_help = (
        "Preset defaults (smoke/coco128/none). "
        "If omitted, uses YOLOZU_CLI_PRESET or 'smoke'."
    )
    p = argparse.ArgumentParser(
        description=(
            "Ultralytics/DETR support helper with fixed 3-layer interface contract: "
            "trainer/runner, repo integration, export/deploy."
        )
    )
    sub = p.add_subparsers(dest="command", required=True)

    layers = sub.add_parser("layers", aliases=["ls"], help="Show fixed 3-layer support matrix.")
    layers.add_argument("-j", "--json", action="store_true", help="Emit JSON.")
    layers.set_defaults(_fn=_cmd_layers)

    ds = sub.add_parser(
        "dataset",
        aliases=["ds", "convert"],
        help="Convert COCO/Ultralytics/internal dataset into YOLOZU internal wrapper.",
    )
    ds.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    ds.add_argument("-f", "--from", dest="from_format", default=None, choices=format_choices)
    ds.add_argument("-d", "--dataset", default=None, help="Dataset root or data.yaml path.")
    ds.add_argument("-s", "--split", default=None, help="Split override.")
    ds.add_argument("-i", "--instances-json", default=None, help="COCO instances JSON (for coco_instances mode).")
    ds.add_argument("-g", "--images-dir", default=None, help="COCO images dir (for coco_instances mode).")
    ds.add_argument("-o", "--output", default="runs/support_ultralytics_detr/dataset", help="Output directory for converted wrapper.")
    ds.add_argument("-r", "--report", default="reports/support_ultralytics_detr.dataset.json", help="Report JSON output path.")
    ds.add_argument("-F", "--force", action="store_true", help="Overwrite generated wrapper outputs.")
    ds.set_defaults(_fn=_cmd_dataset)

    tul = sub.add_parser(
        "train-ultralytics",
        aliases=["tu", "ultra-train"],
        help="Ultralytics YOLO fine-tune wrapper + normalized prediction template.",
    )
    tul.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    tul.add_argument("-m", "--model", default=None, help="Ultralytics model path/id (e.g., yolo11n.pt).")
    tul.add_argument("-f", "--from", dest="from_format", default=None, choices=format_choices)
    tul.add_argument("-d", "--dataset", default=None, help="Dataset root or data.yaml path.")
    tul.add_argument("-s", "--split", default=None, help="Split for training (preset/env fallback when omitted).")
    tul.add_argument("-i", "--instances-json", default=None, help="COCO instances JSON (for coco_instances mode).")
    tul.add_argument("-g", "--images-dir", default=None, help="COCO images dir (for coco_instances mode).")
    tul.add_argument("-e", "--epochs", type=int, default=100, help="Epochs (default: 100).")
    tul.add_argument("-z", "--imgsz", type=int, default=640, help="Image size (default: 640).")
    tul.add_argument("-b", "--batch", type=int, default=16, help="Batch size (default: 16).")
    tul.add_argument("-w", "--workers", type=int, default=8, help="Dataloader workers (default: 8).")
    tul.add_argument("-D", "--device", default="auto", help="Device (default: auto).")
    tul.add_argument("-p", "--project", default="runs/ultralytics_finetune", help="Training project dir.")
    tul.add_argument("-N", "--name", default="exp", help="Run name.")
    tul.add_argument("-W", "--work-dir", default="runs/support_ultralytics_detr/ultralytics", help="Work/cache dir.")
    tul.add_argument("-o", "--output", default="reports/support_ultralytics_detr.train_ultralytics.json", help="Report JSON output path.")
    tul.add_argument("-n", "--dry-run", action="store_true", help="Do not execute runtime training.")
    tul.add_argument("-F", "--force", action="store_true", help="Overwrite generated wrapper outputs.")
    tul.set_defaults(_fn=_cmd_train_ultralytics)

    thf = sub.add_parser(
        "train-hf-detr",
        aliases=["th", "hf-train"],
        help="HF DETR/RT-DETR training entry wrapper (template + script bridge).",
    )
    thf.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    thf.add_argument("-m", "--model-id", default=None, help="HF model id (e.g. facebook/detr-resnet-50).")
    thf.add_argument("-f", "--from", dest="from_format", default=None, choices=format_choices)
    thf.add_argument("-d", "--dataset", default=None, help="Dataset root or descriptor.")
    thf.add_argument("-s", "--split", default=None, help="Split for training (preset/env fallback when omitted).")
    thf.add_argument("-i", "--instances-json", default=None, help="COCO instances JSON (for coco_instances mode).")
    thf.add_argument("-g", "--images-dir", default=None, help="COCO images dir (for coco_instances mode).")
    thf.add_argument("-e", "--epochs", type=int, default=10, help="Epochs (default: 10).")
    thf.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size (default: 4).")
    thf.add_argument("-l", "--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4).")
    thf.add_argument("-k", "--max-steps", type=int, default=200, help="Max optimization steps (default: 200).")
    thf.add_argument("-t", "--train-script", default=None, help="Optional external HF train script for non-dry execution.")
    thf.add_argument("-p", "--python", default=sys.executable, help="Python executable for --train-script.")
    thf.add_argument("-W", "--work-dir", default="runs/support_ultralytics_detr/hf_detr", help="Work/cache dir.")
    thf.add_argument("-o", "--output", default="reports/support_ultralytics_detr.train_hf_detr.json", help="Report JSON output path.")
    thf.add_argument("-n", "--dry-run", action="store_true", help="Do not execute runtime training.")
    thf.add_argument("-F", "--force", action="store_true", help="Overwrite generated wrapper outputs.")
    thf.set_defaults(_fn=_cmd_train_hf_detr)

    eonnx = sub.add_parser(
        "export-onnx",
        aliases=["eo", "onnx"],
        help="Export ONNX for the YOLO-family runtime/HF DETR and optionally build TensorRT engine.",
    )
    eonnx.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    eonnx.add_argument("-p", "--provider", choices=("ultralytics", "hf_detr"), default=None, help="Model provider.")
    eonnx.add_argument("-m", "--model", default=None, help="Model path/id.")
    eonnx.add_argument("-o", "--output", default=None, help="ONNX output path.")
    eonnx.add_argument("-r", "--report", default="reports/support_ultralytics_detr.export_onnx.json", help="Report JSON output path.")
    eonnx.add_argument("-z", "--imgsz", type=int, default=640, help="Image size for export (default: 640).")
    eonnx.add_argument("-O", "--opset", type=int, default=17, help="ONNX opset (default: 17).")
    eonnx.add_argument("--dynamic", default=True, action=argparse.BooleanOptionalAction, help="Dynamic ONNX axes (default: true).")
    eonnx.add_argument("-H", "--half", action="store_true", help="FP16 export where supported.")
    eonnx.add_argument("-S", "--simplify", action="store_true", help="Simplify ONNX graph where supported.")
    eonnx.add_argument("-d", "--device", default="cpu", help="Export device (default: cpu).")
    eonnx.add_argument("-y", "--python", default=sys.executable, help="Python executable for optional TensorRT build.")
    eonnx.add_argument("-t", "--trt-engine", default=None, help="Optional TensorRT engine output path.")
    eonnx.add_argument("-q", "--trt-precision", default="fp16", choices=("fp16", "fp32", "int8"), help="TensorRT precision.")
    eonnx.add_argument("-n", "--dry-run", action="store_true", help="Do not execute runtime export.")
    eonnx.set_defaults(_fn=_cmd_export_onnx)

    pnorm = sub.add_parser(
        "predict-normalize",
        aliases=["pn", "normalize"],
        help="Normalize predictions into YOLOZU interface contract.",
    )
    pnorm.add_argument("-P", "--preset", choices=preset_choices, default=None, help=preset_help)
    pnorm.add_argument("-i", "--input", default=None, help="Input predictions JSON (external/raw).")
    pnorm.add_argument("-o", "--output", default=None, help="Normalized predictions JSON output path.")
    pnorm.add_argument("-r", "--report", default="reports/support_ultralytics_detr.predict_normalize.json", help="Report JSON output path.")
    pnorm.add_argument("-c", "--classes", default=None, help="Optional labels/<split>/classes.json for class mapping.")
    pnorm.add_argument("-A", "--assume-class-id-is-category-id", action="store_true", help="Treat class_id as category_id before remap.")
    pnorm.add_argument("-s", "--strict", action="store_true", help="Strict canonicalization (error on out-of-range/unknown keys).")
    pnorm.add_argument("-m", "--ultralytics-model", default=None, help="Optional Ultralytics model to export predictions before normalization.")
    pnorm.add_argument("-d", "--dataset", default=None, help="Dataset root for --ultralytics-model mode.")
    pnorm.add_argument("-S", "--split", default=None, help="Dataset split for --ultralytics-model mode.")
    pnorm.add_argument("-n", "--ultralytics-dry-run", action="store_true", help="Pass --dry-run to export_predictions_ultralytics.")
    pnorm.add_argument("-y", "--python", default=sys.executable, help="Python executable for helper subprocesses.")
    pnorm.add_argument("-w", "--work-dir", default="runs/support_ultralytics_detr/predict", help="Work/cache dir.")
    pnorm.set_defaults(_fn=_cmd_predict_normalize)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    fn = getattr(args, "_fn", None)
    if fn is None:
        raise SystemExit("missing handler")
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())

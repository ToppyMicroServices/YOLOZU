#!/usr/bin/env python3
"""Ultralytics/DETR integration support tool.

Provides three-layer support helpers:
1) trainer/runner,
2) repo integration wrappers,
3) export/deploy (ONNX + optional TensorRT handoff).
"""

from __future__ import annotations

import argparse
import json
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
    resolution = resolve_internal_dataset(
        source_format=str(args.from_format),
        dataset=str(args.dataset) if args.dataset else None,
        split=str(args.split) if args.split else None,
        output=str(args.output),
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
        "dataset_wrapper": str(resolution.dataset_wrapper) if resolution.dataset_wrapper else None,
        "notes": list(resolution.notes),
        "layers": {
            "trainer_runner": "dataset conversion pre-step",
            "repo_impl": "resolve_internal_dataset",
            "export_deploy": "compatible with ONNX/TensorRT export wrappers",
        },
    }
    report_path = Path(str(args.report)).resolve()
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
    work_dir = Path(str(args.work_dir)).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    resolution = resolve_internal_dataset(
        source_format=str(args.from_format),
        dataset=str(args.dataset),
        split=str(args.split) if args.split else None,
        output=work_dir / "dataset",
        instances_json=str(args.instances_json) if args.instances_json else None,
        images_dir=str(args.images_dir) if args.images_dir else None,
        force=bool(args.force),
    )

    dataset_path = Path(str(args.dataset)).resolve()
    if dataset_path.is_file() and dataset_path.suffix.lower() in (".yaml", ".yml") and resolution.source_format == "ultralytics":
        data_yaml = dataset_path
    else:
        data_yaml = write_ultralytics_data_yaml(
            dataset_root=resolution.dataset_root,
            split=resolution.split,
            output=work_dir / "ultralytics_data.yaml",
        )

    template = _ultralytics_train_template(
        model=str(args.model),
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

            model = YOLO(str(args.model))
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
        "model": str(args.model),
        "dataset_root": str(resolution.dataset_root),
        "split": resolution.split,
        "data_yaml": str(data_yaml),
        "template_train_command": template,
        "template_predict_normalize_command": (
            "python3 tools/support_ultralytics_detr.py predict-normalize "
            f"--ultralytics-model {args.model} --dataset {resolution.dataset_root} "
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
    report_path = Path(str(args.output)).resolve()
    _write_json(report_path, report)
    print(str(report_path))
    return 0 if ok else 1


def _cmd_train_hf_detr(args: argparse.Namespace) -> int:
    work_dir = Path(str(args.work_dir)).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    resolution = resolve_internal_dataset(
        source_format=str(args.from_format),
        dataset=str(args.dataset),
        split=str(args.split) if args.split else None,
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
            str(args.model_id),
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
            str(args.model_id),
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
        "model_id": str(args.model_id),
        "dataset_root": str(resolution.dataset_root),
        "split": resolution.split,
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
    report_path = Path(str(args.output)).resolve()
    _write_json(report_path, report)
    print(str(report_path))
    return 0 if ok else 1


def _cmd_export_onnx(args: argparse.Namespace) -> int:
    out_path = Path(str(args.output)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_written: str | None = None
    runtime_error: str | None = None
    trt_info: dict[str, Any] | None = None

    provider = str(args.provider)
    template = ""

    if provider == "ultralytics":
        template = (
            "yolo export "
            f"model={args.model} format=onnx imgsz={int(args.imgsz)} opset={int(args.opset)} "
            f"dynamic={bool(args.dynamic)} half={bool(args.half)}"
        )
        if not bool(args.dry_run):
            try:
                from ultralytics import YOLO  # type: ignore

                model = YOLO(str(args.model))
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
            f"{args.python} - <<'PY'  # export {args.model} to ONNX\n"
            "# uses transformers AutoModelForObjectDetection + torch.onnx.export\nPY"
        )
        if not bool(args.dry_run):
            try:
                import torch  # type: ignore
                from transformers import AutoModelForObjectDetection  # type: ignore

                model = AutoModelForObjectDetection.from_pretrained(str(args.model))
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
        "model": str(args.model),
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
    report_path = Path(str(args.report)).resolve()
    _write_json(report_path, report)
    print(str(report_path))
    return 0 if ok else 1


def _cmd_predict_normalize(args: argparse.Namespace) -> int:
    input_path = Path(str(args.input)).resolve() if args.input else None
    work_dir = Path(str(args.work_dir)).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    export_report: dict[str, Any] | None = None

    if input_path is None and args.ultralytics_model:
        if not args.dataset:
            raise SystemExit("--dataset is required when using --ultralytics-model")
        raw_export = work_dir / "ultralytics_raw_predictions.json"
        cmd = [
            str(args.python),
            str(repo_root / "tools" / "export_predictions_ultralytics.py"),
            "--model",
            str(args.ultralytics_model),
            "--dataset",
            str(args.dataset),
            "--split",
            str(args.split or "val"),
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
            report_path = Path(str(args.report)).resolve()
            _write_json(report_path, report)
            print(str(report_path))
            return 1
        input_path = raw_export

    if input_path is None:
        raise SystemExit("provide --input or --ultralytics-model with --dataset")

    canonical = canonicalize_predictions_file(
        input_path=input_path,
        output_path=Path(str(args.output)).resolve(),
        strict=bool(args.strict),
        classes_json=(str(args.classes) if args.classes else None),
        assume_class_id_is_category_id=bool(args.assume_class_id_is_category_id),
    )
    report = {
        "task": "predict_normalize",
        "timestamp": _now_utc(),
        "ok": True,
        "strict": bool(args.strict),
        "input": str(input_path),
        "output": str(Path(str(args.output)).resolve()),
        "canonicalization": canonical,
        "export": export_report,
        "layers": {
            "trainer_runner": "N/A",
            "repo_impl": "canonicalize_predictions_file",
            "export_deploy": "normalized predictions usable for ONNX/TensorRT parity/eval",
        },
    }
    report_path = Path(str(args.report)).resolve()
    _write_json(report_path, report)
    print(str(report_path))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Ultralytics/DETR support helper with fixed 3-layer interface contract: "
            "trainer/runner, repo integration, export/deploy."
        )
    )
    sub = p.add_subparsers(dest="command", required=True)

    layers = sub.add_parser("layers", help="Show fixed 3-layer support matrix.")
    layers.add_argument("--json", action="store_true", help="Emit JSON.")
    layers.set_defaults(_fn=_cmd_layers)

    ds = sub.add_parser("dataset", help="Convert COCO/Ultralytics/internal dataset into YOLOZU internal wrapper.")
    ds.add_argument("--from", dest="from_format", default="auto", choices=("auto", "internal", "ultralytics", "coco", "coco_instances"))
    ds.add_argument("--dataset", default=None, help="Dataset root or data.yaml path.")
    ds.add_argument("--split", default=None, help="Split override.")
    ds.add_argument("--instances-json", default=None, help="COCO instances JSON (for coco_instances mode).")
    ds.add_argument("--images-dir", default=None, help="COCO images dir (for coco_instances mode).")
    ds.add_argument("--output", default="runs/support_ultralytics_detr/dataset", help="Output directory for converted wrapper.")
    ds.add_argument("--report", default="reports/support_ultralytics_detr.dataset.json", help="Report JSON output path.")
    ds.add_argument("--force", action="store_true", help="Overwrite generated wrapper outputs.")
    ds.set_defaults(_fn=_cmd_dataset)

    tul = sub.add_parser("train-ultralytics", help="Ultralytics YOLO fine-tune wrapper + normalized prediction template.")
    tul.add_argument("--model", required=True, help="Ultralytics model path/id (e.g., yolo11n.pt).")
    tul.add_argument("--from", dest="from_format", default="auto", choices=("auto", "internal", "ultralytics", "coco", "coco_instances"))
    tul.add_argument("--dataset", required=True, help="Dataset root or data.yaml path.")
    tul.add_argument("--split", default="train", help="Split for training (default: train).")
    tul.add_argument("--instances-json", default=None, help="COCO instances JSON (for coco_instances mode).")
    tul.add_argument("--images-dir", default=None, help="COCO images dir (for coco_instances mode).")
    tul.add_argument("--epochs", type=int, default=100, help="Epochs (default: 100).")
    tul.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640).")
    tul.add_argument("--batch", type=int, default=16, help="Batch size (default: 16).")
    tul.add_argument("--workers", type=int, default=8, help="Dataloader workers (default: 8).")
    tul.add_argument("--device", default="auto", help="Device (default: auto).")
    tul.add_argument("--project", default="runs/ultralytics_finetune", help="Training project dir.")
    tul.add_argument("--name", default="exp", help="Run name.")
    tul.add_argument("--work-dir", default="runs/support_ultralytics_detr/ultralytics", help="Work/cache dir.")
    tul.add_argument("--output", default="reports/support_ultralytics_detr.train_ultralytics.json", help="Report JSON output path.")
    tul.add_argument("--dry-run", action="store_true", help="Do not execute runtime training.")
    tul.add_argument("--force", action="store_true", help="Overwrite generated wrapper outputs.")
    tul.set_defaults(_fn=_cmd_train_ultralytics)

    thf = sub.add_parser("train-hf-detr", help="HF DETR/RT-DETR training entry wrapper (template + script bridge).")
    thf.add_argument("--model-id", default="facebook/detr-resnet-50", help="HF model id (default: facebook/detr-resnet-50).")
    thf.add_argument("--from", dest="from_format", default="auto", choices=("auto", "internal", "ultralytics", "coco", "coco_instances"))
    thf.add_argument("--dataset", required=True, help="Dataset root or descriptor.")
    thf.add_argument("--split", default="train", help="Split for training (default: train).")
    thf.add_argument("--instances-json", default=None, help="COCO instances JSON (for coco_instances mode).")
    thf.add_argument("--images-dir", default=None, help="COCO images dir (for coco_instances mode).")
    thf.add_argument("--epochs", type=int, default=10, help="Epochs (default: 10).")
    thf.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4).")
    thf.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4).")
    thf.add_argument("--max-steps", type=int, default=200, help="Max optimization steps (default: 200).")
    thf.add_argument("--train-script", default=None, help="Optional external HF train script for non-dry execution.")
    thf.add_argument("--python", default=sys.executable, help="Python executable for --train-script.")
    thf.add_argument("--work-dir", default="runs/support_ultralytics_detr/hf_detr", help="Work/cache dir.")
    thf.add_argument("--output", default="reports/support_ultralytics_detr.train_hf_detr.json", help="Report JSON output path.")
    thf.add_argument("--dry-run", action="store_true", help="Do not execute runtime training.")
    thf.add_argument("--force", action="store_true", help="Overwrite generated wrapper outputs.")
    thf.set_defaults(_fn=_cmd_train_hf_detr)

    eonnx = sub.add_parser("export-onnx", help="Export ONNX for Ultralytics/HF DETR and optionally build TensorRT engine.")
    eonnx.add_argument("--provider", choices=("ultralytics", "hf_detr"), required=True, help="Model provider.")
    eonnx.add_argument("--model", required=True, help="Model path/id.")
    eonnx.add_argument("--output", required=True, help="ONNX output path.")
    eonnx.add_argument("--report", default="reports/support_ultralytics_detr.export_onnx.json", help="Report JSON output path.")
    eonnx.add_argument("--imgsz", type=int, default=640, help="Image size for export (default: 640).")
    eonnx.add_argument("--opset", type=int, default=17, help="ONNX opset (default: 17).")
    eonnx.add_argument("--dynamic", default=True, action=argparse.BooleanOptionalAction, help="Dynamic ONNX axes (default: true).")
    eonnx.add_argument("--half", action="store_true", help="FP16 export where supported.")
    eonnx.add_argument("--simplify", action="store_true", help="Simplify ONNX graph where supported.")
    eonnx.add_argument("--device", default="cpu", help="Export device (default: cpu).")
    eonnx.add_argument("--python", default=sys.executable, help="Python executable for optional TensorRT build.")
    eonnx.add_argument("--trt-engine", default=None, help="Optional TensorRT engine output path.")
    eonnx.add_argument("--trt-precision", default="fp16", choices=("fp16", "fp32", "int8"), help="TensorRT precision.")
    eonnx.add_argument("--dry-run", action="store_true", help="Do not execute runtime export.")
    eonnx.set_defaults(_fn=_cmd_export_onnx)

    pnorm = sub.add_parser("predict-normalize", help="Normalize predictions into YOLOZU interface contract.")
    pnorm.add_argument("--input", default=None, help="Input predictions JSON (external/raw).")
    pnorm.add_argument("--output", required=True, help="Normalized predictions JSON output path.")
    pnorm.add_argument("--report", default="reports/support_ultralytics_detr.predict_normalize.json", help="Report JSON output path.")
    pnorm.add_argument("--classes", default=None, help="Optional labels/<split>/classes.json for class mapping.")
    pnorm.add_argument("--assume-class-id-is-category-id", action="store_true", help="Treat class_id as category_id before remap.")
    pnorm.add_argument("--strict", action="store_true", help="Strict canonicalization (error on out-of-range/unknown keys).")
    pnorm.add_argument("--ultralytics-model", default=None, help="Optional Ultralytics model to export predictions before normalization.")
    pnorm.add_argument("--dataset", default=None, help="Dataset root for --ultralytics-model mode.")
    pnorm.add_argument("--split", default="val", help="Dataset split for --ultralytics-model mode.")
    pnorm.add_argument("--ultralytics-dry-run", action="store_true", help="Pass --dry-run to export_predictions_ultralytics.")
    pnorm.add_argument("--python", default=sys.executable, help="Python executable for helper subprocesses.")
    pnorm.add_argument("--work-dir", default="runs/support_ultralytics_detr/predict", help="Work/cache dir.")
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

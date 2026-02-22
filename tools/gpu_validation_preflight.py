import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.doctor import build_doctor_report


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split YOLOZU GPU validation sweep into local-executable and GPU-required scopes."
    )
    p.add_argument("--output", default="reports/gpu_validation_preflight.json", help="Where to write JSON report.")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when local prerequisites for local-executable scope are not met.",
    )
    return p.parse_args(argv)


def _path_ok(path: str) -> dict[str, Any]:
    p = repo_root / path
    return {"name": path, "ok": bool(p.exists()), "detail": str(p)}


def _tool_ok(name: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(shutil.which(name)), "detail": shutil.which(name)}


def _status_from_checks(*, local_checks: list[dict[str, Any]], gpu_required: bool, gpu_checks: list[dict[str, Any]]) -> str:
    local_ok = all(bool(c.get("ok")) for c in local_checks)
    gpu_ok = all(bool(c.get("ok")) for c in gpu_checks) if gpu_checks else True
    if not local_ok:
        return "blocked_local"
    if gpu_required and not gpu_ok:
        return "needs_gpu_runtime"
    if gpu_required:
        return "ready_for_gpu"
    return "ready_local"


def _step(
    *,
    step_id: str,
    title: str,
    gpu_required: bool,
    gpu_recommended: bool,
    local_checks: list[dict[str, Any]],
    gpu_checks: list[dict[str, Any]],
    local_commands: list[str],
    gpu_commands: list[str],
    dod: list[str],
) -> dict[str, Any]:
    status = _status_from_checks(local_checks=local_checks, gpu_required=gpu_required, gpu_checks=gpu_checks)
    return {
        "id": step_id,
        "title": title,
        "gpu_required": bool(gpu_required),
        "gpu_recommended": bool(gpu_recommended),
        "status": status,
        "local_checks": local_checks,
        "gpu_checks": gpu_checks,
        "commands": {"local": local_commands, "gpu": gpu_commands},
        "definition_of_done": dod,
    }


def _summarize_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    local_executable = [s["id"] for s in steps if s["status"] == "ready_local"]
    local_preflight_ready = [s["id"] for s in steps if s["status"] in {"ready_local", "ready_for_gpu", "needs_gpu_runtime"}]
    gpu_execution_required = [s["id"] for s in steps if bool(s.get("gpu_required"))]
    gpu_execution_optional = [s["id"] for s in steps if (not bool(s.get("gpu_required"))) and bool(s.get("gpu_recommended"))]
    blocked_local = [s["id"] for s in steps if s["status"] == "blocked_local"]
    needs_gpu_runtime = [s["id"] for s in steps if s["status"] == "needs_gpu_runtime"]
    ready_for_gpu = [s["id"] for s in steps if s["status"] == "ready_for_gpu"]
    return {
        "local_executable_steps": local_executable,
        "local_preflight_ready_steps": local_preflight_ready,
        "gpu_execution_required_steps": gpu_execution_required,
        "gpu_execution_optional_steps": gpu_execution_optional,
        "ready_for_gpu_steps": ready_for_gpu,
        "blocked_local_steps": blocked_local,
        "needs_gpu_runtime_steps": needs_gpu_runtime,
    }


def build_report() -> dict[str, Any]:
    doctor, _ = build_doctor_report(cwd=repo_root)
    runtime = doctor.get("runtime_capabilities") or {}
    torch_rt = runtime.get("torch") or {}
    ort_rt = runtime.get("onnxruntime") or {}
    trt_rt = runtime.get("tensorrt") or {}
    opencv_rt = runtime.get("opencv") or {}
    cuda_rt = runtime.get("cuda") or {}

    torch_installed = bool(torch_rt.get("installed"))
    torch_cuda = bool(torch_rt.get("cuda_available"))
    torch_gpu_count = int(torch_rt.get("device_count") or 0)
    ort_installed = bool(ort_rt.get("installed"))
    ort_cuda = bool(ort_rt.get("cuda_provider"))
    trtexec_available = bool(trt_rt.get("trtexec_available"))
    trt_py_available = bool(trt_rt.get("python_module_available"))
    opencv_installed = bool(opencv_rt.get("module_available"))
    opencv_cuda_devices = int(opencv_rt.get("cuda_enabled_device_count") or 0)
    nvidia_smi_available = bool(cuda_rt.get("nvidia_smi_available"))

    steps: list[dict[str, Any]] = []

    steps.append(
        _step(
            step_id="train_rtdetr_pose_amp_ddp_resume",
            title="RT-DETR pose training (AMP on/off, DDP, resume)",
            gpu_required=True,
            gpu_recommended=True,
            local_checks=[
                _path_ok("rtdetr_pose/tools/train_minimal.py"),
                _path_ok("rtdetr_pose/configs/base.json"),
                {"name": "torch_installed", "ok": torch_installed, "detail": torch_rt.get("version")},
                _tool_ok("torchrun"),
            ],
            gpu_checks=[
                {"name": "nvidia_smi_available", "ok": nvidia_smi_available, "detail": None},
                {"name": "torch_cuda_available", "ok": torch_cuda, "detail": torch_rt.get("cuda_version")},
                {"name": "at_least_2_gpus_for_ddp", "ok": torch_gpu_count >= 2, "detail": torch_gpu_count},
            ],
            local_commands=[
                "python3 rtdetr_pose/tools/train_minimal.py --help",
            ],
            gpu_commands=[
                "TORCH_DISTRIBUTED_DEFAULT_FIND_UNUSED_PARAMETERS=1 torchrun --nproc_per_node=2 rtdetr_pose/tools/train_minimal.py --config rtdetr_pose/configs/base.json --dataset-root /workspace/tmp/smoke --split train --val-split val --device cuda --amp fp16 --batch-size 2 --val-batch-size 2 --epochs 10 --log-every 2 --val-every 5 --checkpoint-every 100 --run-dir /workspace/runs/rtdetr_pose_ddp_fp16 --export-onnx --onnx-out /workspace/runs/rtdetr_pose_ddp_fp16/model.onnx --onnx-meta-out /workspace/runs/rtdetr_pose_ddp_fp16/model.onnx.meta.json",
            ],
            dod=[
                "runs/<id>/checkpoints/best.* created",
                "ONNX + ONNX meta created from training run",
                "DDP 2GPU run completed",
            ],
        )
    )

    steps.append(
        _step(
            step_id="onnx_export_ort_parity",
            title="ONNX export + ORT parity (CUDA preferred)",
            gpu_required=False,
            gpu_recommended=True,
            local_checks=[
                _path_ok("tools/export_trt.py"),
                _path_ok("tools/rtdetr_pose_backend_suite.py"),
                {"name": "torch_installed", "ok": torch_installed, "detail": torch_rt.get("version")},
                {"name": "onnxruntime_installed", "ok": ort_installed, "detail": ort_rt.get("version")},
            ],
            gpu_checks=[
                {"name": "onnxruntime_cuda_provider", "ok": ort_cuda, "detail": ort_rt.get("providers")},
            ],
            local_commands=[
                "python3 tools/export_trt.py --config rtdetr_pose/configs/base.json --checkpoint /path/to/checkpoint.pt --device cpu --skip-engine --onnx models/rtdetr_pose.onnx --onnx-meta reports/rtdetr_pose.onnx.meta.json",
                "python3 tools/rtdetr_pose_backend_suite.py --config rtdetr_pose/configs/base.json --checkpoint /path/to/checkpoint.pt --onnx models/rtdetr_pose.onnx --backends torch,onnxrt --device cpu --output reports/rtdetr_pose_backend_suite_cpu.json",
            ],
            gpu_commands=[
                "python3 tools/rtdetr_pose_backend_suite.py --config rtdetr_pose/configs/base.json --checkpoint /path/to/checkpoint.pt --onnx models/rtdetr_pose.onnx --backends torch,onnxrt --device cuda --output reports/rtdetr_pose_backend_suite_cuda.json",
            ],
            dod=[
                "ONNX meta contains opset/hash/input/preprocess",
                "Torch vs ORT parity stats reported",
            ],
        )
    )

    steps.append(
        _step(
            step_id="trt_build_export_eval_latency",
            title="TensorRT build + exporter + eval + latency",
            gpu_required=True,
            gpu_recommended=True,
            local_checks=[
                _path_ok("tools/build_trt_engine.py"),
                _path_ok("tools/export_predictions_trt.py"),
                _path_ok("tools/measure_trt_latency.py"),
                _path_ok("tools/run_trt_pipeline.py"),
            ],
            gpu_checks=[
                {"name": "nvidia_smi_available", "ok": nvidia_smi_available, "detail": None},
                {"name": "trtexec_available", "ok": trtexec_available, "detail": trt_rt.get("trtexec_version")},
                {"name": "tensorrt_python_or_trtexec", "ok": trtexec_available or trt_py_available, "detail": None},
            ],
            local_commands=[
                "python3 tools/build_trt_engine.py --onnx models/rtdetr_pose.onnx --engine engines/rtdetr_pose_fp16.plan --precision fp16 --dry-run",
            ],
            gpu_commands=[
                "python3 tools/build_trt_engine.py --onnx /workspace/runs/rtdetr_pose_ddp_fp16/model.onnx --engine /workspace/runs/rtdetr_pose_ddp_fp16/model_fp16.plan --precision fp16 --input-name images --min-shape 1x3x64x64 --opt-shape 1x3x64x64 --max-shape 1x3x64x64 --meta-output /workspace/reports/trt_engine_meta.json",
                "python3 tools/export_predictions_trt.py --dataset /workspace/tmp/smoke --split val --engine /workspace/runs/rtdetr_pose_ddp_fp16/model_fp16.plan --combined-output output0 --boxes-scale abs --wrap --strict --output /workspace/reports/pred_trt.json",
                "python3 tools/measure_trt_latency.py --engine /workspace/runs/rtdetr_pose_ddp_fp16/model_fp16.plan --input-name images --shape 1x3x64x64 --output /workspace/reports/latency_trt.json",
            ],
            dod=[
                "predictions_trt.json generated",
                "parity_trt.json + latency report generated",
                "engine metadata records runtime versions",
            ],
        )
    )

    steps.append(
        _step(
            step_id="safe_ttt_ctta_stability",
            title="Safe TTT/CTTA stability checks (Tent/CoTTA/EATA/SAR)",
            gpu_required=True,
            gpu_recommended=True,
            local_checks=[
                _path_ok("tools/export_predictions.py"),
                _path_ok("docs/ttt_protocol.md"),
                _path_ok("docs/tta_support_matrix.md"),
            ],
            gpu_checks=[
                {"name": "torch_cuda_available", "ok": torch_cuda, "detail": torch_rt.get("cuda_version")},
            ],
            local_commands=[
                "python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-preset safe --ttt-reset sample --dry-run --wrap --output reports/pred_ttt_safe_dryrun.json",
            ],
            gpu_commands=[
                "python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-method tent --ttt-preset safe --ttt-reset sample --ttt-log-out reports/ttt_tent.json --wrap --output reports/pred_ttt_tent.json",
                "python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-method cotta --ttt-preset safe --ttt-reset sample --ttt-log-out reports/ttt_cotta.json --wrap --output reports/pred_ttt_cotta.json",
                "python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-method eata --ttt-preset safe --ttt-reset sample --ttt-log-out reports/ttt_eata.json --wrap --output reports/pred_ttt_eata.json",
                "python3 tools/export_predictions.py --adapter rtdetr_pose --ttt --ttt-method sar --ttt-preset safe --ttt-reset sample --ttt-log-out reports/ttt_sar.json --wrap --output reports/pred_ttt_sar.json",
            ],
            dod=[
                "baseline vs adaptation metrics reported",
                "rollback/instability counters captured in logs",
            ],
        )
    )

    steps.append(
        _step(
            step_id="opencv_dnn_backend_parity",
            title="OpenCV-DNN parity (CPU baseline + CUDA target)",
            gpu_required=True,
            gpu_recommended=True,
            local_checks=[
                _path_ok("tools/export_predictions_opencv_dnn_rtdetr.py"),
                {"name": "opencv_python_installed", "ok": opencv_installed, "detail": opencv_rt.get("python_package_version")},
            ],
            gpu_checks=[
                {"name": "opencv_cuda_enabled_device_count>0", "ok": opencv_cuda_devices > 0, "detail": opencv_cuda_devices},
            ],
            local_commands=[
                "python3 tools/export_predictions_opencv_dnn_rtdetr.py --dataset data/smoke --split val --dry-run --output reports/pred_rtdetr_opencv_cpu_dryrun.json",
            ],
            gpu_commands=[
                "python3 tools/export_predictions_opencv_dnn_rtdetr.py --dataset /workspace/tmp/smoke --split val --onnx /workspace/runs/rtdetr_pose_ddp_fp16/model.onnx --dnn-backend cuda --dnn-target cuda --score-thr 0.01 --output /workspace/reports/pred_rtdetr_opencv_cuda.json --meta-output /workspace/reports/pred_rtdetr_opencv_cuda.meta.json --strict",
            ],
            dod=[
                "CPU and CUDA OpenCV-DNN outputs generated",
                "export_settings.runtime contains backend/target/version",
            ],
        )
    )

    steps.append(
        _step(
            step_id="gpu_doctor_matrix",
            title="Doctor matrix capture (driver/CUDA/providers/SM/bf16)",
            gpu_required=False,
            gpu_recommended=True,
            local_checks=[
                _path_ok("yolozu/doctor.py"),
                _path_ok("tools/yolozu.py"),
            ],
            gpu_checks=[
                {"name": "nvidia_smi_available", "ok": nvidia_smi_available, "detail": None},
                {"name": "torch_cuda_available", "ok": torch_cuda, "detail": torch_rt.get("cuda_version")},
                {"name": "onnxruntime_cuda_provider", "ok": ort_cuda, "detail": ort_rt.get("providers")},
            ],
            local_commands=[
                "python3 tools/yolozu.py doctor --output reports/doctor_local.json",
            ],
            gpu_commands=[
                "python3 tools/yolozu.py doctor --output /workspace/reports/doctor_gpu.json",
            ],
            dod=[
                "doctor json includes nvidia-smi/torch/onnxruntime/tensorrt/opencv runtime capabilities",
                "drift_hints emitted for mixed-runtime mismatch",
            ],
        )
    )

    summary = _summarize_steps(steps)

    return {
        "kind": "yolozu_gpu_validation_preflight",
        "schema_version": 1,
        "timestamp": _now_utc(),
        "issue": "YOLOZU-zisn",
        "summary": summary,
        "steps": steps,
        "doctor_snapshot": {
            "runtime_capabilities": runtime,
            "tools": doctor.get("tools"),
            "warnings": doctor.get("warnings"),
            "errors": doctor.get("errors"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    report = build_report()

    out_path = Path(str(args.output))
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(out_path))

    if bool(args.strict):
        blocked = report.get("summary", {}).get("blocked_local_steps") or []
        return 1 if blocked else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

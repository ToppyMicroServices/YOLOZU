"""Post-training finalization helpers for train_minimal."""

from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from yolozu.metrics_report import build_report, write_csv_row, write_json

from rtdetr_pose.train_utils import _now_utc, collect_rng_state, load_checkpoint_into, run_onnxrt_parity, save_checkpoint_bundle, unwrap_model


def _write_onnx_export_meta(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def finalize_training(
    *,
    args: Any,
    is_main: bool,
    ddp_enabled: bool,
    model: Any,
    optim: Any,
    sched: Any,
    scaler: Any,
    ema: Any,
    device: Any,
    run_contract: Any,
    run_dir: Path | None,
    run_record: dict[str, Any],
    global_step: int,
    last_loss_dict: dict[str, Any] | None,
    last_epoch_avg: float | None,
    last_epoch_steps: int,
    last_grad_norm: float | None,
    last_data_time_s: float | None,
    last_step_time_s: float | None,
    last_throughput: float | None,
    last_max_vram_mb: float | None,
    ewc_accum: Any = None,
    si_accum: Any = None,
    save_ewc_state_fn: Callable[[str, Any], None] | None = None,
    save_si_state_fn: Callable[[str, Any], None] | None = None,
) -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required")

    if is_main and (args.metrics_json or args.metrics_csv):
        losses_out: dict[str, float] = {}
        if last_loss_dict is not None:
            losses_out = {
                k: float(v.detach().cpu())
                for k, v in last_loss_dict.items()
                if hasattr(v, "detach")
            }
        metrics_out = {"epochs": int(args.epochs), "max_steps": int(args.max_steps)}
        if last_epoch_avg is not None:
            metrics_out["loss_avg_last_epoch"] = float(last_epoch_avg)
        if last_grad_norm is not None:
            metrics_out["grad_norm_last"] = float(last_grad_norm)
        if last_data_time_s is not None:
            metrics_out["data_time_last_s"] = float(last_data_time_s)
        if last_step_time_s is not None:
            metrics_out["step_time_last_s"] = float(last_step_time_s)
        if last_throughput is not None:
            metrics_out["throughput_last_img_s"] = float(last_throughput)
        if last_max_vram_mb is not None:
            metrics_out["max_vram_last_mb"] = float(last_max_vram_mb)

        summary = build_report(
            losses=losses_out,
            metrics=metrics_out,
            meta={"kind": "train_run", "run_record": run_record},
        )
        if args.metrics_json:
            write_json(args.metrics_json, summary)
        if args.metrics_csv:
            write_csv_row(args.metrics_csv, summary)

    if is_main and args.checkpoint_out:
        ckpt_path = Path(args.checkpoint_out)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(unwrap_model(model).state_dict(), ckpt_path)

    if is_main and args.checkpoint_bundle_out:
        save_checkpoint_bundle(
            args.checkpoint_bundle_out,
            model=unwrap_model(model),
            optim=optim,
            sched=sched,
            scaler=scaler,
            ema=ema,
            args=args,
            epoch=int(args.epochs) - 1,
            global_step=int(global_step),
            last_epoch_steps=int(last_epoch_steps),
            last_epoch_avg=last_epoch_avg,
            last_loss_dict=last_loss_dict,
            run_record=run_record,
            rng_state=collect_rng_state(),
        )

    if is_main and getattr(args, "best_checkpoint_out", None) and args.checkpoint_bundle_out:
        best_path = Path(str(args.best_checkpoint_out))
        if not best_path.exists():
            best_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(str(args.checkpoint_bundle_out), str(best_path))
            except Exception:
                save_checkpoint_bundle(
                    str(best_path),
                    model=unwrap_model(model),
                    optim=optim,
                    sched=sched,
                    scaler=scaler,
                    ema=ema,
                    args=args,
                    epoch=int(args.epochs) - 1,
                    global_step=int(global_step),
                    last_epoch_steps=int(last_epoch_steps),
                    last_epoch_avg=last_epoch_avg,
                    last_loss_dict=last_loss_dict,
                    run_record=run_record,
                    rng_state=collect_rng_state(),
                )

    if (
        is_main
        and ewc_accum is not None
        and args.ewc_state_out
        and save_ewc_state_fn is not None
    ):
        save_ewc_state_fn(str(args.ewc_state_out), ewc_accum.finalize(unwrap_model(model)))

    if (
        is_main
        and si_accum is not None
        and args.si_state_out
        and save_si_state_fn is not None
    ):
        save_si_state_fn(str(args.si_state_out), si_accum.finalize(unwrap_model(model)))

    onnx_path: Path | None = None
    onnx_meta_path: Path | None = None
    onnx_export_status = "disabled"
    onnx_export_error: dict[str, str] | None = None
    onnx_export_device = "cpu"
    if is_main and args.onnx_out:
        try:
            from rtdetr_pose.export import export_onnx
        except Exception as exc:  # pragma: no cover
            print(
                f"WARNING: ONNX export skipped — could not import rtdetr_pose.export ({exc}). "
                "Install 'onnx' to enable post-training ONNX export.",
                file=sys.stderr,
            )
            onnx_export_status = "import_failed"
            onnx_export_error = {"type": type(exc).__name__, "message": str(exc)}
            export_onnx = None  # type: ignore[assignment]

        onnx_path = Path(str(args.onnx_out)) if export_onnx is not None else None
        if args.onnx_meta_out:
            onnx_meta_path = Path(str(args.onnx_meta_out))
        elif args.onnx_out:
            onnx_meta_path = Path(str(args.onnx_out)).with_suffix(Path(str(args.onnx_out)).suffix + ".meta.json")
        if onnx_path is not None:
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            if run_contract is not None and getattr(args, "best_checkpoint_out", None):
                best_path = Path(str(args.best_checkpoint_out))
                if best_path.exists():
                    load_checkpoint_into(
                        unwrap_model(model),
                        None,
                        str(best_path),
                        restore_rng=False,
                    )
            export_device = torch.device("cpu")
            onnx_export_device = str(export_device)
            # Export/parity run after training completes, so switching the live model to CPU
            # avoids a second full model copy at finalize time.
            export_model = unwrap_model(model).eval().to(export_device)
            dummy = torch.zeros(
                (1, 3, int(args.image_size), int(args.image_size)),
                dtype=torch.float32,
                device=export_device,
            )
            try:
                export_onnx(
                    export_model,
                    dummy,
                    str(onnx_path),
                    opset_version=int(args.onnx_opset),
                    dynamic_hw=bool(args.onnx_dynamic_hw),
                )
                onnx_export_status = "ok"
            except Exception as exc:
                print(
                    f"WARNING: ONNX export failed — {exc}. Training results are saved; "
                    "the run continues without a post-training ONNX artifact.",
                    file=sys.stderr,
                )
                onnx_export_status = "failed"
                onnx_export_error = {"type": type(exc).__name__, "message": str(exc)}
                onnx_path = None
        if onnx_meta_path is not None:
            meta = {
                "timestamp_utc": _now_utc(),
                "status": onnx_export_status,
                "onnx": str(onnx_path) if onnx_path is not None else None,
                "requested_output": str(args.onnx_out),
                "opset": int(args.onnx_opset),
                "dynamic_hw": bool(args.onnx_dynamic_hw),
                "dummy_input": {
                    "shape": [1, 3, int(args.image_size), int(args.image_size)],
                    "dtype": "float32",
                },
                "export_device": onnx_export_device,
                "error": onnx_export_error,
                "run_record": run_record,
            }
            _write_onnx_export_meta(onnx_meta_path, meta)

    parity_out = getattr(args, "parity_json_out", None)
    if is_main and parity_out:
        out_path = Path(str(parity_out))
        policy = str(args.parity_policy or ("fail" if run_contract is not None else "warn"))
        if onnx_path is None:
            report = {
                "timestamp_utc": _now_utc(),
                "onnx": None,
                "thresholds": {
                    "score_atol": float(args.parity_score_atol),
                    "bbox_atol": float(args.parity_bbox_atol),
                },
                "policy": policy,
                "passed": False,
                "available": False,
                "reason": "onnx_export_disabled",
                "run_record": run_record,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            if policy == "fail":
                raise SystemExit(f"ONNX parity requested but ONNX export disabled. See: {out_path}")
            print(
                f"WARNING: ONNX parity requested but ONNX export disabled. See: {out_path}",
                file=sys.stderr,
            )
        else:
            run_onnxrt_parity(
                model=unwrap_model(model),
                onnx_path=onnx_path,
                image_size=int(args.image_size),
                seed=int(getattr(args, "seed", 0) or 0),
                score_atol=float(args.parity_score_atol),
                bbox_atol=float(args.parity_bbox_atol),
                out_path=out_path,
                policy=policy,
                run_record=run_record,
            )

    if is_main and run_dir is not None:
        (run_dir / "run_record.json").write_text(
            json.dumps(run_record, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if ddp_enabled:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

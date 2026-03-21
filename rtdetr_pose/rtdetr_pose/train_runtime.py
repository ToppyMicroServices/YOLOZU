"""Runtime helpers for train_minimal."""

from __future__ import annotations

import signal
from typing import Any, Callable

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from yolozu.metrics_report import append_jsonl, build_report
from yolozu.sdft import SdftConfig
from yolozu.simple_map import evaluate_map

from rtdetr_pose.train_utils import decode_detections_from_outputs, save_checkpoint_bundle, unwrap_model


def setup_distillation_and_derpp(
    *,
    args: Any,
    model_cfg: Any,
    model_num_queries: int,
    device: Any,
    is_main: bool,
    build_model_fn: Callable[[Any], Any],
    rtdetr_pose_cls: Any,
    load_checkpoint_into_fn: Callable[..., dict[str, Any]],
) -> tuple[Any, SdftConfig | None, SdftConfig | None]:
    teacher_model = None
    sdft_cfg = None
    if args.self_distill_from:
        # Build a frozen teacher with identical architecture/config.
        if model_cfg is not None:
            teacher_model = build_model_fn(model_cfg)
        else:
            teacher_model = rtdetr_pose_cls(
                num_classes=int(args.num_classes) + 1,
                num_keypoints=int(getattr(args, "num_keypoints", 0) or 0),
                hidden_dim=args.hidden_dim,
                num_queries=model_num_queries,
                num_decoder_layers=2,
                nhead=4,
                use_uncertainty=bool(args.use_uncertainty),
            )
        load_checkpoint_into_fn(teacher_model, None, args.self_distill_from)
        teacher_model.to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)

        keys = tuple(k.strip() for k in str(args.self_distill_keys).split(",") if k.strip())
        sdft_cfg = SdftConfig(
            weight=float(args.self_distill_weight),
            temperature=float(args.self_distill_temperature),
            kl=str(args.self_distill_kl),
            keys=keys,
            logits_weight=float(args.self_distill_logits_weight),
            bbox_weight=float(args.self_distill_bbox_weight),
            other_l1_weight=float(args.self_distill_other_l1_weight),
        )
        if is_main:
            print(
                "self_distill",
                f"from={args.self_distill_from}",
                f"keys={','.join(keys) if keys else '(none)'}",
                f"kl={sdft_cfg.kl}",
                f"temp={sdft_cfg.temperature}",
                f"weight={sdft_cfg.weight}",
            )

    derpp_cfg = None
    if bool(args.derpp):
        derpp_cfg = SdftConfig(
            weight=float(args.derpp_weight),
            temperature=float(args.derpp_temperature),
            kl=str(args.derpp_kl),
            keys=tuple(k.strip() for k in str(args.derpp_keys).split(",") if k.strip()),
            logits_weight=float(args.derpp_logits_weight),
            bbox_weight=float(args.derpp_bbox_weight),
            other_l1_weight=float(args.derpp_other_l1_weight),
        )
        if is_main:
            print(
                "derpp",
                f"enabled=True teacher_key={args.derpp_teacher_key}",
                f"keys={','.join(derpp_cfg.keys) if derpp_cfg.keys else '(none)'}",
                f"kl={derpp_cfg.kl}",
                f"temp={derpp_cfg.temperature}",
                f"weight={derpp_cfg.weight}",
            )

    return teacher_model, sdft_cfg, derpp_cfg


def setup_continual_regularizers(
    *,
    args: Any,
    device: Any,
    model: Any,
    is_main: bool,
    unwrap_model_fn: Callable[[Any], Any],
) -> tuple[Any, Any, Callable[..., Any] | None, Callable[..., Any] | None, Any, Any, Callable[..., Any] | None, Callable[..., Any] | None]:
    ewc_state = None
    ewc_accum = None
    save_ewc_state_fn = None
    ewc_penalty_fn = None
    if bool(getattr(args, "ewc", False)) or args.ewc_state_in or args.ewc_state_out:
        from yolozu.continual_regularizers import EwcAccumulator, ewc_penalty, load_ewc_state, save_ewc_state

        save_ewc_state_fn = save_ewc_state
        ewc_penalty_fn = ewc_penalty
        if args.ewc_state_in:
            ewc_state = load_ewc_state(str(args.ewc_state_in)).to(device)
        if args.ewc_state_out:
            ewc_accum = EwcAccumulator()
        if is_main and bool(getattr(args, "ewc", False)):
            print(
                "ewc",
                f"enabled=True lambda={float(args.ewc_lambda)}",
                f"state_in={args.ewc_state_in}",
                f"state_out={args.ewc_state_out}",
            )

    si_state = None
    si_accum = None
    save_si_state_fn = None
    si_penalty_fn = None
    if bool(getattr(args, "si", False)) or args.si_state_in or args.si_state_out:
        from yolozu.continual_regularizers import SiAccumulator, load_si_state, save_si_state, si_penalty

        save_si_state_fn = save_si_state
        si_penalty_fn = si_penalty
        si_accum = SiAccumulator(epsilon=float(args.si_epsilon))
        if args.si_state_in:
            si_state = load_si_state(str(args.si_state_in)).to(device)
            si_accum.load_state(load_si_state(str(args.si_state_in)))
        si_accum.begin_task(unwrap_model_fn(model))
        if is_main and bool(getattr(args, "si", False)):
            print(
                "si",
                f"enabled=True c={float(args.si_c)} epsilon={float(args.si_epsilon)}",
                f"state_in={args.si_state_in}",
                f"state_out={args.si_state_out}",
            )

    return (
        ewc_state,
        ewc_accum,
        save_ewc_state_fn,
        ewc_penalty_fn,
        si_state,
        si_accum,
        save_si_state_fn,
        si_penalty_fn,
    )


def install_termination_handlers(*, is_main: bool) -> dict[str, bool]:
    terminate_flag = {"terminate": False}

    def _handle_term(signum, _frame):  # type: ignore[no-untyped-def]
        terminate_flag["terminate"] = True
        if is_main:
            print(f"signal_received={int(signum)} saving_last_checkpoint_and_exiting")

    try:
        signal.signal(signal.SIGTERM, _handle_term)
        signal.signal(signal.SIGINT, _handle_term)
    except (AttributeError, OSError, ValueError) as exc:
        logger.debug("failed to install termination handlers", exc_info=exc)
    return terminate_flag


def build_validation_loader(
    *,
    args: Any,
    val_records: list[dict[str, Any]],
    keypoint_flip_pairs: list[tuple[int, int]],
    loader_kwargs: dict[str, Any],
    manifest_dataset_cls: Any,
    dataloader_cls: Any,
) -> Any:
    if not val_records:
        return None

    val_ds = manifest_dataset_cls(
        val_records,
        num_queries=args.num_queries,
        num_classes=args.num_classes,
        num_keypoints=args.num_keypoints,
        keypoint_flip_pairs=keypoint_flip_pairs,
        image_size=args.image_size,
        seed=args.seed,
        use_matcher=False,
        synthetic_pose=False,
        z_from_dobj=False,
        load_aux=False,
        depth_mode=args.depth_mode,
        depth_unit=args.depth_unit,
        depth_scale=args.depth_scale,
        real_images=bool(args.real_images),
        multiscale=False,
        scale_min=1.0,
        scale_max=1.0,
        hflip_prob=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        hsv_prob=0.0,
        gray_prob=0.0,
        gaussian_noise_std=0.0,
        gaussian_noise_prob=0.0,
        blur_prob=0.0,
        blur_sigma=0.0,
        blur_kernel=3,
        intrinsics_jitter=False,
        jitter_dfx=0.0,
        jitter_dfy=0.0,
        jitter_dcx=0.0,
        jitter_dcy=0.0,
        sim_jitter=False,
        sim_jitter_profile=None,
        sim_jitter_extrinsics=False,
        extrinsics_jitter=False,
        jitter_dx=0.0,
        jitter_dy=0.0,
        jitter_dz=0.0,
        jitter_droll=0.0,
        jitter_dpitch=0.0,
        jitter_dyaw=0.0,
    )
    val_batch_size = int(args.batch_size)
    if args.val_batch_size is not None:
        try:
            val_batch_size = int(args.val_batch_size)
        except (TypeError, ValueError, OverflowError):
            val_batch_size = int(args.batch_size)
    val_loader_kwargs = dict(loader_kwargs)
    val_loader_kwargs.update(
        {
            "batch_size": int(val_batch_size),
            "shuffle": False,
            "sampler": None,
        }
    )
    return dataloader_cls(val_ds, **val_loader_kwargs)


def run_validation(
    *,
    args: Any,
    kind: str,
    epoch: int,
    optim_step: int,
    step: int | None,
    model: Any,
    ema: Any,
    device: Any,
    val_loader: Any,
    val_records_map: list[dict[str, Any]],
    best_map50_95: float,
    running: float,
    steps: int,
    last_loss_dict: dict[str, Any] | None,
    optim: Any,
    sched: Any,
    scaler: Any,
    run_record: dict[str, Any],
) -> tuple[tuple[float, float] | None, float]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required")

    if getattr(args, "val_metrics_jsonl", None) is None:
        return None, float(best_map50_95)

    if val_loader is None or not val_records_map:
        report = build_report(
            losses={},
            metrics={"skipped": True, "reason": "no_val_split"},
            meta={"kind": str(kind), "epoch": int(epoch), "optim_step": int(optim_step)},
        )
        append_jsonl(args.val_metrics_jsonl, report)
        return None, float(best_map50_95)

    model_was_training = bool(model.training)
    model.eval()
    if ema is not None and bool(getattr(args, "ema_eval", False)):
        ema.apply_shadow()

    preds: list[dict[str, Any]] = []
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    with torch.no_grad():
        for v_images, v_targets in val_loader:
            v_images = v_images.to(device)
            v_depth = None
            v_depth_valid = None
            if isinstance(v_targets, dict):
                vd = v_targets.get("depth")
                vv = v_targets.get("depth_valid")
                if isinstance(vd, torch.Tensor):
                    v_depth = vd.to(device)
                if isinstance(vv, torch.Tensor):
                    v_depth_valid = vv.to(device=device, dtype=torch.bool)
            v_out = model(v_images, depth=v_depth, depth_valid=v_depth_valid)
            image_paths: list[str] = []
            if isinstance(v_targets, list):
                image_paths = [
                    str(t.get("image_path", "") or "")
                    for t in v_targets
                    if isinstance(t, dict)
                ]
            elif isinstance(v_targets, dict):
                per = v_targets.get("per_sample")
                if isinstance(per, list):
                    image_paths = [
                        str(t.get("image_path", "") or "")
                        for t in per
                        if isinstance(t, dict)
                    ]
            preds.extend(
                decode_detections_from_outputs(
                    v_out,
                    image_paths,
                    score_thresh=float(getattr(args, "val_score_thresh", 0.001) or 0.0),
                    topk=int(getattr(args, "val_topk", 300) or 300),
                )
            )

    res = evaluate_map(val_records_map, preds, iou_thresholds=thresholds)
    map50_95 = float(getattr(res, "map50_95", 0.0))
    prev_best = float(best_map50_95)
    is_best = bool(map50_95 > prev_best)
    next_best = float(map50_95 if is_best else prev_best)

    if is_best and getattr(args, "best_checkpoint_out", None):
        save_checkpoint_bundle(
            args.best_checkpoint_out,
            model=unwrap_model(model),
            optim=optim,
            sched=sched,
            scaler=scaler,
            ema=ema,
            args=args,
            epoch=int(epoch),
            global_step=int(optim_step),
            last_epoch_steps=int(steps),
            last_epoch_avg=(running / max(1, steps)) if steps > 0 else None,
            last_loss_dict=last_loss_dict,
            run_record=run_record,
        )

    metrics = {
        "map50": float(getattr(res, "map50", 0.0)),
        "map50_95": float(map50_95),
        "images": int(len(val_records_map)),
        "best": bool(is_best),
    }
    if step is not None:
        metrics["step"] = int(step)
    report = build_report(
        losses={},
        metrics=metrics,
        meta={"kind": str(kind), "epoch": int(epoch), "optim_step": int(optim_step)},
    )
    append_jsonl(args.val_metrics_jsonl, report)

    if ema is not None and bool(getattr(args, "ema_eval", False)):
        ema.restore()
    if model_was_training:
        model.train()

    return (float(map50_95), prev_best), next_best

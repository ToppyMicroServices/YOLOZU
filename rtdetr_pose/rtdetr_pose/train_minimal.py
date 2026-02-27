"""Minimal RTDETRPose training scaffold.

The heavy lifting is split across submodules:
- train_cli: CLI argument parsing (build_parser, parse_args, etc.)
- train_utils: utility functions (checkpoint I/O, augmentation, scheduling, etc.)
- train_dataset: ManifestDataset, collate, _pad_field
"""

import argparse
import json
import math
import os
import random
import signal
import shutil
import socket
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    DataLoader = None
    Dataset = object

from rtdetr_pose.dataset import build_manifest
from rtdetr_pose.dataset import extract_full_gt_targets, depth_at_bbox_center
from rtdetr_pose.factory import build_losses, build_model
from rtdetr_pose.losses import Losses
from rtdetr_pose.training import build_query_aligned_targets
from rtdetr_pose.model import RTDETRPose
from rtdetr_pose.optim_factory import build_optimizer
from rtdetr_pose.sched_factory import EMA, build_scheduler

from yolozu.metrics_report import append_jsonl, build_report, write_csv_row, write_json
from yolozu.jitter import default_jitter_profile, sample_intrinsics_jitter, sample_extrinsics_jitter
from yolozu.long_tail_metrics import build_fracal_stats
from yolozu.run_record import build_run_record, validate_run_record_contract
from yolozu.sdft import SdftConfig, compute_sdft_loss
from yolozu.simple_map import evaluate_map

# ---------------------------------------------------------------------------
# Re-exports from submodules (backward compatibility)
# ---------------------------------------------------------------------------
from rtdetr_pose.train_cli import (  # noqa: F401
    load_config_file,
    build_parser,
    _default_run_id,
    apply_run_contract_defaults,
    apply_run_dir_defaults,
    parse_args,
)
from rtdetr_pose.train_utils import (  # noqa: F401
    workspace_root,
    _now_utc,
    _normalize_keypoint_names,
    _derive_keypoint_flip_pairs,
    _extract_manifest_keypoints_meta,
    unwrap_model,
    _quantiles,
    _diff_stats,
    _softmax,
    _sigmoid,
    _derive_score_bbox,
    run_onnxrt_parity,
    collect_torch_cuda_meta,
    collect_rng_state,
    restore_rng_state,
    _rotation_matrix_from_rpy,
    compute_warmup_lr,
    parse_milestones,
    apply_denoise_targets,
    flatten_records_for_map,
    decode_detections_from_outputs,
    plan_accumulation_windows,
    compute_linear_schedule,
    compute_mim_schedule,
    compute_stage_weights,
    compute_stage_costs,
    generate_block_mask,
    _rgb_to_hsv,
    _hsv_to_rgb,
    apply_hsv_jitter,
    apply_grayscale,
    _gaussian_kernel2d,
    apply_gaussian_blur,
    create_geom_input_from_bboxes,
    compute_grad_norm,
    load_checkpoint_into,
    save_checkpoint_bundle,
)
from rtdetr_pose.train_dataset import (  # noqa: F401
    ManifestDataset,
    _pad_field,
    collate,
)


def main(argv: list[str] | None = None) -> int:
    if torch is None:  # pragma: no cover
        raise SystemExit("torch is required; install requirements-test.txt")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if bool(getattr(args, "use_amp", False)) and str(getattr(args, "amp", "none") or "none").lower() == "none":
        args.amp = "fp16"

    if bool(getattr(args, "qlora", False)):
        if int(getattr(args, "lora_r", 0) or 0) <= 0:
            raise SystemExit("--qlora requires --lora-r > 0")
        torchao_quant = str(getattr(args, "torchao_quant", "none") or "none").strip().lower()
        if torchao_quant in ("none", "off", "false", "0"):
            args.torchao_quant = "int4wo"
        args.lora_freeze_base = True

    args.depth_mode = str(getattr(args, "depth_mode", "none") or "none").strip().lower()
    args.depth_unit = str(getattr(args, "depth_unit", "unspecified") or "unspecified").strip().lower()
    args.depth_scale = float(getattr(args, "depth_scale", 1.0) or 1.0)
    args.depth_dropout = max(0.0, min(1.0, float(getattr(args, "depth_dropout", 0.0) or 0.0)))

    args, run_contract = apply_run_contract_defaults(args)
    args, run_dir = apply_run_dir_defaults(args)

    if run_contract is not None:
        if not args.config:
            raise SystemExit("--run-contract requires --config (train_setting.yaml).")
        # Enforce that contract-critical inputs are explicitly present in the YAML/JSON config
        # (not just defaulted by argparse). This makes runs reproducible by construction.
        cfg_obj = load_config_file(str(args.config))
        if not isinstance(cfg_obj, dict):
            raise SystemExit(f"{args.config}: config must be an object at top-level")

        missing: list[str] = []
        if "config_version" not in cfg_obj:
            missing.append("config_version")
        if not (isinstance(cfg_obj.get("dataset_root"), str) and str(cfg_obj.get("dataset_root")).strip()):
            missing.append("dataset_root")
        if "seed" not in cfg_obj:
            missing.append("seed")
        if not (isinstance(cfg_obj.get("device"), str) and str(cfg_obj.get("device")).strip()):
            missing.append("device")
        if "ddp" not in cfg_obj:
            missing.append("ddp")
        if "amp" not in cfg_obj and "use_amp" not in cfg_obj:
            missing.append("amp (or use_amp)")

        if missing:
            raise SystemExit(
                f"{args.config}: run contract requires explicit keys in the config: {', '.join(missing)}"
                "\nExample additions:\n  amp: none\n  ddp: false\n"
            )
        if args.config_version is None:
            raise SystemExit("config_version is required for --run-contract (set config_version: 1 in YAML).")
        if int(args.config_version) != 1:
            raise SystemExit(f"unsupported config_version: {args.config_version} (expected: 1)")

    if bool(getattr(args, "print_config", False)):
        payload = vars(args)
        try:
            import yaml  # type: ignore

            print(yaml.safe_dump(payload, sort_keys=True))
        except Exception:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if bool(getattr(args, "dry_run", False)):
        # Ensure a minimal, fast wiring check that still executes at least one optimizer step
        # (including logging/checkpoint/export paths).
        args.epochs = 1
        try:
            args.max_steps = max(1, int(getattr(args, "grad_accum", 1) or 1))
        except Exception:
            args.max_steps = 1

    # Optional DDP (torchrun sets WORLD_SIZE/RANK/LOCAL_RANK)
    world_size_env = int(os.environ.get("WORLD_SIZE", "1") or "1")
    ddp_enabled = bool(args.ddp) or world_size_env > 1
    if bool(args.ddp) and world_size_env <= 1:
        raise SystemExit("--ddp requires torchrun (WORLD_SIZE>1). Example: torchrun --nproc_per_node=2 ... --ddp")
    rank = int(os.environ.get("RANK", "0") or "0") if ddp_enabled else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0") if ddp_enabled else 0
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1") if ddp_enabled else 1
    if ddp_enabled:
        backend = str(args.ddp_backend or ("nccl" if torch.cuda.is_available() else "gloo"))
        torch.distributed.init_process_group(backend=backend, init_method="env://")

    is_main = rank == 0
    if ddp_enabled and not is_main:
        # Keep multi-rank runs readable by silencing stdout on non-main ranks.
        # Stderr is preserved for tracebacks and error diagnostics.
        try:
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
        except TypeError:  # pragma: no cover
            sys.stdout = open(os.devnull, "w")

    sim_profile = None
    if args.sim_jitter:
        sim_profile = default_jitter_profile()
        if args.sim_jitter_profile:
            path = Path(args.sim_jitter_profile)
            if not path.exists():
                raise SystemExit(f"sim jitter profile not found: {path}")
            try:
                sim_profile = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise SystemExit(f"failed to load sim jitter profile: {path}") from exc

    run_record = build_run_record(
        repo_root=workspace_root,
        argv=(sys.argv[1:] if argv is None else argv),
        args=vars(args),
        dataset_root=(args.dataset_root or None),
        extra={
            "timestamp_utc": _now_utc(),
            "ddp": {"enabled": bool(ddp_enabled), "backend": (str(args.ddp_backend) if args.ddp_backend else None), "rank": rank, "local_rank": local_rank, "world_size": world_size},
            "cuda": collect_torch_cuda_meta(),
            "host": {"hostname": socket.gethostname(), "pid": os.getpid()},
        },
    )
    try:
        validate_run_record_contract(run_record, require_git_sha=True)
    except Exception as exc:
        raise SystemExit(f"invalid run_meta contract: {exc}") from exc

    seed = int(getattr(args, "seed", 0) or 0)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    if bool(getattr(args, "deterministic", False)) and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if is_main and getattr(args, "config_resolved_out", None):
        out_path = Path(str(args.config_resolved_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = vars(args)
        try:
            import yaml  # type: ignore

            out_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
        except Exception:
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        dataset_root = workspace_root / "data" / "coco128"
        if not dataset_root.exists():
            dataset_root = workspace_root.parent / "data" / "coco128"

    model_cfg = None
    loss_cfg = None
    model_cfg_path = getattr(args, "model_config", None) or getattr(args, "config", None)
    if model_cfg_path:
        try:
            from rtdetr_pose.config import load_config
        except Exception:
            load_config = None
        if load_config is not None:
            try:
                cfg_obj = load_config(model_cfg_path)
                model_cfg = cfg_obj.model
                loss_cfg = getattr(cfg_obj, "loss", None)
            except Exception:
                model_cfg = None
                loss_cfg = None
    if model_cfg is not None:
        args.num_queries = int(model_cfg.num_queries)
        args.num_classes = int(model_cfg.num_classes)
        if getattr(model_cfg, "num_keypoints", None) is not None:
            args.num_keypoints = int(getattr(model_cfg, "num_keypoints"))

    records = None
    keypoint_names: list[str] = []
    keypoint_skeleton: list[list[int]] = []
    if args.records_json:
        records_path = Path(str(args.records_json))
        if not records_path.is_absolute():
            records_path = (workspace_root / records_path).resolve()
            if not records_path.exists():
                records_path = (workspace_root.parent / Path(str(args.records_json))).resolve()
        if not records_path.exists():
            raise SystemExit(f"records json not found: {records_path}")
        loaded = json.loads(records_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "images" in loaded:
            loaded = loaded.get("images")
        if not isinstance(loaded, list):
            raise SystemExit(f"records json must be a list or {{images:[...]}}: {records_path}")
        records = [r for r in loaded if isinstance(r, dict)]
    else:
        manifest = build_manifest(dataset_root, split=args.split)
        records = manifest.get("images") or []
        keypoint_names, keypoint_skeleton = _extract_manifest_keypoints_meta(manifest)
    if args.extra_records_json:
        extra_path = Path(str(args.extra_records_json))
        if not extra_path.is_absolute():
            extra_path = (workspace_root / extra_path).resolve()
            if not extra_path.exists():
                extra_path = (workspace_root.parent / Path(str(args.extra_records_json))).resolve()
        if not extra_path.exists():
            raise SystemExit(f"extra records json not found: {extra_path}")
        loaded = json.loads(extra_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "images" in loaded:
            loaded = loaded.get("images")
        if not isinstance(loaded, list):
            raise SystemExit(f"extra records json must be a list or {{images:[...]}}: {extra_path}")
        extra = [r for r in loaded if isinstance(r, dict)]
        if extra:
            records = list(records) + extra
    if not records:
        raise SystemExit(
            f"No records found under {dataset_root}. "
            "Fetch coco128 first: bash tools/fetch_coco128.sh"
        )

    if int(getattr(args, "num_keypoints", 0) or 0) <= 0 and keypoint_names:
        args.num_keypoints = int(len(keypoint_names))
        if is_main:
            print(
                "auto_num_keypoints "
                f"count={int(args.num_keypoints)} source=dataset_meta"
            )

    keypoint_flip_pairs = _derive_keypoint_flip_pairs(keypoint_names) if keypoint_names else []
    if is_main and keypoint_names:
        print(
            "keypoint_meta "
            f"count={int(len(keypoint_names))} skeleton_edges={int(len(keypoint_skeleton))} flip_pairs={int(len(keypoint_flip_pairs))}"
        )

    if is_main:
        stats = {
            "mask": 0,
            "depth": 0,
            "pose": 0,
            "intrinsics": 0,
            "cad_points": 0,
        }
        for rec in records:
            if rec.get("mask_path") is not None:
                stats["mask"] += 1
            if rec.get("depth_path") is not None:
                stats["depth"] += 1
            if rec.get("R_gt") is not None or rec.get("t_gt") is not None or rec.get("pose") is not None:
                stats["pose"] += 1
            if rec.get("K_gt") is not None or rec.get("intrinsics") is not None:
                stats["intrinsics"] += 1
            if rec.get("cad_points") is not None:
                stats["cad_points"] += 1
        print(
            "dataset_stats "
            + " ".join(f"{key}={value}" for key, value in sorted(stats.items()))
        )
        depth_mode = str(getattr(args, "depth_mode", "none") or "none").strip().lower()
        depth_unit = str(getattr(args, "depth_unit", "unspecified") or "unspecified").strip().lower()
        depth_used = bool(depth_mode != "none" and int(stats.get("depth", 0)) > 0)
        run_record["depth_used"] = bool(depth_used)
        run_record["depth_unit"] = depth_unit
        run_record["depth_scale"] = float(getattr(args, "depth_scale", 1.0) or 1.0)
        run_record["depth_mode"] = depth_mode

    if is_main and getattr(args, "run_meta_out", None):
        out_path = Path(str(args.run_meta_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(run_record, indent=2, sort_keys=True), encoding="utf-8")

    if is_main and getattr(args, "fracal_stats_out", None):
        stats_path = Path(str(args.fracal_stats_out))
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        fracal_stats = build_fracal_stats(
            records,
            task=str(getattr(args, "fracal_stats_task", "bbox") or "bbox"),
            allow_rgb_masks=bool(getattr(args, "fracal_allow_rgb_masks", False)),
        )
        fracal_stats["source"] = {
            "kind": "train_records",
            "dataset_root": str(dataset_root),
            "split": (None if args.records_json else str(getattr(args, "split", "") or "")),
            "records_json": (str(args.records_json) if getattr(args, "records_json", None) else None),
        }
        stats_path.write_text(json.dumps(fracal_stats, indent=2, sort_keys=True), encoding="utf-8")
        print(
            "fracal_stats "
            f"task={str(fracal_stats.get('task', 'bbox'))} classes={int(fracal_stats.get('summary', {}).get('classes', 0))} "
            f"instances_total={int(fracal_stats.get('summary', {}).get('instances_total', 0))} "
            f"out={stats_path}"
        )

    val_split = str(args.val_split) if args.val_split else None
    if val_split is None:
        candidate = dataset_root / "images" / "val2017"
        if candidate.exists():
            val_split = "val2017"

    val_records: list[dict[str, Any]] = []
    if val_split:
        try:
            val_manifest = build_manifest(dataset_root, split=val_split)
            val_records = val_manifest.get("images") or []
            if not isinstance(val_records, list):
                val_records = []
        except Exception:
            val_records = []

    if val_records and int(getattr(args, "val_max_images", 0) or 0) > 0:
        val_records = list(val_records)[: int(args.val_max_images)]
    val_records_map = flatten_records_for_map(val_records)

    derpp_keys = ()
    if bool(args.derpp):
        derpp_keys = tuple(k.strip() for k in str(args.derpp_keys).split(",") if k.strip())

    ds = ManifestDataset(
        records,
        num_queries=args.num_queries,
        num_classes=args.num_classes,
        num_keypoints=args.num_keypoints,
        keypoint_flip_pairs=keypoint_flip_pairs,
        image_size=args.image_size,
        seed=args.seed,
        use_matcher=args.use_matcher,
        synthetic_pose=args.synthetic_pose,
        z_from_dobj=args.z_from_dobj,
        load_aux=args.load_aux,
        depth_mode=args.depth_mode,
        depth_unit=args.depth_unit,
        depth_scale=args.depth_scale,
        real_images=args.real_images,
        multiscale=args.multiscale,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        hflip_prob=args.hflip_prob,
        hsv_h=float(getattr(args, "hsv_h", 0.0) or 0.0),
        hsv_s=float(getattr(args, "hsv_s", 0.0) or 0.0),
        hsv_v=float(getattr(args, "hsv_v", 0.0) or 0.0),
        hsv_prob=float(getattr(args, "hsv_prob", 1.0) or 1.0),
        gray_prob=float(getattr(args, "gray_prob", 0.0) or 0.0),
        gaussian_noise_std=float(getattr(args, "gaussian_noise_std", 0.0) or 0.0),
        gaussian_noise_prob=float(getattr(args, "gaussian_noise_prob", 1.0) or 1.0),
        blur_prob=float(getattr(args, "blur_prob", 0.0) or 0.0),
        blur_sigma=float(getattr(args, "blur_sigma", 0.0) or 0.0),
        blur_kernel=int(getattr(args, "blur_kernel", 3) or 3),
        intrinsics_jitter=args.intrinsics_jitter,
        jitter_dfx=args.jitter_dfx,
        jitter_dfy=args.jitter_dfy,
        jitter_dcx=args.jitter_dcx,
        jitter_dcy=args.jitter_dcy,
        sim_jitter=args.sim_jitter,
        sim_jitter_profile=sim_profile,
        sim_jitter_extrinsics=args.sim_jitter_extrinsics,
        extrinsics_jitter=args.extrinsics_jitter,
        jitter_dx=args.jitter_dx,
        jitter_dy=args.jitter_dy,
        jitter_dz=args.jitter_dz,
        jitter_droll=args.jitter_droll,
        jitter_dpitch=args.jitter_dpitch,
        jitter_dyaw=args.jitter_dyaw,
        derpp_enabled=bool(args.derpp),
        derpp_teacher_key=str(args.derpp_teacher_key),
        derpp_keys=derpp_keys,
    )
    sampler = None
    if ddp_enabled:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=bool(args.shuffle),
            seed=int(args.seed),
            drop_last=False,
        )
    num_workers = int(getattr(args, "num_workers", 0) or 0)
    persistent_workers = bool(getattr(args, "persistent_workers", False)) if num_workers > 0 else False
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "shuffle": (bool(args.shuffle) if sampler is None else False),
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": collate,
        "drop_last": False,
        "pin_memory": bool(getattr(args, "pin_memory", False)),
        "persistent_workers": persistent_workers,
        "generator": (torch.Generator().manual_seed(int(args.seed)) if args.deterministic and sampler is None else None),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 2) or 2)
    loader = DataLoader(ds, **loader_kwargs)

    model_num_queries = model_cfg.num_queries if model_cfg is not None else args.num_queries
    args.num_queries = int(model_num_queries)

    if model_cfg is not None:
        if getattr(model_cfg, "enable_mim", None) is not None:
            try:
                model_cfg.enable_mim = bool(args.enable_mim)
            except Exception:
                pass
        if getattr(model_cfg, "mim_geom_channels", None) is not None:
            try:
                model_cfg.mim_geom_channels = int(getattr(model_cfg, "mim_geom_channels", 2) or 2)
            except Exception:
                pass
        if getattr(model_cfg, "depth_mode", None) is not None:
            try:
                model_cfg.depth_mode = str(args.depth_mode)
            except Exception:
                pass
        if getattr(model_cfg, "depth_dropout", None) is not None:
            try:
                model_cfg.depth_dropout = float(args.depth_dropout)
            except Exception:
                pass
        model = build_model(model_cfg)
    else:
        model = RTDETRPose(
            num_classes=int(args.num_classes) + 1,
            num_keypoints=int(getattr(args, "num_keypoints", 0) or 0),
            hidden_dim=args.hidden_dim,
            num_queries=model_num_queries,
            num_decoder_layers=2,
            nhead=4,
            use_uncertainty=bool(args.use_uncertainty),
            enable_mim=bool(args.enable_mim),
            mim_geom_channels=2,
            depth_mode=str(args.depth_mode),
            depth_dropout=float(args.depth_dropout),
        )

    if loss_cfg is not None:
        if args.task_aligner and args.task_aligner != "none":
            try:
                loss_cfg.task_aligner = str(args.task_aligner)
            except Exception:
                pass
        losses_fn = build_losses(loss_cfg)
    else:
        losses_fn = Losses(task_aligner=args.task_aligner)
    base_loss_weights = dict(getattr(losses_fn, "weights", {}) or {})
    absolute_depth_enabled = bool(str(args.depth_mode) != "none" and str(args.depth_unit) == "metric")
    if not absolute_depth_enabled:
        if float(getattr(args, "cost_z", 0.0) or 0.0) != 0.0 or float(getattr(args, "cost_t", 0.0) or 0.0) != 0.0:
            if is_main:
                print(
                    "depth_safety "
                    "disabled=cost_z,cost_t "
                    f"depth_mode={args.depth_mode} depth_unit={args.depth_unit}",
                    file=sys.stderr,
                )
        args.cost_z = 0.0
        args.cost_t = 0.0
        if "z" in base_loss_weights:
            base_loss_weights["z"] = 0.0
        if "t" in base_loss_weights:
            base_loss_weights["t"] = 0.0

    device_str = str(args.device).strip() if args.device is not None else "cpu"
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        if is_main:
            print("warning: cuda requested but not available; falling back to cpu")
        device_str = "cpu"
    if ddp_enabled and device_str.startswith("cuda"):
        if torch.cuda.is_available():
            torch.cuda.set_device(int(local_rank))
        device_str = f"cuda:{int(local_rank)}"
    device = torch.device(device_str)
    model.to(device)

    torchao_recipe = str(getattr(args, "torchao_quant", "none") or "none").strip().lower()
    if torchao_recipe not in ("none", "off", "false", "0"):
        from rtdetr_pose.torchao_integration import apply_torchao_quantization

        model, torchao_report = apply_torchao_quantization(
            unwrap_model(model),
            recipe=torchao_recipe,
            required=bool(getattr(args, "torchao_required", False)),
        )
        if is_main:
            print(
                "torchao",
                f"enabled={bool(torchao_report.enabled)} recipe={torchao_report.recipe}",
                f"applied={bool(torchao_report.applied)} api={torchao_report.api} reason={torchao_report.reason}",
            )

    if int(args.lora_r) > 0:
        from rtdetr_pose.lora import apply_lora, count_trainable_params, mark_only_lora_as_trainable

        replaced = apply_lora(
            unwrap_model(model),
            r=int(args.lora_r),
            alpha=(float(args.lora_alpha) if args.lora_alpha is not None else None),
            dropout=float(args.lora_dropout),
            target=str(args.lora_target),
        )
        trainable_info = None
        if bool(args.lora_freeze_base):
            trainable_info = mark_only_lora_as_trainable(unwrap_model(model), train_bias=str(args.lora_train_bias))
        if is_main:
            print(
                "lora",
                f"enabled=True replaced={int(replaced)} r={int(args.lora_r)} alpha={args.lora_alpha}",
                f"dropout={float(args.lora_dropout)} target={args.lora_target} freeze_base={bool(args.lora_freeze_base)}",
                f"trainable_params={int(count_trainable_params(unwrap_model(model)))} trainable_info={trainable_info}",
            )

    if bool(getattr(args, "torch_compile", False)):
        if not hasattr(torch, "compile"):
            raise SystemExit("--torch-compile requires torch.compile (PyTorch 2.x)")

        backend = str(getattr(args, "torch_compile_backend", "inductor") or "inductor")
        mode_raw = getattr(args, "torch_compile_mode", None)
        mode = None
        if mode_raw is not None:
            mode_str = str(mode_raw).strip()
            if mode_str and mode_str.lower() not in ("none", "null"):
                mode = mode_str
        fullgraph = bool(getattr(args, "torch_compile_fullgraph", False))
        dynamic = getattr(args, "torch_compile_dynamic", None)
        strict = bool(getattr(args, "torch_compile_strict", False))

        try:
            model = torch.compile(  # type: ignore[attr-defined]
                model,
                backend=backend,
                mode=mode,
                fullgraph=bool(fullgraph),
                dynamic=dynamic,
            )
            if is_main:
                print(
                    "torch_compile",
                    f"enabled=True backend={backend} mode={mode} fullgraph={bool(fullgraph)} dynamic={dynamic}",
                )
        except Exception as exc:
            if strict:
                raise SystemExit(f"torch.compile failed: {exc}") from exc
            if is_main:
                print(f"warning: torch.compile failed; continuing without compilation ({exc})")

    ema = None
    if bool(getattr(args, "use_ema", False)):
        ema = EMA(unwrap_model(model), decay=float(getattr(args, "ema_decay", 0.999)))

    optim = build_optimizer(
        unwrap_model(model),
        optimizer=str(getattr(args, "optimizer", "adamw") or "adamw"),
        lr=float(getattr(args, "lr", 1e-4) or 1e-4),
        weight_decay=float(getattr(args, "weight_decay", 0.01) or 0.0),
        momentum=float(getattr(args, "momentum", 0.9) or 0.0),
        nesterov=bool(getattr(args, "nesterov", False)),
        use_param_groups=bool(getattr(args, "use_param_groups", False)),
        backbone_lr_mult=float(getattr(args, "backbone_lr_mult", 1.0) or 1.0),
        head_lr_mult=float(getattr(args, "head_lr_mult", 1.0) or 1.0),
        backbone_wd_mult=float(getattr(args, "backbone_wd_mult", 1.0) or 1.0),
        head_wd_mult=float(getattr(args, "head_wd_mult", 1.0) or 1.0),
        wd_exclude_bias=bool(getattr(args, "wd_exclude_bias", True)),
        wd_exclude_norm=bool(getattr(args, "wd_exclude_norm", True)),
    )

    micro_steps_per_epoch = int(getattr(args, "max_steps", 0) or 0)
    grad_accum = max(1, int(getattr(args, "grad_accum", 1) or 1))
    optim_steps_per_epoch = max(1, (micro_steps_per_epoch + grad_accum - 1) // grad_accum) if micro_steps_per_epoch > 0 else 1
    total_optim_steps = max(1, int(getattr(args, "epochs", 1) or 1) * optim_steps_per_epoch)
    milestones = parse_milestones(getattr(args, "scheduler_milestones", None))
    sched = build_scheduler(
        optim,
        scheduler=str(getattr(args, "scheduler", "none") or "none"),
        total_steps=int(total_optim_steps),
        warmup_steps=int(getattr(args, "lr_warmup_steps", 0) or 0),
        warmup_init_lr=float(getattr(args, "lr_warmup_init", 0.0) or 0.0),
        min_lr=float(getattr(args, "min_lr", 0.0) or 0.0),
        milestones=milestones,
        gamma=float(getattr(args, "scheduler_gamma", 0.1) or 0.1),
    )

    # AMP setup (needed early so resume can restore scaler state).
    amp_mode = str(args.amp or "none").lower()
    scaler = None
    if amp_mode != "none" and device.type != "cuda":
        if is_main:
            print("warning: --amp requested on non-cuda device; disabling AMP")
        amp_mode = "none"
        args.amp = "none"
    if amp_mode != "none":
        if amp_mode == "fp16":
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                try:
                    scaler = torch.amp.GradScaler("cuda")
                except TypeError:
                    scaler = torch.amp.GradScaler(device="cuda")
            else:  # pragma: no cover
                scaler = torch.cuda.amp.GradScaler()
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            else:  # pragma: no cover
                autocast = torch.cuda.amp.autocast(dtype=torch.float16)
        elif amp_mode == "bf16":
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:  # pragma: no cover
                autocast = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            raise SystemExit(f"unknown --amp mode: {args.amp}")
    else:
        autocast = nullcontext()

    start_epoch = 0
    global_step = 0
    if args.resume_from:
        meta = load_checkpoint_into(
            unwrap_model(model),
            optim,
            args.resume_from,
            sched=sched,
            scaler=scaler,
            ema=ema,
            restore_rng=True,
        )
        if meta.get("epoch") is not None:
            try:
                start_epoch = int(meta["epoch"]) + 1
            except Exception:
                start_epoch = 0
        if meta.get("global_step") is not None:
            try:
                global_step = int(meta["global_step"])
            except Exception:
                global_step = 0
        if is_main:
            print(f"resumed_from={meta.get('path')} start_epoch={start_epoch} global_step={global_step}")

    sdft_cfg = None
    teacher_model = None
    if args.self_distill_from:
        # Build a frozen teacher with identical architecture/config.
        if model_cfg is not None:
            teacher_model = build_model(model_cfg)
        else:
            teacher_model = RTDETRPose(
                num_classes=int(args.num_classes) + 1,
                num_keypoints=int(getattr(args, "num_keypoints", 0) or 0),
                hidden_dim=args.hidden_dim,
                num_queries=model_num_queries,
                num_decoder_layers=2,
                nhead=4,
                use_uncertainty=bool(args.use_uncertainty),
            )
        load_checkpoint_into(teacher_model, None, args.self_distill_from)
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

    ewc_state = None
    ewc_accum = None
    if bool(getattr(args, "ewc", False)) or args.ewc_state_in or args.ewc_state_out:
        from yolozu.continual_regularizers import EwcAccumulator, ewc_penalty, load_ewc_state, save_ewc_state

        ewc_state = None
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
    if bool(getattr(args, "si", False)) or args.si_state_in or args.si_state_out:
        from yolozu.continual_regularizers import SiAccumulator, load_si_state, save_si_state, si_penalty

        si_state = None
        si_accum = SiAccumulator(epsilon=float(args.si_epsilon))
        if args.si_state_in:
            si_state = load_si_state(str(args.si_state_in)).to(device)
            si_accum.load_state(load_si_state(str(args.si_state_in)))
        si_accum.begin_task(unwrap_model(model))
        if is_main and bool(getattr(args, "si", False)):
            print(
                "si",
                f"enabled=True c={float(args.si_c)} epsilon={float(args.si_epsilon)}",
                f"state_in={args.si_state_in}",
                f"state_out={args.si_state_out}",
            )

    if ddp_enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(local_rank)] if device.type == "cuda" else None,
            output_device=int(local_rank) if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    terminate_requested = False

    def _handle_term(signum, _frame):  # type: ignore[no-untyped-def]
        nonlocal terminate_requested
        terminate_requested = True
        if is_main:
            print(f"signal_received={int(signum)} saving_last_checkpoint_and_exiting")

    try:
        signal.signal(signal.SIGTERM, _handle_term)
        signal.signal(signal.SIGINT, _handle_term)
    except Exception:
        pass

    val_loader = None
    if val_records:
        val_ds = ManifestDataset(
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
            except Exception:
                val_batch_size = int(args.batch_size)
        val_loader_kwargs = dict(loader_kwargs)
        val_loader_kwargs.update(
            {
                "batch_size": int(val_batch_size),
                "shuffle": False,
                "sampler": None,
            }
        )
        val_loader = DataLoader(val_ds, **val_loader_kwargs)

    model.train()
    last_loss_dict = None
    last_epoch_avg = None
    last_epoch_steps = 0
    last_grad_norm = None
    non_finite_skips = 0
    best_map50_95 = -float("inf")
    last_data_time_s = None
    last_step_time_s = None
    last_throughput = None
    last_max_vram_mb = None
    stop_training = False

    val_every_steps = int(getattr(args, "val_every_steps", 0) or 0)
    early_stop_patience = max(0, int(getattr(args, "early_stop_patience", 0) or 0))
    early_stop_min_delta = float(getattr(args, "early_stop_min_delta", 0.0) or 0.0)
    early_stop_bad = 0

    def _run_validation(*, kind: str, epoch: int, optim_step: int, step: int | None = None) -> tuple[float, float] | None:
        nonlocal best_map50_95

        if getattr(args, "val_metrics_jsonl", None) is None:
            return None

        if val_loader is None or not val_records_map:
            report = build_report(
                losses={},
                metrics={"skipped": True, "reason": "no_val_split"},
                meta={"kind": str(kind), "epoch": int(epoch), "optim_step": int(optim_step)},
            )
            append_jsonl(args.val_metrics_jsonl, report)
            return None

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
        if is_best:
            best_map50_95 = float(map50_95)
            if getattr(args, "best_checkpoint_out", None):
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
                    rng_state=collect_rng_state(),
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

        return float(map50_95), prev_best

    for epoch in range(int(start_epoch), int(args.epochs)):
        if stop_training:
            break
        if sampler is not None:
            sampler.set_epoch(int(epoch))
        if args.hflip_prob_start is not None and args.hflip_prob_end is not None:
            ds.hflip_prob = compute_linear_schedule(
                float(args.hflip_prob_start),
                float(args.hflip_prob_end),
                int(epoch),
                int(args.epochs),
            )
        running = 0.0
        steps = 0
        max_micro_steps = len(loader)
        if args.max_steps and int(args.max_steps) > 0:
            max_micro_steps = min(max_micro_steps, int(args.max_steps))

        grad_accum = max(1, int(args.grad_accum))
        windows = plan_accumulation_windows(max_micro_steps=int(max_micro_steps), grad_accum=int(grad_accum))
        window_idx = 0
        step_in_window = 0
        window_size = windows[0] if windows else int(grad_accum)

        prev_step_end = time.time()
        for images, targets in loader:
            if max_micro_steps and steps >= int(max_micro_steps):
                break

            step_start = time.time()
            data_time_s = float(step_start - prev_step_end)
            if device.type == "cuda":
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                except Exception:
                    pass
            if terminate_requested:
                if is_main and args.checkpoint_bundle_out:
                    save_checkpoint_bundle(
                        args.checkpoint_bundle_out,
                        model=unwrap_model(model),
                        optim=optim,
                        sched=sched,
                        scaler=scaler,
                        ema=ema,
                        args=args,
                        epoch=int(epoch),
                        global_step=int(global_step),
                        last_epoch_steps=int(steps),
                        last_epoch_avg=(running / max(1, steps)) if steps > 0 else None,
                        last_loss_dict=last_loss_dict,
                        run_record=run_record,
                        rng_state=collect_rng_state(),
                    )
                stop_training = True
                break

            if step_in_window == 0:
                optim.zero_grad(set_to_none=True)
                window_size = windows[window_idx] if window_idx < len(windows) else int(grad_accum)

            images = images.to(device)
            per_sample_targets = targets.get("per_sample") if isinstance(targets, dict) else targets
            if not isinstance(per_sample_targets, list):
                per_sample_targets = None
            depth_batch = None
            depth_valid_batch = None
            if isinstance(targets, dict):
                depth_val = targets.get("depth")
                if isinstance(depth_val, torch.Tensor):
                    depth_batch = depth_val.to(device)
                valid_val = targets.get("depth_valid")
                if isinstance(valid_val, torch.Tensor):
                    depth_valid_batch = valid_val.to(device=device, dtype=torch.bool)

            sync_step = step_in_window == int(window_size) - 1
            skip_backward = False
            skip_optim_step = False
            force_sync = False
            ddp_nosync = ddp_enabled and hasattr(model, "no_sync") and not sync_step
            sync_context = model.no_sync() if ddp_nosync else nullcontext()

            with sync_context:
                with autocast:
                    step_weights, _stage = compute_stage_weights(
                        base_loss_weights,
                        global_step=int(global_step),
                        stage_off_steps=int(args.stage_off_steps),
                        stage_k_steps=int(args.stage_k_steps),
                    )
                    if not bool(args.enable_mim) or int(global_step) < int(args.mim_start_step):
                        step_weights["mim"] = 0.0
                        step_weights["entropy"] = 0.0
                    losses_fn.weights = step_weights

                    matcher_costs = {
                        "cost_z": float(args.cost_z),
                        "cost_rot": float(args.cost_rot),
                        "cost_t": float(args.cost_t),
                    }
                    if bool(args.use_matcher):
                        matcher_costs = compute_stage_costs(
                            matcher_costs,
                            global_step=int(global_step),
                            cost_z_start_step=int(args.cost_z_start_step),
                            cost_rot_start_step=int(args.cost_rot_start_step),
                            cost_t_start_step=int(args.cost_t_start_step),
                        )

                    mim_active = (
                        bool(args.enable_mim)
                        and bool(args.use_matcher)
                        and per_sample_targets is not None
                        and int(global_step) >= int(args.mim_start_step)
                    )
                    geom_batch = None
                    mask_batch = None
                    if mim_active:
                        geom_h = int(images.shape[-2])
                        geom_w = int(images.shape[-1])
                        geom_list = []
                        mask_list = []
                        for i, tgt in enumerate(per_sample_targets):
                            bboxes = []
                            z_list = None
                            if isinstance(tgt, dict):
                                bb_t = tgt.get("gt_bbox")
                                if isinstance(bb_t, torch.Tensor):
                                    bboxes = bb_t.tolist()
                                z_t = tgt.get("gt_z")
                                if isinstance(z_t, torch.Tensor) and int(z_t.numel()) > 0:
                                    try:
                                        z_list = z_t.squeeze(-1).tolist()
                                    except Exception:
                                        z_list = None
                            geom_list.append(
                                create_geom_input_from_bboxes(
                                    bboxes,
                                    z_list,
                                    height=geom_h,
                                    width=geom_w,
                                )
                            )
                            mask_gen = torch.Generator()
                            mask_gen.manual_seed(int(args.seed) + int(global_step) * 1000 + int(i))
                            mask_list.append(
                                generate_block_mask(
                                    geom_h,
                                    geom_w,
                                    patch_size=int(args.mim_patch_size),
                                    mask_prob=float(args.mim_mask_prob),
                                    generator=mask_gen,
                                )
                            )
                        geom_batch = torch.stack(geom_list, dim=0).to(device=device)
                        mask_batch = torch.stack(mask_list, dim=0).to(device=device)

                    out = model(
                        images,
                        geom_input=geom_batch,
                        feature_mask=mask_batch,
                        return_mim=bool(mim_active),
                        depth=depth_batch,
                        depth_valid=depth_valid_batch,
                    )

                    sdft_total = None
                    sdft_parts = None
                    if teacher_model is not None and sdft_cfg is not None and float(sdft_cfg.weight) != 0.0:
                        with torch.no_grad():
                            with autocast:
                                teacher_out = teacher_model(images, depth=depth_batch, depth_valid=depth_valid_batch)
                        student_out_sdft = out
                        teacher_out_sdft = teacher_out
                        if (
                            "bbox" in sdft_cfg.keys
                            and isinstance(out.get("bbox"), torch.Tensor)
                            and isinstance(teacher_out.get("bbox"), torch.Tensor)
                        ):
                            student_out_sdft = dict(out)
                            teacher_out_sdft = dict(teacher_out)
                            student_out_sdft["bbox"] = out["bbox"].sigmoid()
                            teacher_out_sdft["bbox"] = teacher_out["bbox"].sigmoid()
                        sdft_total, sdft_parts = compute_sdft_loss(student_out_sdft, teacher_out_sdft, sdft_cfg)

                    derpp_total = None
                    derpp_parts = None
                    derpp_count = 0
                    if derpp_cfg is not None and float(derpp_cfg.weight) != 0.0 and per_sample_targets is not None:
                        indices: list[int] = []
                        teacher_by_key: dict[str, list[torch.Tensor]] = {k: [] for k in derpp_cfg.keys}
                        for i, tgt in enumerate(per_sample_targets):
                            if not isinstance(tgt, dict):
                                continue
                            teacher = tgt.get("derpp_teacher")
                            if not isinstance(teacher, dict):
                                continue
                            ok = True
                            for k in derpp_cfg.keys:
                                if not isinstance(teacher.get(k), torch.Tensor):
                                    ok = False
                                    break
                            if not ok:
                                continue
                            indices.append(int(i))
                            for k in derpp_cfg.keys:
                                teacher_by_key[k].append(teacher[k])

                        if indices:
                            idx = torch.tensor(indices, device=images.device, dtype=torch.long)
                            student_sub: dict[str, torch.Tensor] = {}
                            teacher_sub: dict[str, torch.Tensor] = {}
                            for k in derpp_cfg.keys:
                                s_val = out.get(k)
                                if not isinstance(s_val, torch.Tensor):
                                    continue
                                if int(s_val.shape[0]) <= int(idx.max().item()):
                                    continue
                                student_sub[k] = s_val.index_select(0, idx)
                                try:
                                    teacher_sub[k] = torch.stack(teacher_by_key[k], dim=0).to(device=images.device)
                                except Exception:
                                    teacher_sub.pop(k, None)
                                    student_sub.pop(k, None)
                            if student_sub and teacher_sub and "bbox" in derpp_cfg.keys:
                                if isinstance(student_sub.get("bbox"), torch.Tensor) and isinstance(
                                    teacher_sub.get("bbox"), torch.Tensor
                                ):
                                    student_sub = dict(student_sub)
                                    teacher_sub = dict(teacher_sub)
                                    student_sub["bbox"] = student_sub["bbox"].sigmoid()
                                    teacher_sub["bbox"] = teacher_sub["bbox"].sigmoid()
                            if student_sub and teacher_sub:
                                derpp_count = int(len(indices))
                                derpp_total, derpp_parts = compute_sdft_loss(student_sub, teacher_sub, derpp_cfg)

                    if args.use_matcher:
                        per_sample = per_sample_targets
                        if per_sample is None:
                            raise RuntimeError("use_matcher requires per-sample targets list")
                        aligned = build_query_aligned_targets(
                            out["logits"],
                            out["bbox"],
                            per_sample,
                            num_queries=model_num_queries,
                            cost_cls=args.cost_cls,
                            cost_bbox=args.cost_bbox,
                            log_z_pred=out.get("log_z"),
                            rot6d_pred=out.get("rot6d"),
                            cost_z=float(matcher_costs["cost_z"]),
                            cost_rot=float(matcher_costs["cost_rot"]),
                            offsets_pred=out.get("offsets"),
                            k_delta=out.get("k_delta"),
                            cost_t=float(matcher_costs["cost_t"]),
                            keypoints_pred=out.get("keypoints"),
                        )
                        out = dict(out)
                        # For box regression we train in normalized space.
                        out["bbox"] = aligned["bbox_norm"]
                        targets = {
                            "labels": aligned["labels"],
                            "bbox": aligned["bbox"],
                            "mask": aligned["mask"],
                            "z_gt": aligned["z_gt"],
                            "z_mask": aligned["z_mask"],
                            "R_gt": aligned["R_gt"],
                            "rot_mask": aligned["rot_mask"],
                            "offsets": aligned["offsets"],
                            "off_mask": aligned["off_mask"],
                            "t_gt": aligned["t_gt"],
                            "K_gt": aligned["K_gt"],
                            "image_hw": aligned["image_hw"],
                            "K_mask": aligned["K_mask"],
                            "t_mask": aligned["t_mask"],
                            "keypoints_gt": aligned.get("keypoints_gt"),
                            "keypoints_mask": aligned.get("keypoints_mask"),
                            "M_mask": aligned.get("M_mask"),
                            "D_obj_mask": aligned.get("D_obj_mask"),
                        }
                    else:
                        # legacy padded targets
                        targets = {
                            "labels": torch.stack([t["labels"] for t in targets], dim=0).to(device),
                            "bbox": torch.stack([t["bbox"] for t in targets], dim=0).to(device),
                        }

                    loss_dict = dict(losses_fn(out, targets))
                    loss_supervised = loss_dict["loss"]
                    loss = loss_supervised

                    if sdft_total is not None and sdft_parts is not None and sdft_cfg is not None:
                        loss_dict["loss_supervised"] = loss_supervised
                        loss_dict.update(sdft_parts)
                        loss = loss + float(sdft_cfg.weight) * sdft_total

                    if derpp_total is not None and derpp_parts is not None and derpp_cfg is not None:
                        loss_dict["derpp_samples"] = torch.tensor(int(derpp_count), device=loss.device)
                        loss_dict["loss_derpp"] = derpp_total
                        for k, v in derpp_parts.items():
                            if not isinstance(v, torch.Tensor):
                                continue
                            if str(k) == "loss_sdft":
                                continue
                            suffix = str(k).replace("loss_sdft_", "")
                            loss_dict[f"loss_derpp_{suffix}"] = v
                        loss = loss + float(derpp_cfg.weight) * derpp_total

                    if ewc_state is not None and float(args.ewc_lambda) != 0.0:
                        ewc_raw = ewc_penalty(unwrap_model(model), ewc_state)
                        ewc_term = 0.5 * float(args.ewc_lambda) * ewc_raw
                        loss_dict["loss_ewc"] = ewc_term
                        loss = loss + ewc_term

                    if si_state is not None and float(args.si_c) != 0.0:
                        si_raw = si_penalty(unwrap_model(model), si_state)
                        si_term = 0.5 * float(args.si_c) * si_raw
                        loss_dict["loss_si"] = si_term
                        loss = loss + si_term

                    loss_dict["loss"] = loss
                    last_loss_dict = loss_dict

                    if not bool(torch.isfinite(loss).all()):
                        try:
                            loss_scalar = float(loss.detach().cpu())
                        except Exception:
                            loss_scalar = None

                        if bool(args.stop_on_non_finite_loss):
                            raise SystemExit(f"non-finite loss at epoch={epoch} step={steps + 1}: {loss_scalar}")

                        non_finite_skips += 1
                        max_skips = max(1, int(getattr(args, "non_finite_max_skips", 3) or 3))
                        decay = float(getattr(args, "non_finite_lr_decay", 0.5) or 0.0)
                        if 0.0 < decay < 1.0:
                            for group in optim.param_groups:
                                try:
                                    group["lr"] = float(group.get("lr", 0.0)) * decay
                                except Exception:
                                    pass
                        if is_main and args.metrics_jsonl:
                            lr_now = None
                            try:
                                lr_now = float(optim.param_groups[0].get("lr"))
                            except Exception:
                                lr_now = None
                            metrics = {"non_finite_skips": int(non_finite_skips)}
                            if lr_now is not None:
                                metrics["lr"] = float(lr_now)
                            report = build_report(
                                losses={"loss": loss_scalar} if loss_scalar is not None else {},
                                metrics=metrics,
                                meta={
                                    "kind": "non_finite_loss",
                                    "epoch": int(epoch),
                                    "step": int(steps + 1),
                                    "optim_step": int(global_step),
                                },
                            )
                            append_jsonl(args.metrics_jsonl, report)
                        optim.zero_grad(set_to_none=True)
                        skip_backward = True
                        skip_optim_step = True
                        force_sync = True
                        if non_finite_skips >= max_skips:
                            raise SystemExit(
                                f"non-finite loss persisted: skips={non_finite_skips} (max={max_skips})"
                            )

                    if steps == 0 and args.debug_losses and is_main:
                        printable = {
                            k: float(v.detach().cpu())
                            for k, v in loss_dict.items()
                            if hasattr(v, "detach")
                        }
                        print("loss_breakdown", " ".join(f"{k}={v:.6g}" for k, v in sorted(printable.items())))

                    loss_for_backward = None
                    if not skip_backward:
                        loss_value = float(loss.detach().cpu())
                        running += loss_value
                        loss_for_backward = loss / float(window_size)

                if not skip_backward and loss_for_backward is not None:
                    if scaler is not None:
                        scaler.scale(loss_for_backward).backward()
                    else:
                        loss_for_backward.backward()

            sync_now = bool(sync_step or force_sync)
            did_optim_step = False
            if sync_now:
                if scaler is not None:
                    scaler.unscale_(optim)

                if si_accum is not None:
                    si_accum.capture_before_step(unwrap_model(model))
                if ewc_accum is not None:
                    ewc_accum.accumulate_from_grads(unwrap_model(model))

                grad_norm = None
                if args.clip_grad_norm and float(args.clip_grad_norm) > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))
                elif bool(args.log_grad_norm):
                    grad_norm = compute_grad_norm(model.parameters())
                if grad_norm is not None:
                    try:
                        last_grad_norm = float(grad_norm.detach().cpu())
                    except Exception:
                        last_grad_norm = None

                if grad_norm is not None and not bool(torch.isfinite(grad_norm).all()):
                    if bool(args.stop_on_non_finite_loss):
                        raise SystemExit(f"non-finite grad_norm at epoch={epoch} step={steps + 1}: {last_grad_norm}")
                    non_finite_skips += 1
                    max_skips = max(1, int(getattr(args, "non_finite_max_skips", 3) or 3))
                    decay = float(getattr(args, "non_finite_lr_decay", 0.5) or 0.0)
                    if 0.0 < decay < 1.0:
                        for group in optim.param_groups:
                            try:
                                group["lr"] = float(group.get("lr", 0.0)) * decay
                            except Exception:
                                pass
                    if is_main and args.metrics_jsonl:
                        lr_now = None
                        try:
                            lr_now = float(optim.param_groups[0].get("lr"))
                        except Exception:
                            lr_now = None
                        metrics = {"non_finite_skips": int(non_finite_skips)}
                        if lr_now is not None:
                            metrics["lr"] = float(lr_now)
                        report = build_report(
                            losses={},
                            metrics=metrics,
                            meta={
                                "kind": "non_finite_grad",
                                "epoch": int(epoch),
                                "step": int(steps + 1),
                                "optim_step": int(global_step),
                            },
                        )
                        append_jsonl(args.metrics_jsonl, report)
                    optim.zero_grad(set_to_none=True)
                    skip_optim_step = True
                    if non_finite_skips >= max_skips:
                        raise SystemExit(
                            f"non-finite grad persisted: skips={non_finite_skips} (max={max_skips})"
                        )

                if not skip_backward and not skip_optim_step:
                    if scaler is not None:
                        scaler.step(optim)
                        scaler.update()
                    else:
                        optim.step()
                    did_optim_step = True

                if did_optim_step:
                    if sched is not None:
                        try:
                            sched.step()
                        except Exception:
                            pass

                    if ema is not None:
                        try:
                            ema.update()
                        except Exception:
                            pass

                    if si_accum is not None:
                        si_accum.update_after_step(unwrap_model(model))

                    non_finite_skips = 0
                    global_step += 1

            step_end = time.time()
            last_data_time_s = float(data_time_s)
            last_step_time_s = float(step_end - step_start)
            prev_step_end = step_end
            if last_step_time_s > 0:
                scale = int(world_size) if ddp_enabled else 1
                last_throughput = float(int(images.shape[0]) * scale / last_step_time_s)
            if device.type == "cuda":
                try:
                    last_max_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
                except Exception:
                    last_max_vram_mb = None

            steps += 1

            avg = running / max(1, steps)
            if is_main and (steps == 1 or (args.log_every and steps % int(args.log_every) == 0)):
                print(f"epoch={epoch} step={steps} optim_step={global_step} loss={avg:.4f}")
            if is_main and args.metrics_jsonl and did_optim_step:
                losses_out = {k: float(v.detach().cpu()) for k, v in last_loss_dict.items() if hasattr(v, "detach")} if last_loss_dict is not None else {}
                lr_now = None
                try:
                    lr_now = float(optim.param_groups[0].get("lr"))
                except Exception:
                    lr_now = None
                metrics = {"loss_avg": float(avg), "optim_step": int(global_step)}
                if last_grad_norm is not None:
                    metrics["grad_norm"] = float(last_grad_norm)
                if lr_now is not None:
                    metrics["lr"] = float(lr_now)
                if last_data_time_s is not None:
                    metrics["data_time_s"] = float(last_data_time_s)
                if last_step_time_s is not None:
                    metrics["step_time_s"] = float(last_step_time_s)
                if last_throughput is not None:
                    metrics["throughput_img_s"] = float(last_throughput)
                if last_max_vram_mb is not None:
                    metrics["max_vram_mb"] = float(last_max_vram_mb)
                if ema is not None:
                    metrics["ema_decay"] = float(getattr(ema, "decay", 0.0))
                report = build_report(
                    losses=losses_out,
                    metrics=metrics,
                    meta={
                        "kind": "train_step",
                        "epoch": int(epoch),
                        "step": int(steps),
                        "optim_step": int(global_step),
                    },
                )
                append_jsonl(args.metrics_jsonl, report)

            if (
                is_main
                and did_optim_step
                and args.checkpoint_bundle_out
                and args.checkpoint_every
                and int(args.checkpoint_every) > 0
            ):
                every = int(args.checkpoint_every)
                if global_step % every == 0:
                    bundle_path = Path(args.checkpoint_bundle_out)
                    stepped = bundle_path.with_name(f"{bundle_path.stem}.step{global_step}{bundle_path.suffix or '.pt'}")
                    save_checkpoint_bundle(
                        stepped,
                        model=unwrap_model(model),
                        optim=optim,
                        sched=sched,
                        scaler=scaler,
                        ema=ema,
                        args=args,
                        epoch=epoch,
                        global_step=global_step,
                        last_epoch_steps=steps,
                        last_epoch_avg=(running / max(1, steps)),
                        last_loss_dict=last_loss_dict,
                        run_record=run_record,
                        rng_state=collect_rng_state(),
                    )

            if (
                is_main
                and did_optim_step
                and args.checkpoint_bundle_out
                and args.save_last_every
                and int(args.save_last_every) > 0
            ):
                every = int(args.save_last_every)
                if global_step % every == 0:
                    save_checkpoint_bundle(
                        args.checkpoint_bundle_out,
                        model=unwrap_model(model),
                        optim=optim,
                        sched=sched,
                        scaler=scaler,
                        ema=ema,
                        args=args,
                        epoch=epoch,
                        global_step=global_step,
                        last_epoch_steps=steps,
                        last_epoch_avg=(running / max(1, steps)),
                        last_loss_dict=last_loss_dict,
                        run_record=run_record,
                        rng_state=collect_rng_state(),
                    )

            val_due_steps = bool(
                getattr(args, "val_metrics_jsonl", None)
                and val_every_steps > 0
                and did_optim_step
                and int(global_step) > 0
                and int(global_step) % int(val_every_steps) == 0
            )
            if val_due_steps:
                if ddp_enabled and not is_main:
                    torch.distributed.barrier()
                if is_main:
                    res = _run_validation(
                        kind="val_step",
                        epoch=int(epoch),
                        optim_step=int(global_step),
                        step=int(steps),
                    )
                    if res is not None and early_stop_patience > 0:
                        map50_95, prev_best = res
                        improved = bool(float(map50_95) > float(prev_best) + float(early_stop_min_delta))
                        if improved:
                            early_stop_bad = 0
                        else:
                            early_stop_bad += 1
                            if early_stop_bad >= int(early_stop_patience):
                                stop_training = True
                                report = build_report(
                                    losses={},
                                    metrics={
                                        "early_stop": True,
                                        "patience": int(early_stop_patience),
                                        "bad": int(early_stop_bad),
                                        "min_delta": float(early_stop_min_delta),
                                        "best_map50_95": float(best_map50_95),
                                    },
                                    meta={
                                        "kind": "early_stop",
                                        "epoch": int(epoch),
                                        "optim_step": int(global_step),
                                    },
                                )
                                append_jsonl(args.val_metrics_jsonl, report)
                if ddp_enabled and is_main:
                    torch.distributed.barrier()
                if ddp_enabled:
                    flag_device = device if device.type == "cuda" else torch.device("cpu")
                    flag = torch.tensor([1 if stop_training else 0], dtype=torch.int64, device=flag_device)
                    torch.distributed.broadcast(flag, src=0)
                    stop_training = bool(int(flag.item()))
                if stop_training:
                    break

            if bool(getattr(args, "dry_run", False)) and did_optim_step:
                stop_training = True
                break

            if sync_now:
                window_idx += 1
                step_in_window = 0
            else:
                step_in_window += 1

        avg = running / max(1, steps)
        last_epoch_avg = float(avg)
        last_epoch_steps = int(steps)
        if is_main:
            print(f"epoch={epoch} done steps={steps} optim_step={global_step} loss={avg:.4f}")
        if is_main and args.metrics_jsonl and last_loss_dict is not None:
            losses_out = {k: float(v.detach().cpu()) for k, v in last_loss_dict.items() if hasattr(v, "detach")}
            lr_now = None
            try:
                lr_now = float(optim.param_groups[0].get("lr"))
            except Exception:
                lr_now = None
            metrics = {"loss_avg": float(avg), "steps": int(steps)}
            if last_grad_norm is not None:
                metrics["grad_norm"] = float(last_grad_norm)
            if lr_now is not None:
                metrics["lr"] = float(lr_now)
            if last_data_time_s is not None:
                metrics["data_time_s"] = float(last_data_time_s)
            if last_step_time_s is not None:
                metrics["step_time_s"] = float(last_step_time_s)
            if last_throughput is not None:
                metrics["throughput_img_s"] = float(last_throughput)
            if last_max_vram_mb is not None:
                metrics["max_vram_mb"] = float(last_max_vram_mb)
            if ema is not None:
                metrics["ema_decay"] = float(getattr(ema, "decay", 0.0))
            report = build_report(
                losses=losses_out,
                metrics=metrics,
                meta={"kind": "train_epoch", "epoch": int(epoch)},
            )
            append_jsonl(args.metrics_jsonl, report)

        val_every = int(getattr(args, "val_every", 0) or 0)
        val_due_epoch = bool(val_every > 0 and ((int(epoch) + 1) % val_every == 0 or (int(epoch) + 1) >= int(args.epochs)))
        if val_due_epoch and getattr(args, "val_metrics_jsonl", None):
            if ddp_enabled and not is_main:
                torch.distributed.barrier()
            if is_main:
                res = _run_validation(kind="val_epoch", epoch=int(epoch), optim_step=int(global_step))
                if res is not None and early_stop_patience > 0:
                    map50_95, prev_best = res
                    improved = bool(float(map50_95) > float(prev_best) + float(early_stop_min_delta))
                    if improved:
                        early_stop_bad = 0
                    else:
                        early_stop_bad += 1
                        if early_stop_bad >= int(early_stop_patience):
                            stop_training = True
                            report = build_report(
                                losses={},
                                metrics={
                                    "early_stop": True,
                                    "patience": int(early_stop_patience),
                                    "bad": int(early_stop_bad),
                                    "min_delta": float(early_stop_min_delta),
                                    "best_map50_95": float(best_map50_95),
                                },
                                meta={"kind": "early_stop", "epoch": int(epoch), "optim_step": int(global_step)},
                            )
                            append_jsonl(args.val_metrics_jsonl, report)
            if ddp_enabled and is_main:
                torch.distributed.barrier()
            if ddp_enabled:
                flag_device = device if device.type == "cuda" else torch.device("cpu")
                flag = torch.tensor([1 if stop_training else 0], dtype=torch.int64, device=flag_device)
                torch.distributed.broadcast(flag, src=0)
                stop_training = bool(int(flag.item()))

    if is_main and (args.metrics_json or args.metrics_csv):
        losses_out = {}
        if last_loss_dict is not None:
            losses_out = {k: float(v.detach().cpu()) for k, v in last_loss_dict.items() if hasattr(v, "detach")}
        metrics_out = {"epochs": int(args.epochs), "max_steps": int(args.max_steps)}
        if last_epoch_avg is not None:
            metrics_out["loss_avg_last_epoch"] = float(last_epoch_avg)
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

    if is_main and ewc_accum is not None and args.ewc_state_out:
        save_ewc_state(str(args.ewc_state_out), ewc_accum.finalize(unwrap_model(model)))

    if is_main and si_accum is not None and args.si_state_out:
        save_si_state(str(args.si_state_out), si_accum.finalize(unwrap_model(model)))

    onnx_path = None
    if is_main and args.onnx_out:
        try:
            from rtdetr_pose.export import export_onnx
        except Exception as exc:  # pragma: no cover
            print(
                f"WARNING: ONNX export skipped — could not import rtdetr_pose.export ({exc}). "
                "Install 'onnx' to enable post-training ONNX export.",
                file=sys.stderr,
            )
            export_onnx = None  # type: ignore[assignment]

        onnx_path = Path(str(args.onnx_out)) if export_onnx is not None else None
        if onnx_path is not None:
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            if run_contract is not None and getattr(args, "best_checkpoint_out", None):
                best_path = Path(str(args.best_checkpoint_out))
                if best_path.exists():
                    load_checkpoint_into(unwrap_model(model), None, str(best_path), restore_rng=False)
            dummy = torch.zeros((1, 3, int(args.image_size), int(args.image_size)), dtype=torch.float32, device=device)
            try:
                export_onnx(
                    unwrap_model(model).eval(),
                    dummy,
                    str(onnx_path),
                    opset_version=int(args.onnx_opset),
                    dynamic_hw=bool(args.onnx_dynamic_hw),
                )
            except RuntimeError as exc:
                print(
                    f"WARNING: ONNX export failed — {exc}. Training results are saved; "
                    "install 'onnx' to enable post-training ONNX export.",
                    file=sys.stderr,
                )
                onnx_path = None
            if onnx_path is not None:
                meta_path = Path(str(args.onnx_meta_out)) if args.onnx_meta_out else onnx_path.with_suffix(onnx_path.suffix + ".meta.json")
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                meta = {
                    "timestamp_utc": _now_utc(),
                    "onnx": str(onnx_path),
                    "opset": int(args.onnx_opset),
                    "dynamic_hw": bool(args.onnx_dynamic_hw),
                    "dummy_input": {"shape": [1, 3, int(args.image_size), int(args.image_size)], "dtype": "float32"},
                    "run_record": run_record,
                }
                meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    parity_out = getattr(args, "parity_json_out", None)
    if is_main and parity_out:
        out_path = Path(str(parity_out))
        policy = str(args.parity_policy or ("fail" if run_contract is not None else "warn"))
        if onnx_path is None:
            report = {
                "timestamp_utc": _now_utc(),
                "onnx": None,
                "thresholds": {"score_atol": float(args.parity_score_atol), "bbox_atol": float(args.parity_bbox_atol)},
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
            print(f"WARNING: ONNX parity requested but ONNX export disabled. See: {out_path}", file=sys.stderr)
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

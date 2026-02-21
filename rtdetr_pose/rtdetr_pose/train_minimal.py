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

workspace_root = Path.cwd()

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


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_keypoint_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _derive_keypoint_flip_pairs(keypoint_names: list[str]) -> list[list[int]]:
    if not keypoint_names:
        return []

    index_by_name = {str(name).strip().lower(): idx for idx, name in enumerate(keypoint_names)}
    pairs: list[list[int]] = []
    seen: set[tuple[int, int]] = set()

    for idx, name in enumerate(keypoint_names):
        lower = str(name).strip().lower()
        if not lower:
            continue

        partner_name = None
        if "left" in lower:
            partner_name = lower.replace("left", "right")
        elif "right" in lower:
            partner_name = lower.replace("right", "left")
        elif lower.startswith("l_"):
            partner_name = "r_" + lower[2:]
        elif lower.startswith("r_"):
            partner_name = "l_" + lower[2:]
        elif lower.endswith("_l"):
            partner_name = lower[:-2] + "_r"
        elif lower.endswith("_r"):
            partner_name = lower[:-2] + "_l"

        if not partner_name:
            continue
        j = index_by_name.get(partner_name)
        if j is None or j == idx:
            continue
        key = (idx, j) if idx <= j else (j, idx)
        if key in seen:
            continue
        seen.add(key)
        pairs.append([int(key[0]), int(key[1])])

    return pairs


def _extract_manifest_keypoints_meta(manifest: dict[str, Any] | None) -> tuple[list[str], list[list[int]]]:
    if not isinstance(manifest, dict):
        return [], []
    meta = manifest.get("keypoints_meta")
    if not isinstance(meta, dict):
        return [], []

    names = _normalize_keypoint_names(meta.get("keypoint_names") or [])
    skeleton: list[list[int]] = []
    raw_skeleton = meta.get("skeleton") or []
    if isinstance(raw_skeleton, list) and names:
        seen: set[tuple[int, int]] = set()
        for edge in raw_skeleton:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            try:
                a = int(edge[0])
                b = int(edge[1])
            except Exception:
                continue
            if a <= 0 or b <= 0 or a == b:
                continue
            if a > len(names) or b > len(names):
                continue
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            skeleton.append([a, b])
    return names, skeleton


def load_config_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"config not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except Exception as exc:  # pragma: no cover
            raise SystemExit("PyYAML is required for YAML configs; install requirements.txt") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data or {}

    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    # Fallback: try JSON then YAML.
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover
            raise SystemExit("PyYAML is required for YAML configs; install requirements.txt") from exc
        data = yaml.safe_load(text)
        return data or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal RTDETRPose training scaffold.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML/JSON config file. Values become argparse defaults; explicit CLI flags override.",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Optional RTDETRPose model config (e.g., rtdetr_pose/configs/base.json). Used to infer model defaults.",
    )
    parser.add_argument(
        "--config-version",
        type=int,
        default=None,
        help="Optional config schema version (recommended: 1).",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved config (after applying defaults) and exit 0.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single training step (including logging/checkpoint wiring) then exit 0.",
    )
    parser.add_argument("--dataset-root", type=str, default="", help="Path to data/coco128")
    parser.add_argument("--split", type=str, default="train2017")
    parser.add_argument(
        "--val-split",
        type=str,
        default=None,
        help="Optional validation split (default: val2017 if it exists, else disabled).",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=1,
        help="Run validation every N epochs (0 disables; default: 1).",
    )
    parser.add_argument(
        "--val-every-steps",
        type=int,
        default=0,
        help="Run validation every N optimizer steps (0 disables; default: 0).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if val map50_95 does not improve for N validations (0 disables; default: 0).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum improvement in val map50_95 to reset early-stop counter (default: 0.0).",
    )
    parser.add_argument(
        "--val-max-images",
        type=int,
        default=0,
        help="Optional cap on number of validation images (0 = all).",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Validation batch size (default: --batch-size).",
    )
    parser.add_argument(
        "--val-score-thresh",
        type=float,
        default=0.001,
        help="Score threshold for decoding detections during validation (default: 0.001).",
    )
    parser.add_argument(
        "--val-topk",
        type=int,
        default=300,
        help="Top-K detections per image for validation decode (default: 300).",
    )
    parser.add_argument(
        "--records-json",
        default=None,
        help="Optional JSON file containing a list of training records (overrides dataset-root/split scan).",
    )
    parser.add_argument(
        "--extra-records-json",
        default=None,
        help="Optional JSON file containing extra records to append to the scanned dataset records.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--grad-accum",
        "--gradient-accumulation-steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1). Optimizer steps happen every N batches.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "sgd"),
        default="adamw",
        help="Optimizer type (default: adamw).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer (default: 0.01).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer (default: 0.9).",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        help="Enable Nesterov momentum (SGD only).",
    )
    parser.add_argument(
        "--use-param-groups",
        action="store_true",
        help="Split parameters into backbone/head groups with configurable lr/wd multipliers.",
    )
    parser.add_argument("--backbone-lr-mult", type=float, default=1.0, help="Backbone lr multiplier (default: 1.0).")
    parser.add_argument("--head-lr-mult", type=float, default=1.0, help="Head lr multiplier (default: 1.0).")
    parser.add_argument("--backbone-wd-mult", type=float, default=1.0, help="Backbone wd multiplier (default: 1.0).")
    parser.add_argument("--head-wd-mult", type=float, default=1.0, help="Head wd multiplier (default: 1.0).")
    parser.add_argument(
        "--wd-exclude-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude bias parameters from weight decay (default: true).",
    )
    parser.add_argument(
        "--wd-exclude-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude normalization layers from weight decay (default: true).",
    )
    parser.add_argument(
        "--scheduler",
        choices=("none", "cosine", "onecycle", "multistep"),
        default="none",
        help="Learning-rate scheduler (default: none).",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Minimum LR for cosine scheduler (default: 0.0).",
    )
    parser.add_argument(
        "--scheduler-milestones",
        type=str,
        default="",
        help="Comma-separated milestone steps for multistep scheduler (e.g., 1000,2000).",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.1,
        help="Gamma for multistep scheduler (default: 0.1).",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Enable EMA tracking of model weights.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay (default: 0.999).",
    )
    parser.add_argument(
        "--ema-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use EMA weights for evaluation/export when EMA is enabled (default: false).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
        help="Torch device for training (e.g., cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--amp",
        choices=("none", "fp16", "bf16"),
        default="none",
        help="Automatic mixed precision (cuda only): none|fp16|bf16 (default: none).",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for model forward (experimental; default: false).",
    )
    parser.add_argument(
        "--torch-compile-backend",
        default="inductor",
        help="torch.compile backend (default: inductor).",
    )
    parser.add_argument(
        "--torch-compile-mode",
        default=None,
        help="torch.compile mode (e.g., default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs). Default: none.",
    )
    parser.add_argument(
        "--torch-compile-fullgraph",
        action="store_true",
        help="Pass fullgraph=True to torch.compile (default: false).",
    )
    parser.add_argument(
        "--torch-compile-dynamic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pass dynamic=True/False to torch.compile (default: None).",
    )
    parser.add_argument(
        "--torch-compile-strict",
        action="store_true",
        help="If torch.compile fails, stop the run instead of falling back (default: false).",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Alias for --amp fp16 (back-compat).",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable DistributedDataParallel (single node). Use via torchrun; outputs written on rank0 only.",
    )
    parser.add_argument(
        "--ddp-backend",
        default=None,
        help="DDP backend override (default: nccl for cuda, else gloo).",
    )
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=0.0,
        help="If >0, clip gradients to this max norm before optimizer step.",
    )
    parser.add_argument(
        "--log-grad-norm",
        action="store_true",
        help="Log gradient norm into metrics.jsonl (computed on optimizer steps only).",
    )
    parser.add_argument(
        "--stop-on-non-finite-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop training when the loss becomes non-finite (default: true).",
    )
    parser.add_argument(
        "--non-finite-max-skips",
        type=int,
        default=3,
        help="When non-finite guard is enabled (--no-stop-on-non-finite-loss), stop after this many skips (default: 3).",
    )
    parser.add_argument(
        "--non-finite-lr-decay",
        type=float,
        default=0.5,
        help="When non-finite guard is enabled, multiply LR by this factor on each non-finite event (default: 0.5).",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="Linear warmup steps for learning rate (0 disables warmup).",
    )
    parser.add_argument(
        "--lr-warmup-init",
        type=float,
        default=0.0,
        help="Initial learning rate value at step 0 for warmup.",
    )
    parser.add_argument("--max-steps", type=int, default=30, help="Cap steps per epoch")
    parser.add_argument("--log-every", type=int, default=10, help="Print every N steps")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help="Enable random multiscale resize around --image-size.",
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.8,
        help="Lower bound for multiscale resize (relative to --image-size).",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=1.2,
        help="Upper bound for multiscale resize (relative to --image-size).",
    )
    parser.add_argument(
        "--hflip-prob",
        type=float,
        default=0.0,
        help="Probability of random horizontal flip augmentation.",
    )
    parser.add_argument(
        "--hflip-prob-start",
        type=float,
        default=None,
        help="Optional starting hflip probability for linear schedule.",
    )
    parser.add_argument(
        "--hflip-prob-end",
        type=float,
        default=None,
        help="Optional ending hflip probability for linear schedule.",
    )
    parser.add_argument(
        "--hsv-h",
        type=float,
        default=0.0,
        help="HSV hue jitter magnitude in [0,1] units (e.g., 0.015 ~= 5.4 degrees). Default: 0 (disabled).",
    )
    parser.add_argument(
        "--hsv-s",
        type=float,
        default=0.0,
        help="HSV saturation jitter magnitude (scales S by factor in [1-hsv_s, 1+hsv_s]). Default: 0 (disabled).",
    )
    parser.add_argument(
        "--hsv-v",
        type=float,
        default=0.0,
        help="HSV value/brightness jitter magnitude (scales V by factor in [1-hsv_v, 1+hsv_v]). Default: 0 (disabled).",
    )
    parser.add_argument(
        "--hsv-prob",
        type=float,
        default=1.0,
        help="Probability to apply HSV jitter when any of --hsv-* is enabled (default: 1.0).",
    )
    parser.add_argument(
        "--gray-prob",
        type=float,
        default=0.0,
        help="Probability of converting the image to grayscale (photometric only; default: 0.0).",
    )
    parser.add_argument(
        "--gaussian-noise-std",
        type=float,
        default=0.0,
        help="Stddev for additive Gaussian noise in [0,1] pixel space (default: 0.0 disables).",
    )
    parser.add_argument(
        "--gaussian-noise-prob",
        type=float,
        default=1.0,
        help="Probability to apply Gaussian noise when --gaussian-noise-std>0 (default: 1.0).",
    )
    parser.add_argument(
        "--blur-prob",
        type=float,
        default=0.0,
        help="Probability of applying a small Gaussian blur (photometric only; default: 0.0).",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=0.0,
        help="Max sigma for Gaussian blur; sampled uniformly in (0, sigma]. Default: 0.0 disables.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=3,
        help="Gaussian blur kernel size (odd int, default: 3).",
    )
    parser.add_argument(
        "--intrinsics-jitter",
        action="store_true",
        help="Enable intrinsics jitter augmentation on K_gt.",
    )
    parser.add_argument(
        "--sim-jitter",
        action="store_true",
        help="Enable SIM-style intrinsics jitter using yolozu.jitter profiles.",
    )
    parser.add_argument(
        "--sim-jitter-profile",
        type=str,
        default=None,
        help="Optional JSON file to override the default SIM jitter profile.",
    )
    parser.add_argument(
        "--sim-jitter-extrinsics",
        action="store_true",
        help="Apply SIM profile extrinsics jitter to gt_t/gt_R.",
    )
    parser.add_argument(
        "--extrinsics-jitter",
        action="store_true",
        help="Enable manual extrinsics jitter on gt_t/gt_R.",
    )
    parser.add_argument("--jitter-dx", type=float, default=0.01, help="Translation jitter range in meters.")
    parser.add_argument("--jitter-dy", type=float, default=0.01, help="Translation jitter range in meters.")
    parser.add_argument("--jitter-dz", type=float, default=0.02, help="Translation jitter range in meters.")
    parser.add_argument("--jitter-droll", type=float, default=1.0, help="Roll jitter range in degrees.")
    parser.add_argument("--jitter-dpitch", type=float, default=1.0, help="Pitch jitter range in degrees.")
    parser.add_argument("--jitter-dyaw", type=float, default=2.0, help="Yaw jitter range in degrees.")
    parser.add_argument(
        "--jitter-dfx",
        type=float,
        default=0.02,
        help="Relative fx jitter range (uniform in [-dfx, dfx]).",
    )
    parser.add_argument(
        "--jitter-dfy",
        type=float,
        default=0.02,
        help="Relative fy jitter range (uniform in [-dfy, dfy]).",
    )
    parser.add_argument(
        "--jitter-dcx",
        type=float,
        default=4.0,
        help="Absolute cx jitter range in pixels (uniform in [-dcx, dcx]).",
    )
    parser.add_argument(
        "--jitter-dcy",
        type=float,
        default=4.0,
        help="Absolute cy jitter range in pixels (uniform in [-dcy, dcy]).",
    )
    parser.add_argument(
        "--real-images",
        action="store_true",
        help="Load real images via record['image_path'] (requires Pillow). Default uses synthetic images.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers (default: 0).")
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="DataLoader pin_memory (default: false).",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="DataLoader persistent_workers (requires --num-workers>0; default: false).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch_factor (requires --num-workers>0; default: 2).",
    )
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument(
        "--num-keypoints",
        type=int,
        default=0,
        help="If >0, enable keypoints training with this many keypoints per instance (default: 0 disables).",
    )
    parser.add_argument(
        "--use-uncertainty",
        action="store_true",
        help="Enable uncertainty heads (log_sigma_z/log_sigma_rot) for task alignment.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=0,
        help="Enable LoRA by setting rank r>0 (default: 0 disables).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="LoRA alpha scaling (default: r).",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout on inputs (default: 0.0).",
    )
    parser.add_argument(
        "--lora-target",
        default="head",
        choices=("head", "all_linear", "all_conv1x1", "all_linear_conv1x1"),
        help="Where to apply LoRA (default: head).",
    )
    parser.add_argument(
        "--lora-freeze-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze base weights and train LoRA params only (default: true).",
    )
    parser.add_argument(
        "--lora-train-bias",
        choices=("none", "all"),
        default="none",
        help="If LoRA is enabled, optionally train biases too (default: none).",
    )
    parser.add_argument(
        "--torchao-quant",
        default="none",
        choices=("none", "int8wo", "int4wo"),
        help="Optional torchao quantization recipe (experimental). Requires torchao. (default: none)",
    )
    parser.add_argument(
        "--torchao-required",
        action="store_true",
        help="If enabled, fail the run when torchao isn't installed or quantization fails (default: false).",
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Convenience flag: sets --torchao-quant=int4wo and forces --lora-freeze-base (requires --lora-r>0).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Shuffle dataset each epoch (default: true).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic data order via DataLoader generator (seeded).",
    )
    parser.add_argument("--use-matcher", action="store_true", help="Use Hungarian matching")
    parser.add_argument("--cost-cls", type=float, default=1.0)
    parser.add_argument("--cost-bbox", type=float, default=5.0)
    parser.add_argument("--cost-z", type=float, default=0.0, help="Optional matching cost for depth")
    parser.add_argument(
        "--cost-rot",
        type=float,
        default=0.0,
        help="Optional matching cost for rotation (geodesic angle)",
    )
    parser.add_argument(
        "--synthetic-pose",
        action="store_true",
        help="Generate synthetic z/R GT per instance (scaffold only)",
    )
    parser.add_argument(
        "--z-from-dobj",
        action="store_true",
        help="When GT t is missing, derive z (and optionally t if K is available) from D_obj at bbox center",
    )
    parser.add_argument(
        "--load-aux",
        action="store_true",
        help="Allow loading mask/depth arrays from paths (.json/.npy) for z-from-dobj; default keeps lazy paths",
    )
    parser.add_argument(
        "--depth-mode",
        choices=("none", "sidecar", "fuse_mid"),
        default="none",
        help="Depth handling mode: none (default), sidecar (read depth_path metadata), or fuse_mid (mid-fusion with depth sidecar).",
    )
    parser.add_argument(
        "--depth-unit",
        choices=("unspecified", "relative", "metric"),
        default="unspecified",
        help="Depth unit semantics for safety gating (default: unspecified).",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to sidecar depth values (default: 1.0).",
    )
    parser.add_argument(
        "--depth-dropout",
        type=float,
        default=0.0,
        help="Modality dropout probability for depth when --depth-mode=fuse_mid (default: 0.0).",
    )
    parser.add_argument(
        "--enable-mim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable masked reconstruction branch (MIM) and related losses (default: false).",
    )
    parser.add_argument(
        "--mim-mask-prob",
        type=float,
        default=0.6,
        help="Block mask probability for MIM feature masking (default: 0.6).",
    )
    parser.add_argument(
        "--mim-patch-size",
        type=int,
        default=16,
        help="Block mask patch size for MIM feature masking (default: 16).",
    )
    parser.add_argument(
        "--mim-start-step",
        type=int,
        default=0,
        help="Start applying MIM/entropy loss weights after this optimizer step (default: 0).",
    )
    parser.add_argument(
        "--cost-t",
        type=float,
        default=0.0,
        help="Optional matching cost for translation recovered from (bbox, offsets, z, K')",
    )
    parser.add_argument(
        "--cost-z-start-step",
        type=int,
        default=0,
        help="Enable cost_z in the matcher after this optimizer step (default: 0).",
    )
    parser.add_argument(
        "--cost-rot-start-step",
        type=int,
        default=0,
        help="Enable cost_rot in the matcher after this optimizer step (default: 0).",
    )
    parser.add_argument(
        "--cost-t-start-step",
        type=int,
        default=0,
        help="Enable cost_t in the matcher after this optimizer step (default: 0).",
    )
    parser.add_argument(
        "--stage-off-steps",
        type=int,
        default=0,
        help="Train offsets only for the first N optimizer steps (sets loss weight k=0). Default: 0 (disabled).",
    )
    parser.add_argument(
        "--stage-k-steps",
        type=int,
        default=0,
        help="Then train GlobalKHead only for the next N optimizer steps (sets loss weight off=0). Default: 0 (disabled).",
    )
    parser.add_argument(
        "--debug-losses",
        action="store_true",
        help="Print loss dict breakdown on step 1",
    )
    parser.add_argument(
        "--task-aligner",
        choices=("none", "uncertainty"),
        default="none",
        help="Multi-task loss alignment strategy (default: none).",
    )
    parser.add_argument("--metrics-jsonl", default=None, help="Append per-step loss/metric report JSONL here.")
    parser.add_argument("--val-metrics-jsonl", default=None, help="Append validation metrics JSONL here.")
    parser.add_argument("--metrics-json", default=None, help="Write final run summary JSON here.")
    parser.add_argument("--metrics-csv", default=None, help="Write final run summary CSV (single row) here.")
    parser.add_argument(
        "--config-resolved-out",
        default=None,
        help="Write resolved config YAML here (useful for run contracts).",
    )
    parser.add_argument(
        "--run-meta-out",
        default=None,
        help="Write run metadata JSON here (useful for run contracts).",
    )
    parser.add_argument(
        "--best-checkpoint-out",
        default=None,
        help="Optional path to write the best checkpoint bundle (model+optimizer+state).",
    )

    # Continual learning / self-distillation (SDFT-inspired)
    parser.add_argument(
        "--self-distill-from",
        default=None,
        help="Optional teacher checkpoint to distill against (to reduce catastrophic forgetting).",
    )
    parser.add_argument(
        "--self-distill-weight",
        type=float,
        default=1.0,
        help="Global multiplier for self-distillation loss (only used when --self-distill-from is set).",
    )
    parser.add_argument(
        "--self-distill-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for logits distillation (>=1 recommended).",
    )
    parser.add_argument(
        "--self-distill-kl",
        choices=("forward", "reverse", "sym"),
        default="reverse",
        help="KL direction for logits distillation (default: reverse, SDFT-style).",
    )
    parser.add_argument(
        "--self-distill-keys",
        type=str,
        default="logits,bbox",
        help="Comma-separated model output keys to distill (default: logits,bbox).",
    )
    parser.add_argument(
        "--self-distill-logits-weight",
        type=float,
        default=1.0,
        help="Per-key weight for logits distillation term.",
    )
    parser.add_argument(
        "--self-distill-bbox-weight",
        type=float,
        default=1.0,
        help="Per-key weight for bbox distillation term (compared in sigmoid space).",
    )
    parser.add_argument(
        "--self-distill-other-l1-weight",
        type=float,
        default=1.0,
        help="Per-key L1 weight for any other distilled tensor outputs.",
    )

    # DER++ replay distillation (optional; requires per-sample teacher outputs in records)
    parser.add_argument(
        "--derpp",
        action="store_true",
        help="Enable DER++-style replay distillation (uses per-sample teacher outputs stored in records).",
    )
    parser.add_argument(
        "--derpp-teacher-key",
        type=str,
        default="derpp_teacher_npz",
        help="Record key holding DER++ teacher outputs (dict) or a path to an .npz/.json/.pt/.pth/.safetensors (default: derpp_teacher_npz).",
    )
    parser.add_argument(
        "--derpp-weight",
        type=float,
        default=1.0,
        help="Global multiplier for DER++ distillation loss (default: 1.0).",
    )
    parser.add_argument(
        "--derpp-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for DER++ logits distillation (>=1 recommended).",
    )
    parser.add_argument(
        "--derpp-kl",
        choices=("forward", "reverse", "sym"),
        default="reverse",
        help="KL direction for DER++ logits distillation (default: reverse, SDFT-style).",
    )
    parser.add_argument(
        "--derpp-keys",
        type=str,
        default="logits,bbox",
        help="Comma-separated model output keys to distill for DER++ (default: logits,bbox).",
    )
    parser.add_argument(
        "--derpp-logits-weight",
        type=float,
        default=1.0,
        help="Per-key weight for DER++ logits distillation term.",
    )
    parser.add_argument(
        "--derpp-bbox-weight",
        type=float,
        default=1.0,
        help="Per-key weight for DER++ bbox distillation term (compared in sigmoid space).",
    )
    parser.add_argument(
        "--derpp-other-l1-weight",
        type=float,
        default=1.0,
        help="Per-key L1 weight for any other DER++ distilled tensor outputs.",
    )

    # Continual learning regularizers (optional)
    parser.add_argument(
        "--ewc",
        action="store_true",
        help="Enable EWC regularization (penalty uses --ewc-state-in; importance saved to --ewc-state-out).",
    )
    parser.add_argument("--ewc-lambda", type=float, default=1.0, help="EWC penalty weight (default: 1.0).")
    parser.add_argument("--ewc-state-in", default=None, help="EWC state (.pt) path from a previous task.")
    parser.add_argument("--ewc-state-out", default=None, help="Write EWC state (.pt) for this task.")

    parser.add_argument(
        "--si",
        action="store_true",
        help="Enable Synaptic Intelligence regularization (penalty uses --si-state-in; importance saved to --si-state-out).",
    )
    parser.add_argument("--si-c", type=float, default=1.0, help="SI penalty weight (default: 1.0).")
    parser.add_argument("--si-epsilon", type=float, default=1e-3, help="SI epsilon for importance normalization.")
    parser.add_argument("--si-state-in", default=None, help="SI state (.pt) path from a previous task.")
    parser.add_argument("--si-state-out", default=None, help="Write SI state (.pt) for this task.")

    # Checkpointing / resume
    parser.add_argument("--resume-from", default=None, help="Resume weights or full checkpoint bundle from this path.")
    parser.add_argument(
        "--checkpoint-bundle-out",
        default=None,
        help="Write a full checkpoint bundle (model+optimizer+progress) to this path at end.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="If >0 and --checkpoint-bundle-out is set, also save intermediate bundles every N steps.",
    )
    parser.add_argument(
        "--save-last-every",
        type=int,
        default=0,
        help="If >0 and --checkpoint-bundle-out is set, periodically overwrite the last checkpoint every N optimizer steps.",
    )

    # Back-compat: weights-only checkpoint
    parser.add_argument("--checkpoint-out", default=None, help="Write model state_dict to this path at end.")

    # Reproducible artifacts
    parser.add_argument(
        "--run-contract",
        action="store_true",
        help="Enable production-style run contract layout under --runs-dir/<run-id>/.",
    )
    parser.add_argument("--runs-dir", default="runs", help="Base directory for --run-contract (default: runs).")
    parser.add_argument("--run-id", default=None, help="Run identifier (default: <utc timestamp> when --run-contract).")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the contracted last checkpoint (requires --run-contract).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="If set, write standard artifacts into this folder (run_record.json, metrics.jsonl/json/csv, checkpoint*.pt, model.onnx).",
    )
    parser.add_argument(
        "--fracal-stats-out",
        default=None,
        help="Optional JSON path to write FRACAL class-frequency stats computed from training records.",
    )
    parser.add_argument(
        "--fracal-stats-task",
        choices=("bbox", "seg", "pose"),
        default="bbox",
        help="Task for FRACAL stats generation (default: bbox).",
    )
    parser.add_argument(
        "--fracal-allow-rgb-masks",
        action="store_true",
        help="(fracal-stats-task=seg) Treat RGB masks as valid by using channel-0 as foreground.",
    )
    parser.add_argument(
        "--export-onnx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export ONNX when --onnx-out is set (default: true).",
    )
    parser.add_argument(
        "--onnx-out",
        default=None,
        help="Optional ONNX export output path (overrides --run-dir default).",
    )
    parser.add_argument(
        "--onnx-meta-out",
        default=None,
        help="Optional ONNX export metadata JSON path (default: <onnx-out>.meta.json).",
    )
    parser.add_argument("--onnx-opset", type=int, default=18, help="ONNX opset version (default: 18).")
    parser.add_argument(
        "--onnx-dynamic-hw",
        action="store_true",
        help="Export ONNX with dynamic height/width axes (batch is always dynamic).",
    )
    parser.add_argument(
        "--parity-json-out",
        default=None,
        help="Optional JSON path to write Torch vs ONNXRuntime parity stats.",
    )
    parser.add_argument(
        "--parity-score-atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for derived score parity (default: 1e-4).",
    )
    parser.add_argument(
        "--parity-bbox-atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for sigmoid(bbox) parity (default: 1e-4).",
    )
    parser.add_argument(
        "--parity-policy",
        choices=("warn", "fail"),
        default=None,
        help="Parity gate behavior (warn|fail). Default: warn (non-contract) / fail (run-contract).",
    )
    return parser


def _default_run_id() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def apply_run_contract_defaults(args: argparse.Namespace) -> tuple[argparse.Namespace, dict[str, Path] | None]:
    enabled = bool(getattr(args, "run_contract", False)) or bool(getattr(args, "run_id", None))
    if not enabled:
        return args, None

    if getattr(args, "run_dir", None):
        raise SystemExit("--run-contract cannot be combined with --run-dir (choose one artifact layout).")

    run_id = str(getattr(args, "run_id", "") or "").strip()
    if not run_id:
        run_id = _default_run_id()
        args.run_id = run_id

    runs_dir = Path(str(getattr(args, "runs_dir", "runs") or "runs"))
    run_dir = runs_dir / run_id
    checkpoints_dir = run_dir / "checkpoints"
    reports_dir = run_dir / "reports"
    exports_dir = run_dir / "exports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    def _default_path(current: str | None, path: Path) -> str:
        return current if current else str(path)

    args.metrics_jsonl = _default_path(getattr(args, "metrics_jsonl", None), reports_dir / "train_metrics.jsonl")
    args.val_metrics_jsonl = _default_path(getattr(args, "val_metrics_jsonl", None), reports_dir / "val_metrics.jsonl")
    args.config_resolved_out = _default_path(getattr(args, "config_resolved_out", None), reports_dir / "config_resolved.yaml")
    args.run_meta_out = _default_path(getattr(args, "run_meta_out", None), reports_dir / "run_meta.json")
    args.fracal_stats_out = _default_path(
        getattr(args, "fracal_stats_out", None),
        reports_dir / f"fracal_stats_{str(getattr(args, 'fracal_stats_task', 'bbox') or 'bbox').strip().lower()}.json",
    )
    args.parity_json_out = _default_path(getattr(args, "parity_json_out", None), reports_dir / "onnx_parity.json")
    if getattr(args, "parity_policy", None) is None:
        args.parity_policy = "fail"

    args.checkpoint_bundle_out = _default_path(getattr(args, "checkpoint_bundle_out", None), checkpoints_dir / "last.pt")
    args.best_checkpoint_out = _default_path(getattr(args, "best_checkpoint_out", None), checkpoints_dir / "best.pt")

    # Default ONNX export path (can still be disabled via --no-export-onnx).
    args.onnx_out = _default_path(getattr(args, "onnx_out", None), exports_dir / "model.onnx")
    args.onnx_meta_out = _default_path(
        getattr(args, "onnx_meta_out", None),
        exports_dir / "model.onnx.meta.json",
    )

    # Convenience: --resume means resume from contracted last checkpoint.
    if bool(getattr(args, "resume", False)) and not getattr(args, "resume_from", None):
        args.resume_from = str(checkpoints_dir / "last.pt")

    if int(getattr(args, "save_last_every", 0) or 0) <= 0:
        args.save_last_every = 100
    try:
        if float(getattr(args, "clip_grad_norm", 0.0) or 0.0) <= 0.0:
            args.clip_grad_norm = 1.0
    except Exception:
        pass

    return args, {
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "reports_dir": reports_dir,
        "exports_dir": exports_dir,
    }


def apply_run_dir_defaults(args: argparse.Namespace) -> tuple[argparse.Namespace, Path | None]:
    run_dir = None
    if args.run_dir:
        run_dir = Path(str(args.run_dir))
        run_dir.mkdir(parents=True, exist_ok=True)

        def _default_path(current: str | None, name: str) -> str:
            return current if current else str(run_dir / name)

        args.metrics_jsonl = _default_path(args.metrics_jsonl, "metrics.jsonl")
        args.metrics_json = _default_path(args.metrics_json, "metrics.json")
        args.metrics_csv = _default_path(args.metrics_csv, "metrics.csv")
        args.checkpoint_out = _default_path(args.checkpoint_out, "checkpoint.pt")
        args.checkpoint_bundle_out = _default_path(args.checkpoint_bundle_out, "checkpoint_bundle.pt")
        args.onnx_out = _default_path(args.onnx_out, "model.onnx")
        if getattr(args, "ewc", False):
            args.ewc_state_out = _default_path(getattr(args, "ewc_state_out", None), "ewc_state.pt")
        if getattr(args, "si", False):
            args.si_state_out = _default_path(getattr(args, "si_state_out", None), "si_state.pt")

    if not bool(getattr(args, "export_onnx", True)):
        args.onnx_out = None

    return args, run_dir


def unwrap_model(model: "torch.nn.Module") -> "torch.nn.Module":
    # Unwrap common wrappers (DDP, torch.compile OptimizedModule, etc.).
    while True:
        if hasattr(model, "module"):
            try:
                model = model.module
                continue
            except Exception:
                pass
        if hasattr(model, "_orig_mod"):
            try:
                model = model._orig_mod  # type: ignore[attr-defined]
                continue
            except Exception:
                pass
        return model


def _quantiles(values: "Any", qs: tuple[int, ...] = (50, 90, 95, 99)) -> dict[str, float]:
    import numpy as np  # type: ignore

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    out: dict[str, float] = {}
    if flat.size == 0:
        for q in qs:
            out[f"p{int(q)}"] = 0.0
        return out
    for q in qs:
        out[f"p{int(q)}"] = float(np.quantile(flat, float(q) / 100.0))
    return out


def _diff_stats(a: "Any", b: "Any") -> dict[str, Any]:
    import numpy as np  # type: ignore

    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {"ok": False, "reason": "shape_mismatch", "a_shape": list(a.shape), "b_shape": list(b.shape)}
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    finite = np.isfinite(diff)
    if not bool(finite.all()):
        return {
            "ok": False,
            "reason": "non_finite_diff",
            "shape": list(diff.shape),
            "non_finite": int((~finite).sum()),
        }
    out: dict[str, Any] = {
        "ok": True,
        "shape": list(diff.shape),
        "max": float(diff.max()) if diff.size else 0.0,
        "mean": float(diff.mean()) if diff.size else 0.0,
    }
    out.update(_quantiles(diff))
    return out


def _softmax(x: "Any", axis: int = -1) -> "Any":
    import numpy as np  # type: ignore

    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    denom = ex.sum(axis=axis, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return (ex / denom).astype(np.float32)


def _sigmoid(x: "Any") -> "Any":
    import numpy as np  # type: ignore

    x = np.asarray(x, dtype=np.float64)
    y = 1.0 / (1.0 + np.exp(-x))
    return y.astype(np.float32)


def _derive_score_bbox(outputs: dict[str, "Any"]) -> tuple["Any", "Any"]:
    logits = outputs.get("logits")
    bbox = outputs.get("bbox")
    if logits is None or bbox is None:
        raise ValueError("outputs must contain logits and bbox")
    probs = _softmax(logits, axis=-1)
    score = probs.max(axis=-1)
    bbox_sig = _sigmoid(bbox)
    return score.astype("float32"), bbox_sig.astype("float32")


def run_onnxrt_parity(
    *,
    model: "torch.nn.Module",
    onnx_path: Path,
    image_size: int,
    seed: int,
    score_atol: float,
    bbox_atol: float,
    out_path: Path,
    policy: str,
    run_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "timestamp_utc": _now_utc(),
        "onnx": str(onnx_path),
        "thresholds": {"score_atol": float(score_atol), "bbox_atol": float(bbox_atol)},
        "policy": str(policy),
        "passed": False,
        "available": False,
        "reason": None,
        "run_record": run_record,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        report["reason"] = f"missing_numpy:{exc}"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        if str(policy) == "fail":
            raise SystemExit(f"ONNX parity unavailable (numpy missing). See: {out_path}")
        print(f"WARNING: ONNX parity unavailable (numpy missing). See: {out_path}", file=sys.stderr)
        return report

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        report["reason"] = f"missing_onnxruntime:{exc}"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        if str(policy) == "fail":
            raise SystemExit(f"ONNX parity unavailable (onnxruntime missing). See: {out_path}")
        print(f"WARNING: ONNX parity unavailable (onnxruntime missing). See: {out_path}", file=sys.stderr)
        return report

    if not onnx_path.exists():
        report["reason"] = "onnx_not_found"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        if str(policy) == "fail":
            raise SystemExit(f"ONNX parity unavailable (onnx not found). See: {out_path}")
        print(f"WARNING: ONNX parity unavailable (onnx not found). See: {out_path}", file=sys.stderr)
        return report

    report["available"] = True
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        report["available"] = False
        report["reason"] = f"onnxruntime_init_failed:{exc}"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        if str(policy) == "fail":
            raise SystemExit(f"ONNX parity unavailable (onnxruntime init failed). See: {out_path}") from exc
        print(f"WARNING: ONNX parity unavailable (onnxruntime init failed). See: {out_path}", file=sys.stderr)
        return report

    input_name = None
    try:
        if sess.get_inputs():
            input_name = sess.get_inputs()[0].name
    except Exception:
        input_name = None
    if not input_name:
        input_name = "images"

    gen = torch.Generator(device="cpu")
    try:
        gen.manual_seed(int(seed))
    except Exception:
        pass
    x = torch.rand((1, 3, int(image_size), int(image_size)), generator=gen, dtype=torch.float32, device="cpu")
    report["input"] = {"shape": [1, 3, int(image_size), int(image_size)], "dtype": "float32", "seed": int(seed)}

    ref = model.eval().cpu()
    with torch.no_grad():
        out_torch = ref(x)
    if not isinstance(out_torch, dict):
        raise SystemExit("unexpected torch output type for parity (expected dict).")

    torch_outputs: dict[str, Any] = {}
    for key in ("logits", "bbox"):
        value = out_torch.get(key)
        if value is None:
            continue
        if hasattr(value, "detach"):
            torch_outputs[key] = value.detach().cpu().numpy()

    ort_outputs = sess.run(None, {str(input_name): np.asarray(x.numpy(), dtype=np.float32)})
    names = []
    try:
        names = [o.name for o in sess.get_outputs()]
    except Exception:
        names = []
    if not names:
        names = ["logits", "bbox"]
    cand_outputs: dict[str, Any] = {str(name): val for name, val in zip(names, ort_outputs)}

    score_t, bbox_t = _derive_score_bbox(torch_outputs)
    score_o, bbox_o = _derive_score_bbox(cand_outputs)
    score_stats = _diff_stats(score_t, score_o)
    bbox_stats = _diff_stats(bbox_t, bbox_o)

    score_max = float(score_stats.get("max", float("inf"))) if bool(score_stats.get("ok")) else float("inf")
    bbox_max = float(bbox_stats.get("max", float("inf"))) if bool(bbox_stats.get("ok")) else float("inf")
    passed = bool(
        bool(score_stats.get("ok"))
        and bool(bbox_stats.get("ok"))
        and score_max <= float(score_atol)
        and bbox_max <= float(bbox_atol)
    )

    report["derived"] = {
        "score": score_stats,
        "bbox_sigmoid": bbox_stats,
        "score_max": score_max,
        "bbox_max": bbox_max,
    }
    report["onnxrt"] = {"providers": list(sess.get_providers()), "input_name": str(input_name)}
    report["passed"] = passed

    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if not passed:
        msg = (
            f"ONNX parity failed: score_max={score_max:.6g} bbox_max={bbox_max:.6g} "
            f"(score_atol={float(score_atol):.6g}, bbox_atol={float(bbox_atol):.6g}). See: {out_path}"
        )
        if str(policy) == "fail":
            raise SystemExit(msg)
        print(f"WARNING: {msg}", file=sys.stderr)

    return report


def collect_torch_cuda_meta() -> dict[str, Any]:
    if torch is None:
        return {"available": False}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch.cuda.is_available() is false"}
    try:
        idx = int(torch.cuda.current_device())
    except Exception:
        idx = 0
    meta: dict[str, Any] = {
        "available": True,
        "device_index": idx,
        "device_name": torch.cuda.get_device_name(idx),
        "device_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(idx)),
        "total_memory_mb": int(torch.cuda.get_device_properties(idx).total_memory // (1024 * 1024)),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": (torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None),
    }
    return meta


def collect_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np  # type: ignore

        state["numpy"] = np.random.get_state()
    except Exception:
        state["numpy"] = None
    if torch is not None:
        try:
            state["torch"] = torch.get_rng_state()
        except Exception:
            state["torch"] = None
        if torch.cuda.is_available():
            try:
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                state["torch_cuda"] = None
        else:
            state["torch_cuda"] = None
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    py_state = state.get("python")
    if py_state is not None:
        try:
            random.setstate(py_state)
        except Exception:
            pass
    np_state = state.get("numpy")
    if np_state is not None:
        try:
            import numpy as np  # type: ignore

            np.random.set_state(np_state)
        except Exception:
            pass
    if torch is None:
        return
    torch_state = state.get("torch")
    if torch_state is not None:
        try:
            torch.set_rng_state(torch_state)
        except Exception:
            pass
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception:
            pass


def _rotation_matrix_from_rpy(roll_rad: float, pitch_rad: float, yaw_rad: float) -> "torch.Tensor":
    cr = math.cos(roll_rad)
    sr = math.sin(roll_rad)
    cp = math.cos(pitch_rad)
    sp = math.sin(pitch_rad)
    cy = math.cos(yaw_rad)
    sy = math.sin(yaw_rad)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=torch.float32)
    ry = torch.tensor([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=torch.float32)
    rz = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return rz @ ry @ rx


def compute_warmup_lr(base_lr: float, step: int, warmup_steps: int, warmup_init: float) -> float:
    if warmup_steps <= 0:
        return float(base_lr)
    if step <= 0:
        return float(warmup_init)
    if step >= warmup_steps:
        return float(base_lr)
    alpha = float(step) / float(warmup_steps)
    return float(warmup_init + (base_lr - warmup_init) * alpha)


def parse_milestones(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except Exception:
                continue
        return out
    text = str(value).strip()
    if not text:
        return []
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return out


def apply_denoise_targets(
    targets: list[dict[str, Any]],
    *,
    num_classes: int,
    denoise_count: int,
    bbox_noise: float = 0.0,
    label_noise: float = 0.0,
) -> list[dict[str, Any]]:
    """Append noisy copies of GT targets for denoising-style training.

    This is a lightweight utility used by unit tests and optional training
    extensions. It duplicates `gt_*` tensors (labels/bbox/z/...) `denoise_count`
    times and appends them to the originals.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for apply_denoise_targets")

    copies = max(0, int(denoise_count))
    if copies <= 0:
        return targets

    out: list[dict[str, Any]] = []
    for tgt in targets:
        if not isinstance(tgt, dict):
            out.append(tgt)
            continue

        updated = dict(tgt)

        gt_labels = tgt.get("gt_labels")
        gt_bbox = tgt.get("gt_bbox")
        gt_z = tgt.get("gt_z")

        if not isinstance(gt_labels, torch.Tensor) or gt_labels.numel() == 0:
            out.append(updated)
            continue

        label_list = [gt_labels]
        bbox_list = [gt_bbox] if isinstance(gt_bbox, torch.Tensor) else None
        z_list = [gt_z] if isinstance(gt_z, torch.Tensor) else None

        for _ in range(copies):
            noisy_labels = gt_labels.clone()
            if float(label_noise) > 0.0 and int(num_classes) > 0:
                mask = torch.rand_like(noisy_labels.to(dtype=torch.float32)) < float(label_noise)
                if bool(mask.any()):
                    noisy = torch.randint(0, int(num_classes), (int(mask.sum().item()),), device=noisy_labels.device)
                    noisy_labels = noisy_labels.clone()
                    noisy_labels[mask] = noisy

            label_list.append(noisy_labels)

            if bbox_list is not None and isinstance(gt_bbox, torch.Tensor):
                noisy_bbox = gt_bbox.clone()
                if float(bbox_noise) > 0.0:
                    noisy_bbox = noisy_bbox + torch.randn_like(noisy_bbox) * float(bbox_noise)
                    noisy_bbox = noisy_bbox.clamp(0.0, 1.0)
                bbox_list.append(noisy_bbox)

            if z_list is not None and isinstance(gt_z, torch.Tensor):
                z_list.append(gt_z.clone())

        updated["gt_labels"] = torch.cat(label_list, dim=0)
        if bbox_list is not None:
            updated["gt_bbox"] = torch.cat(bbox_list, dim=0)
        if z_list is not None:
            updated["gt_z"] = torch.cat(z_list, dim=0)

        out.append(updated)

    return out


def flatten_records_for_map(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert rtdetr_pose manifest records into YOLOZU/simple_map GT records."""

    flat: list[dict[str, Any]] = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        image = rec.get("image_path") or rec.get("image") or ""
        image = str(image) if image is not None else ""
        labels_out: list[dict[str, Any]] = []
        for inst in rec.get("labels", []) or []:
            if not isinstance(inst, dict):
                continue
            bb = inst.get("bbox") or {}
            try:
                labels_out.append(
                    {
                        "class_id": int(inst.get("class_id", 0)),
                        "cx": float(bb.get("cx", 0.0)),
                        "cy": float(bb.get("cy", 0.0)),
                        "w": float(bb.get("w", 0.0)),
                        "h": float(bb.get("h", 0.0)),
                    }
                )
            except Exception:
                continue
        flat.append({"image": image, "labels": labels_out})
    return flat


def decode_detections_from_outputs(
    outputs: dict[str, Any],
    image_paths: list[str],
    *,
    score_thresh: float,
    topk: int,
) -> list[dict[str, Any]]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for decode_detections_from_outputs")

    logits = outputs.get("logits")
    bbox = outputs.get("bbox")
    if not isinstance(logits, torch.Tensor) or not isinstance(bbox, torch.Tensor):
        return [{"image": str(p), "detections": []} for p in image_paths]

    probs = logits.softmax(dim=-1)
    scores, class_ids = probs.max(dim=-1)  # (B,Q)
    bg_idx = int(logits.shape[-1]) - 1
    bbox_norm = bbox.sigmoid().clamp(0.0, 1.0)

    batch = int(scores.shape[0])
    out: list[dict[str, Any]] = []
    for i in range(batch):
        image = str(image_paths[i]) if i < len(image_paths) else ""
        sc = scores[i]
        cls = class_ids[i]
        bb = bbox_norm[i]
        keep = (cls != bg_idx) & (sc >= float(score_thresh))
        dets: list[dict[str, Any]] = []
        if bool(keep.any()):
            sc_k = sc[keep]
            cls_k = cls[keep]
            bb_k = bb[keep]

            if int(sc_k.numel()) > int(topk):
                sc_k, idx = torch.topk(sc_k, k=int(topk))
                cls_k = cls_k[idx]
                bb_k = bb_k[idx]
            else:
                order = torch.argsort(sc_k, descending=True)
                sc_k = sc_k[order]
                cls_k = cls_k[order]
                bb_k = bb_k[order]

            for j in range(int(sc_k.shape[0])):
                cx, cy, w, h = [float(v) for v in bb_k[j].tolist()]
                dets.append(
                    {
                        "class_id": int(cls_k[j].item()),
                        "score": float(sc_k[j].item()),
                        "bbox": {"cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h)},
                    }
                )
        out.append({"image": image, "detections": dets})
    return out


def plan_accumulation_windows(*, max_micro_steps: int, grad_accum: int) -> list[int]:
    """Return accumulation window sizes for micro-step training loops.

    Example: max_micro_steps=5, grad_accum=2 -> [2, 2, 1]
    """

    steps_total = int(max_micro_steps)
    if steps_total <= 0:
        return []
    accum = max(1, int(grad_accum))

    windows: list[int] = []
    step = 0
    while step < steps_total:
        window = min(accum, steps_total - step)
        windows.append(int(window))
        step += int(window)
    return windows


def compute_linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(end)
    alpha = min(max(float(step) / float(total_steps - 1), 0.0), 1.0)
    return float(start + (end - start) * alpha)


def compute_mim_schedule(
    *,
    step: int,
    total_steps: int,
    mask_start: float,
    mask_end: float,
    weight_start: float,
    weight_end: float,
    default_mask: float,
    default_weight: float,
) -> tuple[float, float]:
    """Linear schedule for MIM masking ratio and loss weight."""

    steps = int(total_steps)
    if steps <= 0:
        return float(default_mask), float(default_weight)
    s = int(step)
    s = max(0, min(s, steps - 1))
    alpha = 0.0 if steps <= 1 else float(s) / float(steps - 1)
    mask = float(mask_start + (mask_end - mask_start) * alpha)
    weight = float(weight_start + (weight_end - weight_start) * alpha)
    return mask, weight


def compute_stage_weights(
    base: dict[str, float],
    *,
    global_step: int,
    stage_off_steps: int = 0,
    stage_k_steps: int = 0,
) -> tuple[dict[str, float], str]:
    """Return per-step loss weights for simple staged training.

    Stages (by optimizer step):
    - offsets: [0, stage_off_steps)
    - k: [stage_off_steps, stage_off_steps + stage_k_steps)
    - full: afterwards
    """

    out = {str(k): float(v) for k, v in (base or {}).items()}
    step = int(global_step)
    off_n = max(0, int(stage_off_steps))
    k_n = max(0, int(stage_k_steps))

    stage = "full"
    if off_n > 0 and step < off_n:
        stage = "offsets"
        out["k"] = 0.0
    elif k_n > 0 and step < (off_n + k_n):
        stage = "k"
        out["off"] = 0.0
    return out, stage


def compute_stage_costs(
    base: dict[str, float],
    *,
    global_step: int,
    cost_z_start_step: int = 0,
    cost_rot_start_step: int = 0,
    cost_t_start_step: int = 0,
) -> dict[str, float]:
    """Return per-step matcher costs for staged matching."""

    out = {str(k): float(v) for k, v in (base or {}).items()}
    step = int(global_step)
    if step < int(cost_z_start_step):
        out["cost_z"] = 0.0
    if step < int(cost_rot_start_step):
        out["cost_rot"] = 0.0
    if step < int(cost_t_start_step):
        out["cost_t"] = 0.0
    return out


def generate_block_mask(
    height: int,
    width: int,
    *,
    patch_size: int,
    mask_prob: float,
    generator: "torch.Generator",
) -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for generate_block_mask")

    h = max(1, int(height))
    w = max(1, int(width))
    ps = max(1, int(patch_size))
    prob = float(mask_prob)

    grid_h = max(1, (h + ps - 1) // ps)
    grid_w = max(1, (w + ps - 1) // ps)
    mask_grid = torch.rand((grid_h, grid_w), generator=generator) < prob
    mask = mask_grid.repeat_interleave(ps, dim=0).repeat_interleave(ps, dim=1)
    return mask[:h, :w]


def _rgb_to_hsv(image: "torch.Tensor") -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for _rgb_to_hsv")
    if image.ndim != 3 or int(image.shape[0]) != 3:
        raise ValueError("_rgb_to_hsv expects an image tensor shaped [3,H,W]")

    eps = 1e-6
    r, g, b = image[0], image[1], image[2]
    maxc, argmax = torch.max(image, dim=0)
    minc = torch.min(image, dim=0).values
    v = maxc
    delta = maxc - minc
    s = delta / (maxc + eps)

    h = torch.zeros_like(maxc)
    mask = delta > eps
    delta_safe = torch.where(mask, delta, torch.ones_like(delta))

    rc = (g - b) / delta_safe
    gc = (b - r) / delta_safe + 2.0
    bc = (r - g) / delta_safe + 4.0

    h = torch.where(mask & (argmax == 0), rc, h)
    h = torch.where(mask & (argmax == 1), gc, h)
    h = torch.where(mask & (argmax == 2), bc, h)
    h = torch.remainder(h / 6.0, 1.0)
    return torch.stack([h, s, v], dim=0)


def _hsv_to_rgb(hsv: "torch.Tensor") -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for _hsv_to_rgb")
    if hsv.ndim != 3 or int(hsv.shape[0]) != 3:
        raise ValueError("_hsv_to_rgb expects an HSV tensor shaped [3,H,W]")

    h, s, v = hsv[0], hsv[1], hsv[2]
    h = torch.remainder(h, 1.0)
    s = torch.clamp(s, 0.0, 1.0)
    v = torch.clamp(v, 0.0, 1.0)

    h6 = h * 6.0
    i = torch.floor(h6).to(dtype=torch.int64) % 6
    f = h6 - torch.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = torch.empty_like(v)
    g = torch.empty_like(v)
    b = torch.empty_like(v)

    m0 = i == 0
    m1 = i == 1
    m2 = i == 2
    m3 = i == 3
    m4 = i == 4
    m5 = i == 5

    r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
    r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
    r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
    r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
    r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
    r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]

    return torch.stack([r, g, b], dim=0)


def apply_hsv_jitter(
    image: "torch.Tensor",
    *,
    generator: "torch.Generator",
    hgain: float,
    sgain: float,
    vgain: float,
) -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for apply_hsv_jitter")
    hsv = _rgb_to_hsv(image)
    h, s, v = hsv[0], hsv[1], hsv[2]

    if float(hgain) > 0:
        dh = (torch.rand((), generator=generator) * 2.0 - 1.0) * float(hgain)
        h = torch.remainder(h + dh, 1.0)
    if float(sgain) > 0:
        gs = (torch.rand((), generator=generator) * 2.0 - 1.0) * float(sgain)
        s = torch.clamp(s * (1.0 + gs), 0.0, 1.0)
    if float(vgain) > 0:
        gv = (torch.rand((), generator=generator) * 2.0 - 1.0) * float(vgain)
        v = torch.clamp(v * (1.0 + gv), 0.0, 1.0)

    return _hsv_to_rgb(torch.stack([h, s, v], dim=0))


def apply_grayscale(image: "torch.Tensor") -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for apply_grayscale")
    if image.ndim != 3 or int(image.shape[0]) != 3:
        raise ValueError("apply_grayscale expects an image tensor shaped [3,H,W]")

    gray = image[0] * 0.2989 + image[1] * 0.5870 + image[2] * 0.1140
    return gray.unsqueeze(0).expand_as(image)


def _gaussian_kernel2d(
    *,
    kernel_size: int,
    sigma: float,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for _gaussian_kernel2d")

    k = int(kernel_size)
    if k <= 0 or k % 2 == 0:
        raise ValueError("blur_kernel must be a positive odd integer")

    sig = float(sigma)
    if not math.isfinite(sig) or sig <= 0:
        raise ValueError("blur_sigma must be > 0")

    half = k // 2
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    kernel1d = torch.exp(-(coords * coords) / (2.0 * sig * sig))
    kernel1d = kernel1d / torch.clamp(kernel1d.sum(), min=1e-12)
    kernel2d = kernel1d[:, None] * kernel1d[None, :]
    return kernel2d / torch.clamp(kernel2d.sum(), min=1e-12)


def apply_gaussian_blur(
    image: "torch.Tensor",
    *,
    sigma: float,
    kernel_size: int,
) -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for apply_gaussian_blur")
    if image.ndim != 3 or int(image.shape[0]) != 3:
        raise ValueError("apply_gaussian_blur expects an image tensor shaped [3,H,W]")

    k = int(kernel_size)
    pad = k // 2
    kernel2d = _gaussian_kernel2d(kernel_size=k, sigma=float(sigma), device=image.device, dtype=image.dtype)
    weight = kernel2d.view(1, 1, k, k).expand(3, 1, k, k)

    x = image.unsqueeze(0)
    x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate")
    out = torch.nn.functional.conv2d(x, weight, bias=None, stride=1, padding=0, groups=3)
    return out.squeeze(0)


def create_geom_input_from_bboxes(
    bboxes_cxcywh_norm: list[list[float]],
    z_list: list[float] | None,
    *,
    height: int,
    width: int,
) -> "torch.Tensor":
    """Create geometry input tensor from bbox rectangles.

    Output channels follow `tools/example_mim_inference.py`:
    - mask (float32 0/1)
    - normalized depth: mask * log(D / z_ref)
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for create_geom_input_from_bboxes")

    h = max(1, int(height))
    w = max(1, int(width))
    mask = torch.zeros((h, w), dtype=torch.float32)
    depth = torch.ones((h, w), dtype=torch.float32)

    for i, bb in enumerate(bboxes_cxcywh_norm or []):
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        cx, cy, bw, bh = [float(v) for v in bb]
        x0 = int(math.floor((cx - bw * 0.5) * w))
        x1 = int(math.ceil((cx + bw * 0.5) * w))
        y0 = int(math.floor((cy - bh * 0.5) * h))
        y1 = int(math.ceil((cy + bh * 0.5) * h))
        x0 = max(0, min(w - 1, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h, y1))
        if x1 <= x0:
            x1 = min(w, x0 + 1)
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        mask[y0:y1, x0:x1] = 1.0

        if z_list is not None and i < len(z_list):
            try:
                z_val = float(z_list[i])
            except Exception:
                z_val = 1.0
            if z_val > 0:
                depth[y0:y1, x0:x1] = torch.minimum(depth[y0:y1, x0:x1], torch.tensor(z_val, dtype=torch.float32))

    eps = 1e-6
    if bool((mask > 0).any()):
        z_ref = depth[mask > 0].median()
    else:
        z_ref = torch.tensor(1.0, dtype=torch.float32)
    depth_norm = mask * (torch.log(depth + eps) - torch.log(z_ref + eps))
    return torch.stack([mask, depth_norm], dim=0)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = build_parser()

    # Two-stage parse so config can set defaults.
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        cfg = load_config_file(pre.config)
        if not isinstance(cfg, dict):
            raise SystemExit(f"config must be a dict/object at top-level: {pre.config}")
        known_actions = {a.dest: a for a in parser._actions if getattr(a, "dest", None)}

        strict = bool(cfg.get("run_contract") or cfg.get("run_id") or cfg.get("config_version") is not None)
        defaults: dict[str, Any] = {}
        unknown_keys: list[str] = []

        # Back-compat: some older configs used `config: rtdetr_pose/configs/base.json` to mean the model config.
        if (
            "model_config" not in cfg
            and isinstance(cfg.get("config"), str)
            and str(cfg.get("config", "")).strip().lower().endswith(".json")
        ):
            defaults["model_config"] = str(cfg.get("config")).strip()

        for key, value in cfg.items():
            if value is None:
                continue
            dest = str(key)
            if dest == "config":
                # Reserved: this is the trainer settings file path passed via CLI.
                continue
            if dest == "grad_accum":
                dest = "gradient_accumulation_steps"
            action = known_actions.get(dest)
            if action is None:
                unknown_keys.append(str(key))
                continue

            # Basic type/choice validation so YAML mistakes fail fast.
            is_bool_action = isinstance(
                action,
                (
                    argparse._StoreTrueAction,
                    argparse._StoreFalseAction,
                    argparse.BooleanOptionalAction,
                ),
            )
            if is_bool_action:
                if isinstance(value, bool):
                    defaults[dest] = bool(value)
                elif isinstance(value, (int, float)) and float(value) in (0.0, 1.0):
                    defaults[dest] = bool(int(value))
                elif isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0", "yes", "no"):
                    defaults[dest] = value.strip().lower() in ("true", "1", "yes")
                else:
                    raise SystemExit(f"{pre.config}: {key} must be a boolean")
                continue

            casted = value
            if getattr(action, "type", None) is not None:
                try:
                    casted = action.type(value)
                except Exception as exc:
                    raise SystemExit(f"{pre.config}: {key} has invalid type") from exc
            if getattr(action, "choices", None) is not None and casted not in action.choices:
                raise SystemExit(f"{pre.config}: {key} must be one of {sorted(action.choices)}")
            defaults[dest] = casted

        if unknown_keys and strict:
            raise SystemExit(f"{pre.config}: unknown keys (strict mode): {', '.join(sorted(unknown_keys))}")

        parser.set_defaults(**defaults)

    args = parser.parse_args(argv)
    # Back-compat alias used throughout this script.
    try:
        args.grad_accum = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    except Exception:
        args.grad_accum = 1
    return args


def compute_grad_norm(parameters) -> "torch.Tensor":
    """Compute global L2 grad norm over parameters (no clipping)."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for compute_grad_norm")
    norms = []
    for p in parameters:
        g = getattr(p, "grad", None)
        if g is None:
            continue
        if getattr(g, "is_sparse", False):
            try:
                g = g.coalesce().values()
            except Exception:
                continue
        try:
            norms.append(g.detach().norm(2))
        except Exception:
            continue
    if not norms:
        return torch.zeros((), dtype=torch.float32)
    return torch.norm(torch.stack(norms), 2)


def load_checkpoint_into(
    model: "torch.nn.Module",
    optim: "torch.optim.Optimizer | None",
    path: str | Path,
    *,
    sched: Any | None = None,
    scaler: Any | None = None,
    ema: Any | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    meta: dict[str, Any] = {"path": str(path)}

    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
        if optim is not None and isinstance(obj.get("optim_state_dict"), dict):
            try:
                optim.load_state_dict(obj["optim_state_dict"])
                meta["optim_loaded"] = True
            except Exception:
                meta["optim_loaded"] = False
        if sched is not None and isinstance(obj.get("sched_state_dict"), dict):
            try:
                sched.load_state_dict(obj["sched_state_dict"])
                meta["sched_loaded"] = True
            except Exception:
                meta["sched_loaded"] = False
        if scaler is not None and isinstance(obj.get("scaler_state_dict"), dict):
            try:
                scaler.load_state_dict(obj["scaler_state_dict"])
                meta["scaler_loaded"] = True
            except Exception:
                meta["scaler_loaded"] = False
        if ema is not None and isinstance(obj.get("ema_state_dict"), dict):
            try:
                ema.load_state_dict(obj["ema_state_dict"])
                meta["ema_loaded"] = True
            except Exception:
                meta["ema_loaded"] = False
        if restore_rng:
            try:
                restore_rng_state(obj.get("rng_state"))
                meta["rng_restored"] = True
            except Exception:
                meta["rng_restored"] = False
        meta.update({k: obj.get(k) for k in ("epoch", "global_step") if k in obj})
        meta["schema_version"] = obj.get("schema_version")
        return meta

    # Assume weights-only state_dict
    if isinstance(obj, dict):
        model.load_state_dict(obj)
        meta["optim_loaded"] = False
        return meta

    raise SystemExit(f"unrecognized checkpoint format: {path}")


def save_checkpoint_bundle(
    path: str | Path,
    *,
    model: "torch.nn.Module",
    optim: "torch.optim.Optimizer | None",
    sched: Any | None = None,
    scaler: Any | None = None,
    ema: Any | None = None,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    last_epoch_steps: int,
    last_epoch_avg: float | None,
    last_loss_dict: dict[str, Any] | None,
    run_record: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "last_epoch_steps": int(last_epoch_steps),
        "last_epoch_avg": float(last_epoch_avg) if last_epoch_avg is not None else None,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "run_record": run_record
        if run_record is not None
        else build_run_record(
            repo_root=workspace_root,
            args=vars(args),
            dataset_root=(getattr(args, "dataset_root", "") or None),
        ),
    }
    if optim is not None:
        payload["optim_state_dict"] = optim.state_dict()
    if sched is not None and hasattr(sched, "state_dict"):
        try:
            payload["sched_state_dict"] = sched.state_dict()
        except Exception:
            payload["sched_state_dict"] = None
    if scaler is not None and hasattr(scaler, "state_dict"):
        try:
            payload["scaler_state_dict"] = scaler.state_dict()
        except Exception:
            payload["scaler_state_dict"] = None
    if ema is not None and hasattr(ema, "state_dict"):
        try:
            payload["ema_state_dict"] = ema.state_dict()
        except Exception:
            payload["ema_state_dict"] = None
    if rng_state is not None:
        payload["rng_state"] = rng_state
    if last_loss_dict is not None:
        payload["last_loss"] = {
            k: float(v.detach().cpu()) for k, v in last_loss_dict.items() if hasattr(v, "detach")
        }
    torch.save(payload, path)


class ManifestDataset(Dataset):
    def __init__(
        self,
        records,
        *,
        num_queries=300,
        num_classes=80,
        num_keypoints=0,
        keypoint_flip_pairs=(),
        image_size=640,
        seed=0,
        use_matcher=False,
        synthetic_pose=False,
        z_from_dobj=False,
        load_aux=False,
        depth_mode="none",
        depth_unit="unspecified",
        depth_scale=1.0,
        real_images=False,
        multiscale=False,
        scale_min=1.0,
        scale_max=1.0,
        hflip_prob=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        hsv_prob=1.0,
        gray_prob=0.0,
        gaussian_noise_std=0.0,
        gaussian_noise_prob=1.0,
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
        mim_mask_prob=0.0,
        mim_mask_size=16,
        mim_mask_value=0.0,
        derpp_enabled=False,
        derpp_teacher_key="derpp_teacher",
        derpp_keys=(),
    ):
        self.records = records
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.num_keypoints = int(num_keypoints)
        normalized_pairs = []
        seen_pairs = set()
        for pair in keypoint_flip_pairs or ():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            try:
                a = int(pair[0])
                b = int(pair[1])
            except Exception:
                continue
            if a == b:
                continue
            if not (0 <= a < self.num_keypoints and 0 <= b < self.num_keypoints):
                continue
            key = (a, b) if a <= b else (b, a)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            normalized_pairs.append((int(key[0]), int(key[1])))
        self.keypoint_flip_pairs = tuple(normalized_pairs)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.use_matcher = bool(use_matcher)
        self.synthetic_pose = bool(synthetic_pose)
        self.z_from_dobj = bool(z_from_dobj)
        self.load_aux = bool(load_aux)
        self.depth_mode = str(depth_mode or "none").strip().lower()
        self.depth_unit = str(depth_unit or "unspecified").strip().lower()
        self.depth_scale = float(depth_scale)
        self.mim_mask_prob = float(mim_mask_prob)
        self.mim_mask_size = int(mim_mask_size)
        self.mim_mask_value = float(mim_mask_value)
        self.real_images = bool(real_images)
        self.multiscale = bool(multiscale)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.hflip_prob = float(hflip_prob)
        self.hsv_h = float(hsv_h)
        self.hsv_s = float(hsv_s)
        self.hsv_v = float(hsv_v)
        self.hsv_prob = float(hsv_prob)
        self.gray_prob = float(gray_prob)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.gaussian_noise_prob = float(gaussian_noise_prob)
        self.blur_prob = float(blur_prob)
        self.blur_sigma = float(blur_sigma)
        self.blur_kernel = int(blur_kernel)
        self.intrinsics_jitter = bool(intrinsics_jitter)
        self.jitter_dfx = float(jitter_dfx)
        self.jitter_dfy = float(jitter_dfy)
        self.jitter_dcx = float(jitter_dcx)
        self.jitter_dcy = float(jitter_dcy)
        self.sim_jitter = bool(sim_jitter)
        self.sim_jitter_profile = sim_jitter_profile
        self.sim_jitter_extrinsics = bool(sim_jitter_extrinsics)
        self.extrinsics_jitter = bool(extrinsics_jitter)
        self.jitter_dx = float(jitter_dx)
        self.jitter_dy = float(jitter_dy)
        self.jitter_dz = float(jitter_dz)
        self.jitter_droll = float(jitter_droll)
        self.jitter_dpitch = float(jitter_dpitch)
        self.jitter_dyaw = float(jitter_dyaw)
        self.derpp_enabled = bool(derpp_enabled)
        self.derpp_teacher_key = str(derpp_teacher_key) if derpp_teacher_key is not None else ""
        self.derpp_keys = tuple(str(k) for k in (derpp_keys or ()))
        self._derpp_teacher_warned: set[str] = set()

    def _load_derpp_teacher(self, value: Any) -> dict[str, "torch.Tensor"] | None:
        if not self.derpp_enabled or not self.derpp_teacher_key:
            return None
        if value is None:
            return None

        keys = tuple(k for k in self.derpp_keys if k)
        if not keys:
            return None

        def _maybe_squeeze(t: "torch.Tensor") -> "torch.Tensor":
            if t.ndim >= 1 and int(t.shape[0]) == 1:
                return t.squeeze(0)
            return t

        def _warn_once(msg: str) -> None:
            warned = getattr(self, "_derpp_teacher_warned", None)
            if not isinstance(warned, set):
                warned = set()
                setattr(self, "_derpp_teacher_warned", warned)
            if msg in warned:
                return
            warned.add(msg)
            print(msg, file=sys.stderr)

        if isinstance(value, dict):
            out: dict[str, torch.Tensor] = {}
            for k in keys:
                v = value.get(k)
                if isinstance(v, torch.Tensor):
                    out[k] = _maybe_squeeze(v.detach().to(dtype=torch.float32, device="cpu").clone())
                elif isinstance(v, (list, tuple)):
                    try:
                        out[k] = _maybe_squeeze(torch.tensor(v, dtype=torch.float32))
                    except Exception:
                        continue
            return out or None

        if isinstance(value, (str, Path)) and str(value):
            path = Path(value)
            if not path.is_absolute():
                path = (workspace_root / path).resolve()
                if not path.exists():
                    path = (workspace_root.parent / Path(value)).resolve()
            if not path.exists():
                return None
            if path.suffix.lower() == ".json":
                import json

                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    return None
                if isinstance(loaded, dict):
                    return self._load_derpp_teacher(loaded)
                return None
            if path.suffix.lower() in (".pt", ".pth"):
                try:
                    loaded = torch.load(path, map_location="cpu", weights_only=True)
                except Exception:
                    return None
                if isinstance(loaded, dict):
                    return self._load_derpp_teacher(loaded)
                if isinstance(loaded, torch.Tensor):
                    if len(keys) != 1:
                        return None
                    return {keys[0]: _maybe_squeeze(loaded.detach().to(dtype=torch.float32, device="cpu").clone())}
                if isinstance(loaded, (list, tuple)):
                    if len(keys) != 1:
                        return None
                    try:
                        return {keys[0]: _maybe_squeeze(torch.tensor(loaded, dtype=torch.float32))}
                    except Exception:
                        return None
                return None
            if path.suffix.lower() == ".safetensors":
                try:
                    from safetensors.torch import load_file  # type: ignore
                except Exception:
                    _warn_once(
                        "DERPP teacher: safetensors is not installed; cannot load "
                        f"{path}. Install it with `python -m pip install safetensors`."
                    )
                    return None
                try:
                    loaded = load_file(str(path), device="cpu")
                except Exception:
                    return None
                if isinstance(loaded, dict):
                    return self._load_derpp_teacher(loaded)
                return None
            if path.suffix.lower() in (".npy", ".npz"):
                try:
                    import numpy as np
                except Exception:
                    return None
                try:
                    loaded = np.load(path, allow_pickle=False)
                except Exception:
                    return None
                out: dict[str, torch.Tensor] = {}
                if hasattr(loaded, "files"):
                    for k in keys:
                        if k in loaded.files:
                            try:
                                out[k] = _maybe_squeeze(torch.from_numpy(loaded[k]).to(dtype=torch.float32))
                            except Exception:
                                continue
                else:
                    if len(keys) == 1:
                        try:
                            out[keys[0]] = _maybe_squeeze(torch.from_numpy(loaded).to(dtype=torch.float32))
                        except Exception:
                            return None
                return out or None
        return None

    def _load_rgb_image_tensor(self, image_path: Path, target_size: int) -> "torch.Tensor | None":
        if not image_path.exists():
            return None
        try:
            from PIL import Image
        except Exception as exc:
            raise SystemExit(
                "Pillow is required for --real-images. Install it (e.g. pip install Pillow) or omit --real-images."
            ) from exc
        try:
            import numpy as np
        except Exception as exc:
            raise SystemExit(
                "NumPy is required for --real-images. Install it (e.g. pip install numpy) or omit --real-images."
            ) from exc

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            return None

        if target_size > 0:
            img = img.resize((target_size, target_size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        arr = arr / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_2d(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return value
        if not self.load_aux:
            return None
        if isinstance(value, str):
            path = Path(value)
            if path.suffix.lower() == ".json":
                import json

                try:
                    return json.loads(path.read_text())
                except Exception:
                    return None
            if path.suffix.lower() in (".npy", ".npz"):
                try:
                    import numpy as np
                except Exception:
                    return None
                try:
                    loaded = np.load(path)
                except Exception:
                    return None
                if hasattr(loaded, "files"):
                    if not loaded.files:
                        return None
                    return loaded[loaded.files[0]]
                return loaded
        return None

    def _load_depth_sidecar(self, value, target_size: int, flip: bool):
        if self.depth_mode == "none" or value is None:
            return None, False

        depth = None
        if isinstance(value, torch.Tensor):
            depth = value.detach().to(dtype=torch.float32, device="cpu")
        elif isinstance(value, (list, tuple)):
            try:
                depth = torch.tensor(value, dtype=torch.float32)
            except Exception:
                depth = None
        elif isinstance(value, str) and value:
            path = Path(value)
            if not path.is_absolute():
                cand = (workspace_root / path).resolve()
                if cand.exists():
                    path = cand
                else:
                    cand2 = (workspace_root.parent / path).resolve()
                    if cand2.exists():
                        path = cand2
            if path.exists():
                suffix = path.suffix.lower()
                if suffix == ".json":
                    try:
                        loaded = json.loads(path.read_text(encoding="utf-8"))
                        depth = torch.tensor(loaded, dtype=torch.float32)
                    except Exception:
                        depth = None
                elif suffix in (".npy", ".npz"):
                    try:
                        import numpy as np

                        loaded = np.load(path, allow_pickle=False)
                        if hasattr(loaded, "files"):
                            if loaded.files:
                                depth = torch.from_numpy(loaded[loaded.files[0]]).to(dtype=torch.float32)
                        else:
                            depth = torch.from_numpy(loaded).to(dtype=torch.float32)
                    except Exception:
                        depth = None
                else:
                    try:
                        from PIL import Image
                        import numpy as np

                        arr = np.asarray(Image.open(path), dtype=np.float32)
                        depth = torch.from_numpy(arr).to(dtype=torch.float32)
                    except Exception:
                        depth = None

        if depth is None:
            return None, False
        if depth.ndim == 3:
            if int(depth.shape[0]) in (1, 3, 4) and int(depth.shape[1]) > 4 and int(depth.shape[2]) > 4:
                depth = depth[0]
            else:
                depth = depth[..., 0]
        if depth.ndim != 2:
            return None, False

        depth = depth.contiguous()
        if target_size > 0 and (int(depth.shape[0]) != int(target_size) or int(depth.shape[1]) != int(target_size)):
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(int(target_size), int(target_size)),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        if flip:
            depth = torch.flip(depth, dims=(1,))

        depth = depth * float(self.depth_scale)
        finite_any = bool(torch.isfinite(depth).any().item())
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        positive_any = bool((depth > 0).any().item())
        valid = bool(finite_any and positive_any)
        return depth.unsqueeze(0), valid

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        gen = torch.Generator()
        gen.manual_seed(self.seed + int(idx))

        base_size = max(1, int(self.image_size))
        scale = 1.0
        if self.multiscale:
            lo = min(self.scale_min, self.scale_max)
            hi = max(self.scale_min, self.scale_max)
            if hi > 0 and lo > 0:
                scale = float(torch.rand((), generator=gen) * (hi - lo) + lo)
        target_size = max(1, int(round(base_size * scale)))
        scale_factor = float(target_size) / float(base_size)

        flip = False
        if self.hflip_prob > 0:
            flip = bool(torch.rand((), generator=gen) < float(self.hflip_prob))

        # This is a training-loop scaffold.
        # By default we use synthetic images (keeps deps minimal), but an optional
        # mode can load real JPEGs from record['image_path'].
        image = None
        if self.real_images:
            image_path_raw = record.get("image_path")
            if image_path_raw:
                image = self._load_rgb_image_tensor(Path(str(image_path_raw)), target_size)
        if image is None:
            image = torch.rand(3, target_size, target_size, generator=gen)

        if image.shape[-1] != target_size or image.shape[-2] != target_size:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if flip:
            image = torch.flip(image, dims=(2,))

        if self.hsv_prob > 0.0 and (self.hsv_h > 0.0 or self.hsv_s > 0.0 or self.hsv_v > 0.0):
            if bool(torch.rand((), generator=gen) < float(self.hsv_prob)):
                image = apply_hsv_jitter(
                    image,
                    generator=gen,
                    hgain=float(self.hsv_h),
                    sgain=float(self.hsv_s),
                    vgain=float(self.hsv_v),
                )

        if self.gray_prob > 0.0:
            if bool(torch.rand((), generator=gen) < float(self.gray_prob)):
                image = apply_grayscale(image)

        if self.blur_prob > 0.0 and self.blur_sigma > 0.0:
            if bool(torch.rand((), generator=gen) < float(self.blur_prob)):
                sigma = float(torch.rand((), generator=gen) * float(self.blur_sigma))
                sigma = max(sigma, 1e-3)
                image = apply_gaussian_blur(image, sigma=sigma, kernel_size=int(self.blur_kernel))

        if self.gaussian_noise_std > 0.0:
            prob = float(self.gaussian_noise_prob)
            if prob >= 1.0 or bool(torch.rand((), generator=gen) < prob):
                noise = torch.randn(image.shape, generator=gen, device=image.device, dtype=image.dtype)
                image = image + noise * float(self.gaussian_noise_std)
                image = torch.clamp(image, 0.0, 1.0)

        depth_tensor = None
        depth_valid = False
        if self.depth_mode != "none":
            depth_source = record.get("depth_path")
            if depth_source is None:
                depth_source = record.get("depth")
            depth_tensor, depth_valid = self._load_depth_sidecar(depth_source, target_size=target_size, flip=flip)

        image_raw = None
        mim_mask_ratio = None
        if self.mim_mask_prob and float(self.mim_mask_prob) > 0 and int(self.mim_mask_size) > 0:
            image_raw = image.clone()
            mask = generate_block_mask(
                target_size,
                target_size,
                patch_size=int(self.mim_mask_size),
                mask_prob=float(self.mim_mask_prob),
                generator=gen,
            )
            mim_mask_ratio = mask.float().mean()
            if bool(mask.any()):
                image = image.masked_fill(mask.unsqueeze(0).expand_as(image), float(self.mim_mask_value))

        instances = record.get("labels") or []
        if not instances:
            mask_value = record.get("mask")
            if mask_value is None:
                mask_value = record.get("M")
            if mask_value is None:
                mask_value = record.get("mask_path")
            mask_format = str(record.get("mask_format") or "")
            if not mask_format and bool(record.get("mask_instances", False)):
                mask_format = "instance"
            mask_class_id = record.get("mask_class_id")

            # Minimal mask-to-labels support for unit tests and inline-record use.
            if mask_value is not None and isinstance(mask_value, (list, tuple)) and mask_value:
                try:
                    h = len(mask_value)
                    w = len(mask_value[0]) if isinstance(mask_value[0], (list, tuple)) else 0
                except Exception:
                    h = 0
                    w = 0

                if h > 0 and w > 0:
                    unique_vals = set()
                    for row in mask_value:
                        if not isinstance(row, (list, tuple)):
                            continue
                        for v in row:
                            try:
                                unique_vals.add(int(v))
                            except Exception:
                                continue
                    unique_vals.discard(0)

                    derived = []
                    if mask_format.lower() in ("instance", "instances"):
                        class_id = int(mask_class_id) if mask_class_id is not None else 0
                        for inst_id in sorted(unique_vals):
                            x_min = y_min = None
                            x_max = y_max = None
                            for y, row in enumerate(mask_value):
                                if not isinstance(row, (list, tuple)):
                                    continue
                                for x, v in enumerate(row):
                                    try:
                                        if int(v) != int(inst_id):
                                            continue
                                    except Exception:
                                        continue
                                    x_min = x if x_min is None else min(x_min, x)
                                    x_max = x if x_max is None else max(x_max, x)
                                    y_min = y if y_min is None else min(y_min, y)
                                    y_max = y if y_max is None else max(y_max, y)
                            if x_min is None or y_min is None or x_max is None or y_max is None:
                                continue
                            cx = (x_min + x_max + 1) / 2.0 / float(w)
                            cy = (y_min + y_max + 1) / 2.0 / float(h)
                            bw = (x_max - x_min + 1) / float(w)
                            bh = (y_max - y_min + 1) / float(h)
                            derived.append({"class_id": class_id, "bbox": {"cx": cx, "cy": cy, "w": bw, "h": bh}})
                    else:
                        # Treat mask values as class ids (semantic mask -> one bbox per class).
                        for class_val in sorted(unique_vals):
                            x_min = y_min = None
                            x_max = y_max = None
                            for y, row in enumerate(mask_value):
                                if not isinstance(row, (list, tuple)):
                                    continue
                                for x, v in enumerate(row):
                                    try:
                                        if int(v) != int(class_val):
                                            continue
                                    except Exception:
                                        continue
                                    x_min = x if x_min is None else min(x_min, x)
                                    x_max = x if x_max is None else max(x_max, x)
                                    y_min = y if y_min is None else min(y_min, y)
                                    y_max = y if y_max is None else max(y_max, y)
                            if x_min is None or y_min is None or x_max is None or y_max is None:
                                continue
                            cx = (x_min + x_max + 1) / 2.0 / float(w)
                            cy = (y_min + y_max + 1) / 2.0 / float(h)
                            bw = (x_max - x_min + 1) / float(w)
                            bh = (y_max - y_min + 1) / float(h)
                            derived.append(
                                {"class_id": int(class_val), "bbox": {"cx": cx, "cy": cy, "w": bw, "h": bh}}
                            )

                    if derived:
                        instances = derived
        if self.use_matcher:
            full = extract_full_gt_targets(record, num_instances=len(instances))
            gt_labels = []
            gt_bbox = []
            gt_z = []
            gt_z_mask = []
            gt_R = []
            gt_R_mask = []
            gt_t = []
            gt_t_mask = []
            gt_offsets = []
            gt_offsets_mask = []
            gt_keypoints = []
            gt_keypoints_mask = []
            gt_M_mask = []
            gt_D_obj_mask = []
            gt_M = []
            gt_D_obj = []

            # We don't decode images, so default HW is the generated tensor size.
            image_hw = torch.tensor([float(target_size), float(target_size)], dtype=torch.float32)

            # Prefer real intrinsics when present; else synthesize if requested.
            K_gt = None
            if full.get("K_gt") is not None:
                K_gt = torch.tensor(full["K_gt"], dtype=torch.float32)
            elif self.synthetic_pose:
                w = float(self.image_size)
                h = float(self.image_size)
                fx = w
                fy = w
                cx = w * 0.5
                cy = h * 0.5
                K_gt = torch.tensor(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                )

            if K_gt is not None and scale_factor != 1.0:
                K_gt = K_gt.clone()
                K_gt[0, 0] *= scale_factor
                K_gt[1, 1] *= scale_factor
                K_gt[0, 2] *= scale_factor
                K_gt[1, 2] *= scale_factor
            if K_gt is not None and flip:
                K_gt = K_gt.clone()
                K_gt[0, 2] = float(target_size - 1.0) - K_gt[0, 2]
            if K_gt is not None and self.sim_jitter and self.sim_jitter_profile:
                jitter = sample_intrinsics_jitter(self.sim_jitter_profile, seed=self.seed + int(idx))
                K_gt = K_gt.clone()
                K_gt[0, 0] = K_gt[0, 0] * (1.0 + float(jitter.get("dfx", 0.0)))
                K_gt[1, 1] = K_gt[1, 1] * (1.0 + float(jitter.get("dfy", 0.0)))
                K_gt[0, 2] = K_gt[0, 2] + float(jitter.get("dcx", 0.0))
                K_gt[1, 2] = K_gt[1, 2] + float(jitter.get("dcy", 0.0))
            elif K_gt is not None and self.intrinsics_jitter:
                K_gt = K_gt.clone()
                dfx = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dfx)
                dfy = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dfy)
                dcx = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dcx)
                dcy = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dcy)
                K_gt[0, 0] = K_gt[0, 0] * (1.0 + dfx)
                K_gt[1, 1] = K_gt[1, 1] * (1.0 + dfy)
                K_gt[0, 2] = K_gt[0, 2] + dcx
                K_gt[1, 2] = K_gt[1, 2] + dcy

            for inst_i, inst in enumerate(instances):
                class_id = int(inst.get("class_id", -1))
                if not (0 <= class_id < self.num_classes):
                    continue
                bb = inst.get("bbox") or {}
                cx = float(bb.get("cx", 0.0))
                cy = float(bb.get("cy", 0.0))
                w = float(bb.get("w", 0.0))
                h = float(bb.get("h", 0.0))
                if flip:
                    cx = 1.0 - cx
                gt_labels.append(class_id)
                gt_bbox.append([cx, cy, w, h])

                if int(self.num_keypoints) > 0:
                    k_count = int(self.num_keypoints)
                    kps = inst.get("keypoints")
                    kp_xy = [[0.0, 0.0] for _ in range(k_count)]
                    kp_mask = [False for _ in range(k_count)]
                    if isinstance(kps, list) and kps:
                        for ki, kp in enumerate(kps[:k_count]):
                            if not isinstance(kp, dict):
                                continue
                            try:
                                x = float(kp.get("x", 0.0))
                                y = float(kp.get("y", 0.0))
                            except Exception:
                                continue
                            v = kp.get("v", 0.0)
                            try:
                                v_i = int(float(v))
                            except Exception:
                                v_i = 0
                            if flip:
                                x = 1.0 - x
                            kp_xy[ki] = [x, y]
                            kp_mask[ki] = v_i > 0
                    if flip and self.keypoint_flip_pairs:
                        for a, b in self.keypoint_flip_pairs:
                            kp_xy[a], kp_xy[b] = kp_xy[b], kp_xy[a]
                            kp_mask[a], kp_mask[b] = kp_mask[b], kp_mask[a]
                    gt_keypoints.append(kp_xy)
                    gt_keypoints_mask.append(kp_mask)

                # Real (t/R/offsets) if present, otherwise optional synthetic fallback.
                t_i = full.get("t_gt", [None])[inst_i] if full.get("t_gt") is not None else None
                r_i = full.get("R_gt", [None])[inst_i] if full.get("R_gt") is not None else None
                off_i = full.get("offsets_gt", [None])[inst_i] if full.get("offsets_gt") is not None else None

                m_i = full.get("M", [None])[inst_i] if full.get("M") is not None else None
                d_i = full.get("D_obj", [None])[inst_i] if full.get("D_obj") is not None else None
                m_loaded = self._load_2d(m_i)
                d_loaded = self._load_2d(d_i)
                gt_M_mask.append(bool(full.get("M_mask", [False])[inst_i]) if full.get("M_mask") else False)
                gt_D_obj_mask.append(
                    bool(full.get("D_obj_mask", [False])[inst_i]) if full.get("D_obj_mask") else False
                )
                gt_M.append(m_loaded)
                gt_D_obj.append(d_loaded)

                # z/t
                if t_i is not None:
                    t_val = [float(v) for v in t_i]
                    z_val = float(t_val[2])
                    gt_t.append(t_val)
                    gt_t_mask.append(True)
                    gt_z.append(z_val)
                    gt_z_mask.append(True)
                elif self.synthetic_pose:
                    z_val = float(torch.rand((), generator=gen) * 0.9 + 0.1)
                    gt_z.append(z_val)
                    gt_z_mask.append(True)
                    if K_gt is None:
                        raise RuntimeError("synthetic_pose requires K_gt")
                    cx_n = float(bb.get("cx", 0.0))
                    cy_n = float(bb.get("cy", 0.0))
                    u = cx_n * float(image_hw[1])
                    v = cy_n * float(image_hw[0])
                    x = (u - float(K_gt[0, 2])) / float(K_gt[0, 0]) * z_val
                    y = (v - float(K_gt[1, 2])) / float(K_gt[1, 1]) * z_val
                    gt_t.append([x, y, z_val])
                    gt_t_mask.append(True)
                else:
                    # Optional: derive z (and optionally t) from D_obj at bbox center.
                    z_val = None
                    if self.z_from_dobj and d_loaded is not None:
                        z_val = depth_at_bbox_center(d_loaded, bb, mask=m_loaded)
                    if z_val is not None:
                        gt_z.append(float(z_val))
                        gt_z_mask.append(True)
                        if K_gt is not None:
                            cx_n = float(bb.get("cx", 0.0))
                            cy_n = float(bb.get("cy", 0.0))
                            u = cx_n * float(image_hw[1])
                            v = cy_n * float(image_hw[0])
                            x = (u - float(K_gt[0, 2])) / float(K_gt[0, 0]) * float(z_val)
                            y = (v - float(K_gt[1, 2])) / float(K_gt[1, 1]) * float(z_val)
                            gt_t.append([x, y, float(z_val)])
                            gt_t_mask.append(True)
                        else:
                            gt_t.append([0.0, 0.0, float(z_val)])
                            gt_t_mask.append(False)
                    else:
                        gt_z.append(0.0)
                        gt_z_mask.append(False)
                        gt_t.append([0.0, 0.0, 0.0])
                        gt_t_mask.append(False)

                # R
                if r_i is not None:
                    gt_R.append(torch.tensor(r_i, dtype=torch.float32))
                    gt_R_mask.append(True)
                elif self.synthetic_pose:
                    a = torch.randn(3, 3, generator=gen)
                    q, _ = torch.linalg.qr(a)
                    if torch.det(q) < 0:
                        q[:, 0] = -q[:, 0]
                    gt_R.append(q)
                    gt_R_mask.append(True)
                else:
                    gt_R.append(torch.eye(3, dtype=torch.float32))
                    gt_R_mask.append(False)

                # offsets
                if off_i is not None:
                    du = float(off_i[0]) * scale_factor
                    dv = float(off_i[1]) * scale_factor
                    if flip:
                        du = -du
                    gt_offsets.append([du, dv])
                    gt_offsets_mask.append(True)
                elif self.synthetic_pose:
                    gt_offsets.append([0.0, 0.0])
                    gt_offsets_mask.append(True)
                else:
                    gt_offsets.append([0.0, 0.0])
                    gt_offsets_mask.append(False)

            if (self.sim_jitter and self.sim_jitter_profile and self.sim_jitter_extrinsics) or self.extrinsics_jitter:
                if self.sim_jitter and self.sim_jitter_profile and self.sim_jitter_extrinsics:
                    jitter = sample_extrinsics_jitter(self.sim_jitter_profile, seed=self.seed + int(idx))
                    dx = float(jitter.get("dx", 0.0))
                    dy = float(jitter.get("dy", 0.0))
                    dz = float(jitter.get("dz", 0.0))
                    droll = float(jitter.get("droll", 0.0))
                    dpitch = float(jitter.get("dpitch", 0.0))
                    dyaw = float(jitter.get("dyaw", 0.0))
                else:
                    dx = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dx)
                    dy = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dy)
                    dz = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dz)
                    droll = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_droll)
                    dpitch = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dpitch)
                    dyaw = float((torch.rand((), generator=gen) * 2.0 - 1.0) * self.jitter_dyaw)

                for j in range(len(gt_t)):
                    if gt_t_mask[j]:
                        gt_t[j] = [gt_t[j][0] + dx, gt_t[j][1] + dy, gt_t[j][2] + dz]
                if any(gt_R_mask):
                    r_delta = _rotation_matrix_from_rpy(
                        math.radians(droll),
                        math.radians(dpitch),
                        math.radians(dyaw),
                    )
                    gt_R = [r_delta @ r if mask else r for r, mask in zip(gt_R, gt_R_mask)]

            num_inst = len(gt_labels)
            m_tensor = None
            d_tensor = None
            if self.load_aux and num_inst > 0:
                for item in gt_M:
                    if item is not None:
                        m_tensor = torch.as_tensor(item, dtype=torch.float32)
                        break
                if m_tensor is not None:
                    m_hw = tuple(m_tensor.shape)
                    stacked = []
                    for item in gt_M:
                        if item is None:
                            stacked.append(torch.zeros(m_hw, dtype=torch.float32))
                            continue
                        t = torch.as_tensor(item, dtype=torch.float32)
                        if tuple(t.shape) != m_hw:
                            stacked.append(torch.zeros(m_hw, dtype=torch.float32))
                        else:
                            stacked.append(t)
                    m_tensor = torch.stack(stacked, dim=0)

                for item in gt_D_obj:
                    if item is not None:
                        d_tensor = torch.as_tensor(item, dtype=torch.float32)
                        break
                if d_tensor is not None:
                    d_hw = tuple(d_tensor.shape)
                    stacked = []
                    for item in gt_D_obj:
                        if item is None:
                            stacked.append(torch.zeros(d_hw, dtype=torch.float32))
                            continue
                        t = torch.as_tensor(item, dtype=torch.float32)
                        if tuple(t.shape) != d_hw:
                            stacked.append(torch.zeros(d_hw, dtype=torch.float32))
                        else:
                            stacked.append(t)
                    d_tensor = torch.stack(stacked, dim=0)
            if num_inst == 0:
                gt_labels_t = torch.empty((0,), dtype=torch.long)
                gt_bbox_t = torch.empty((0, 4), dtype=torch.float32)
                gt_z_t = torch.empty((0, 1), dtype=torch.float32)
                gt_z_mask_t = torch.empty((0,), dtype=torch.bool)
                gt_R_t = torch.empty((0, 3, 3), dtype=torch.float32)
                gt_R_mask_t = torch.empty((0,), dtype=torch.bool)
                gt_t_t = torch.empty((0, 3), dtype=torch.float32)
                gt_t_mask_t = torch.empty((0,), dtype=torch.bool)
                gt_offsets_t = torch.empty((0, 2), dtype=torch.float32)
                gt_offsets_mask_t = torch.empty((0,), dtype=torch.bool)
                if int(self.num_keypoints) > 0:
                    k_count = int(self.num_keypoints)
                    gt_keypoints_t = torch.empty((0, k_count, 2), dtype=torch.float32)
                    gt_keypoints_mask_t = torch.empty((0, k_count), dtype=torch.bool)
                else:
                    gt_keypoints_t = torch.empty((0, 0, 2), dtype=torch.float32)
                    gt_keypoints_mask_t = torch.empty((0, 0), dtype=torch.bool)
                gt_M_mask_t = torch.empty((0,), dtype=torch.bool)
                gt_D_obj_mask_t = torch.empty((0,), dtype=torch.bool)
            else:
                gt_labels_t = torch.tensor(gt_labels, dtype=torch.long)
                gt_bbox_t = torch.tensor(gt_bbox, dtype=torch.float32)
                gt_z_t = torch.tensor(gt_z, dtype=torch.float32).unsqueeze(-1)
                gt_z_mask_t = torch.tensor(gt_z_mask, dtype=torch.bool)
                gt_R_t = torch.stack(gt_R, dim=0)
                gt_R_mask_t = torch.tensor(gt_R_mask, dtype=torch.bool)
                gt_t_t = torch.tensor(gt_t, dtype=torch.float32)
                gt_t_mask_t = torch.tensor(gt_t_mask, dtype=torch.bool)
                gt_offsets_t = torch.tensor(gt_offsets, dtype=torch.float32)
                gt_offsets_mask_t = torch.tensor(gt_offsets_mask, dtype=torch.bool)
                if int(self.num_keypoints) > 0:
                    gt_keypoints_t = torch.tensor(gt_keypoints, dtype=torch.float32)
                    gt_keypoints_mask_t = torch.tensor(gt_keypoints_mask, dtype=torch.bool)
                else:
                    gt_keypoints_t = torch.empty((num_inst, 0, 2), dtype=torch.float32)
                    gt_keypoints_mask_t = torch.empty((num_inst, 0), dtype=torch.bool)
                gt_M_mask_t = torch.tensor(gt_M_mask, dtype=torch.bool)
                gt_D_obj_mask_t = torch.tensor(gt_D_obj_mask, dtype=torch.bool)

            targets = {
                "image_path": str(record.get("image_path", "") or ""),
                "gt_labels": gt_labels_t,
                "gt_bbox": gt_bbox_t,
                "gt_z": gt_z_t,
                "gt_z_mask": gt_z_mask_t,
                "gt_R": gt_R_t,
                "gt_R_mask": gt_R_mask_t,
                "gt_t": gt_t_t,
                "gt_t_mask": gt_t_mask_t,
                "gt_offsets": gt_offsets_t,
                "gt_offsets_mask": gt_offsets_mask_t,
                "gt_keypoints": gt_keypoints_t,
                "gt_keypoints_mask": gt_keypoints_mask_t,
                "gt_M_mask": gt_M_mask_t,
                "gt_D_obj_mask": gt_D_obj_mask_t,
                **({"gt_M": m_tensor} if m_tensor is not None else {}),
                **({"gt_D_obj": d_tensor} if d_tensor is not None else {}),
                **({"K_gt": K_gt} if K_gt is not None else {}),
                "image_hw": image_hw,
                "depth_valid": torch.tensor(bool(depth_valid), dtype=torch.bool),
            }
            derpp_teacher = self._load_derpp_teacher(record.get(self.derpp_teacher_key))
            if derpp_teacher is not None:
                targets["derpp_teacher"] = derpp_teacher
            out = {"image": image, "targets": targets}
            if depth_tensor is not None:
                out["depth"] = depth_tensor
            out["depth_valid"] = bool(depth_valid)
            if image_raw is not None and mim_mask_ratio is not None:
                out["image_raw"] = image_raw
                out["mim_mask_ratio"] = float(mim_mask_ratio)
            return out

        labels = torch.full((self.num_queries,), -1, dtype=torch.long)
        bbox = torch.zeros((self.num_queries, 4), dtype=torch.float32)
        for qi, inst in enumerate(instances[: self.num_queries]):
            class_id = int(inst.get("class_id", -1))
            if 0 <= class_id < self.num_classes:
                labels[qi] = class_id
            bb = inst.get("bbox") or {}
            cx = float(bb.get("cx", 0.0))
            if flip:
                cx = 1.0 - cx
            bbox[qi, 0] = cx
            bbox[qi, 1] = float(bb.get("cy", 0.0))
            bbox[qi, 2] = float(bb.get("w", 0.0))
            bbox[qi, 3] = float(bb.get("h", 0.0))

        targets = {"labels": labels, "bbox": bbox}
        targets["image_path"] = str(record.get("image_path", "") or "")
        derpp_teacher = self._load_derpp_teacher(record.get(self.derpp_teacher_key))
        if derpp_teacher is not None:
            targets["derpp_teacher"] = derpp_teacher
        out = {"image": image, "targets": targets}
        if depth_tensor is not None:
            out["depth"] = depth_tensor
        if self.depth_mode != "none":
            out["depth_valid"] = bool(depth_valid)
        if image_raw is not None and mim_mask_ratio is not None:
            out["image_raw"] = image_raw
            out["mim_mask_ratio"] = float(mim_mask_ratio)
        return out


def _pad_field(targets, key, max_len, *, pad_value=0.0, dtype=None):
    if max_len == 0:
        return None
    tail = None
    for tgt in targets:
        value = tgt.get(key)
        if isinstance(value, torch.Tensor):
            if dtype is None:
                dtype = value.dtype
            tail = value.shape[1:]
            break
    if dtype is None:
        return None
    if tail is None:
        tail = ()

    rows = []
    for tgt in targets:
        value = tgt.get(key)
        if not isinstance(value, torch.Tensor):
            value = torch.empty((0, *tail), dtype=dtype)
        pad_len = max_len - value.shape[0]
        if pad_len < 0:
            raise ValueError(f"{key} has more instances than max_len")
        if pad_len == 0:
            padded = value
        else:
            pad = torch.full((pad_len, *tail), pad_value, dtype=dtype)
            padded = torch.cat([value, pad], dim=0)
        rows.append(padded)
    return torch.stack(rows, dim=0)


def collate(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = [item["targets"] for item in batch]

    extra: dict[str, torch.Tensor] = {}
    if any("image_raw" in item for item in batch):
        raws = []
        for item in batch:
            raw = item.get("image_raw")
            if isinstance(raw, torch.Tensor):
                raws.append(raw)
            else:
                raws.append(item["image"])
        extra["image_raw"] = torch.stack(raws, dim=0)
    if any("mim_mask_ratio" in item for item in batch):
        ratios = []
        for item in batch:
            try:
                ratios.append(float(item.get("mim_mask_ratio", 0.0)))
            except Exception:
                ratios.append(0.0)
        extra["mim_mask_ratio"] = torch.tensor(ratios, dtype=torch.float32)
    if any(("depth" in item) or ("depth_valid" in item) for item in batch):
        depth_rows = []
        depth_valid_rows = []
        for item in batch:
            depth_item = item.get("depth")
            if isinstance(depth_item, torch.Tensor) and depth_item.ndim == 3 and int(depth_item.shape[0]) == 1:
                depth_rows.append(depth_item.to(dtype=torch.float32))
                try:
                    depth_valid_rows.append(bool(item.get("depth_valid", True)))
                except Exception:
                    depth_valid_rows.append(True)
            else:
                depth_rows.append(torch.zeros((1, int(images.shape[-2]), int(images.shape[-1])), dtype=torch.float32))
                depth_valid_rows.append(False)
        extra["depth"] = torch.stack(depth_rows, dim=0)
        extra["depth_valid"] = torch.tensor(depth_valid_rows, dtype=torch.bool)

    if not targets or "gt_labels" not in targets[0]:
        return images, targets

    counts = torch.tensor(
        [int(tgt.get("gt_labels").shape[0]) if isinstance(tgt.get("gt_labels"), torch.Tensor) else 0 for tgt in targets],
        dtype=torch.long,
    )
    max_len = int(counts.max().item()) if counts.numel() else 0
    if max_len == 0:
        padded = {
            "gt_count": counts,
            "gt_mask": torch.zeros((len(targets), 0), dtype=torch.bool),
        }
        out = {"per_sample": targets, "padded": padded, **extra}
        return images, out

    padded = {
        "gt_count": counts,
        "gt_mask": (torch.arange(max_len).unsqueeze(0) < counts.unsqueeze(1)),
        "gt_labels": _pad_field(targets, "gt_labels", max_len, pad_value=-1, dtype=torch.long),
        "gt_bbox": _pad_field(targets, "gt_bbox", max_len, pad_value=0.0, dtype=torch.float32),
    }

    optional_fields = [
        ("gt_z", 0.0, torch.float32),
        ("gt_z_mask", False, torch.bool),
        ("gt_R", 0.0, torch.float32),
        ("gt_R_mask", False, torch.bool),
        ("gt_t", 0.0, torch.float32),
        ("gt_t_mask", False, torch.bool),
        ("gt_offsets", 0.0, torch.float32),
        ("gt_offsets_mask", False, torch.bool),
        ("gt_keypoints", 0.0, torch.float32),
        ("gt_keypoints_mask", False, torch.bool),
        ("gt_M_mask", False, torch.bool),
        ("gt_D_obj_mask", False, torch.bool),
        ("gt_M", 0.0, torch.float32),
        ("gt_D_obj", 0.0, torch.float32),
    ]
    for key, pad_value, dtype in optional_fields:
        if any(key in tgt for tgt in targets):
            padded[key] = _pad_field(targets, key, max_len, pad_value=pad_value, dtype=dtype)

    if any("depth_valid" in tgt for tgt in targets):
        depth_valid_list = []
        for tgt in targets:
            value = tgt.get("depth_valid")
            if isinstance(value, torch.Tensor):
                depth_valid_list.append(bool(value.to(dtype=torch.bool).item()))
            else:
                depth_valid_list.append(False)
        padded["depth_valid"] = torch.tensor(depth_valid_list, dtype=torch.bool)

    out = {"per_sample": targets, "padded": padded, **extra}
    return images, out


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
            raise SystemExit("rtdetr_pose.export.export_onnx is required for ONNX export") from exc

        onnx_path = Path(str(args.onnx_out))
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        if run_contract is not None and getattr(args, "best_checkpoint_out", None):
            best_path = Path(str(args.best_checkpoint_out))
            if best_path.exists():
                load_checkpoint_into(unwrap_model(model), None, str(best_path), restore_rng=False)
        dummy = torch.zeros((1, 3, int(args.image_size), int(args.image_size)), dtype=torch.float32, device=device)
        export_onnx(
            unwrap_model(model).eval(),
            dummy,
            str(onnx_path),
            opset_version=int(args.onnx_opset),
            dynamic_hw=bool(args.onnx_dynamic_hw),
        )
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

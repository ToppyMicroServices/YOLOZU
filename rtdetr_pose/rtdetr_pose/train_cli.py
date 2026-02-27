"""CLI argument parsing for train_minimal."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


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

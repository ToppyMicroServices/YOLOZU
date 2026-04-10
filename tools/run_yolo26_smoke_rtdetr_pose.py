import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_DIR = repo_root / "configs" / "yolo26_rtdetr_pose"
DEFAULT_BUCKETS = ("n",)


def _now_utc_compact() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO26 bucket smoke run using the in-repo RT-DETR reference trainer (rtdetr_pose).",
    )
    p.add_argument("--dataset", default=str(repo_root / "data" / "coco128"), help="YOLO-format dataset root.")
    p.add_argument("--split", default="train2017", help="Dataset split (default: train2017, matches coco128 layout).")
    p.add_argument(
        "--buckets",
        default=",".join(DEFAULT_BUCKETS),
        help="Comma-separated bucket list (n,s,m,l,x). Default: n",
    )
    p.add_argument(
        "--config-dir",
        default=str(DEFAULT_CONFIG_DIR),
        help="Folder containing per-bucket configs (expects yolo26{bucket}.json).",
    )
    p.add_argument("--device", default="cpu", help="Torch device for training/export_predictions (default: cpu).")
    p.add_argument("--image-size", type=int, default=64, help="Training image size (default: 64).")
    p.add_argument("--max-steps", type=int, default=5, help="Training steps cap per bucket (default: 5).")
    p.add_argument("--max-images", type=int, default=32, help="Max images for predictions export (default: 32).")
    p.add_argument("--run-dir", default=None, help="Run directory (default: runs/yolo26_smoke/<utc>).")
    p.add_argument("--skip-fetch", action="store_true", help="Do not attempt to fetch coco128 if missing.")
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and reuse existing checkpoint/model.onnx in each bucket run dir.",
    )
    p.add_argument("--strict", action="store_true", help="Strict predictions schema validation.")
    return p.parse_args(argv)


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(shlex.join(cmd))
    subprocess.run(cmd, cwd=str(repo_root), check=True, env=env)


def _ensure_coco128(dataset_root: Path, split: str, *, skip_fetch: bool) -> None:
    if skip_fetch:
        return
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    if images_dir.exists() and labels_dir.exists():
        return

    env = os.environ.copy()
    env["YOLOZU_INSECURE_SSL"] = "1"
    cmd = ["bash", "tools/fetch_coco128.sh"]
    print("note: coco128 missing; fetching via tools/fetch_coco128.sh (YOLOZU_INSECURE_SSL=1)")
    _run(cmd, env=env)


def _bucket_config_path(config_dir: Path, bucket: str) -> Path:
    path = config_dir / f"yolo26{bucket}.json"
    if not path.exists():
        raise SystemExit(f"missing config for bucket {bucket!r}: {path}")
    return path


def _run_bucket(
    *,
    bucket: str,
    dataset_root: Path,
    split: str,
    config_dir: Path,
    device: str,
    image_size: int,
    max_steps: int,
    max_images: int,
    run_dir: Path,
    skip_train: bool,
    strict: bool,
) -> dict[str, str]:
    python = sys.executable
    bucket_dir = run_dir / f"yolo26{bucket}"
    bucket_dir.mkdir(parents=True, exist_ok=True)

    config_path = _bucket_config_path(config_dir, bucket)
    checkpoint = bucket_dir / "checkpoint.pt"
    onnx_path = bucket_dir / "model.onnx"

    commands: list[str] = []

    if not skip_train:
        cmd = [
            python,
            "rtdetr_pose/tools/train_minimal.py",
            "--config",
            str(config_path),
            "--dataset-root",
            str(dataset_root),
            "--split",
            str(split),
            "--device",
            str(device),
            "--real-images",
            "--image-size",
            str(int(image_size)),
            "--max-steps",
            str(int(max_steps)),
            "--run-dir",
            str(bucket_dir),
            "--checkpoint-out",
            str(checkpoint),
            "--onnx-out",
            str(onnx_path),
        ]
        commands.append(shlex.join(cmd))
        _run(cmd)

    if not checkpoint.exists():
        raise SystemExit(f"missing checkpoint: {checkpoint}")
    if not onnx_path.exists():
        raise SystemExit(f"missing onnx: {onnx_path}")

    pred_path = bucket_dir / f"pred_yolo26{bucket}.json"
    cmd = [
        python,
        "tools/export_predictions.py",
        "--adapter",
        "rtdetr_pose",
        "--dataset",
        str(dataset_root),
        "--split",
        str(split),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--device",
        str(device),
        "--max-images",
        str(int(max_images)),
        "--wrap",
        "--output",
        str(pred_path),
    ]
    if strict:
        cmd.append("--strict")
    commands.append(shlex.join(cmd))
    _run(cmd)

    suite_path = bucket_dir / "eval_suite_dry_run.json"
    cmd = [
        python,
        "tools/eval_suite.py",
        "--dataset",
        str(dataset_root),
        "--split",
        str(split),
        "--predictions-glob",
        str(pred_path),
        "--bbox-format",
        "cxcywh_norm",
        "--dry-run",
        "--output",
        str(suite_path),
        "--max-images",
        str(int(max_images)),
    ]
    commands.append(shlex.join(cmd))
    _run(cmd)

    (bucket_dir / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    return {
        "bucket": bucket,
        "run_dir": str(bucket_dir),
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "predictions": str(pred_path),
        "eval_suite_dry_run": str(suite_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = Path(args.dataset)
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root

    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = repo_root / config_dir

    buckets = _split_csv(str(args.buckets))
    if not buckets:
        raise SystemExit("--buckets must be non-empty")
    for bucket in buckets:
        if bucket not in ("n", "s", "m", "l", "x"):
            raise SystemExit(f"invalid bucket: {bucket!r} (expected n,s,m,l,x)")

    run_dir = Path(args.run_dir) if args.run_dir else (repo_root / "runs" / "yolo26_smoke" / _now_utc_compact())
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    _ensure_coco128(dataset_root, str(args.split), skip_fetch=bool(args.skip_fetch))

    artifacts: list[dict[str, str]] = []
    for bucket in buckets:
        artifacts.append(
            _run_bucket(
                bucket=bucket,
                dataset_root=dataset_root,
                split=str(args.split),
                config_dir=config_dir,
                device=str(args.device),
                image_size=int(args.image_size),
                max_steps=int(args.max_steps),
                max_images=int(args.max_images),
                run_dir=run_dir,
                skip_train=bool(args.skip_train),
                strict=bool(args.strict),
            )
        )

    run_record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_dir.name,
        "dataset": str(dataset_root),
        "split": str(args.split),
        "device": str(args.device),
        "config_dir": str(config_dir),
        "buckets": buckets,
        "artifacts": artifacts,
    }
    (run_dir / "run.json").write_text(json.dumps(run_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

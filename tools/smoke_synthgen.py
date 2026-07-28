import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args(argv):
    p = argparse.ArgumentParser(
        description=(
            "Run SynthGen intake smoke (validate -> overlay -> eval), optionally "
            "generating and qualifying a fresh cross-repo handoff first."
        )
    )
    p.add_argument(
        "--dataset-root",
        default=None,
        help="Existing SynthGen dataset root (default: bundled mini-shard).",
    )
    p.add_argument("--schema-id", default="animal_v1", help="Schema used for overlay sample selection.")
    p.add_argument(
        "--predictions",
        default=None,
        help="Predictions JSON used for eval (default: bundled fixture or generated oracle self-check).",
    )
    p.add_argument("--max-samples", type=int, default=2, help="Max samples for contract validation.")
    p.add_argument("--output-dir", default="reports", help="Directory for smoke artifacts.")
    p.add_argument(
        "--synthgen-repo",
        default=None,
        help="Optional YOLOZU-synthgen checkout. Generate, export, and qualify a fresh handoff.",
    )
    p.add_argument(
        "--synthgen-python",
        default=None,
        help="Generator Python (default: <repo>/.venv312/bin/python, then .venv/bin/python).",
    )
    p.add_argument("--backend", choices=("placeholder", "open3d"), default="open3d")
    p.add_argument("--mode", choices=("render_only", "bg_only_inpaint"), default="bg_only_inpaint")
    p.add_argument("--global-seed", type=int, default=20260727)
    p.add_argument("--num-train", type=int, default=3)
    p.add_argument("--num-val", type=int, default=2)
    return p.parse_args(argv)


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_json: bool = False,
):
    completed = subprocess.run(
        cmd,
        check=True,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        capture_output=capture_json,
    )
    if capture_json:
        return json.loads(completed.stdout)
    return None


def _resolve_generator_python(repo: Path, explicit: str | None) -> Path:
    candidates = [Path(explicit).expanduser()] if explicit else [repo / ".venv312/bin/python", repo / ".venv/bin/python"]
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else Path.cwd() / candidate
        if path.is_file():
            # Keep the venv launcher path. Resolving its interpreter symlink
            # would bypass the venv's site-packages.
            return path.absolute()
    raise FileNotFoundError(
        "generator Python not found; pass --synthgen-python or create "
        f"{repo}/.venv312/bin/python"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_first_row(dataset_root: Path) -> tuple[Path, dict]:
    shards = sorted((dataset_root / "shards").glob("train_*.jsonl"))
    if not shards:
        raise ValueError(f"no train shards found under {dataset_root}")
    shard = shards[0]
    for line in shard.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return shard, json.loads(line)
    raise ValueError(f"no sample rows found in {shard}")


def _prepare_generator_handoff(args, repo_root: Path, output_dir: Path) -> tuple[Path, Path, dict]:
    import numpy as np

    synthgen_repo = Path(args.synthgen_repo).expanduser().resolve()
    if not (synthgen_repo / "manifest.json").is_file() or not (synthgen_repo / "src/yolozu_synthgen").is_dir():
        raise ValueError(f"not a YOLOZU-synthgen checkout: {synthgen_repo}")
    if args.dataset_root is not None or args.predictions is not None:
        raise ValueError("--synthgen-repo cannot be combined with --dataset-root or --predictions")
    if args.num_train < 1 or args.num_val < 0:
        raise ValueError("--num-train must be >= 1 and --num-val must be >= 0")

    generated_root = output_dir / "generated_handoff"
    if generated_root.exists():
        raise FileExistsError(
            f"refusing to replace existing generated handoff: {generated_root}; choose a fresh --output-dir"
        )
    generated_root.mkdir(parents=True)

    synthgen_python = _resolve_generator_python(synthgen_repo, args.synthgen_python)
    synthgen_env = {**os.environ, "PYTHONPATH": str(synthgen_repo / "src")}
    module = [str(synthgen_python), "-m", "yolozu_synthgen"]
    demo_root = generated_root / "demo"
    demo_summary = _run(
        [
            *module,
            "generate-demo-dataset",
            "--backend",
            args.backend,
            "--output-dir",
            str(demo_root),
            "--num-train",
            str(args.num_train),
            "--num-val",
            str(args.num_val),
            "--global-seed",
            str(args.global_seed),
        ],
        cwd=synthgen_repo,
        env=synthgen_env,
        capture_json=True,
    )

    render_export = generated_root / "render_export"
    _run(
        [
            *module,
            "export-yolozu-synthgen",
            "--shards-root",
            str(demo_root / "shards"),
            "--output-dir",
            str(render_export),
        ],
        cwd=synthgen_repo,
        env=synthgen_env,
        capture_json=True,
    )

    scene_spec = sorted((demo_root / "scene_specs").glob("train-*.json"))[0]
    if args.mode == "render_only":
        dataset_root = render_export
        mode_shards = demo_root / "shards"
    else:
        mode_shards = generated_root / "mode_shards"
        _run(
            [
                *module,
                "augment-sample",
                "--backend",
                args.backend,
                "--asset-keypoints",
                str(synthgen_repo / "examples/asset_keypoints_demo.jsonl"),
                "--mode",
                args.mode,
                "--output-dir",
                str(mode_shards),
                str(scene_spec),
            ],
            cwd=synthgen_repo,
            env=synthgen_env,
            capture_json=True,
        )
        dataset_root = generated_root / "mode_export"
        _run(
            [
                *module,
                "export-yolozu-synthgen",
                "--shards-root",
                str(mode_shards),
                "--output-dir",
                str(dataset_root),
            ],
            cwd=synthgen_repo,
            env=synthgen_env,
            capture_json=True,
        )

    validation = _run(
        [*module, "validate-yolozu-export", "--input-root", str(dataset_root)],
        cwd=synthgen_repo,
        env=synthgen_env,
        capture_json=True,
    )

    shard, row = _load_first_row(dataset_root)
    predictions = shard.parent / "oracle_interface_predictions.json"
    predictions.write_text(json.dumps([row], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    original_sample = sorted((demo_root / "shards").glob("shard-*/sample-*"))[0]
    mode_sample = sorted(mode_shards.glob("shard-*/sample-*"))[0]
    truth_fields = ("depth_ndc.npy", "inst_id.npy", "sem_id.npy", "kpts2d.npy")
    truth_equal = {
        name: bool(np.array_equal(np.load(original_sample / name), np.load(mode_sample / name)))
        for name in truth_fields
    }
    if not all(truth_equal.values()):
        raise ValueError(f"renderer-owned truth changed under {args.mode}: {truth_equal}")
    qa_report = json.loads((mode_sample / "qa.json").read_text(encoding="utf-8"))
    if qa_report.get("accepted") is not True:
        raise ValueError(f"generator QA rejected {args.mode}: {qa_report.get('reject_reasons')}")

    bridge_env = {**os.environ, "PYTHONPATH": str(synthgen_repo / "src")}
    training_loader = _run(
        [
            sys.executable,
            "-m",
            "yolozu_synthgen",
            "training-loader-smoke",
            "--input-root",
            str(render_export),
            "--batch-size",
            "2",
        ],
        cwd=synthgen_repo,
        env=bridge_env,
        capture_json=True,
    )
    yolozu_bridge = _run(
        [
            sys.executable,
            "-m",
            "yolozu_synthgen",
            "yolozu-bridge-smoke",
            "--input-root",
            str(render_export),
            "--batch-size",
            "2",
            "--yolozu-repo",
            str(repo_root),
        ],
        cwd=synthgen_repo,
        env=bridge_env,
        capture_json=True,
    )

    generator_commit = subprocess.run(
        ["git", "-C", str(synthgen_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    yolozu_commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    image_path = Path(row["image"])
    if not image_path.is_absolute():
        image_path = shard.parent / image_path
    return dataset_root, predictions, {
        "scope": "oracle interface self-check; not model accuracy",
        "mode": args.mode,
        "backend": args.backend,
        "global_seed": args.global_seed,
        "generator_commit": generator_commit,
        "yolozu_commit": yolozu_commit,
        "demo": demo_summary,
        "validation": validation,
        "qa_report": qa_report,
        "truth_equal": truth_equal,
        "training_loader": training_loader,
        "yolozu_bridge": yolozu_bridge,
        "hashes": {
            "predictions": _sha256(predictions),
            "image": _sha256(image_path.resolve()),
        },
    }


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cwd = Path.cwd()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = cwd / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    qualification = None
    if args.synthgen_repo is not None:
        dataset_root, predictions, qualification = _prepare_generator_handoff(args, repo_root, output_dir)
    else:
        dataset_root = Path(args.dataset_root or "data/smoke/synthgen_minishard")
        predictions = Path(
            args.predictions or "data/smoke/synthgen_minishard/predictions_synthgen_smoke.json"
        )

    if not dataset_root.is_absolute():
        dataset_root = cwd / dataset_root
    if not predictions.is_absolute():
        predictions = cwd / predictions

    shard = dataset_root / "shards" / "train_000.jsonl"
    overlay = output_dir / "smoke_synthgen_overlay.png"
    report = output_dir / "smoke_synthgen_eval.json"
    summary = output_dir / "smoke_synthgen_summary.json"

    _run(
        [
            sys.executable,
            str(repo_root / "tools" / "validate_synthgen_contract.py"),
            "--input",
            str(shard),
            "--max-samples",
            str(int(args.max_samples)),
        ]
    )
    _run(
        [
            sys.executable,
            str(repo_root / "tools" / "render_synthgen_overlay.py"),
            "--dataset-root",
            str(dataset_root),
            "--schema-id",
            str(args.schema_id),
            "--sample-index",
            "0",
            "--output",
            str(overlay),
        ]
    )
    _run(
        [
            sys.executable,
            str(repo_root / "tools" / "eval_synthgen.py"),
            "--dataset-root",
            str(dataset_root),
            "--predictions",
            str(predictions),
            "--output",
            str(report),
        ]
    )

    summary_payload = {
        "dataset_root": str(dataset_root),
        "shard": str(shard),
        "predictions": str(predictions),
        "overlay": str(overlay),
        "eval_report": str(report),
        "schema_id": str(args.schema_id),
        "max_samples": int(args.max_samples),
        "status": "ok",
    }
    if qualification is not None:
        summary_payload["qualification"] = qualification
    summary.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

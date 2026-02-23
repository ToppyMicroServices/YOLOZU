import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args(argv):
    p = argparse.ArgumentParser(description="Run SynthGen intake smoke (validate -> overlay -> eval).")
    p.add_argument("--dataset-root", default="data/smoke/synthgen_minishard", help="SynthGen mini-shard root.")
    p.add_argument("--schema-id", default="animal_v1", help="Schema used for overlay sample selection.")
    p.add_argument(
        "--predictions",
        default="data/smoke/synthgen_minishard/predictions_synthgen_smoke.json",
        help="Predictions JSON used for eval smoke.",
    )
    p.add_argument("--max-samples", type=int, default=2, help="Max samples for contract validation.")
    p.add_argument("--output-dir", default="reports", help="Directory for smoke artifacts.")
    return p.parse_args(argv)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cwd = Path.cwd()
    repo_root = Path(__file__).resolve().parents[1]

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = cwd / dataset_root
    predictions = Path(args.predictions)
    if not predictions.is_absolute():
        predictions = cwd / predictions
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = cwd / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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
    summary.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

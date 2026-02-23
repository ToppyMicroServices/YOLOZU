import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.data.synthgen_shard_dataset import SynthGenShardDataset  # noqa: E402
from yolozu.synthgen_eval import evaluate_synthgen_predictions  # noqa: E402


def _resolve_path(path: str | Path, *, base_dir: Path, root_dir: Path) -> Path | None:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    cand1 = base_dir / p
    if cand1.exists():
        return cand1
    cand2 = root_dir / p
    if cand2.exists():
        return cand2
    return None


def _load_artifact(value: Any, *, base_dir: Path, root_dir: Path) -> Any:
    if isinstance(value, str):
        resolved = _resolve_path(value, base_dir=base_dir, root_dir=root_dir)
        if resolved is None:
            return value
        suffix = resolved.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            with Image.open(resolved) as img:
                return np.asarray(img)
        if suffix == ".npy":
            return np.load(resolved, allow_pickle=False)
        if suffix == ".json":
            return json.loads(resolved.read_text(encoding="utf-8"))
        return resolved.read_text(encoding="utf-8")
    return value


def _load_predictions(path: Path, *, dataset_root: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("predictions"), list):
        entries = raw["predictions"]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError("predictions must be list or {predictions:[...]}")

    out: dict[str, dict[str, Any]] = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("sample_id") or entry.get("image_id") or entry.get("image") or idx)
        record = dict(entry)
        base_dir = path.parent
        for field in ("sem_id", "inst_id", "depth_ndc", "kpts2d"):
            if field in record:
                record[field] = _load_artifact(record[field], base_dir=base_dir, root_dir=dataset_root)
        out[key] = record
    return out


def parse_args(argv):
    p = argparse.ArgumentParser(description="Evaluate SynthGen prediction artifacts (kpts/seg/depth).")
    p.add_argument("--dataset-root", required=True, help="SynthGen shard dataset root.")
    p.add_argument("--predictions", required=True, help="Predictions JSON path (sample_id keyed records).")
    p.add_argument("--schema-id", default=None, help="Optional schema filter.")
    p.add_argument("--max-samples", type=int, default=0, help="Optional cap (0 means all).")
    p.add_argument("--num-classes", type=int, default=0, help="Semantic class count (0=auto).")
    p.add_argument("--output", default="reports/synthgen_eval.json", help="Output report JSON.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = Path.cwd() / dataset_root
    pred_path = Path(args.predictions)
    if not pred_path.is_absolute():
        pred_path = Path.cwd() / pred_path

    dataset = SynthGenShardDataset(dataset_root, schema_id=args.schema_id)
    limit = int(args.max_samples)
    samples = list(dataset if limit <= 0 else [dataset[i] for i in range(min(limit, len(dataset)))])
    predictions_index = _load_predictions(pred_path, dataset_root=dataset_root)

    report = evaluate_synthgen_predictions(
        samples=samples,
        predictions_index=predictions_index,
        num_classes=(int(args.num_classes) if int(args.num_classes) > 0 else None),
    )
    report["meta"] = {
        "dataset_root": str(dataset_root),
        "predictions": str(pred_path),
        "schema_id": args.schema_id,
        "sample_count": len(samples),
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

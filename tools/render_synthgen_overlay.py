import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.data.synthgen_shard_dataset import SynthGenShardDataset  # noqa: E402
from yolozu.viz.synthgen_overlay import render_synthgen_overlay  # noqa: E402


def parse_args(argv):
    p = argparse.ArgumentParser(description="Render SynthGen semantic/instance/keypoint overlay.")
    p.add_argument("--dataset-root", required=True, help="SynthGen shard dataset root.")
    p.add_argument("--schema-id", default=None, help="Optional schema filter (animal_v1/mechanical_v1).")
    p.add_argument("--sample-index", type=int, default=0, help="Sample index after filtering.")
    p.add_argument("--alpha", type=float, default=0.45, help="Semantic overlay alpha.")
    p.add_argument("--output", required=True, help="Output PNG path.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = Path.cwd() / dataset_root
    dataset = SynthGenShardDataset(dataset_root, schema_id=args.schema_id)
    if len(dataset) <= 0:
        raise SystemExit(f"no samples found under {dataset_root} (schema_id={args.schema_id})")

    idx = int(args.sample_index)
    if idx < 0 or idx >= len(dataset):
        raise SystemExit(f"sample index out of range: {idx} (len={len(dataset)})")

    sample = dataset[idx]
    image = render_synthgen_overlay(sample, alpha=float(args.alpha))
    out = Path(args.output)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(out)


if __name__ == "__main__":
    main()

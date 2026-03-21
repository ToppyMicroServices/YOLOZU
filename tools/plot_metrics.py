import argparse
import json
from pathlib import Path


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="reports/train_metrics.jsonl", help="Path to metrics JSONL.")
    p.add_argument("--out", default="reports/train_loss.png", help="Output image path.")
    p.add_argument("--key", default="loss_avg", help="Metric key to plot (default: loss_avg).")
    return p.parse_args(argv)


def _load_jsonl(path: Path):
    rows = []
    if not path.exists():
        raise SystemExit(f"metrics jsonl not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main(argv=None):
    args = _parse_args(argv)
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting; install requirements-dev.txt") from exc

    jsonl_path = Path(args.jsonl)
    rows = _load_jsonl(jsonl_path)
    x = []
    y = []
    for row in rows:
        metrics = row.get("metrics") or {}
        meta = row.get("meta") or {}
        if args.key not in metrics:
            continue
        step = meta.get("global_step") or meta.get("step") or len(x)
        x.append(int(step))
        y.append(float(metrics[args.key]))

    if not x:
        raise SystemExit(f"no metric '{args.key}' found in {jsonl_path}")

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label=args.key)
    plt.xlabel("step")
    plt.ylabel(args.key)
    plt.title(f"{args.key} over steps")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(out_path)


if __name__ == "__main__":
    main()

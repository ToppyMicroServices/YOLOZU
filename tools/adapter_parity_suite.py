import argparse
import json
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]

SUPPORTED_ADAPTERS = (
    "mmdet",
    "detectron2",
    "ultralytics",
    "rtdetr",
    "opencv_dnn",
    "custom_cpp",
)


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adapter-predictions",
        action="append",
        required=True,
        help="Repeatable adapter=predictions.json mapping.",
    )
    p.add_argument("--reference-adapter", default=None, help="Reference adapter key from --adapter-predictions.")
    p.add_argument("--image-size", default=None, help="Optional image size for parity (e.g., 640 or 640,640).")
    p.add_argument("--iou-thresh", type=float, default=0.99)
    p.add_argument("--score-atol", type=float, default=1e-4)
    p.add_argument("--bbox-atol", type=float, default=1e-4)
    p.add_argument("--output", default="reports/adapter_parity_suite.json")
    return p.parse_args(argv)


def _parse_item(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"invalid --adapter-predictions value: {text!r} (expected adapter=path)")
    adapter, path = text.split("=", 1)
    adapter = adapter.strip()
    path = path.strip()
    if adapter not in SUPPORTED_ADAPTERS:
        raise ValueError(f"unsupported adapter: {adapter!r}; supported: {', '.join(SUPPORTED_ADAPTERS)}")
    if not path:
        raise ValueError(f"empty path for adapter: {adapter}")
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return adapter, p


def _run_parity(
    *,
    reference: Path,
    candidate: Path,
    image_size: str | None,
    iou_thresh: float,
    score_atol: float,
    bbox_atol: float,
) -> tuple[bool, dict | None, str]:
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "check_predictions_parity.py"),
        "--reference",
        str(reference),
        "--candidate",
        str(candidate),
        "--iou-thresh",
        str(iou_thresh),
        "--score-atol",
        str(score_atol),
        "--bbox-atol",
        str(bbox_atol),
    ]
    if image_size:
        cmd.extend(["--image-size", str(image_size)])

    proc = subprocess.run(cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    payload = None
    err = ""
    text = (proc.stdout or "").strip()
    if text:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            err = text
    if proc.returncode != 0 and not err:
        err = (proc.stderr or "").strip()
    return proc.returncode == 0, payload, err


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    mapping: dict[str, Path] = {}
    for item in args.adapter_predictions:
        adapter, path = _parse_item(item)
        if adapter in mapping:
            raise SystemExit(f"duplicate adapter mapping: {adapter}")
        if not path.exists():
            raise SystemExit(f"file not found for {adapter}: {path}")
        mapping[adapter] = path

    if len(mapping) < 2:
        raise SystemExit("at least two --adapter-predictions values are required")

    reference_adapter = args.reference_adapter or sorted(mapping.keys())[0]
    if reference_adapter not in mapping:
        raise SystemExit(f"reference adapter not provided: {reference_adapter}")

    reference_path = mapping[reference_adapter]
    results = []
    all_ok = True
    for adapter, candidate_path in sorted(mapping.items()):
        if adapter == reference_adapter:
            continue
        ok, report, error = _run_parity(
            reference=reference_path,
            candidate=candidate_path,
            image_size=args.image_size,
            iou_thresh=float(args.iou_thresh),
            score_atol=float(args.score_atol),
            bbox_atol=float(args.bbox_atol),
        )
        all_ok = all_ok and ok
        results.append(
            {
                "adapter": adapter,
                "path": str(candidate_path),
                "ok": ok,
                "error": error or None,
                "report": report,
            }
        )

    payload = {
        "supported_adapters": list(SUPPORTED_ADAPTERS),
        "reference_adapter": reference_adapter,
        "reference_path": str(reference_path),
        "ok": all_ok,
        "comparisons": results,
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)

    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

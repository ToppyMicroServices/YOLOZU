#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.run_record import build_run_record


REQUIRED_BACKENDS = ("torch", "onnxrt", "trt", "opencv_dnn", "custom_cpp")


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_run_id() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backend parity matrix automation across torch/onnxrt/trt/opencv_dnn/custom_cpp predictions.",
    )
    p.add_argument(
        "--backend-predictions",
        action="append",
        required=True,
        help="Repeatable backend=predictions.json mapping.",
    )
    p.add_argument("--reference-backend", default="torch", choices=REQUIRED_BACKENDS)
    p.add_argument("--image-size", default=None, help="Optional fixed image size (e.g. 640 or 640,640).")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--iou-thresh", type=float, default=0.99)
    p.add_argument("--score-atol", type=float, default=1e-4)
    p.add_argument("--bbox-atol", type=float, default=1e-4)
    p.add_argument("--run-id", default=None)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-html", default=None)
    return p.parse_args(argv)


def _parse_backend_mapping(values: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"invalid --backend-predictions value: {value!r} (expected backend=path)")
        backend, path_text = value.split("=", 1)
        backend = backend.strip().lower()
        if backend not in REQUIRED_BACKENDS:
            raise SystemExit(f"unsupported backend: {backend!r}; expected one of {', '.join(REQUIRED_BACKENDS)}")
        p = Path(path_text.strip())
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        mapping[backend] = p

    missing = [b for b in REQUIRED_BACKENDS if b not in mapping]
    if missing:
        raise SystemExit(f"missing backend predictions for: {', '.join(missing)}")

    for backend, path in mapping.items():
        if not path.exists():
            raise SystemExit(f"backend predictions file not found ({backend}): {path}")

    return mapping


def _run_parity(
    *,
    reference: Path,
    candidate: Path,
    image_size: str | None,
    max_images: int | None,
    iou_thresh: float,
    score_atol: float,
    bbox_atol: float,
) -> tuple[int, dict[str, Any] | None, str]:
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
    if max_images is not None:
        cmd.extend(["--max-images", str(int(max_images))])

    proc = subprocess.run(cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    payload = None
    err = ""
    out = (proc.stdout or "").strip()
    if out:
        try:
            payload = json.loads(out)
        except Exception:
            err = out
    if proc.returncode != 0 and not err:
        err = (proc.stderr or "").strip()
    return proc.returncode, payload, err


def _summarize_parity(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {
            "images": 0,
            "failure_images": 0,
            "total_failures": 0,
            "matched": 0,
            "extra_candidate": 0,
            "score_abs_max": None,
            "bbox_abs_max": None,
        }

    results = report.get("results") or []
    failure_images = 0
    total_failures = 0
    matched = 0
    extra_candidate = 0
    score_abs_max = 0.0
    bbox_abs_max = 0.0

    for item in results:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("ok", False)):
            failure_images += 1
        failures = item.get("failures") or []
        total_failures += len(failures)
        counts = item.get("counts") or {}
        matched += int(counts.get("matched") or 0)
        extra_candidate += int(counts.get("extra_cand") or 0)

        for failure in failures:
            if not isinstance(failure, dict):
                continue
            if failure.get("type") != "value_mismatch":
                continue
            ref = failure.get("ref") or {}
            cand = failure.get("cand") or {}
            try:
                score_abs_max = max(score_abs_max, abs(float(ref.get("score")) - float(cand.get("score"))))
            except (TypeError, ValueError):
                continue
            ref_bbox = ref.get("bbox") or {}
            cand_bbox = cand.get("bbox") or {}
            for key in ("cx", "cy", "w", "h"):
                try:
                    bbox_abs_max = max(bbox_abs_max, abs(float(ref_bbox.get(key)) - float(cand_bbox.get(key))))
                except Exception:
                    continue

    return {
        "images": int(report.get("images") or 0),
        "failure_images": int(failure_images),
        "total_failures": int(total_failures),
        "matched": int(matched),
        "extra_candidate": int(extra_candidate),
        "score_abs_max": float(score_abs_max),
        "bbox_abs_max": float(bbox_abs_max),
    }


def _fixed_input_fingerprint(
    *,
    backend_files: dict[str, Path],
    reference_backend: str,
    image_size: str | None,
    max_images: int | None,
    iou_thresh: float,
    score_atol: float,
    bbox_atol: float,
) -> str:
    payload = {
        "reference_backend": reference_backend,
        "thresholds": {
            "image_size": image_size,
            "max_images": max_images,
            "iou_thresh": iou_thresh,
            "score_atol": score_atol,
            "bbox_atol": bbox_atol,
        },
        "backend_files": {
            backend: {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for backend, path in sorted(backend_files.items())
        },
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render_html(report: dict[str, Any]) -> str:
    rows = []
    for item in report.get("matrix") or []:
        backend = html.escape(str(item.get("backend")))
        ok = "PASS" if bool(item.get("ok")) else "FAIL"
        summary = item.get("summary") or {}
        rows.append(
            "<tr>"
            f"<td>{backend}</td>"
            f"<td>{ok}</td>"
            f"<td>{int(summary.get('images') or 0)}</td>"
            f"<td>{int(summary.get('failure_images') or 0)}</td>"
            f"<td>{int(summary.get('total_failures') or 0)}</td>"
            f"<td>{float(summary.get('score_abs_max') or 0.0):.6g}</td>"
            f"<td>{float(summary.get('bbox_abs_max') or 0.0):.6g}</td>"
            "</tr>"
        )

    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Backend Parity Matrix</title>"
        "<style>body{font-family:system-ui,Arial,sans-serif;margin:20px}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px;text-align:left}"
        "th{background:#f5f5f5}</style></head><body>"
        f"<h1>Backend Parity Matrix</h1><p>timestamp={html.escape(str(report.get('timestamp_utc')))}</p>"
        f"<p>reference={html.escape(str(report.get('reference_backend')))} | fingerprint={html.escape(str(report.get('fixed_input_fingerprint')))}</p>"
        "<table><thead><tr><th>Backend</th><th>Status</th><th>Images</th><th>Failure Images</th><th>Total Failures</th><th>Score Abs Max</th><th>BBox Abs Max</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    backend_files = _parse_backend_mapping(args.backend_predictions)

    run_id = str(args.run_id or _default_run_id())
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = (repo_root / run_dir).resolve()
    else:
        run_dir = (repo_root / str(args.runs_dir) / "backend_parity_matrix" / run_id).resolve()
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_json = Path(args.output_json) if args.output_json else reports_dir / "backend_parity_matrix.json"
    output_html = Path(args.output_html) if args.output_html else reports_dir / "backend_parity_matrix.html"
    if not output_json.is_absolute():
        output_json = (repo_root / output_json).resolve()
    if not output_html.is_absolute():
        output_html = (repo_root / output_html).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    reference_backend = str(args.reference_backend)
    reference_path = backend_files[reference_backend]

    matrix = []
    all_ok = True
    for backend in REQUIRED_BACKENDS:
        if backend == reference_backend:
            continue
        rc, parity_report, error = _run_parity(
            reference=reference_path,
            candidate=backend_files[backend],
            image_size=args.image_size,
            max_images=args.max_images,
            iou_thresh=float(args.iou_thresh),
            score_atol=float(args.score_atol),
            bbox_atol=float(args.bbox_atol),
        )
        ok = rc == 0
        all_ok = all_ok and ok
        matrix.append(
            {
                "backend": backend,
                "candidate_path": str(backend_files[backend]),
                "ok": ok,
                "error": error or None,
                "summary": _summarize_parity(parity_report),
                "parity_report": parity_report,
            }
        )

    thresholds = {
        "image_size": args.image_size,
        "max_images": args.max_images,
        "iou_thresh": float(args.iou_thresh),
        "score_atol": float(args.score_atol),
        "bbox_atol": float(args.bbox_atol),
    }

    report = {
        "schema_version": 1,
        "timestamp_utc": _now_utc(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "reports_dir": str(reports_dir),
        "reference_backend": reference_backend,
        "backend_files": {
            backend: {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for backend, path in sorted(backend_files.items())
        },
        "thresholds": thresholds,
        "fixed_input_fingerprint": _fixed_input_fingerprint(
            backend_files=backend_files,
            reference_backend=reference_backend,
            image_size=args.image_size,
            max_images=args.max_images,
            iou_thresh=float(args.iou_thresh),
            score_atol=float(args.score_atol),
            bbox_atol=float(args.bbox_atol),
        ),
        "ok": all_ok,
        "matrix": matrix,
        "run_record": build_run_record(
            repo_root=repo_root,
            argv=(sys.argv[1:] if argv is None else argv),
            args=vars(args),
            extra={
                "command_str": shlex.join([sys.executable, str(repo_root / "tools" / "backend_parity_matrix.py"), *(sys.argv[1:] if argv is None else argv)]),
            },
        ),
    }

    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    output_html.write_text(_render_html(report), encoding="utf-8")
    print(output_json)
    print(output_html)

    if not all_ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

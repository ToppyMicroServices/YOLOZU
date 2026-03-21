import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.boxes import iou_cxcywh_norm_dict
from yolozu.cli_args import require_non_negative_int, resolve_input_path
from yolozu.image_keys import add_image_aliases
from yolozu.keypoints import normalize_keypoints
from yolozu.predictions import load_predictions_entries

_NUMERIC_ERRORS = (TypeError, ValueError, OverflowError)


@dataclass(frozen=True)
class _Det:
    class_id: int
    score: float
    bbox: dict
    keypoints: list[dict]


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="Reference predictions JSON (e.g. PyTorch).")
    p.add_argument("--candidate", required=True, help="Candidate predictions JSON (e.g. ONNXRuntime/TensorRT).")
    p.add_argument("--iou-thresh", type=float, default=0.99, help="IoU threshold to consider a match.")
    p.add_argument("--score-atol", type=float, default=1e-4, help="Absolute tolerance for score differences.")
    p.add_argument("--bbox-atol", type=float, default=1e-4, help="Absolute tolerance for bbox cx/cy/w/h differences.")
    p.add_argument("--kp-atol", type=float, default=1e-4, help="Absolute tolerance for keypoint x/y differences (normalized coords).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images.")
    return p.parse_args(argv)


def _bbox_tuple_norm(bbox: dict) -> tuple[float, float, float, float]:
    return float(bbox["cx"]), float(bbox["cy"]), float(bbox["w"]), float(bbox["h"])


def _bbox_iou_cxcywh_norm(a: dict, b: dict) -> float:
    return iou_cxcywh_norm_dict(a, b)


def _close(a: float, b: float, atol: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(float(a) - float(b)) <= float(atol)


def _load_index(path: str) -> tuple[list[str], dict[str, list[_Det]]]:
    entries = load_predictions_entries(path)
    images = []
    out: dict[str, list[_Det]] = {}
    for e in entries:
        image = str(e.get("image", ""))
        if not image:
            continue
        images.append(image)
        dets: list[_Det] = []
        for d in (e.get("detections") or []):
            if not isinstance(d, dict):
                continue
            if "class_id" not in d or "bbox" not in d or "score" not in d or "keypoints" not in d:
                continue
            try:
                kps = normalize_keypoints(d.get("keypoints"), where="det.keypoints")
            except ValueError:
                continue
            dets.append(_Det(class_id=int(d["class_id"]), score=float(d["score"]), bbox=d["bbox"], keypoints=kps))
        add_image_aliases(out, image, dets)
    return images, out


def _match_image(
    *,
    image_key: str,
    ref: list[_Det],
    cand: list[_Det],
    iou_thresh: float,
    score_atol: float,
    bbox_atol: float,
    kp_atol: float,
) -> dict:
    used = set()
    matches = []
    failures = []

    for i, r in enumerate(ref):
        best = None
        best_iou = -1.0
        for j, c in enumerate(cand):
            if j in used:
                continue
            if c.class_id != r.class_id:
                continue
            try:
                iou = _bbox_iou_cxcywh_norm(r.bbox, c.bbox)
            except _NUMERIC_ERRORS:
                continue
            if iou > best_iou:
                best_iou = iou
                best = j

        if best is None or best_iou < float(iou_thresh):
            failures.append(
                {
                    "type": "missing_match",
                    "ref_index": i,
                    "class_id": r.class_id,
                    "ref_score": r.score,
                    "best_iou": None if best is None else best_iou,
                }
            )
            continue

        used.add(best)
        c = cand[best]

        rb = _bbox_tuple_norm(r.bbox)
        cb = _bbox_tuple_norm(c.bbox)
        bbox_ok = all(_close(a, b, bbox_atol) for a, b in zip(rb, cb))
        score_ok = _close(r.score, c.score, score_atol)

        kp_ok = True
        kp_max = 0.0
        if len(r.keypoints) != len(c.keypoints):
            kp_ok = False
        else:
            for rk, ck in zip(r.keypoints, c.keypoints):
                dx = abs(float(rk["x"]) - float(ck["x"]))
                dy = abs(float(rk["y"]) - float(ck["y"]))
                kp_max = max(kp_max, float(dx), float(dy))
                if dx > float(kp_atol) or dy > float(kp_atol):
                    kp_ok = False

        matches.append(
            {
                "ref_index": i,
                "cand_index": best,
                "class_id": r.class_id,
                "iou": best_iou,
                "score_ref": r.score,
                "score_cand": c.score,
                "score_ok": score_ok,
                "bbox_ok": bbox_ok,
                "keypoints_ok": kp_ok,
                "keypoints_max_abs_diff": kp_max,
            }
        )

        if not (bbox_ok and score_ok and kp_ok):
            failures.append(
                {
                    "type": "value_mismatch",
                    "ref_index": i,
                    "cand_index": best,
                    "class_id": r.class_id,
                    "iou": best_iou,
                    "ref": {"score": r.score, "bbox": r.bbox, "keypoints": r.keypoints},
                    "cand": {"score": c.score, "bbox": c.bbox, "keypoints": c.keypoints},
                }
            )

    extras = [j for j in range(len(cand)) if j not in used]
    return {
        "image": image_key,
        "counts": {"ref": len(ref), "cand": len(cand), "matched": len(matches), "extra_cand": len(extras)},
        "matches": matches,
        "extras": extras,
        "failures": failures,
        "ok": len(failures) == 0,
    }


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        max_images = require_non_negative_int(args.max_images, flag_name="--max-images")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    ref_path = resolve_input_path(args.reference, cwd=Path.cwd(), repo_root=repo_root)
    cand_path = resolve_input_path(args.candidate, cwd=Path.cwd(), repo_root=repo_root)

    images, ref_idx = _load_index(str(ref_path))
    _, cand_idx = _load_index(str(cand_path))
    if max_images is not None:
        images = images[: int(max_images)]
    if not images:
        raise SystemExit("no images found in reference predictions")

    per_image = []
    ok = True
    for image in images:
        ref = ref_idx.get(image, [])
        cand = cand_idx.get(image, [])
        result = _match_image(
            image_key=image,
            ref=ref,
            cand=cand,
            iou_thresh=args.iou_thresh,
            score_atol=args.score_atol,
            bbox_atol=args.bbox_atol,
            kp_atol=args.kp_atol,
        )
        per_image.append(result)
        ok = ok and bool(result["ok"])

    report = {
        "reference": str(ref_path),
        "candidate": str(cand_path),
        "iou_thresh": args.iou_thresh,
        "score_atol": args.score_atol,
        "bbox_atol": args.bbox_atol,
        "kp_atol": args.kp_atol,
        "images": len(per_image),
        "ok": ok,
        "results": per_image,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

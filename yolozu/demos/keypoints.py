from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _utc_run_id() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def _require_deps() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        from PIL import Image, ImageDraw

        import torch
        import torchvision
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("demo keypoints requires torch, torchvision, numpy, and Pillow") from exc
    return np, (Image, ImageDraw), (torch, torchvision)


_COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# COCO keypoint skeleton (1-indexed in the common spec; we store as 0-indexed).
_COCO_SKELETON = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
]


def _pick_default_image() -> Path | None:
    candidates = [
        # Prefer images that reliably yield a strong person/keypoints detection.
        Path("data") / "smoke" / "images" / "val" / "000000000036.jpg",
        Path("data") / "smoke" / "images" / "val" / "000000000049.jpg",
        Path("data") / "smoke" / "images" / "val" / "000000000061.jpg",
        Path("data") / "smoke" / "images" / "val" / "000000000064.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def run_keypoints_demo(
    *,
    image: str | Path | None = None,
    run_dir: str | Path | None = None,
    device: str = "auto",
    score_threshold: float = 0.7,
    max_persons: int = 3,
    output_name: str = "keypoints_demo_report.json",
) -> Path:
    """Keypoints demo using a pretrained Keypoint R-CNN (torchvision)."""

    np, (Image, ImageDraw), (torch, torchvision) = _require_deps()

    if image is None:
        picked = _pick_default_image()
        if picked is None:
            raise FileNotFoundError(
                "no default demo image found under data/smoke; pass --image <path>"
            )
        image = picked

    image_path = Path(image)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    if run_dir is None:
        run_dir = Path("demo_output") / "keypoints" / _utc_run_id()
    else:
        run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    orig_out = run_dir / "image.png"
    overlay_out = run_dir / "keypoints_overlay.png"

    img = Image.open(image_path).convert("RGB")
    img.save(orig_out)

    # Device resolution (same policy as instance-seg demo).
    if str(device).strip().lower() == "auto":
        if torch.cuda.is_available():
            torch_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch_device = torch.device("mps")
        else:
            torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(str(device))

    weights_name: str | None = None
    preprocess = None
    try:
        from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights  # type: ignore

        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        weights_name = getattr(weights, "name", None) or str(weights)
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights)
        preprocess = weights.transforms()
    except Exception:
        # Fallback for older torchvision.
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        weights_name = "pretrained=True"

        try:
            import torchvision.transforms.functional as TVF  # type: ignore

            preprocess = lambda im: TVF.to_tensor(im)  # noqa: E731
        except Exception:
            preprocess = None

    if preprocess is None:
        raise RuntimeError("torchvision transforms not available")

    model.to(torch_device)
    model.eval()

    with torch.no_grad():
        x = preprocess(img).to(torch_device)
        out = model([x])

    if not out or not isinstance(out, list) or not isinstance(out[0], dict):
        raise RuntimeError("unexpected torchvision output")

    det = out[0]
    boxes = det.get("boxes")
    scores = det.get("scores")
    labels = det.get("labels")
    keypoints = det.get("keypoints")

    if boxes is None or scores is None or labels is None or keypoints is None:
        raise RuntimeError("unexpected torchvision output (missing boxes/scores/labels/keypoints)")

    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    keypoints = keypoints.detach().cpu().numpy()  # NxKx3

    # Keep only person label (COCO=1) + score threshold.
    keep: list[int] = []
    for i, (sc, lab) in enumerate(zip(scores.tolist(), labels.tolist(), strict=False)):
        if int(lab) != 1:
            continue
        if float(sc) < float(score_threshold):
            continue
        keep.append(int(i))
        if len(keep) >= int(max_persons):
            break

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    persons: list[dict[str, Any]] = []
    for idx in keep:
        b = boxes[idx].tolist()  # xyxy
        sc = float(scores[idx])
        kpts = keypoints[idx].tolist()

        # bbox
        x0, y0, x1, y1 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)

        # skeleton (draw only visible-ish points; v in {0,1,2})
        def _vis(p: list[float]) -> bool:
            try:
                return float(p[2]) > 0.0
            except Exception:
                return False

        # lines
        for a, c in _COCO_SKELETON:
            a0 = int(a) - 1
            c0 = int(c) - 1
            if a0 < 0 or c0 < 0 or a0 >= len(kpts) or c0 >= len(kpts):
                continue
            pa = kpts[a0]
            pc = kpts[c0]
            if not (_vis(pa) and _vis(pc)):
                continue
            draw.line([float(pa[0]), float(pa[1]), float(pc[0]), float(pc[1])], fill=(255, 0, 0), width=3)

        # points
        for p in kpts:
            if not _vis(p):
                continue
            x, y = float(p[0]), float(p[1])
            r = 3
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 0), outline=(0, 0, 0))

        persons.append(
            {
                "score": sc,
                "bbox_xyxy": [x0, y0, x1, y1],
                "keypoints": kpts,
            }
        )

    overlay.save(overlay_out)

    payload = {
        "kind": "keypoints_demo",
        "schema_version": 1,
        "settings": {
            "image": str(image_path),
            "run_dir": str(run_dir),
            "device": str(torch_device),
            "score_threshold": float(score_threshold),
            "max_persons": int(max_persons),
        },
        "meta": {
            "backend": "torchvision.keypointrcnn_resnet50_fpn",
            "weights": weights_name,
            "torch": getattr(torch, "__version__", None),
            "torchvision": getattr(torchvision, "__version__", None),
            "keypoints": {
                "names": list(_COCO_KEYPOINT_NAMES),
                "skeleton": [[int(a), int(b)] for (a, b) in _COCO_SKELETON],
            },
        },
        "result": {
            "num_persons": int(len(persons)),
            "persons": persons,
            "artifacts": {
                "image": str(orig_out),
                "overlay": str(overlay_out),
            },
        },
    }

    out_path = Path(run_dir) / str(output_name)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path

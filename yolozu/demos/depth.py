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
        from PIL import Image

        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("demo depth requires torch, numpy, and Pillow") from exc
    return np, Image, torch


def _pick_default_image() -> Path | None:
    candidates = [
        Path("data") / "smoke" / "images" / "val" / "000000000036.jpg",
        Path("data") / "smoke" / "images" / "val" / "000000000049.jpg",
        Path("data") / "smoke" / "images" / "val" / "000000000061.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def run_depth_demo(
    *,
    image: str | Path | None = None,
    run_dir: str | Path | None = None,
    device: str = "auto",
    model: str = "midas_small",
    invert: bool = True,
    output_name: str = "depth_demo_report.json",
) -> Path:
    """Monocular depth demo via MiDaS (torch.hub).

    Notes:
    - This demo downloads weights on first run (requires network).
    - Output depth is relative (not metric) and normalized for visualization.
    """

    np, Image, torch = _require_deps()

    if image is None:
        picked = _pick_default_image()
        if picked is None:
            raise FileNotFoundError("no default demo image found under data/smoke; pass --image <path>")
        image = picked

    image_path = Path(image)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    if run_dir is None:
        run_dir = Path("demo_output") / "depth" / _utc_run_id()
    else:
        run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    orig_out = run_dir / "image.png"
    depth_gray_out = run_dir / "depth_gray.png"
    overlay_out = run_dir / "depth_overlay.png"

    img = Image.open(image_path).convert("RGB")
    img.save(orig_out)

    if str(device).strip().lower() == "auto":
        if torch.cuda.is_available():
            torch_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch_device = torch.device("mps")
        else:
            torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(str(device))

    model_key = str(model).strip().lower()
    if model_key in ("midas_small", "small"):
        hub_model = "MiDaS_small"
        input_size = 256
    elif model_key in ("dpt_hybrid", "hybrid"):
        hub_model = "DPT_Hybrid"
        input_size = 384
    elif model_key in ("dpt_large", "large"):
        hub_model = "DPT_Large"
        input_size = 384
    else:
        raise ValueError("unknown depth model (expected: midas_small|dpt_hybrid|dpt_large)")

    def _preprocess_pil(im: Any, *, size: int) -> Any:
        # MiDaS expects a normalized RGB tensor. We intentionally keep this
        # dependency-light (no cv2) so the demo can run with yolozu[demo].
        im2 = im.resize((int(size), int(size)), resample=Image.BICUBIC).convert("RGB")
        arr = np.asarray(im2).astype("float32") / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype).view(3, 1, 1)
        t = (t - mean) / std
        return t.unsqueeze(0)

    # torch.hub is intentionally used to avoid extra pip deps beyond yolozu[demo].
    try:
        midas = torch.hub.load("intel-isl/MiDaS", hub_model)
    except ModuleNotFoundError as exc:
        # Recent MiDaS revisions import timm even for the smaller variants.
        if "timm" in str(exc):
            raise RuntimeError(
                "demo depth requires timm (MiDaS dependency). Install: python3 -m pip install -U 'yolozu[demo]'"
            ) from exc
        raise
    midas.to(torch_device)
    midas.eval()

    input_batch = _preprocess_pil(img, size=int(input_size)).to(torch_device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

    depth = prediction.squeeze().detach().cpu().float().numpy()

    dmin = float(depth.min())
    dmax = float(depth.max())
    denom = max(1e-12, dmax - dmin)
    norm = (depth - dmin) / denom
    if bool(invert):
        norm = 1.0 - norm
    depth_u8 = (norm * 255.0).clip(0.0, 255.0).astype("uint8")

    depth_img = Image.fromarray(depth_u8, mode="L")
    depth_img.save(depth_gray_out)

    # Simple overlay: grayscale depth blended on the original.
    depth_rgb = depth_img.convert("RGB")
    overlay = Image.blend(img, depth_rgb, alpha=0.55)
    overlay.save(overlay_out)

    payload = {
        "kind": "depth_demo",
        "schema_version": 1,
        "settings": {
            "image": str(image_path),
            "run_dir": str(run_dir),
            "device": str(torch_device),
            "model": str(model_key),
            "invert": bool(invert),
        },
        "meta": {
            "backend": "torch.hub.intel-isl/MiDaS",
            "hub_model": hub_model,
            "preprocess": {
                "type": "pil_resize_square_imagenet_norm",
                "size": int(input_size),
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "torch": getattr(torch, "__version__", None),
        },
        "result": {
            "depth": {
                "min": dmin,
                "max": dmax,
            },
            "artifacts": {
                "image": str(orig_out),
                "depth_gray": str(depth_gray_out),
                "overlay": str(overlay_out),
            },
        },
    }

    out_path = Path(run_dir) / str(output_name)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path

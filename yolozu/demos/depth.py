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


def _require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "demo depth (depth_anything) requires transformers. "
            "Install: python3 -m pip install -U 'yolozu[demo]'"
        ) from exc
    return AutoImageProcessor, AutoModelForDepthEstimation


def _resolve_torch_device(torch: Any, device: str) -> Any:
    if str(device).strip().lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(str(device))


def _normalize_depth_to_u8(depth: Any, *, invert: bool) -> tuple[float, float, Any]:
    dmin = float(depth.min())
    dmax = float(depth.max())
    denom = max(1e-12, dmax - dmin)
    norm = (depth - dmin) / denom
    if bool(invert):
        norm = 1.0 - norm
    depth_u8 = (norm * 255.0).clip(0.0, 255.0).astype("uint8")
    return dmin, dmax, depth_u8


def _sanitize_tag(tag: str) -> str:
    out = []
    for ch in str(tag):
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _run_midas_depth(
    *,
    np: Any,
    Image: Any,
    torch: Any,
    img: Any,
    torch_device: Any,
    model_key: str,
) -> tuple[Any, dict[str, Any]]:
    model_key = str(model_key).strip().lower()
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
        raise ValueError("unknown MiDaS/DPT model (expected: midas_small|dpt_hybrid|dpt_large)")

    def _preprocess_pil(im: Any, *, size: int) -> Any:
        im2 = im.resize((int(size), int(size)), resample=Image.BICUBIC).convert("RGB")
        arr = np.asarray(im2).astype("float32") / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype).view(3, 1, 1)
        t = (t - mean) / std
        return t.unsqueeze(0)

    try:
        midas = torch.hub.load("intel-isl/MiDaS", hub_model)
    except ModuleNotFoundError as exc:
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

    meta = {
        "backend": "torch.hub.intel-isl/MiDaS",
        "hub_model": hub_model,
        "preprocess": {
            "type": "pil_resize_square_imagenet_norm",
            "size": int(input_size),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "torch": getattr(torch, "__version__", None),
    }
    return depth, meta


def _run_depth_anything(
    *,
    np: Any,
    torch: Any,
    img: Any,
    torch_device: Any,
    model_id: str,
) -> tuple[Any, dict[str, Any]]:
    AutoImageProcessor, AutoModelForDepthEstimation = _require_transformers()

    processor = AutoImageProcessor.from_pretrained(str(model_id))
    model = AutoModelForDepthEstimation.from_pretrained(str(model_id))
    model.to(torch_device)
    model.eval()

    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(torch_device) for (k, v) in dict(inputs).items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred = getattr(outputs, "predicted_depth", None)
        if pred is None:
            raise RuntimeError("unexpected transformers depth output (missing predicted_depth)")
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
    depth = pred.squeeze().detach().cpu().float().numpy()

    meta = {
        "backend": "transformers.depth_estimation",
        "model_id": str(model_id),
        "torch": getattr(torch, "__version__", None),
        "transformers": None,
    }
    try:
        import transformers as _tf

        meta["transformers"] = getattr(_tf, "__version__", None)
    except Exception:
        pass
    return depth, meta


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
    model: str = "depth_anything",
    invert: bool = True,
    compare: bool = False,
    output_name: str = "depth_demo_report.json",
) -> Path:
    """Monocular depth demo.

    Notes:
    - This demo downloads weights on first run (requires network).
    - Output depth is relative (not metric) and normalized for visualization.
    - If compare=True, runs (midas_small, dpt_hybrid, depth_anything) and writes per-model artifacts with suffixes.
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

    torch_device = _resolve_torch_device(torch, str(device))

    depth_anything_id = "depth-anything/Depth-Anything-V2-Small-hf"
    requested = str(model).strip().lower()
    if requested in ("depth_anything", "depth-anything", "depth_anything_v2_small", "anything"):
        requested = "depth_anything"

    if bool(compare):
        model_keys = ["midas_small", "dpt_hybrid", "depth_anything"]
    else:
        model_keys = [requested]

    results: list[dict[str, Any]] = []
    for mk in model_keys:
        mk_norm = str(mk).strip().lower()
        tag = _sanitize_tag(mk_norm)

        if mk_norm in ("midas_small", "small", "dpt_hybrid", "hybrid", "dpt_large", "large"):
            depth, meta = _run_midas_depth(np=np, Image=Image, torch=torch, img=img, torch_device=torch_device, model_key=mk_norm)
        elif mk_norm == "depth_anything":
            depth, meta = _run_depth_anything(np=np, torch=torch, img=img, torch_device=torch_device, model_id=depth_anything_id)
        else:
            raise ValueError(
                "unknown depth model (expected: depth_anything|midas_small|dpt_hybrid|dpt_large)"
            )

        dmin, dmax, depth_u8 = _normalize_depth_to_u8(depth, invert=bool(invert))
        depth_img = Image.fromarray(depth_u8, mode="L")

        if bool(compare):
            depth_gray_path = run_dir / f"depth_gray_{tag}.png"
            overlay_path = run_dir / f"depth_overlay_{tag}.png"
        else:
            depth_gray_path = depth_gray_out
            overlay_path = overlay_out

        depth_img.save(depth_gray_path)
        depth_rgb = depth_img.convert("RGB")
        overlay = Image.blend(img, depth_rgb, alpha=0.55)
        overlay.save(overlay_path)

        results.append(
            {
                "model": str(mk_norm),
                "depth": {"min": float(dmin), "max": float(dmax)},
                "artifacts": {
                    "depth_gray": str(depth_gray_path),
                    "overlay": str(overlay_path),
                },
                "meta": meta,
            }
        )

    payload = {
        "kind": "depth_demo",
        "schema_version": 1,
        "settings": {
            "image": str(image_path),
            "run_dir": str(run_dir),
            "device": str(torch_device),
            "model": str(requested),
            "compare": bool(compare),
            "invert": bool(invert),
        },
        "result": {
            "models": results,
            "artifacts": {
                "image": str(orig_out),
            },
        },
    }

    # Back-compat conveniences for single-model runs.
    if len(results) == 1:
        payload["meta"] = dict(results[0].get("meta") or {})
        payload["result"]["depth"] = dict(results[0].get("depth") or {})
        payload["result"]["artifacts"].update(dict(results[0].get("artifacts") or {}))

    out_path = Path(run_dir) / str(output_name)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path

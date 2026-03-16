"""Training utility functions for train_minimal."""

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

workspace_root = Path.cwd()

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from yolozu.run_record import build_run_record

logger = logging.getLogger(__name__)


def _debug_swallow(context: str, exc: Exception) -> None:
    logger.debug("%s: %s", context, exc, exc_info=True)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_keypoint_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _derive_keypoint_flip_pairs(keypoint_names: list[str]) -> list[list[int]]:
    if not keypoint_names:
        return []

    index_by_name = {str(name).strip().lower(): idx for idx, name in enumerate(keypoint_names)}
    pairs: list[list[int]] = []
    seen: set[tuple[int, int]] = set()

    for idx, name in enumerate(keypoint_names):
        lower = str(name).strip().lower()
        if not lower:
            continue

        partner_name = None
        if "left" in lower:
            partner_name = lower.replace("left", "right")
        elif "right" in lower:
            partner_name = lower.replace("right", "left")
        elif lower.startswith("l_"):
            partner_name = "r_" + lower[2:]
        elif lower.startswith("r_"):
            partner_name = "l_" + lower[2:]
        elif lower.endswith("_l"):
            partner_name = lower[:-2] + "_r"
        elif lower.endswith("_r"):
            partner_name = lower[:-2] + "_l"

        if not partner_name:
            continue
        j = index_by_name.get(partner_name)
        if j is None or j == idx:
            continue
        key = (idx, j) if idx <= j else (j, idx)
        if key in seen:
            continue
        seen.add(key)
        pairs.append([int(key[0]), int(key[1])])

    return pairs


def _extract_manifest_keypoints_meta(manifest: dict[str, Any] | None) -> tuple[list[str], list[list[int]]]:
    if not isinstance(manifest, dict):
        return [], []
    meta = manifest.get("keypoints_meta")
    if not isinstance(meta, dict):
        return [], []

    names = _normalize_keypoint_names(meta.get("keypoint_names") or [])
    skeleton: list[list[int]] = []
    raw_skeleton = meta.get("skeleton") or []
    if isinstance(raw_skeleton, list) and names:
        seen: set[tuple[int, int]] = set()
        for edge in raw_skeleton:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            try:
                a = int(edge[0])
                b = int(edge[1])
            except Exception:
                continue
            if a <= 0 or b <= 0 or a == b:
                continue
            if a > len(names) or b > len(names):
                continue
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            skeleton.append([a, b])
    return names, skeleton


def unwrap_model(model: "torch.nn.Module") -> "torch.nn.Module":
    # Unwrap common wrappers (DDP, torch.compile OptimizedModule, etc.).
    while True:
        if hasattr(model, "module"):
            try:
                model = model.module
                continue
            except Exception as exc:
                _debug_swallow("model.module unwrap skipped", exc)
        if hasattr(model, "_orig_mod"):
            try:
                model = model._orig_mod  # type: ignore[attr-defined]
                continue
            except Exception as exc:
                _debug_swallow("model._orig_mod unwrap skipped", exc)
        return model


def _quantiles(values: "Any", qs: tuple[int, ...] = (50, 90, 95, 99)) -> dict[str, float]:
    import numpy as np  # type: ignore

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    out: dict[str, float] = {}
    if flat.size == 0:
        for q in qs:
            out[f"p{int(q)}"] = 0.0
        return out
    for q in qs:
        out[f"p{int(q)}"] = float(np.quantile(flat, float(q) / 100.0))
    return out


def _diff_stats(a: "Any", b: "Any") -> dict[str, Any]:
    import numpy as np  # type: ignore

    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {"ok": False, "reason": "shape_mismatch", "a_shape": list(a.shape), "b_shape": list(b.shape)}
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    finite = np.isfinite(diff)
    if not bool(finite.all()):
        return {
            "ok": False,
            "reason": "non_finite_diff",
            "shape": list(diff.shape),
            "non_finite": int((~finite).sum()),
        }
    out: dict[str, Any] = {
        "ok": True,
        "shape": list(diff.shape),
        "max": float(diff.max()) if diff.size else 0.0,
        "mean": float(diff.mean()) if diff.size else 0.0,
    }
    out.update(_quantiles(diff))
    return out


def _softmax(x: "Any", axis: int = -1) -> "Any":
    import numpy as np  # type: ignore

    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    denom = ex.sum(axis=axis, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return (ex / denom).astype(np.float32)


def _sigmoid(x: "Any") -> "Any":
    import numpy as np  # type: ignore

    x = np.asarray(x, dtype=np.float64)
    y = 1.0 / (1.0 + np.exp(-x))
    return y.astype(np.float32)


def _derive_score_bbox(outputs: dict[str, "Any"]) -> tuple["Any", "Any"]:
    logits = outputs.get("logits")
    bbox = outputs.get("bbox")
    if logits is None or bbox is None:
        raise ValueError("outputs must contain logits and bbox")
    probs = _softmax(logits, axis=-1)
    score = probs.max(axis=-1)
    bbox_sig = _sigmoid(bbox)
    return score.astype("float32"), bbox_sig.astype("float32")


def run_onnxrt_parity(
    *,
    model: "torch.nn.Module",
    onnx_path: Path,
    image_size: int,
    seed: int,
    score_atol: float,
    bbox_atol: float,
    out_path: Path,
    policy: str,
    run_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "timestamp_utc": _now_utc(),
        "onnx": str(onnx_path),
        "thresholds": {"score_atol": float(score_atol), "bbox_atol": float(bbox_atol)},
        "policy": str(policy),
        "passed": False,
        "available": False,
        "reason": None,
        "run_record": run_record,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        report["reason"] = f"missing_numpy:{exc}"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        if str(policy) == "fail":
            raise SystemExit(f"ONNX parity unavailable (numpy missing). See: {out_path}")
        print(f"WARNING: ONNX parity unavailable (numpy missing). See: {out_path}", file=sys.stderr)
        return report

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        report["reason"] = f"missing_onnxruntime:{exc}"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        if str(policy) == "fail":
            raise SystemExit(f"ONNX parity unavailable (onnxruntime missing). See: {out_path}")
        print(f"WARNING: ONNX parity unavailable (onnxruntime missing). See: {out_path}", file=sys.stderr)
        return report

    if not onnx_path.exists():
        report["reason"] = "onnx_not_found"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        if str(policy) == "fail":
            raise SystemExit(f"ONNX parity unavailable (onnx not found). See: {out_path}")
        print(f"WARNING: ONNX parity unavailable (onnx not found). See: {out_path}", file=sys.stderr)
        return report

    report["available"] = True
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        report["available"] = False
        report["reason"] = f"onnxruntime_init_failed:{exc}"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        if str(policy) == "fail":
            raise SystemExit(f"ONNX parity unavailable (onnxruntime init failed). See: {out_path}") from exc
        print(f"WARNING: ONNX parity unavailable (onnxruntime init failed). See: {out_path}", file=sys.stderr)
        return report

    input_name = None
    try:
        if sess.get_inputs():
            input_name = sess.get_inputs()[0].name
    except Exception:
        input_name = None
    if not input_name:
        input_name = "images"

    gen = torch.Generator(device="cpu")
    try:
        gen.manual_seed(int(seed))
    except Exception as exc:
        _debug_swallow("generator seed setup skipped", exc)
    x = torch.rand((1, 3, int(image_size), int(image_size)), generator=gen, dtype=torch.float32, device="cpu")
    report["input"] = {"shape": [1, 3, int(image_size), int(image_size)], "dtype": "float32", "seed": int(seed)}

    ref = model.eval().cpu()
    with torch.no_grad():
        out_torch = ref(x)
    if not isinstance(out_torch, dict):
        raise SystemExit("unexpected torch output type for parity (expected dict).")

    torch_outputs: dict[str, Any] = {}
    for key in ("logits", "bbox"):
        value = out_torch.get(key)
        if value is None:
            continue
        if hasattr(value, "detach"):
            torch_outputs[key] = value.detach().cpu().numpy()

    ort_outputs = sess.run(None, {str(input_name): np.asarray(x.numpy(), dtype=np.float32)})
    names = []
    try:
        names = [o.name for o in sess.get_outputs()]
    except Exception:
        names = []
    if not names:
        names = ["logits", "bbox"]
    cand_outputs: dict[str, Any] = {str(name): val for name, val in zip(names, ort_outputs)}

    score_t, bbox_t = _derive_score_bbox(torch_outputs)
    score_o, bbox_o = _derive_score_bbox(cand_outputs)
    score_stats = _diff_stats(score_t, score_o)
    bbox_stats = _diff_stats(bbox_t, bbox_o)

    score_max = float(score_stats.get("max", float("inf"))) if bool(score_stats.get("ok")) else float("inf")
    bbox_max = float(bbox_stats.get("max", float("inf"))) if bool(bbox_stats.get("ok")) else float("inf")
    passed = bool(
        bool(score_stats.get("ok"))
        and bool(bbox_stats.get("ok"))
        and score_max <= float(score_atol)
        and bbox_max <= float(bbox_atol)
    )

    report["derived"] = {
        "score": score_stats,
        "bbox_sigmoid": bbox_stats,
        "score_max": score_max,
        "bbox_max": bbox_max,
    }
    report["onnxrt"] = {"providers": list(sess.get_providers()), "input_name": str(input_name)}
    report["passed"] = passed

    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if not passed:
        msg = (
            f"ONNX parity failed: score_max={score_max:.6g} bbox_max={bbox_max:.6g} "
            f"(score_atol={float(score_atol):.6g}, bbox_atol={float(bbox_atol):.6g}). See: {out_path}"
        )
        if str(policy) == "fail":
            raise SystemExit(msg)
        print(f"WARNING: {msg}", file=sys.stderr)

    return report


def collect_torch_cuda_meta() -> dict[str, Any]:
    if torch is None:
        return {"available": False}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch.cuda.is_available() is false"}
    try:
        idx = int(torch.cuda.current_device())
    except Exception:
        idx = 0
    meta: dict[str, Any] = {
        "available": True,
        "device_index": idx,
        "device_name": torch.cuda.get_device_name(idx),
        "device_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(idx)),
        "total_memory_mb": int(torch.cuda.get_device_properties(idx).total_memory // (1024 * 1024)),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": (torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None),
    }
    return meta


def collect_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np  # type: ignore

        state["numpy"] = np.random.get_state()
    except Exception:
        state["numpy"] = None
    if torch is not None:
        try:
            state["torch"] = torch.get_rng_state()
        except Exception:
            state["torch"] = None
        if torch.cuda.is_available():
            try:
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
            except Exception:
                state["torch_cuda"] = None
        else:
            state["torch_cuda"] = None
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    py_state = state.get("python")
    if py_state is not None:
        try:
            random.setstate(py_state)
        except Exception as exc:
            _debug_swallow("python RNG restore skipped", exc)
    np_state = state.get("numpy")
    if np_state is not None:
        try:
            import numpy as np  # type: ignore

            np.random.set_state(np_state)
        except Exception as exc:
            _debug_swallow("numpy RNG restore skipped", exc)
    if torch is None:
        return
    torch_state = state.get("torch")
    if torch_state is not None:
        try:
            torch.set_rng_state(torch_state)
        except Exception as exc:
            _debug_swallow("torch RNG restore skipped", exc)
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception as exc:
            _debug_swallow("cuda RNG restore skipped", exc)


def _rotation_matrix_from_rpy(roll_rad: float, pitch_rad: float, yaw_rad: float) -> "torch.Tensor":
    cr = math.cos(roll_rad)
    sr = math.sin(roll_rad)
    cp = math.cos(pitch_rad)
    sp = math.sin(pitch_rad)
    cy = math.cos(yaw_rad)
    sy = math.sin(yaw_rad)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=torch.float32)
    ry = torch.tensor([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=torch.float32)
    rz = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return rz @ ry @ rx


def compute_warmup_lr(base_lr: float, step: int, warmup_steps: int, warmup_init: float) -> float:
    if warmup_steps <= 0:
        return float(base_lr)
    if step <= 0:
        return float(warmup_init)
    if step >= warmup_steps:
        return float(base_lr)
    alpha = float(step) / float(warmup_steps)
    return float(warmup_init + (base_lr - warmup_init) * alpha)


def parse_milestones(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except Exception:
                continue
        return out
    text = str(value).strip()
    if not text:
        return []
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return out


def apply_denoise_targets(
    targets: list[dict[str, Any]],
    *,
    num_classes: int,
    denoise_count: int,
    bbox_noise: float = 0.0,
    label_noise: float = 0.0,
) -> list[dict[str, Any]]:
    """Append noisy copies of GT targets for denoising-style training.

    This is a lightweight utility used by unit tests and optional training
    extensions. It duplicates `gt_*` tensors (labels/bbox/z/...) `denoise_count`
    times and appends them to the originals.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for apply_denoise_targets")

    copies = max(0, int(denoise_count))
    if copies <= 0:
        return targets

    out: list[dict[str, Any]] = []
    for tgt in targets:
        if not isinstance(tgt, dict):
            out.append(tgt)
            continue

        updated = dict(tgt)

        gt_labels = tgt.get("gt_labels")
        gt_bbox = tgt.get("gt_bbox")
        gt_z = tgt.get("gt_z")

        if not isinstance(gt_labels, torch.Tensor) or gt_labels.numel() == 0:
            out.append(updated)
            continue

        label_list = [gt_labels]
        bbox_list = [gt_bbox] if isinstance(gt_bbox, torch.Tensor) else None
        z_list = [gt_z] if isinstance(gt_z, torch.Tensor) else None

        for _ in range(copies):
            noisy_labels = gt_labels.clone()
            if float(label_noise) > 0.0 and int(num_classes) > 0:
                mask = torch.rand_like(noisy_labels.to(dtype=torch.float32)) < float(label_noise)
                if bool(mask.any()):
                    noisy = torch.randint(0, int(num_classes), (int(mask.sum().item()),), device=noisy_labels.device)
                    noisy_labels = noisy_labels.clone()
                    noisy_labels[mask] = noisy

            label_list.append(noisy_labels)

            if bbox_list is not None and isinstance(gt_bbox, torch.Tensor):
                noisy_bbox = gt_bbox.clone()
                if float(bbox_noise) > 0.0:
                    noisy_bbox = noisy_bbox + torch.randn_like(noisy_bbox) * float(bbox_noise)
                    noisy_bbox = noisy_bbox.clamp(0.0, 1.0)
                bbox_list.append(noisy_bbox)

            if z_list is not None and isinstance(gt_z, torch.Tensor):
                z_list.append(gt_z.clone())

        updated["gt_labels"] = torch.cat(label_list, dim=0)
        if bbox_list is not None:
            updated["gt_bbox"] = torch.cat(bbox_list, dim=0)
        if z_list is not None:
            updated["gt_z"] = torch.cat(z_list, dim=0)

        out.append(updated)

    return out


def flatten_records_for_map(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert rtdetr_pose manifest records into YOLOZU/simple_map GT records."""

    flat: list[dict[str, Any]] = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        image = rec.get("image_path") or rec.get("image") or ""
        image = str(image) if image is not None else ""
        labels_out: list[dict[str, Any]] = []
        for inst in rec.get("labels", []) or []:
            if not isinstance(inst, dict):
                continue
            bb = inst.get("bbox") or {}
            try:
                labels_out.append(
                    {
                        "class_id": int(inst.get("class_id", 0)),
                        "cx": float(bb.get("cx", 0.0)),
                        "cy": float(bb.get("cy", 0.0)),
                        "w": float(bb.get("w", 0.0)),
                        "h": float(bb.get("h", 0.0)),
                    }
                )
            except Exception:
                continue
        flat.append({"image": image, "labels": labels_out})
    return flat


def decode_detections_from_outputs(
    outputs: dict[str, Any],
    image_paths: list[str],
    *,
    score_thresh: float,
    topk: int,
) -> list[dict[str, Any]]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for decode_detections_from_outputs")

    logits = outputs.get("logits")
    bbox = outputs.get("bbox")
    if not isinstance(logits, torch.Tensor) or not isinstance(bbox, torch.Tensor):
        return [{"image": str(p), "detections": []} for p in image_paths]

    probs = logits.softmax(dim=-1)
    scores, class_ids = probs.max(dim=-1)  # (B,Q)
    bg_idx = int(logits.shape[-1]) - 1
    bbox_norm = bbox.sigmoid().clamp(0.0, 1.0)

    batch = int(scores.shape[0])
    out: list[dict[str, Any]] = []
    for i in range(batch):
        image = str(image_paths[i]) if i < len(image_paths) else ""
        sc = scores[i]
        cls = class_ids[i]
        bb = bbox_norm[i]
        keep = (cls != bg_idx) & (sc >= float(score_thresh))
        dets: list[dict[str, Any]] = []
        if bool(keep.any()):
            sc_k = sc[keep]
            cls_k = cls[keep]
            bb_k = bb[keep]

            if int(sc_k.numel()) > int(topk):
                sc_k, idx = torch.topk(sc_k, k=int(topk))
                cls_k = cls_k[idx]
                bb_k = bb_k[idx]
            else:
                order = torch.argsort(sc_k, descending=True)
                sc_k = sc_k[order]
                cls_k = cls_k[order]
                bb_k = bb_k[order]

            for j in range(int(sc_k.shape[0])):
                cx, cy, w, h = [float(v) for v in bb_k[j].tolist()]
                dets.append(
                    {
                        "class_id": int(cls_k[j].item()),
                        "score": float(sc_k[j].item()),
                        "bbox": {"cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h)},
                    }
                )
        out.append({"image": image, "detections": dets})
    return out


def plan_accumulation_windows(*, max_micro_steps: int, grad_accum: int) -> list[int]:
    """Return accumulation window sizes for micro-step training loops.

    Example: max_micro_steps=5, grad_accum=2 -> [2, 2, 1]
    """

    steps_total = int(max_micro_steps)
    if steps_total <= 0:
        return []
    accum = max(1, int(grad_accum))

    windows: list[int] = []
    step = 0
    while step < steps_total:
        window = min(accum, steps_total - step)
        windows.append(int(window))
        step += int(window)
    return windows


def compute_linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(end)
    alpha = min(max(float(step) / float(total_steps - 1), 0.0), 1.0)
    return float(start + (end - start) * alpha)


def compute_mim_schedule(
    *,
    step: int,
    total_steps: int,
    mask_start: float,
    mask_end: float,
    weight_start: float,
    weight_end: float,
    default_mask: float,
    default_weight: float,
) -> tuple[float, float]:
    """Linear schedule for MIM masking ratio and loss weight."""

    steps = int(total_steps)
    if steps <= 0:
        return float(default_mask), float(default_weight)
    s = int(step)
    s = max(0, min(s, steps - 1))
    alpha = 0.0 if steps <= 1 else float(s) / float(steps - 1)
    mask = float(mask_start + (mask_end - mask_start) * alpha)
    weight = float(weight_start + (weight_end - weight_start) * alpha)
    return mask, weight


def compute_stage_weights(
    base: dict[str, float],
    *,
    global_step: int,
    stage_off_steps: int = 0,
    stage_k_steps: int = 0,
) -> tuple[dict[str, float], str]:
    """Return per-step loss weights for simple staged training.

    Stages (by optimizer step):
    - offsets: [0, stage_off_steps)
    - k: [stage_off_steps, stage_off_steps + stage_k_steps)
    - full: afterwards
    """

    out = {str(k): float(v) for k, v in (base or {}).items()}
    step = int(global_step)
    off_n = max(0, int(stage_off_steps))
    k_n = max(0, int(stage_k_steps))

    stage = "full"
    if off_n > 0 and step < off_n:
        stage = "offsets"
        out["k"] = 0.0
    elif k_n > 0 and step < (off_n + k_n):
        stage = "k"
        out["off"] = 0.0
    return out, stage


def compute_stage_costs(
    base: dict[str, float],
    *,
    global_step: int,
    cost_z_start_step: int = 0,
    cost_rot_start_step: int = 0,
    cost_t_start_step: int = 0,
) -> dict[str, float]:
    """Return per-step matcher costs for staged matching."""

    out = {str(k): float(v) for k, v in (base or {}).items()}
    step = int(global_step)
    if step < int(cost_z_start_step):
        out["cost_z"] = 0.0
    if step < int(cost_rot_start_step):
        out["cost_rot"] = 0.0
    if step < int(cost_t_start_step):
        out["cost_t"] = 0.0
    return out


def generate_block_mask(
    height: int,
    width: int,
    *,
    patch_size: int,
    mask_prob: float,
    generator: "torch.Generator",
) -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for generate_block_mask")

    h = max(1, int(height))
    w = max(1, int(width))
    ps = max(1, int(patch_size))
    prob = float(mask_prob)

    grid_h = max(1, (h + ps - 1) // ps)
    grid_w = max(1, (w + ps - 1) // ps)
    mask_grid = torch.rand((grid_h, grid_w), generator=generator) < prob
    mask = mask_grid.repeat_interleave(ps, dim=0).repeat_interleave(ps, dim=1)
    return mask[:h, :w]


def _rgb_to_hsv(image: "torch.Tensor") -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for _rgb_to_hsv")
    if image.ndim != 3 or int(image.shape[0]) != 3:
        raise ValueError("_rgb_to_hsv expects an image tensor shaped [3,H,W]")

    eps = 1e-6
    r, g, b = image[0], image[1], image[2]
    maxc, argmax = torch.max(image, dim=0)
    minc = torch.min(image, dim=0).values
    v = maxc
    delta = maxc - minc
    s = delta / (maxc + eps)

    h = torch.zeros_like(maxc)
    mask = delta > eps
    delta_safe = torch.where(mask, delta, torch.ones_like(delta))

    rc = (g - b) / delta_safe
    gc = (b - r) / delta_safe + 2.0
    bc = (r - g) / delta_safe + 4.0

    h = torch.where(mask & (argmax == 0), rc, h)
    h = torch.where(mask & (argmax == 1), gc, h)
    h = torch.where(mask & (argmax == 2), bc, h)
    h = torch.remainder(h / 6.0, 1.0)
    return torch.stack([h, s, v], dim=0)


def _hsv_to_rgb(hsv: "torch.Tensor") -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for _hsv_to_rgb")
    if hsv.ndim != 3 or int(hsv.shape[0]) != 3:
        raise ValueError("_hsv_to_rgb expects an HSV tensor shaped [3,H,W]")

    h, s, v = hsv[0], hsv[1], hsv[2]
    h = torch.remainder(h, 1.0)
    s = torch.clamp(s, 0.0, 1.0)
    v = torch.clamp(v, 0.0, 1.0)

    h6 = h * 6.0
    i = torch.floor(h6).to(dtype=torch.int64) % 6
    f = h6 - torch.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = torch.empty_like(v)
    g = torch.empty_like(v)
    b = torch.empty_like(v)

    m0 = i == 0
    m1 = i == 1
    m2 = i == 2
    m3 = i == 3
    m4 = i == 4
    m5 = i == 5

    r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
    r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
    r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
    r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
    r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
    r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]

    return torch.stack([r, g, b], dim=0)


def apply_hsv_jitter(
    image: "torch.Tensor",
    *,
    generator: "torch.Generator",
    hgain: float,
    sgain: float,
    vgain: float,
) -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for apply_hsv_jitter")
    hsv = _rgb_to_hsv(image)
    h, s, v = hsv[0], hsv[1], hsv[2]

    if float(hgain) > 0:
        dh = (torch.rand((), generator=generator) * 2.0 - 1.0) * float(hgain)
        h = torch.remainder(h + dh, 1.0)
    if float(sgain) > 0:
        gs = (torch.rand((), generator=generator) * 2.0 - 1.0) * float(sgain)
        s = torch.clamp(s * (1.0 + gs), 0.0, 1.0)
    if float(vgain) > 0:
        gv = (torch.rand((), generator=generator) * 2.0 - 1.0) * float(vgain)
        v = torch.clamp(v * (1.0 + gv), 0.0, 1.0)

    return _hsv_to_rgb(torch.stack([h, s, v], dim=0))


def apply_grayscale(image: "torch.Tensor") -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for apply_grayscale")
    if image.ndim != 3 or int(image.shape[0]) != 3:
        raise ValueError("apply_grayscale expects an image tensor shaped [3,H,W]")

    gray = image[0] * 0.2989 + image[1] * 0.5870 + image[2] * 0.1140
    return gray.unsqueeze(0).expand_as(image)


def _gaussian_kernel2d(
    *,
    kernel_size: int,
    sigma: float,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for _gaussian_kernel2d")

    k = int(kernel_size)
    if k <= 0 or k % 2 == 0:
        raise ValueError("blur_kernel must be a positive odd integer")

    sig = float(sigma)
    if not math.isfinite(sig) or sig <= 0:
        raise ValueError("blur_sigma must be > 0")

    half = k // 2
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    kernel1d = torch.exp(-(coords * coords) / (2.0 * sig * sig))
    kernel1d = kernel1d / torch.clamp(kernel1d.sum(), min=1e-12)
    kernel2d = kernel1d[:, None] * kernel1d[None, :]
    return kernel2d / torch.clamp(kernel2d.sum(), min=1e-12)


def apply_gaussian_blur(
    image: "torch.Tensor",
    *,
    sigma: float,
    kernel_size: int,
) -> "torch.Tensor":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for apply_gaussian_blur")
    if image.ndim != 3 or int(image.shape[0]) != 3:
        raise ValueError("apply_gaussian_blur expects an image tensor shaped [3,H,W]")

    k = int(kernel_size)
    pad = k // 2
    kernel2d = _gaussian_kernel2d(kernel_size=k, sigma=float(sigma), device=image.device, dtype=image.dtype)
    weight = kernel2d.view(1, 1, k, k).expand(3, 1, k, k)

    x = image.unsqueeze(0)
    x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate")
    out = torch.nn.functional.conv2d(x, weight, bias=None, stride=1, padding=0, groups=3)
    return out.squeeze(0)


def create_geom_input_from_bboxes(
    bboxes_cxcywh_norm: list[list[float]],
    z_list: list[float] | None,
    *,
    height: int,
    width: int,
) -> "torch.Tensor":
    """Create geometry input tensor from bbox rectangles.

    Output channels follow `tools/example_mim_inference.py`:
    - mask (float32 0/1)
    - normalized depth: mask * log(D / z_ref)
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for create_geom_input_from_bboxes")

    h = max(1, int(height))
    w = max(1, int(width))
    mask = torch.zeros((h, w), dtype=torch.float32)
    depth = torch.ones((h, w), dtype=torch.float32)

    for i, bb in enumerate(bboxes_cxcywh_norm or []):
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        cx, cy, bw, bh = [float(v) for v in bb]
        x0 = int(math.floor((cx - bw * 0.5) * w))
        x1 = int(math.ceil((cx + bw * 0.5) * w))
        y0 = int(math.floor((cy - bh * 0.5) * h))
        y1 = int(math.ceil((cy + bh * 0.5) * h))
        x0 = max(0, min(w - 1, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h, y1))
        if x1 <= x0:
            x1 = min(w, x0 + 1)
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        mask[y0:y1, x0:x1] = 1.0

        if z_list is not None and i < len(z_list):
            try:
                z_val = float(z_list[i])
            except Exception:
                z_val = 1.0
            if z_val > 0:
                depth[y0:y1, x0:x1] = torch.minimum(depth[y0:y1, x0:x1], torch.tensor(z_val, dtype=torch.float32))

    eps = 1e-6
    if bool((mask > 0).any()):
        z_ref = depth[mask > 0].median()
    else:
        z_ref = torch.tensor(1.0, dtype=torch.float32)
    depth_norm = mask * (torch.log(depth + eps) - torch.log(z_ref + eps))
    return torch.stack([mask, depth_norm], dim=0)


def compute_grad_norm(parameters) -> "torch.Tensor":
    """Compute global L2 grad norm over parameters (no clipping)."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for compute_grad_norm")
    norms = []
    for p in parameters:
        g = getattr(p, "grad", None)
        if g is None:
            continue
        if getattr(g, "is_sparse", False):
            try:
                g = g.coalesce().values()
            except Exception:
                continue
        try:
            norms.append(g.detach().norm(2))
        except Exception:
            continue
    if not norms:
        return torch.zeros((), dtype=torch.float32)
    return torch.norm(torch.stack(norms), 2)


def load_checkpoint_into(
    model: "torch.nn.Module",
    optim: "torch.optim.Optimizer | None",
    path: str | Path,
    *,
    sched: Any | None = None,
    scaler: Any | None = None,
    ema: Any | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    meta: dict[str, Any] = {"path": str(path)}

    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
        if optim is not None and isinstance(obj.get("optim_state_dict"), dict):
            try:
                optim.load_state_dict(obj["optim_state_dict"])
                meta["optim_loaded"] = True
            except Exception:
                meta["optim_loaded"] = False
        if sched is not None and isinstance(obj.get("sched_state_dict"), dict):
            try:
                sched.load_state_dict(obj["sched_state_dict"])
                meta["sched_loaded"] = True
            except Exception:
                meta["sched_loaded"] = False
        if scaler is not None and isinstance(obj.get("scaler_state_dict"), dict):
            try:
                scaler.load_state_dict(obj["scaler_state_dict"])
                meta["scaler_loaded"] = True
            except Exception:
                meta["scaler_loaded"] = False
        if ema is not None and isinstance(obj.get("ema_state_dict"), dict):
            try:
                ema.load_state_dict(obj["ema_state_dict"])
                meta["ema_loaded"] = True
            except Exception:
                meta["ema_loaded"] = False
        if restore_rng:
            try:
                restore_rng_state(obj.get("rng_state"))
                meta["rng_restored"] = True
            except Exception:
                meta["rng_restored"] = False
        meta.update({k: obj.get(k) for k in ("epoch", "global_step") if k in obj})
        meta["schema_version"] = obj.get("schema_version")
        return meta

    # Assume weights-only state_dict
    if isinstance(obj, dict):
        model.load_state_dict(obj)
        meta["optim_loaded"] = False
        return meta

    raise SystemExit(f"unrecognized checkpoint format: {path}")


def save_checkpoint_bundle(
    path: str | Path,
    *,
    model: "torch.nn.Module",
    optim: "torch.optim.Optimizer | None",
    sched: Any | None = None,
    scaler: Any | None = None,
    ema: Any | None = None,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    last_epoch_steps: int,
    last_epoch_avg: float | None,
    last_loss_dict: dict[str, Any] | None,
    run_record: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "last_epoch_steps": int(last_epoch_steps),
        "last_epoch_avg": float(last_epoch_avg) if last_epoch_avg is not None else None,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "run_record": run_record
        if run_record is not None
        else build_run_record(
            repo_root=workspace_root,
            args=vars(args),
            dataset_root=(getattr(args, "dataset_root", "") or None),
        ),
    }
    if optim is not None:
        payload["optim_state_dict"] = optim.state_dict()
    if sched is not None and hasattr(sched, "state_dict"):
        try:
            payload["sched_state_dict"] = sched.state_dict()
        except Exception:
            payload["sched_state_dict"] = None
    if scaler is not None and hasattr(scaler, "state_dict"):
        try:
            payload["scaler_state_dict"] = scaler.state_dict()
        except Exception:
            payload["scaler_state_dict"] = None
    if ema is not None and hasattr(ema, "state_dict"):
        try:
            payload["ema_state_dict"] = ema.state_dict()
        except Exception:
            payload["ema_state_dict"] = None
    if rng_state is not None:
        payload["rng_state"] = rng_state
    if last_loss_dict is not None:
        payload["last_loss"] = {
            k: float(v.detach().cpu()) for k, v in last_loss_dict.items() if hasattr(v, "detach")
        }
    torch.save(payload, path)

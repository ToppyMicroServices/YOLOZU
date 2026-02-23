from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SYNTHGEN_SKELETONS: dict[str, list[tuple[int, int]]] = {
    "animal_v1": [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6)],
    "mechanical_v1": [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)],
}


def _palette_color(class_id: int) -> tuple[int, int, int]:
    cid = int(class_id)
    return (
        int((37 * cid + 17) % 256),
        int((67 * cid + 29) % 256),
        int((97 * cid + 43) % 256),
    )


def _render_sem_overlay(image: np.ndarray, sem: np.ndarray, *, alpha: float) -> np.ndarray:
    out = image.astype(np.float32).copy()
    unique = np.unique(sem)
    for cls in unique:
        mask = sem == cls
        color = np.asarray(_palette_color(int(cls)), dtype=np.float32).reshape(1, 1, 3)
        out[mask] = out[mask] * (1.0 - float(alpha)) + color * float(alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _instance_contour_mask(inst_map: np.ndarray) -> np.ndarray:
    h, w = inst_map.shape[:2]
    contour = np.zeros((h, w), dtype=bool)
    contour[1:, :] |= inst_map[1:, :] != inst_map[:-1, :]
    contour[:-1, :] |= inst_map[:-1, :] != inst_map[1:, :]
    contour[:, 1:] |= inst_map[:, 1:] != inst_map[:, :-1]
    contour[:, :-1] |= inst_map[:, :-1] != inst_map[:, 1:]
    return contour


def _draw_keypoints(
    draw: ImageDraw.ImageDraw,
    kpts: np.ndarray,
    *,
    schema_id: str,
    point_radius: int = 3,
) -> None:
    if kpts.ndim != 3 or kpts.shape[-1] < 3:
        return
    skeleton = SYNTHGEN_SKELETONS.get(str(schema_id), [])
    for inst_idx in range(int(kpts.shape[0])):
        pts = kpts[inst_idx]
        for kp in pts:
            x, y, vis = float(kp[0]), float(kp[1]), float(kp[2])
            if int(round(vis)) <= 0:
                continue
            draw.ellipse(
                [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                fill=(255, 255, 0),
                outline=(0, 0, 0),
                width=1,
            )
        for a, b in skeleton:
            if a >= pts.shape[0] or b >= pts.shape[0]:
                continue
            va = int(round(float(pts[a, 2])))
            vb = int(round(float(pts[b, 2])))
            if va <= 0 or vb <= 0:
                continue
            draw.line([float(pts[a, 0]), float(pts[a, 1]), float(pts[b, 0]), float(pts[b, 1])], fill=(0, 255, 255), width=2)


def render_synthgen_overlay(
    sample: dict[str, Any],
    *,
    alpha: float = 0.45,
    draw_semantic: bool = True,
    draw_instances: bool = True,
    draw_keypoints: bool = True,
) -> Image.Image:
    """Render a debug overlay for SynthGen sample inspection."""

    image = np.asarray(sample["image"]).astype(np.uint8, copy=False)
    canvas = image
    if draw_semantic and "sem_id" in sample:
        canvas = _render_sem_overlay(canvas, np.asarray(sample["sem_id"]), alpha=float(alpha))

    out = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(out)

    if draw_instances and "inst_id" in sample:
        contour = _instance_contour_mask(np.asarray(sample["inst_id"]))
        ys, xs = np.where(contour)
        for x, y in zip(xs.tolist(), ys.tolist()):
            draw.point((int(x), int(y)), fill=(255, 0, 0))

    if draw_keypoints and "kpts2d" in sample:
        _draw_keypoints(draw, np.asarray(sample["kpts2d"]), schema_id=str(sample.get("schema_id", "")))

    return out

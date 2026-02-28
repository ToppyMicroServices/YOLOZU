"""Torchvision v2 transforms bridge for detection data augmentation.

Provides composable transform pipelines that jointly transform images,
bounding boxes, and keypoints using ``torchvision.transforms.v2`` (stable
since torchvision 0.17).

Usage::

    from yolozu.training.transforms_bridge import (
        build_detection_transforms,
        build_eval_transforms,
        transforms_v2_available,
    )

    if transforms_v2_available():
        train_tfm = build_detection_transforms(size=(640, 640))
        eval_tfm = build_eval_transforms(size=(640, 640))
"""

from __future__ import annotations

__all__ = [
    "transforms_v2_available",
    "build_detection_transforms",
    "build_eval_transforms",
]


def transforms_v2_available() -> bool:
    """Return ``True`` if ``torchvision.transforms.v2`` is importable."""
    try:
        from torchvision.transforms import v2 as _v2  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def build_detection_transforms(
    *,
    size: tuple[int, int] = (640, 640),
    hflip_prob: float = 0.5,
    color_jitter: bool = True,
    scale_range: tuple[float, float] = (0.8, 1.2),
    normalize_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    normalize_std: tuple[float, ...] = (0.229, 0.224, 0.225),
):
    """Build a training-time augmentation pipeline for detection.

    Parameters
    ----------
    size : tuple[int, int]
        Target (height, width) after resize.
    hflip_prob : float
        Probability of horizontal flip.
    color_jitter : bool
        Whether to apply photometric distortion.
    scale_range : tuple[float, float]
        Min/max scale factor for random resize.
    normalize_mean, normalize_std : tuple[float, ...]
        ImageNet normalization constants.

    Returns
    -------
    torchvision.transforms.v2.Compose
        The composed transform pipeline.
    """
    if not transforms_v2_available():
        raise RuntimeError(
            "torchvision.transforms.v2 is not available — "
            "install torchvision >= 0.17"
        )

    import torch  # type: ignore
    from torchvision.transforms import v2  # type: ignore

    transforms_list = []

    # Random resize with scale jitter.
    if scale_range != (1.0, 1.0):
        transforms_list.append(
            v2.RandomResize(
                int(size[0] * scale_range[0]),
                int(size[0] * scale_range[1]),
            )
        )

    # Resize to exact target size.
    transforms_list.append(v2.Resize(size))

    # Horizontal flip.
    if hflip_prob > 0:
        transforms_list.append(v2.RandomHorizontalFlip(p=hflip_prob))

    # Photometric distortion.
    if color_jitter:
        transforms_list.append(v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03))

    # Convert PIL → Tensor, then normalize.
    transforms_list.append(v2.ToImage())
    transforms_list.append(v2.ToDtype(torch.float32, scale=True))
    transforms_list.append(v2.Normalize(mean=list(normalize_mean), std=list(normalize_std)))

    return v2.Compose(transforms_list)


def build_eval_transforms(
    *,
    size: tuple[int, int] = (640, 640),
    normalize_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    normalize_std: tuple[float, ...] = (0.229, 0.224, 0.225),
):
    """Build an evaluation-time transform pipeline (resize + normalize only).

    Parameters
    ----------
    size : tuple[int, int]
        Target (height, width).
    normalize_mean, normalize_std : tuple[float, ...]
        ImageNet normalization constants.

    Returns
    -------
    torchvision.transforms.v2.Compose
        The composed transform pipeline.
    """
    if not transforms_v2_available():
        raise RuntimeError(
            "torchvision.transforms.v2 is not available — "
            "install torchvision >= 0.17"
        )

    import torch  # type: ignore
    from torchvision.transforms import v2  # type: ignore

    return v2.Compose([
        v2.Resize(size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(normalize_mean), std=list(normalize_std)),
    ])

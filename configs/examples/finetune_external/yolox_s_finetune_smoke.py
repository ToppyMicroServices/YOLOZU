"""YOLOX smoke exp template for YOLOZU external training.

This file is intentionally Apache-2.0-friendly and can be executed even when
YOLOX is not installed. When YOLOX is available, the same file can act as a
minimal external launcher bridge because it subclasses ``yolox.exp.Exp``.

The dataset path is injected through environment variables so YOLOZU can keep
dataset resolution and reporting on its side:

- ``YOLOZU_DATASET_ROOT``
- ``YOLOZU_SPLIT``
- ``YOLOZU_BATCH_SIZE``
- ``YOLOZU_MAX_EPOCHS``
- ``YOLOZU_IMAGE_SIZE``

YOLO-family defaults intentionally differ from the RT-DETR reference lane:
this template keeps letterbox-shaped square inputs, SGD-style optimization,
and NMS-applied export/eval assumptions explicit.
"""

from __future__ import annotations

import os


try:  # pragma: no cover - only exercised when YOLOX is installed.
    from yolox.exp import Exp as _BaseExp  # type: ignore
except Exception:  # pragma: no cover - dry-run/projection environments.
    class _BaseExp:  # type: ignore[override]
        pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


class Exp(_BaseExp):
    def __init__(self) -> None:
        super().__init__()
        dataset_root = os.environ.get("YOLOZU_DATASET_ROOT", "data/smoke")
        split = os.environ.get("YOLOZU_SPLIT", "train")
        image_size = _env_int("YOLOZU_IMAGE_SIZE", 96)

        self.num_classes = _env_int("YOLOZU_NUM_CLASSES", 1)
        self.depth = 0.33
        self.width = 0.50
        self.max_epoch = _env_int("YOLOZU_MAX_EPOCHS", 1)
        self.data_num_workers = 0
        self.input_size = (image_size, image_size)
        self.test_size = (image_size, image_size)
        self.multiscale_range = 0
        self.enable_mixup = False
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.hsv_prob = 0.0
        self.flip_prob = 0.0
        self.warmup_epochs = 0
        self.no_aug_epochs = 1
        self.optimizer = "SGD"
        self.basic_lr_per_img = 0.001 / max(1, _env_int("YOLOZU_BATCH_SIZE", 2))
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.nesterov = True
        self.preprocess = "letterbox"
        self.decode_postprocess = "nms"
        self.nmsthre = 0.65
        self.test_conf = 0.01
        self.print_interval = 1
        self.eval_interval = 1
        self.output_dir = "runs/yolox_finetune"
        self.exp_name = "yolox_smoke"

        # Standard YOLOX data knobs. External launchers can override these.
        self.data_dir = str(dataset_root)
        self.train_ann = f"instances_{split}.json"
        self.val_ann = f"instances_{split}.json"
        self.test_ann = f"instances_{split}.json"


def get_exp() -> Exp:
    return Exp()


exp = Exp()

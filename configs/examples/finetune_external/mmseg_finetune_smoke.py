"""MMSeg finetune smoke template for YOLOZU.

This config is intentionally small and relies on environment variables so the
same file works in local and CI runners.

Expected env vars (set by YOLOZU wrappers):
- YOLOZU_DATASET_ROOT
- YOLOZU_SPLIT
"""

import os

_base_ = "mmseg::pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py"
if not _base_:
    raise RuntimeError("_base_ must point to an MMSeg base config")

dataset_root = os.getenv("YOLOZU_DATASET_ROOT", "data/cityscapes")
split = os.getenv("YOLOZU_SPLIT", "train")

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    dataset=dict(
        data_root=dataset_root,
        data_prefix=dict(
            img_path=f"images/{split}",
            seg_map_path=f"labels/{split}",
        ),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dict(
        data_root=dataset_root,
        data_prefix=dict(
            img_path=f"images/{split}",
            seg_map_path=f"labels/{split}",
        ),
    ),
)

test_dataloader = val_dataloader

train_cfg = dict(max_iters=20, val_interval=20)
default_hooks = dict(logger=dict(interval=1), checkpoint=dict(interval=20, save_best=None))

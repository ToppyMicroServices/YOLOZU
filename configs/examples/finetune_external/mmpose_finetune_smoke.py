"""MMPose finetune smoke template for YOLOZU.

This config is intentionally small and relies on environment variables so the
same file works in local and CI runners.

Expected env vars (set by YOLOZU wrappers):
- YOLOZU_DATASET_ROOT
- YOLOZU_SPLIT
"""

import os

_base_ = "mmpose::body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py"
if not _base_:
    raise RuntimeError("_base_ must point to an MMPose base config")

dataset_root = os.getenv("YOLOZU_DATASET_ROOT", "data/coco")
split = os.getenv("YOLOZU_SPLIT", "train")

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    dataset=dict(
        data_root=dataset_root,
        ann_file=f"annotations/person_keypoints_{split}.json",
        data_prefix=dict(img="images/"),
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=0,
    dataset=dict(
        data_root=dataset_root,
        ann_file=f"annotations/person_keypoints_{split}.json",
        data_prefix=dict(img="images/"),
    ),
)

test_dataloader = val_dataloader

train_cfg = dict(max_epochs=1, val_interval=1)
default_hooks = dict(logger=dict(interval=1), checkpoint=dict(interval=1, save_best=None))

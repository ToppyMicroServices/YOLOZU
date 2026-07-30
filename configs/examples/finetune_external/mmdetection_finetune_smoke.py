"""MMDetection finetune smoke template for YOLOZU.

This config is intentionally minimal and uses environment variables so the same
file works in local and CI runners.

Expected env vars (set by tools/run_external_finetune_smoke.py):
- YOLOZU_DATASET_ROOT
- YOLOZU_SPLIT
- YOLOZU_MAX_EPOCHS
- YOLOZU_BATCH_SIZE
"""

_base_ = "mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"
if not _base_:
    raise RuntimeError("_base_ must point to an MMDetection base config")

dataset_root = "{{$YOLOZU_DATASET_ROOT:data/coco}}"
split = "{{$YOLOZU_SPLIT:train2017}}"
max_epochs = int("{{$YOLOZU_MAX_EPOCHS:1}}")
batch_size = int("{{$YOLOZU_BATCH_SIZE:2}}")

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_root=dataset_root,
        ann_file=f"annotations/instances_{split}.json",
        data_prefix=dict(img=f"images/{split}/"),
    ),
)

val_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_root=dataset_root,
        ann_file=f"annotations/instances_{split}.json",
        data_prefix=dict(img=f"images/{split}/"),
    ),
)

test_dataloader = val_dataloader

train_cfg = dict(max_epochs=max_epochs, val_interval=1)

# Force scratch-style smoke run by default (no external checkpoint fetch).
load_from = None
model = dict(backbone=dict(init_cfg=None))

# Keep smoke runs light.
default_hooks = dict(
    logger=dict(interval=1),
    checkpoint=dict(interval=1, save_best=None),
)

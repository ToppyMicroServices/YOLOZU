"""Dataset helpers (conversion, discovery, manifests).

Importing this package auto-registers all built-in dataset adapters
(COCO, Pascal VOC, Cityscapes, ADE20K) with the unified registry.
"""

# Auto-register built-in adapters by importing the modules that call
# ``register_adapter()`` at module scope.
from . import ade20k as ade20k  # noqa: F401
from . import cityscapes as cityscapes  # noqa: F401
from . import coco as coco  # noqa: F401
from . import pascal_voc as pascal_voc  # noqa: F401


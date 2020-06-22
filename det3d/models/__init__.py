# from .anchor_heads import *  # noqa: F401,F403
import importlib
spconv_spec = importlib.util.find_spec("spconv")
found = spconv_spec is not None
if found:
    from .backbones import *  # noqa: F401,F403
else:
    print("No spconv, sparse convolution disabled!")
from .bbox_heads import *  # noqa: F401,F403
from .builder import (
    build_backbone,
    build_detector,
    build_head,
    build_loss,
    build_neck,
    build_roi_extractor,
    build_shared_head,
)
from .detectors import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .readers import *
from .registry import (
    BACKBONES,
    DETECTORS,
    HEADS,
    LOSSES,
    NECKS,
    READERS,
    ROI_EXTRACTORS,
    SHARED_HEADS,
)

# from .roi_extractors import *  # noqa: F401,F403
# from .shared_heads import *  # noqa: F401,F403

__all__ = [
    "READERS",
    "BACKBONES",
    "NECKS",
    "ROI_EXTRACTORS",
    "SHARED_HEADS",
    "HEADS",
    "LOSSES",
    "DETECTORS",
    "build_backbone",
    "build_neck",
    "build_roi_extractor",
    "build_shared_head",
    "build_head",
    "build_loss",
    "build_detector",
]

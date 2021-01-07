from det3d.utils import Registry

READERS = Registry("reader")
BACKBONES = Registry("backbone")
NECKS = Registry("neck")
HEADS = Registry("head")
LOSSES = Registry("loss")
DETECTORS = Registry("detector")
SECOND_STAGE = Registry("second_stage")
ROI_HEAD = Registry("roi_head")
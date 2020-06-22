import logging
from abc import ABCMeta, abstractmethod

import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn
from det3d import torchie


class BaseDetector(nn.Module):
    """Base class for detectors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_reader(self):
        # Whether input data need to be processed by Input Feature Extractor
        return hasattr(self, "reader") and self.reader is not None

    @property
    def with_neck(self):
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, "shared_head") and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, "bbox_head") and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, "mask_head") and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info("load model from: {}".format(pretrained))

    def forward_test(self, imgs, **kwargs):
        pass

    def forward(self, example, return_loss=True, **kwargs):
        pass

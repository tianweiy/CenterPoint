import time

import numpy as np
import spconv
import torch
from det3d.models.utils import Empty, change_default_args
from det3d.torchie.cnn import constant_init, kaiming_init
from det3d.torchie.trainer import load_checkpoint
from spconv import SparseConv3d, SubMConv3d
from torch import nn
from torch.nn import BatchNorm1d
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from .. import builder
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer


def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out


@BACKBONES.register_module
class SpMiddleFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleFHD", **kwargs
    ):
        super(SpMiddleFHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="subm0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SubMConv3d(16, 16, 3, bias=False, indice_key="subm0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1", bias=False),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1", bias=False),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseConv3d(
                64, 64, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseConv3d(
                64, 64, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
        )

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, "conv2_offset"):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        coors = coors.int()

        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret


@BACKBONES.register_module
class SpMiddleFHDNobn(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleFHD", **kwargs
    ):
        super(SpMiddleFHDNobn, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=True, indice_key="subm0"),
            # build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SubMConv3d(16, 16, 3, bias=True, indice_key="subm0"),
            # build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=True
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            # build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1", bias=True),
            # build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1", bias=True),
            # build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=True
            ),  # [800, 600, 21] -> [400, 300, 11]
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2", bias=True),
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2", bias=True),
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2", bias=True),
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseConv3d(
                64, 64, 3, 2, padding=[0, 1, 1], bias=True
            ),  # [400, 300, 11] -> [200, 150, 5]
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3", bias=True),
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3", bias=True),
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3", bias=True),
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseConv3d(
                64, 64, (3, 1, 1), (2, 1, 1), bias=True
            ),  # [200, 150, 5] -> [200, 150, 2]
            # build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
        )

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, "conv2_offset"):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        coors = coors.int()

        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret


@BACKBONES.register_module
class SpMiddleResNetFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHD", **kwargs
    ):
        super(SpMiddleResNetFHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseConv3d(
                64, 128, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret


@BACKBONES.register_module
class SASSDFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SASSDFHD", **kwargs
    ):
        super(SASSDFHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="subm0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SubMConv3d(16, 16, 3, bias=False, indice_key="subm0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1", bias=False),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm1", bias=False),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseConv3d(
                64, 64, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm3", bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseConv3d(
                64, 64, (1, 1, 1), (1, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
        )

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, "conv2_offset"):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        coors = coors.int()

        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret


@BACKBONES.register_module
class SASSDResNetFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SASSDResNetFHD", **kwargs
    ):
        super(SASSDResNetFHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="subm0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseConv3d(
                64, 64, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res3"),
            SparseConv3d(
                64, 64, (1, 1, 1), (1, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
        )

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, "conv2_offset"):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        coors = coors.int()

        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret




@BACKBONES.register_module
class RCNNSpMiddleFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="RCNNSpMiddleFHD", **kwargs
    ):
        super(RCNNSpMiddleFHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="subm0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SubMConv3d(16, 16, 3, bias=False, indice_key="subm0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(),
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [32, 80, 41] -> [16, 40, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SubMConv3d(32, 32, 3, bias=False, indice_key="subm1"),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            # SubMConv3d(32, 32, 3, bias=False, indice_key="subm1"),
            # build_norm_layer(norm_cfg, 32)[1],
            # nn.ReLU(),
            SparseConv3d(
                32, 64, 3, 2, bias=False, padding=1
            ),  # [16, 40, 21] -> [8, 20, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, bias=False, indice_key="subm2"),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            # SubMConv3d(64, 64, 3, bias=False, indice_key="subm2"),
            # build_norm_layer(norm_cfg, 64)[1],
            # nn.ReLU(),
            # SubMConv3d(64, 64, 3, bias=False, indice_key="subm2"),
            # build_norm_layer(norm_cfg, 64)[1],
            # nn.ReLU(),
            SparseConv3d(
                64, 64, 3, 2, bias=False, padding=[1, 1, 0]
            ),  # [8, 20, 11] -> [4, 10, 5]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SubMConv3d(64, 64, 3, bias=False, indice_key="subm3"),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            # SubMConv3d(64, 64, 3, bias=False, indice_key="subm3"),
            # build_norm_layer(norm_cfg, 64)[1],
            # nn.ReLU(),
            # SubMConv3d(64, 64, 3, bias=False, indice_key="subm3"),
            # build_norm_layer(norm_cfg, 64)[1],
            # nn.ReLU(),
            SparseConv3d(
                64, 64, (1, 1, 3), (1, 1, 2), bias=False
            ),  # [4, 10, 5] -> [4, 10, 2]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [0, 0, 1]

        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        ret = self.middle_conv(ret)

        ret = ret.dense()

        ret = ret.permute(0, 1, 4, 2, 3).contiguous()
        N, C, W, D, H = ret.shape
        ret = ret.view(N, C * W, D, H)

        return ret

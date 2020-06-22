import numpy as np
from det3d.core.bbox import box_np_ops


class AnchorGeneratorStride:
    def __init__(
        self,
        sizes=[1.6, 3.9, 1.56],
        anchor_strides=[0.4, 0.4, 1.0],
        anchor_offsets=[0.2, -39.8, -1.78],
        rotations=[0, np.pi / 2],
        velocities=[0, 0],
        class_name=None,
        match_threshold=-1,
        unmatch_threshold=-1,
        dtype=np.float32,
    ):
        self._sizes = sizes
        self._anchor_strides = anchor_strides
        self._anchor_offsets = anchor_offsets
        self._rotations = rotations
        self._velocities = velocities
        self._dtype = dtype
        self._class_name = class_name
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold

    @property
    def class_name(self):
        return self._class_name

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    @property
    def ndim(self):
        # return 7 + len(self._custom_values)
        return self._anchors.shape[-1]

    def generate(self, feature_map_size):
        self._anchors = box_np_ops.create_anchors_3d_stride(
            feature_map_size,
            self._sizes,
            self._anchor_strides,
            self._anchor_offsets,
            self._rotations,
            self._velocities,
            self._dtype,
        )
        return self._anchors


class AnchorGeneratorRange:
    def __init__(
        self,
        anchor_ranges,
        sizes=[1.6, 3.9, 1.56],
        rotations=[0, np.pi / 2],
        velocities=[0, 0],
        class_name=None,
        match_threshold=-1,
        unmatch_threshold=-1,
        dtype=np.float32,
    ):
        self._sizes = sizes
        self._anchor_ranges = anchor_ranges
        self._rotations = rotations
        self._velocities = velocities
        self._dtype = dtype
        self._class_name = class_name
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold

    @property
    def class_name(self):
        return self._class_name

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    @property
    def ndim(self):
        # return 7 + len(self._custom_values)
        return self._anchors.shape[-1]

    def generate(self, feature_map_size):
        self._anchors = box_np_ops.create_anchors_3d_range(
            feature_map_size,
            self._anchor_ranges,
            self._sizes,
            self._rotations,
            self._velocities,
            self._dtype,
        )
        return self._anchors


class BevAnchorGeneratorRange:
    def __init__(
        self,
        anchor_ranges,
        sizes=[1.6, 3.9],
        rotations=[0, np.pi / 2],
        velocities=[0, 0],
        class_name=None,
        match_threshold=-1,
        unmatch_threshold=-1,
        dtype=np.float32,
    ):
        self._sizes = sizes
        self._anchor_ranges = anchor_ranges
        self._rotations = rotations
        self._velocities = velocities
        self._dtype = dtype
        self._class_name = class_name
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold

    @property
    def class_name(self):
        return self._class_name

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 2]).shape[0]
        return num_rot * num_size

    @property
    def ndim(self):
        # return 7 + len(self._custom_values)
        return self._anchors.shape[-1]

    def generate(self, feature_map_size):
        self._anchors = box_np_ops.create_anchors_bev_range(
            feature_map_size,
            self._anchor_ranges,
            self._sizes,
            self._rotations,
            self._velocities,
            self._dtype,
        )
        return self._anchors

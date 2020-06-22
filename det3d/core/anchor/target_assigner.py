from collections import OrderedDict

import numpy as np
from det3d.core.anchor.target_ops import create_target_np
from det3d.core.bbox import box_np_ops, region_similarity


class TargetAssigner:
    def __init__(
        self,
        box_coder,
        anchor_generators,
        region_similarity_calculator=None,
        positive_fraction=None,
        sample_size=512,
    ):
        self._region_similarity_calculator = region_similarity_calculator
        self._box_coder = box_coder
        self._anchor_generators = anchor_generators
        self._positive_fraction = positive_fraction
        self._sample_size = sample_size

    @property
    def box_coder(self):
        return self._box_coder

    @property
    def classes(self):
        return [a.class_name for a in self._anchor_generators]

    def assign(
        self,
        anchors,
        gt_boxes,
        anchors_mask=None,
        gt_classes=None,
        matched_thresholds=None,
        unmatched_thresholds=None,
    ):
        if anchors_mask is not None:
            prune_anchor_fn = lambda _: np.where(anchors_mask)[0]
        else:
            prune_anchor_fn = None

        def similarity_fn(anchors, gt_boxes):
            anchors_rbv = anchors[:, [0, 1, 3, 4, -1]]
            gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, -1]]
            return self._region_similarity_calculator.compare(anchors_rbv, gt_boxes_rbv)

        def box_encoding_fn(boxes, anchors):
            return self._box_coder.encode(boxes, anchors)

        return create_target_np(
            anchors,
            gt_boxes[gt_classes == class_name],
            similarity_fn,
            box_encoding_fn,
            prune_anchor_fn=prune_anchor_fn,
            gt_classes=gt_classes[gt_classes == class_name],
            matched_threshold=matched_thresholds,
            unmatched_threshold=unmatched_thresholds,
            positive_fraction=self._positive_fraction,
            rpn_batch_size=self._sample_size,
            norm_by_num_examples=False,
            box_code_size=self.box_coder.code_size,
        )

    def assign_v2(
        self, anchors_dict, gt_boxes, anchors_mask=None, gt_classes=None, gt_names=None
    ):
        def similarity_fn(anchors, gt_boxes):
            anchors_rbv = anchors[:, [0, 1, 3, 4, -1]]
            gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, -1]]
            return self._region_similarity_calculator.compare(anchors_rbv, gt_boxes_rbv)

        def box_encoding_fn(boxes, anchors):
            return self._box_coder.encode(boxes, anchors)

        targets_list = []
        anchor_loc_idx = 0
        for class_name, anchor_dict in anchors_dict.items():
            mask = np.array([c == class_name for c in gt_names], dtype=np.bool_)
            feature_map_size = anchor_dict["anchors"].shape[:3]
            num_loc = anchor_dict["anchors"].shape[-2]

            if anchors_mask is not None:
                anchors_mask = anchors_mask.reshape(*feature_map_size, -1)
                anchors_mask_class = anchors_mask[
                    ..., anchor_loc_idx : anchor_loc_idx + num_loc
                ].reshape(-1)
                prune_anchor_fn = lambda _: np.where(anchors_mask_class)[0]
            else:
                prune_anchor_fn = None

            targets = create_target_np(
                anchor_dict["anchors"].reshape(-1, self.box_coder.n_dim),#code_size),
                gt_boxes[mask],
                similarity_fn,
                box_encoding_fn,
                prune_anchor_fn=prune_anchor_fn,
                gt_classes=gt_classes[mask],
                matched_threshold=anchor_dict["matched_thresholds"],
                unmatched_threshold=anchor_dict["unmatched_thresholds"],
                positive_fraction=self._positive_fraction,
                rpn_batch_size=self._sample_size,
                norm_by_num_examples=False,
                box_code_size=self.box_coder.code_size,
            )
            anchor_loc_idx += num_loc
            targets_list.append(targets)

        targets_dict = {
            "labels": [t["labels"] for t in targets_list],
            "bbox_targets": [t["bbox_targets"] for t in targets_list],
            "bbox_outside_weights": [t["bbox_outside_weights"] for t in targets_list],
        }
        targets_dict["bbox_targets"] = np.concatenate(
            [
                v.reshape(*feature_map_size, -1, self.box_coder.code_size)
                for v in targets_dict["bbox_targets"]
            ],
            axis=-2,
        )
        targets_dict["bbox_targets"] = targets_dict["bbox_targets"].reshape(
            -1, self.box_coder.code_size
        )
        targets_dict["labels"] = np.concatenate(
            [v.reshape(*feature_map_size, -1) for v in targets_dict["labels"]], axis=-1
        )
        targets_dict["bbox_outside_weights"] = np.concatenate(
            [
                v.reshape(*feature_map_size, -1)
                for v in targets_dict["bbox_outside_weights"]
            ],
            axis=-1,
        )
        targets_dict["labels"] = targets_dict["labels"].reshape(-1)
        targets_dict["bbox_outside_weights"] = targets_dict[
            "bbox_outside_weights"
        ].reshape(-1)

        return targets_dict

    def generate_anchors(self, feature_map_size):
        anchors_list = []
        matched_thresholds = [a.match_threshold for a in self._anchor_generators]
        unmatched_thresholds = [a.unmatch_threshold for a in self._anchor_generators]
        match_list, unmatch_list = [], []
        for anchor_generator, match_thresh, unmatch_thresh in zip(
            self._anchor_generators, matched_thresholds, unmatched_thresholds
        ):
            anchors = anchor_generator.generate(feature_map_size)
            anchors = anchors.reshape([*anchors.shape[:3], -1, anchors.shape[-1]])
            anchors_list.append(anchors)
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(np.full([num_anchors], match_thresh, anchors.dtype))
            unmatch_list.append(np.full([num_anchors], unmatch_thresh, anchors.dtype))
        anchors = np.concatenate(anchors_list, axis=-2)
        matched_thresholds = np.concatenate(match_list, axis=0)
        unmatched_thresholds = np.concatenate(unmatch_list, axis=0)
        return {
            "anchors": anchors,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
        }

    def generate_anchors_dict(self, feature_map_size):
        anchors_list = []
        matched_thresholds = [a.match_threshold for a in self._anchor_generators]
        unmatched_thresholds = [a.unmatch_threshold for a in self._anchor_generators]
        match_list, unmatch_list = [], []
        anchors_dict = {a.class_name: {} for a in self._anchor_generators}
        anchors_dict = OrderedDict(anchors_dict)
        for anchor_generator, match_thresh, unmatch_thresh in zip(
            self._anchor_generators, matched_thresholds, unmatched_thresholds
        ):
            anchors = anchor_generator.generate(feature_map_size)
            anchors = anchors.reshape([*anchors.shape[:3], -1, anchors.shape[-1]])
            anchors_list.append(anchors)
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(np.full([num_anchors], match_thresh, anchors.dtype))
            unmatch_list.append(np.full([num_anchors], unmatch_thresh, anchors.dtype))
            class_name = anchor_generator.class_name
            anchors_dict[class_name]["anchors"] = anchors
            anchors_dict[class_name]["matched_thresholds"] = match_list[-1]
            anchors_dict[class_name]["unmatched_thresholds"] = unmatch_list[-1]
        return anchors_dict

    @property
    def num_anchors_per_location(self):
        num = 0
        for a_generator in self._anchor_generators:
            num += a_generator.num_anchors_per_localization
        return num

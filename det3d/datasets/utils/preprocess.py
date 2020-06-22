import pathlib
import time
from collections import defaultdict
import torch
import cv2
import numpy as np

from det3d.core import box_np_ops
from det3d.core import preprocess as prep
from det3d.core.geometry import points_in_convex_polygon_3d_jit
from det3d.datasets import kitti
import itertools


def prcnn_rpn_collate_batch(batch_list):

    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)

    batch_size = len(batch_list)
    ret = {}

    for key, elems in example_merged.items():
        if key in ["gt_boxes3d"]:
            task_max_gts = []
            for task_id in range(len(elems[0])):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, len(elems[k][task_id]))
                task_max_gts.append(max_gt)
            res = []
            for idx, max_gt in enumerate(task_max_gts):
                batch_task_gt_boxes3d = np.zeros((batch_size, max_gt, 7))
                for i in range(batch_size):
                    batch_task_gt_boxes3d[i, : len(elems[i][idx]), :] = elems[i][idx]
                res.append(batch_task_gt_boxes3d)
            ret[key] = res
        elif key == "metadata":
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = np.stack(v1, axis=0)
        elif key in ["pts_input"]:
            ret[key] = np.concatenate(
                [elems[k][np.newaxis, ...] for k in range(batch_size)]
            )
        elif key in ["rpn_cls_label", "rpn_reg_label"]:
            ret[key] = []
            for task_id in range(len(elems[0])):
                branch_out = np.concatenate(
                    [elems[k][task_id][np.newaxis, ...] for k in range(batch_size)]
                )
                ret[key].append(branch_out)
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret


type_map = {
    "voxels": torch.float32,
    "bev_map": torch.float32,
    "anchors": torch.float32,
    "reg_targets": torch.float32,
    "reg_weights": torch.float32,
    "coordinates": torch.int32,
    "num_points": torch.int32,
    "labels": torch.int32,
    "points": torch.float32,
    "anchors_mask": torch.uint8,
    "calib": torch.float32,
    "num_voxels": torch.int64,
}


def collate_sequence_batch(batch_list):
    example_current_frame_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example["current_frame"].items():
            example_current_frame_merged[k].append(v)
    batch_size = len(batch_list)
    ret_current_frame = {}
    for key, elems in example_current_frame_merged.items():
        if key in ["voxels", "num_points", "num_gt", "voxel_labels"]:
            ret_current_frame[key] = np.concatenate(elems, axis=0)
        elif key in ["gt_boxes"]:
            task_max_gts = []
            for task_id in range(len(elems[0])):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, len(elems[k][task_id]))
                task_max_gts.append(max_gt)
            res = []
            for idx, max_gt in enumerate(task_max_gts):
                batch_task_gt_boxes3d = np.zeros((batch_size, max_gt, 9))
                for i in range(batch_size):
                    batch_task_gt_boxes3d[i, : len(elems[i][idx]), :] = elems[i][idx]
                res.append(batch_task_gt_boxes3d)
            ret_current_frame[key] = res
        elif key == "metadata":
            ret_current_frame[key] = elems
        elif key == "calib":
            ret_current_frame[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret_current_frame[key][k1] = [v1]
                    else:
                        ret_current_frame[key][k1].append(v1)
            for k1, v1 in ret_current_frame[key].items():
                ret_current_frame[key][k1] = np.stack(v1, axis=0)
        elif key in ["coordinates", "points"]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret_current_frame[key] = np.concatenate(coors, axis=0)
        elif key in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels"]:
            ret_current_frame[key] = defaultdict(list)
            for elem in elems:
                for idx, ele in enumerate(elem):
                    ret_current_frame[key][str(idx)].append(ele)
        else:
            ret_current_frame[key] = np.stack(elems, axis=0)

    example_keyframe_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example["keyframe"].items():
            example_keyframe_merged[k].append(v)
    batch_size = len(batch_list)
    ret_keyframe = {}
    for key, elems in example_keyframe_merged.items():
        if key in ["voxels", "num_points", "num_gt", "voxel_labels"]:
            ret_keyframe[key] = np.concatenate(elems, axis=0)
        elif key == "calib":
            ret_keyframe[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret_keyframe[key][k1] = [v1]
                    else:
                        ret_keyframe[key][k1].append(v1)
            for k1, v1 in ret_keyframe[key].items():
                ret_keyframe[key][k1] = np.stack(v1, axis=0)
        elif key in ["coordinates", "points"]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret_keyframe[key] = np.concatenate(coors, axis=0)
        else:
            ret_keyframe[key] = np.stack(elems, axis=0)

    rets = {}
    rets["current_frame"] = ret_current_frame
    rets["keyframe"] = ret_keyframe
    return rets


def collate_batch(batch_list):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    batch_size = len(batch_list)
    ret = {}
    # voxel_nums_list = example_merged["num_voxels"]
    # example_merged.pop("num_voxels")
    for key, elems in example_merged.items():
        if key in ["voxels", "num_points", "num_gt", "voxel_labels", "ground_plane"]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key in [
            "gt_boxes",
        ]:
            task_max_gts = []
            for task_id in range(len(elems[0])):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, len(elems[k][task_id]))
                task_max_gts.append(max_gt)
            res = []
            for idx, max_gt in enumerate(task_max_gts):
                batch_task_gt_boxes3d = np.zeros((batch_size, max_gt, 9))
                for i in range(batch_size):
                    batch_task_gt_boxes3d[i, : len(elems[i][idx]), :] = elems[i][idx]
                res.append(batch_task_gt_boxes3d)
            ret[key] = res
        elif key == "metadata":
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = np.stack(v1, axis=0)
        elif key in ["coordinates", "points"]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        elif key in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels"]:
            ret[key] = defaultdict(list)
            for elem in elems:
                for idx, ele in enumerate(elem):
                    ret[key][str(idx)].append(ele)
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret


def collate_batch_kitti(batch_list):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    batch_size = len(batch_list)
    ret = {}
    # voxel_nums_list = example_merged["num_voxels"]
    # example_merged.pop("num_voxels")
    for key, elems in example_merged.items():
        if key in ["voxels", "num_points", "num_gt", "voxel_labels"]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key in [
            "gt_boxes",
        ]:
            task_max_gts = []
            for task_id in range(len(elems[0])):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, len(elems[k][task_id]))
                task_max_gts.append(max_gt)
            res = []
            for idx, max_gt in enumerate(task_max_gts):
                batch_task_gt_boxes3d = np.zeros((batch_size, max_gt, 7))
                for i in range(batch_size):
                    batch_task_gt_boxes3d[i, : len(elems[i][idx]), :] = elems[i][idx]
                res.append(batch_task_gt_boxes3d)
            ret[key] = res
        elif key == "metadata":
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = np.stack(v1, axis=0)
        elif key in ["coordinates", "points"]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        elif key in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels"]:
            ret[key] = defaultdict(list)
            for elem in elems:
                for idx, ele in enumerate(elem):
                    ret[key][str(idx)].append(ele)
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret


def collate_batch_torch(batch_list):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    batch_size = len(batch_list)
    ret = {}
    # voxel_nums_list = example_merged["num_voxels"]
    # example_merged.pop("num_voxels")
    for key, elems in example_merged.items():
        if key in ["voxels", "num_points", "num_gt", "voxel_labels"]:
            ret[key] = torch.tensor(np.concatenate(elems, axis=0), dtype=type_map[key])
        elif key in [
            "gt_boxes",
        ]:
            task_max_gts = []
            for task_id in range(len(elems[0])):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, len(elems[k][task_id]))
                task_max_gts.append(max_gt)
            res = []
            for idx, max_gt in enumerate(task_max_gts):
                batch_task_gt_boxes3d = np.zeros((batch_size, max_gt, 9))
                for i in range(batch_size):
                    batch_task_gt_boxes3d[i, : len(elems[i][idx]), :] = elems[i][idx]
                res.append(batch_task_gt_boxes3d)
            ret[key] = res
        elif key == "metadata":
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = torch.tensor(np.stack(v1, axis=0), dtype=type_map[key])
        elif key in ["coordinates", "points"]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret[key] = torch.tensor(np.concatenate(coors, axis=0), dtype=type_map[key])
        elif key in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels"]:
            ret[key] = defaultdict(list)
            for elem in elems:
                for idx, ele in enumerate(elem):
                    ret[key][str(idx)].append(torch.tensor(ele, dtype=type_map[key]))
        else:
            ret[key] = torch.tensor(np.stack(elems, axis=0), dtype=type_map[key])

    return ret


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def prep_pointcloud(
    input_dict,
    root_path,
    voxel_generator,
    target_assigners,
    prep_cfg=None,
    db_sampler=None,
    remove_outside_points=False,
    training=True,
    create_targets=True,
    num_point_features=4,
    anchor_cache=None,
    random_crop=False,
    reference_detections=None,
    out_size_factor=2,
    out_dtype=np.float32,
    min_points_in_gt=-1,
    logger=None,
):
    """
    convert point cloud to voxels, create targets if ground truths exists.
    input_dict format: dataset.get_sensor_data format
    """
    assert prep_cfg is not None

    task_class_names = [target_assigner.classes for target_assigner in target_assigners]
    class_names = list(itertools.chain(*task_class_names))

    # res = voxel_generator.generate(
    #     points, max_voxels)
    # voxels = res["voxels"]
    # coordinates = res["coordinates"]
    # num_points = res["num_points_per_voxel"]

    num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

    example = {
        "voxels": voxels,
        "num_points": num_points,
        "points": points,
        "coordinates": coordinates,
        # "num_voxels": np.array([voxels.shape[0]], dtype=np.int64),
        "num_voxels": num_voxels,
        # "ground_plane": input_dict["ground_plane"],
        # "gt_dict": gt_dict,
    }

    if training:
        example["gt_boxes"] = gt_dict["gt_boxes"]
    else:
        example["gt_boxes"] = [input_dict["lidar"]["annotations"]["boxes"]]

    if calib is not None:
        example["calib"] = calib

    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]

    if anchor_cache is not None:
        anchorss = anchor_cache["anchors"]
        anchors_bvs = anchor_cache["anchors_bv"]
        anchors_dicts = anchor_cache["anchors_dict"]
    else:
        rets = [
            target_assigner.generate_anchors(feature_map_size)
            for target_assigner in target_assigners
        ]
        anchorss = [ret["anchors"].reshape([-1, 7]) for ret in rets]
        anchors_dicts = [
            target_assigner.generate_anchors_dict(feature_map_size)
            for target_assigner in target_assigners
        ]
        anchors_bvs = [
            box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
            for anchors in anchorss
        ]

    example["anchors"] = anchorss

    if anchor_area_threshold >= 0:
        example["anchors_mask"] = []
        for idx, anchors_bv in enumerate(anchors_bvs):
            anchors_mask = None
            # slow with high resolution. recommend disable this forever.
            coors = coordinates
            dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
                coors, tuple(grid_size[::-1][1:])
            )
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            anchors_area = box_np_ops.fused_get_anchors_area(
                dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size
            )
            anchors_mask = anchors_area > anchor_area_threshold
            # example['anchors_mask'] = anchors_mask.astype(np.uint8)
            example["anchors_mask"].append(anchors_mask)

    if not training:
        return example

    # voxel_labels = box_np_ops.assign_label_to_voxel(gt_boxes, coordinates,
    #                                                 voxel_size, coors_range)
    """
    example.update({
        'gt_boxes': gt_boxes.astype(out_dtype),
        'num_gt': np.array([gt_boxes.shape[0]]),
        # 'voxel_labels': voxel_labels,
    })
    """
    if create_targets:
        targets_dicts = []
        for idx, target_assigner in enumerate(target_assigners):
            if "anchors_mask" in example:
                anchors_mask = example["anchors_mask"][idx]
            else:
                anchors_mask = None
            targets_dict = target_assigner.assign_v2(
                anchors_dicts[idx],
                gt_dict["gt_boxes"][idx],
                anchors_mask,
                gt_classes=gt_dict["gt_classes"][idx],
                gt_names=gt_dict["gt_names"][idx],
            )
            targets_dicts.append(targets_dict)

        example.update(
            {
                "labels": [targets_dict["labels"] for targets_dict in targets_dicts],
                "reg_targets": [
                    targets_dict["bbox_targets"] for targets_dict in targets_dicts
                ],
                "reg_weights": [
                    targets_dict["bbox_outside_weights"]
                    for targets_dict in targets_dicts
                ],
            }
        )

    return example


def prep_sequence_pointcloud(
    input_dict,
    root_path,
    voxel_generator,
    target_assigners,
    prep_cfg=None,
    db_sampler=None,
    remove_outside_points=False,
    training=True,
    create_targets=True,
    num_point_features=4,
    anchor_cache=None,
    random_crop=False,
    reference_detections=None,
    out_size_factor=2,
    out_dtype=np.float32,
    min_points_in_gt=-1,
    logger=None,
):
    """
    convert point cloud to voxels, create targets if ground truths exists.
    input_dict format: dataset.get_sensor_data format
    """
    assert prep_cfg is not None

    remove_environment = prep_cfg.REMOVE_ENVIRONMENT
    max_voxels = prep_cfg.MAX_VOXELS_NUM
    shuffle_points = prep_cfg.SHUFFLE
    anchor_area_threshold = prep_cfg.ANCHOR_AREA_THRES

    if training:
        remove_unknown = prep_cfg.REMOVE_UNKOWN_EXAMPLES
        gt_rotation_noise = prep_cfg.GT_ROT_NOISE
        gt_loc_noise_std = prep_cfg.GT_LOC_NOISE
        global_rotation_noise = prep_cfg.GLOBAL_ROT_NOISE
        global_scaling_noise = prep_cfg.GLOBAL_SCALE_NOISE
        global_random_rot_range = prep_cfg.GLOBAL_ROT_PER_OBJ_RANGE
        global_translate_noise_std = prep_cfg.GLOBAL_TRANS_NOISE
        gt_points_drop = prep_cfg.GT_DROP_PERCENTAGE
        gt_drop_max_keep = prep_cfg.GT_DROP_MAX_KEEP_POINTS
        remove_points_after_sample = prep_cfg.REMOVE_POINTS_AFTER_SAMPLE
        min_points_in_gt = prep_cfg.get("MIN_POINTS_IN_GT", -1)

    task_class_names = [target_assigner.classes for target_assigner in target_assigners]
    class_names = list(itertools.chain(*task_class_names))

    # points_only = input_dict["lidar"]["points"]
    # times = input_dict["lidar"]["times"]
    # points = np.hstack([points_only, times])
    try:
        points = input_dict["current_frame"]["lidar"]["combined"]
    except Exception:
        points = input_dict["current_frame"]["lidar"]["points"]
    keyframe_points = input_dict["keyframe"]["lidar"]["combined"]

    if training:
        anno_dict = input_dict["current_frame"]["lidar"]["annotations"]
        gt_dict = {
            "gt_boxes": anno_dict["boxes"],
            "gt_names": np.array(anno_dict["names"]).reshape(-1),
        }

        if "difficulty" not in anno_dict:
            difficulty = np.zeros([anno_dict["boxes"].shape[0]], dtype=np.int32)
            gt_dict["difficulty"] = difficulty
        else:
            gt_dict["difficulty"] = anno_dict["difficulty"]
        # if use_group_id and "group_ids" in anno_dict:
        #     group_ids = anno_dict["group_ids"]
        #     gt_dict["group_ids"] = group_ids

    calib = None
    if "calib" in input_dict:
        calib = input_dict["current_frame"]["calib"]

    if reference_detections is not None:
        assert calib is not None and "image" in input_dict["current_frame"]
        C, R, T = box_np_ops.projection_matrix_to_CRT_kitti(P2)
        frustums = box_np_ops.get_frustum_v2(reference_detections, C)
        frustums -= T
        frustums = np.einsum("ij, akj->aki", np.linalg.inv(R), frustums)
        frustums = box_np_ops.camera_to_lidar(frustums, rect, Trv2c)
        surfaces = box_np_ops.corner_to_surfaces_3d_jit(frustums)
        masks = points_in_convex_polygon_3d_jit(points, surfaces)
        points = points[masks.any(-1)]

    if remove_outside_points:
        assert calib is not None
        image_shape = input_dict["current_frame"]["image"]["image_shape"]
        points = box_np_ops.remove_outside_points(
            points, calib["rect"], calib["Trv2c"], calib["P2"], image_shape
        )
    if remove_environment is True and training:
        selected = kitti.keep_arrays_by_name(gt_names, target_assigner.classes)
        _dict_select(gt_dict, selected)
        masks = box_np_ops.points_in_rbbox(points, gt_dict["gt_boxes"])
        points = points[masks.any(-1)]

    if training:
        # boxes_lidar = gt_dict["gt_boxes"]
        # cv2.imshow('pre-noise', bev_map)
        selected = kitti.drop_arrays_by_name(
            gt_dict["gt_names"], ["DontCare", "ignore"]
        )
        _dict_select(gt_dict, selected)
        if remove_unknown:
            remove_mask = gt_dict["difficulty"] == -1
            """
            gt_boxes_remove = gt_boxes[remove_mask]
            gt_boxes_remove[:, 3:6] += 0.25
            points = prep.remove_points_in_boxes(points, gt_boxes_remove)
            """
            keep_mask = np.logical_not(remove_mask)
            _dict_select(gt_dict, keep_mask)
        gt_dict.pop("difficulty")

        if min_points_in_gt > 0:
            # points_count_rbbox takes 10ms with 10 sweeps nuscenes data
            point_counts = box_np_ops.points_count_rbbox(points, gt_dict["gt_boxes"])
            mask = point_counts >= min_points_in_gt
            _dict_select(gt_dict, mask)

        gt_boxes_mask = np.array(
            [n in class_names for n in gt_dict["gt_names"]], dtype=np.bool_
        )

        # db_sampler = None
        if db_sampler is not None:
            group_ids = None
            # if "group_ids" in gt_dict:
            #     group_ids = gt_dict["group_ids"]
            sampled_dict = db_sampler.sample_all(
                root_path,
                gt_dict["gt_boxes"],
                gt_dict["gt_names"],
                num_point_features,
                random_crop,
                gt_group_ids=group_ids,
                calib=calib,
            )

            if sampled_dict is not None:
                sampled_gt_names = sampled_dict["gt_names"]
                sampled_gt_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                sampled_gt_masks = sampled_dict["gt_masks"]
                gt_dict["gt_names"] = np.concatenate(
                    [gt_dict["gt_names"], sampled_gt_names], axis=0
                )
                gt_dict["gt_boxes"] = np.concatenate(
                    [gt_dict["gt_boxes"], sampled_gt_boxes]
                )
                gt_boxes_mask = np.concatenate(
                    [gt_boxes_mask, sampled_gt_masks], axis=0
                )

                # if group_ids is not None:
                #     sampled_group_ids = sampled_dict["group_ids"]
                #     gt_dict["group_ids"] = np.concatenate(
                #         [gt_dict["group_ids"], sampled_group_ids])

                if remove_points_after_sample:
                    masks = box_np_ops.points_in_rbbox(points, sampled_gt_boxes)
                    points = points[np.logical_not(masks.any(-1))]

                points = np.concatenate([sampled_points, points], axis=0)

        pc_range = voxel_generator.point_cloud_range

        # group_ids = None
        # if "group_ids" in gt_dict:
        #     group_ids = gt_dict["group_ids"]

        # prep.noise_per_object_v3_(
        #     gt_dict["gt_boxes"],
        #     points,
        #     gt_boxes_mask,
        #     rotation_perturb=gt_rotation_noise,
        #     center_noise_std=gt_loc_noise_std,
        #     global_random_rot_range=global_random_rot_range,
        #     group_ids=group_ids,
        #     num_try=100)

        # should remove unrelated objects after noise per object
        # for k, v in gt_dict.items():
        #     print(k, v.shape)

        _dict_select(gt_dict, gt_boxes_mask)

        gt_classes = np.array(
            [class_names.index(n) + 1 for n in gt_dict["gt_names"]], dtype=np.int32
        )
        gt_dict["gt_classes"] = gt_classes

        # concatenate
        points_current = points.shape[0]
        points_keyframe = keyframe_points.shape[0]
        points = np.concatenate((points, keyframe_points), axis=0)

        # data aug
        gt_dict["gt_boxes"], points = prep.random_flip(gt_dict["gt_boxes"], points)
        gt_dict["gt_boxes"], points = prep.global_rotation(
            gt_dict["gt_boxes"], points, rotation=global_rotation_noise
        )
        gt_dict["gt_boxes"], points = prep.global_scaling_v2(
            gt_dict["gt_boxes"], points, *global_scaling_noise
        )
        prep.global_translate_(gt_dict["gt_boxes"], points, global_translate_noise_std)

        # slice
        points_keyframe = points[points_current:, :]
        points = points[:points_current, :]

        bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
        mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
        _dict_select(gt_dict, mask)

        task_masks = []
        flag = 0
        for class_name in task_class_names:
            task_masks.append(
                [
                    np.where(gt_dict["gt_classes"] == class_name.index(i) + 1 + flag)
                    for i in class_name
                ]
            )
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        task_names = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            task_name = []
            for m in mask:
                task_box.append(gt_dict["gt_boxes"][m])
                task_class.append(gt_dict["gt_classes"][m] - flag2)
                task_name.append(gt_dict["gt_names"][m])
            task_boxes.append(np.concatenate(task_box, axis=0))
            task_classes.append(np.concatenate(task_class))
            task_names.append(np.concatenate(task_name))
            flag2 += len(mask)

        for task_box in task_boxes:
            # limit rad to [-pi, pi]
            task_box[:, -1] = box_np_ops.limit_period(
                task_box[:, -1], offset=0.5, period=2 * np.pi
            )

        # print(gt_dict.keys())
        gt_dict["gt_classes"] = task_classes
        gt_dict["gt_names"] = task_names
        gt_dict["gt_boxes"] = task_boxes

    # if shuffle_points:
    #     # shuffle is a little slow.
    #     np.random.shuffle(points)

    # [0, -40, -3, 70.4, 40, 1]
    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range
    grid_size = voxel_generator.grid_size
    # [352, 400]

    # points = points[:int(points.shape[0] * 0.1), :]
    voxels, coordinates, num_points = voxel_generator.generate(points, max_voxels)

    # res = voxel_generator.generate(
    #     points, max_voxels)
    # voxels = res["voxels"]
    # coordinates = res["coordinates"]
    # num_points = res["num_points_per_voxel"]

    num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

    # key frame voxel
    keyframe_info = voxel_generator.generate(keyframe_points, max_voxels)
    keyframe_info = keyframe_voxels, keyframe_coordinates, keyframe_num_points

    keyframe_num_voxels = np.array([keyframe_voxels.shape[0]], dtype=np.int64)

    example = {
        "voxels": voxels,
        "num_points": num_points,
        "points": points,
        "coordinates": coordinates,
        "num_voxels": num_voxels,
    }

    example_keyframe = {
        "voxels": keyframe_voxels,
        "num_points": keyframe_num_points,
        "points": keyframe_points,
        "coordinates": keyframe_coordinates,
        "num_voxels": keyframe_num_voxels,
    }

    if training:
        example["gt_boxes"] = gt_dict["gt_boxes"]

    if calib is not None:
        example["calib"] = calib

    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]

    if anchor_cache is not None:
        anchorss = anchor_cache["anchors"]
        anchors_bvs = anchor_cache["anchors_bv"]
        anchors_dicts = anchor_cache["anchors_dict"]
    else:
        rets = [
            target_assigner.generate_anchors(feature_map_size)
            for target_assigner in target_assigners
        ]
        anchorss = [ret["anchors"].reshape([-1, 7]) for ret in rets]
        anchors_dicts = [
            target_assigner.generate_anchors_dict(feature_map_size)
            for target_assigner in target_assigners
        ]
        anchors_bvs = [
            box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
            for anchors in anchorss
        ]

    example["anchors"] = anchorss

    if anchor_area_threshold >= 0:
        example["anchors_mask"] = []
        for idx, anchors_bv in enumerate(anchors_bvs):
            anchors_mask = None
            # slow with high resolution. recommend disable this forever.
            coors = coordinates
            dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
                coors, tuple(grid_size[::-1][1:])
            )
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            anchors_area = box_np_ops.fused_get_anchors_area(
                dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size
            )
            anchors_mask = anchors_area > anchor_area_threshold
            # example['anchors_mask'] = anchors_mask.astype(np.uint8)
            example["anchors_mask"].append(anchors_mask)

    example_sequences = {}
    example_sequences["current_frame"] = example
    example_sequences["keyframe"] = example_keyframe

    if not training:
        return example_sequences

    # voxel_labels = box_np_ops.assign_label_to_voxel(gt_boxes, coordinates,
    #                                                 voxel_size, coors_range)
    """
    example.update({
        'gt_boxes': gt_boxes.astype(out_dtype),
        'num_gt': np.array([gt_boxes.shape[0]]),
        # 'voxel_labels': voxel_labels,
    })
    """
    if create_targets:
        targets_dicts = []
        for idx, target_assigner in enumerate(target_assigners):
            if "anchors_mask" in example:
                anchors_mask = example["anchors_mask"][idx]
            else:
                anchors_mask = None
            targets_dict = target_assigner.assign_v2(
                anchors_dicts[idx],
                gt_dict["gt_boxes"][idx],
                anchors_mask,
                gt_classes=gt_dict["gt_classes"][idx],
                gt_names=gt_dict["gt_names"][idx],
            )
            targets_dicts.append(targets_dict)

        example_sequences["current_frame"].update(
            {
                "labels": [targets_dict["labels"] for targets_dict in targets_dicts],
                "reg_targets": [
                    targets_dict["bbox_targets"] for targets_dict in targets_dicts
                ],
                "reg_weights": [
                    targets_dict["bbox_outside_weights"]
                    for targets_dict in targets_dicts
                ],
            }
        )
    return example_sequences


def prep_pointcloud_rpn(
    input_dict,
    root_path,
    task_class_names=[],
    prep_cfg=None,
    db_sampler=None,
    remove_outside_points=False,
    training=True,
    num_point_features=4,
    random_crop=False,
    reference_detections=None,
    out_dtype=np.float32,
    min_points_in_gt=-1,
    logger=None,
):
    """
    convert point cloud to voxels, create targets if ground truths exists.
    input_dict format: dataset.get_sensor_data format
    """
    assert prep_cfg is not None

    remove_environment = prep_cfg.REMOVE_UNKOWN_EXAMPLES

    if training:
        remove_unknown = prep_cfg.REMOVE_UNKOWN_EXAMPLES
        gt_rotation_noise = prep_cfg.GT_ROT_NOISE
        gt_loc_noise_std = prep_cfg.GT_LOC_NOISE
        global_rotation_noise = prep_cfg.GLOBAL_ROT_NOISE
        global_scaling_noise = prep_cfg.GLOBAL_SCALE_NOISE
        global_random_rot_range = prep_cfg.GLOBAL_ROT_PER_OBJ_RANGE
        global_translate_noise_std = prep_cfg.GLOBAL_TRANS_NOISE
        gt_points_drop = prep_cfg.GT_DROP_PERCENTAGE
        gt_drop_max_keep = prep_cfg.GT_DROP_MAX_KEEP_POINTS
        remove_points_after_sample = prep_cfg.REMOVE_POINTS_AFTER_SAMPLE

    class_names = list(itertools.chain(*task_class_names))

    # points_only = input_dict["lidar"]["points"]
    # times = input_dict["lidar"]["times"]
    # points = np.hstack([points_only, times])
    points = input_dict["lidar"]["points"]

    if training:
        anno_dict = input_dict["lidar"]["annotations"]
        gt_dict = {
            "gt_boxes": anno_dict["boxes"],
            "gt_names": np.array(anno_dict["names"]).reshape(-1),
        }

        if "difficulty" not in anno_dict:
            difficulty = np.zeros([anno_dict["boxes"].shape[0]], dtype=np.int32)
            gt_dict["difficulty"] = difficulty
        else:
            gt_dict["difficulty"] = anno_dict["difficulty"]
        # if use_group_id and "group_ids" in anno_dict:
        #     group_ids = anno_dict["group_ids"]
        #     gt_dict["group_ids"] = group_ids

    calib = None
    if "calib" in input_dict:
        calib = input_dict["calib"]

    if reference_detections is not None:
        assert calib is not None and "image" in input_dict
        C, R, T = box_np_ops.projection_matrix_to_CRT_kitti(P2)
        frustums = box_np_ops.get_frustum_v2(reference_detections, C)
        frustums -= T
        frustums = np.einsum("ij, akj->aki", np.linalg.inv(R), frustums)
        frustums = box_np_ops.camera_to_lidar(frustums, rect, Trv2c)
        surfaces = box_np_ops.corner_to_surfaces_3d_jit(frustums)
        masks = points_in_convex_polygon_3d_jit(points, surfaces)
        points = points[masks.any(-1)]

    if remove_outside_points:
        assert calib is not None
        image_shape = input_dict["image"]["image_shape"]
        points = box_np_ops.remove_outside_points(
            points, calib["rect"], calib["Trv2c"], calib["P2"], image_shape
        )
    if remove_environment is True and training:
        selected = kitti.keep_arrays_by_name(gt_names, target_assigner.classes)
        _dict_select(gt_dict, selected)
        masks = box_np_ops.points_in_rbbox(points, gt_dict["gt_boxes"])
        points = points[masks.any(-1)]

    if training:
        selected = kitti.drop_arrays_by_name(
            gt_dict["gt_names"], ["DontCare", "ignore"]
        )
        _dict_select(gt_dict, selected)
        if remove_unknown:
            remove_mask = gt_dict["difficulty"] == -1
            """
            gt_boxes_remove = gt_boxes[remove_mask]
            gt_boxes_remove[:, 3:6] += 0.25
            points = prep.remove_points_in_boxes(points, gt_boxes_remove)
            """
            keep_mask = np.logical_not(remove_mask)
            _dict_select(gt_dict, keep_mask)
        gt_dict.pop("difficulty")

        gt_boxes_mask = np.array(
            [n in class_names for n in gt_dict["gt_names"]], dtype=np.bool_
        )

        # db_sampler = None
        if db_sampler is not None:
            group_ids = None
            # if "group_ids" in gt_dict:
            #     group_ids = gt_dict["group_ids"]
            sampled_dict = db_sampler.sample_all(
                root_path,
                gt_dict["gt_boxes"],
                gt_dict["gt_names"],
                num_point_features,
                random_crop,
                gt_group_ids=group_ids,
                calib=calib,
            )

            if sampled_dict is not None:
                sampled_gt_names = sampled_dict["gt_names"]
                sampled_gt_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                sampled_gt_masks = sampled_dict["gt_masks"]
                gt_dict["gt_names"] = np.concatenate(
                    [gt_dict["gt_names"], sampled_gt_names], axis=0
                )
                gt_dict["gt_boxes"] = np.concatenate(
                    [gt_dict["gt_boxes"], sampled_gt_boxes]
                )
                gt_boxes_mask = np.concatenate(
                    [gt_boxes_mask, sampled_gt_masks], axis=0
                )

                # if group_ids is not None:
                #     sampled_group_ids = sampled_dict["group_ids"]
                #     gt_dict["group_ids"] = np.concatenate(
                #         [gt_dict["group_ids"], sampled_group_ids])

                if remove_points_after_sample:
                    masks = box_np_ops.points_in_rbbox(points, sampled_gt_boxes)
                    points = points[np.logical_not(masks.any(-1))]

                points = np.concatenate([sampled_points, points], axis=0)

        # group_ids = None
        # if "group_ids" in gt_dict:
        #     group_ids = gt_dict["group_ids"]

        prep.noise_per_object_v3_(
            gt_dict["gt_boxes"],
            points,
            gt_boxes_mask,
            rotation_perturb=gt_rotation_noise,
            center_noise_std=gt_loc_noise_std,
            global_random_rot_range=global_random_rot_range,
            group_ids=None,
            num_try=100,
        )

        # should remove unrelated objects after noise per object
        # for k, v in gt_dict.items():
        #     print(k, v.shape)

        _dict_select(gt_dict, gt_boxes_mask)

        gt_classes = np.array(
            [class_names.index(n) + 1 for n in gt_dict["gt_names"]], dtype=np.int32
        )
        gt_dict["gt_classes"] = gt_classes

        gt_dict["gt_boxes"], points = prep.random_flip(gt_dict["gt_boxes"], points)
        gt_dict["gt_boxes"], points = prep.global_rotation(
            gt_dict["gt_boxes"], points, rotation=global_rotation_noise
        )
        gt_dict["gt_boxes"], points = prep.global_scaling_v2(
            gt_dict["gt_boxes"], points, *global_scaling_noise
        )
        prep.global_translate_(gt_dict["gt_boxes"], points, global_translate_noise_std)

        task_masks = []
        flag = 0
        for class_name in task_class_names:
            task_masks.append(
                [
                    np.where(gt_dict["gt_classes"] == class_name.index(i) + 1 + flag)
                    for i in class_name
                ]
            )
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        task_names = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            task_name = []
            for m in mask:
                task_box.append(gt_dict["gt_boxes"][m])
                task_class.append(gt_dict["gt_classes"][m] - flag2)
                task_name.append(gt_dict["gt_names"][m])
            task_boxes.append(np.concatenate(task_box, axis=0))
            task_classes.append(np.concatenate(task_class))
            task_names.append(np.concatenate(task_name))
            flag2 += len(mask)

        for task_box in task_boxes:
            # limit rad to [-pi, pi]
            task_box[:, -1] = box_np_ops.limit_period(
                task_box[:, -1], offset=0.5, period=2 * np.pi
            )

        # print(gt_dict.keys())
        gt_dict["gt_classes"] = task_classes
        gt_dict["gt_names"] = task_names
        gt_dict["gt_boxes"] = task_boxes

        example = {
            "pts_input": points,
            "pts_rect": None,
            "pts_features": None,
            "gt_boxes3d": gt_dict["gt_boxes"],
            "rpn_cls_label": [],
            "rpn_reg_label": [],
        }

        if calib is not None:
            example["calib"] = calib

        return example

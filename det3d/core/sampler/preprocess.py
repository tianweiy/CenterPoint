import abc
import sys
import time
from collections import OrderedDict
from functools import reduce

import numba
import numpy as np

from det3d.core.bbox import box_np_ops
from det3d.core.bbox.geometry import (
    is_line_segment_intersection_jit,
    points_in_convex_polygon_3d_jit,
    points_in_convex_polygon_jit,
)
import copy


class BatchSampler:
    def __init__(
        self, sampled_list, name=None, epoch=None, shuffle=True, drop_reminder=False
    ):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx :].copy()
            self._reset()
        else:
            ret = self._indices[self._idx : self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        # if self._name is not None:
        #     print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]
        # return np.random.choice(self._sampled_list, num)


class DataBasePreprocessing:
    def __call__(self, db_infos):
        return self._preprocess(db_infos)

    @abc.abstractclassmethod
    def _preprocess(self, db_infos):
        pass


class DBFilterByDifficulty(DataBasePreprocessing):
    def __init__(self, removed_difficulties, logger=None):
        self._removed_difficulties = removed_difficulties
        logger.info(f"{removed_difficulties}")

    def _preprocess(self, db_infos):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info
                for info in dinfos
                if info["difficulty"] not in self._removed_difficulties
            ]
        return new_db_infos


class DBFilterByMinNumPoint(DataBasePreprocessing):
    def __init__(self, min_gt_point_dict, logger=None):
        self._min_gt_point_dict = min_gt_point_dict
        logger.info(f"{min_gt_point_dict}")

    def _preprocess(self, db_infos):
        for name, min_num in self._min_gt_point_dict.items():
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info["num_points_in_gt"] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos


class DataBasePreprocessor:
    def __init__(self, preprocessors):
        self._preprocessors = preprocessors

    def __call__(self, db_infos):
        for prepor in self._preprocessors:
            db_infos = prepor(db_infos)
        return db_infos


def filter_gt_box_outside_range(gt_boxes, limit_range):
    """remove gtbox outside training range.
    this function should be applied after other prep functions
    Args:
        gt_boxes ([type]): [description]
        limit_range ([type]): [description]
    """
    gt_boxes_bv = box_np_ops.center_to_corner_box2d(
        gt_boxes[:, [0, 1]], gt_boxes[:, [3, 3 + 1]], gt_boxes[:, -1]
    )
    bounding_box = box_np_ops.minmax_to_corner_2d(
        np.asarray(limit_range)[np.newaxis, ...]
    )
    ret = points_in_convex_polygon_jit(gt_boxes_bv.reshape(-1, 2), bounding_box)
    return np.any(ret.reshape(-1, 4), axis=1)


def filter_gt_box_outside_range_by_center(gt_boxes, limit_range):
    """remove gtbox outside training range.
    this function should be applied after other prep functions
    Args:
        gt_boxes ([type]): [description]
        limit_range ([type]): [description]
    """
    gt_box_centers = gt_boxes[:, :2]
    bounding_box = box_np_ops.minmax_to_corner_2d(
        np.asarray(limit_range)[np.newaxis, ...]
    )
    ret = points_in_convex_polygon_jit(gt_box_centers, bounding_box)
    return ret.reshape(-1)


def filter_gt_low_points(gt_boxes, points, num_gt_points, point_num_threshold=2):
    points_mask = np.ones([points.shape[0]], np.bool)
    gt_boxes_mask = np.ones([gt_boxes.shape[0]], np.bool)
    for i, num in enumerate(num_gt_points):
        if num <= point_num_threshold:
            masks = box_np_ops.points_in_rbbox(points, gt_boxes[i : i + 1])
            masks = masks.reshape([-1])
            points_mask &= np.logical_not(masks)
            gt_boxes_mask[i] = False
    return gt_boxes[gt_boxes_mask], points[points_mask]


def mask_points_in_corners(points, box_corners):
    surfaces = box_np_ops.corner_to_surfaces_3d(box_corners)
    mask = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return mask


@numba.njit
def _rotation_matrix_3d_(rot_mat_T, angle, axis):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = -rot_sin
        rot_mat_T[2, 0] = rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = -rot_sin
        rot_mat_T[2, 1] = rot_sin
        rot_mat_T[2, 2] = rot_cos


@numba.njit
def _rotation_box2d_jit_(corners, angle, rot_mat_T):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[0, 0] = rot_cos
    rot_mat_T[0, 1] = -rot_sin
    rot_mat_T[1, 0] = rot_sin
    rot_mat_T[1, 1] = rot_cos
    corners[:] = corners @ rot_mat_T


@numba.jit(nopython=True)
def _box_single_to_corner_jit(boxes):
    num_box = boxes.shape[0]
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners = boxes.reshape(num_box, 1, 5)[:, :, 2:4] * corners_norm.reshape(1, 4, 2)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    box_corners = np.zeros((num_box, 4, 2), dtype=boxes.dtype)
    for i in range(num_box):
        rot_sin = np.sin(boxes[i, -1])
        rot_cos = np.cos(boxes[i, -1])
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
        box_corners[i] = corners[i] @ rot_mat_T + boxes[i, :2]
    return box_corners


@numba.njit
def noise_per_box(boxes, valid_mask, loc_noises, rot_noises):
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    # print(valid_mask)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_corners[:] = box_corners[i]
                current_corners -= boxes[i, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j], rot_mat_T)
                current_corners += boxes[i, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners
                )
                coll_mat[0, i] = False
                # print(coll_mat)
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    break
    return success_mask


@numba.njit
def noise_per_box_group(boxes, valid_mask, loc_noises, rot_noises, group_nums):
    # WARNING: this function need boxes to be sorted by group id.
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_groups = group_nums.shape[0]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    max_group_num = group_nums.max()
    current_corners = np.zeros((max_group_num, 4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    # print(valid_mask)
    idx = 0
    for num in group_nums:
        if valid_mask[idx]:
            for j in range(num_tests):
                for i in range(num):
                    current_corners[i] = box_corners[i + idx]
                    current_corners[i] -= boxes[i + idx, :2]
                    _rotation_box2d_jit_(
                        current_corners[i], rot_noises[idx + i, j], rot_mat_T
                    )
                    current_corners[i] += (
                        boxes[i + idx, :2] + loc_noises[i + idx, j, :2]
                    )
                coll_mat = box_collision_test(
                    current_corners[:num].reshape(num, 4, 2), box_corners
                )
                for i in range(num):  # remove self-coll
                    coll_mat[i, idx : idx + num] = False
                if not coll_mat.any():
                    for i in range(num):
                        success_mask[i + idx] = j
                        box_corners[i + idx] = current_corners[i]
                    break
        idx += num
    return success_mask


@numba.njit
def noise_per_box_group_v2_(
    boxes, valid_mask, loc_noises, rot_noises, group_nums, global_rot_noises
):
    # WARNING: this function need boxes to be sorted by group id.
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    max_group_num = group_nums.max()
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    current_corners = np.zeros((max_group_num, 4, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((max_group_num, 2), dtype=boxes.dtype)

    current_grot = np.zeros((max_group_num,), dtype=boxes.dtype)
    dst_grot = np.zeros((max_group_num,), dtype=boxes.dtype)

    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)

    # print(valid_mask)
    idx = 0
    for num in group_nums:
        if valid_mask[idx]:
            for j in range(num_tests):
                for i in range(num):
                    current_box[0, :] = boxes[i + idx]
                    current_radius = np.sqrt(
                        current_box[0, 0] ** 2 + current_box[0, 1] ** 2
                    )
                    current_grot[i] = np.arctan2(current_box[0, 0], current_box[0, 1])
                    dst_grot[i] = current_grot[i] + global_rot_noises[idx + i, j]
                    dst_pos[i, 0] = current_radius * np.sin(dst_grot[i])
                    dst_pos[i, 1] = current_radius * np.cos(dst_grot[i])
                    current_box[0, :2] = dst_pos[i]
                    current_box[0, -1] += dst_grot[i] - current_grot[i]

                    rot_sin = np.sin(current_box[0, -1])
                    rot_cos = np.cos(current_box[0, -1])
                    rot_mat_T[0, 0] = rot_cos
                    rot_mat_T[0, 1] = -rot_sin
                    rot_mat_T[1, 0] = rot_sin
                    rot_mat_T[1, 1] = rot_cos
                    current_corners[i] = (
                        current_box[0, 2:4] * corners_norm @ rot_mat_T
                        + current_box[0, :2]
                    )
                    current_corners[i] -= current_box[0, :2]

                    _rotation_box2d_jit_(
                        current_corners[i], rot_noises[idx + i, j], rot_mat_T
                    )
                    current_corners[i] += (
                        current_box[0, :2] + loc_noises[i + idx, j, :2]
                    )
                coll_mat = box_collision_test(
                    current_corners[:num].reshape(num, 4, 2), box_corners
                )
                for i in range(num):  # remove self-coll
                    coll_mat[i, idx : idx + num] = False
                if not coll_mat.any():
                    for i in range(num):
                        success_mask[i + idx] = j
                        box_corners[i + idx] = current_corners[i]
                        loc_noises[i + idx, j, :2] += dst_pos[i] - boxes[i + idx, :2]
                        rot_noises[i + idx, j] += dst_grot[i] - current_grot[i]
                    break
        idx += num
    return success_mask


@numba.njit
def noise_per_box_v2_(boxes, valid_mask, loc_noises, rot_noises, global_rot_noises):
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((2,), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_box[0, :] = boxes[i]
                current_radius = np.sqrt(boxes[i, 0] ** 2 + boxes[i, 1] ** 2)
                current_grot = np.arctan2(boxes[i, 0], boxes[i, 1])
                dst_grot = current_grot + global_rot_noises[i, j]
                dst_pos[0] = current_radius * np.sin(dst_grot)
                dst_pos[1] = current_radius * np.cos(dst_grot)
                current_box[0, :2] = dst_pos
                current_box[0, -1] += dst_grot - current_grot

                rot_sin = np.sin(current_box[0, -1])
                rot_cos = np.cos(current_box[0, -1])
                rot_mat_T[0, 0] = rot_cos
                rot_mat_T[0, 1] = -rot_sin
                rot_mat_T[1, 0] = rot_sin
                rot_mat_T[1, 1] = rot_cos
                current_corners[:] = (
                    current_box[0, 2:4] * corners_norm @ rot_mat_T + current_box[0, :2]
                )
                current_corners -= current_box[0, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j], rot_mat_T)
                current_corners += current_box[0, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners
                )
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    loc_noises[i, j, :2] += dst_pos - boxes[i, :2]
                    rot_noises[i, j] += dst_grot - current_grot
                    break
    return success_mask


@numba.njit
def points_transform_(
    points, centers, point_masks, loc_transform, rot_transform, valid_mask
):
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        _rotation_matrix_3d_(rot_mat_T[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    points[i, :3] -= centers[j, :3]
                    points[i : i + 1, :3] = points[i : i + 1, :3] @ rot_mat_T[j]
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break  # only apply first box's transform


@numba.njit
def box3d_transform_(boxes, loc_transform, rot_transform, valid_mask):
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]


def _select_transform(transform, indices):
    result = np.zeros((transform.shape[0], *transform.shape[2:]), dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result


@numba.njit
def group_transform_(loc_noise, rot_noise, locs, rots, group_center, valid_mask):
    # loc_noise: [N, M, 3], locs: [N, 3]
    # rot_noise: [N, M]
    # group_center: [N, 3]
    num_try = loc_noise.shape[1]
    r = 0.0
    x = 0.0
    y = 0.0
    rot_center = 0.0
    for i in range(loc_noise.shape[0]):
        if valid_mask[i]:
            x = locs[i, 0] - group_center[i, 0]
            y = locs[i, 1] - group_center[i, 1]
            r = np.sqrt(x ** 2 + y ** 2)
            # calculate rots related to group center
            rot_center = np.arctan2(x, y)
            for j in range(num_try):
                loc_noise[i, j, 0] += r * (
                    np.sin(rot_center + rot_noise[i, j]) - np.sin(rot_center)
                )
                loc_noise[i, j, 1] += r * (
                    np.cos(rot_center + rot_noise[i, j]) - np.cos(rot_center)
                )


@numba.njit
def group_transform_v2_(
    loc_noise, rot_noise, locs, rots, group_center, grot_noise, valid_mask
):
    # loc_noise: [N, M, 3], locs: [N, 3]
    # rot_noise: [N, M]
    # group_center: [N, 3]
    num_try = loc_noise.shape[1]
    r = 0.0
    x = 0.0
    y = 0.0
    rot_center = 0.0
    for i in range(loc_noise.shape[0]):
        if valid_mask[i]:
            x = locs[i, 0] - group_center[i, 0]
            y = locs[i, 1] - group_center[i, 1]
            r = np.sqrt(x ** 2 + y ** 2)
            # calculate rots related to group center
            rot_center = np.arctan2(x, y)
            for j in range(num_try):
                loc_noise[i, j, 0] += r * (
                    np.sin(rot_center + rot_noise[i, j] + grot_noise[i, j])
                    - np.sin(rot_center + grot_noise[i, j])
                )
                loc_noise[i, j, 1] += r * (
                    np.cos(rot_center + rot_noise[i, j] + grot_noise[i, j])
                    - np.cos(rot_center + grot_noise[i, j])
                )


def set_group_noise_same_(loc_noise, rot_noise, group_ids):
    gid_to_index_dict = {}
    for i, gid in enumerate(group_ids):
        if gid not in gid_to_index_dict:
            gid_to_index_dict[gid] = i
    for i in range(loc_noise.shape[0]):
        loc_noise[i] = loc_noise[gid_to_index_dict[group_ids[i]]]
        rot_noise[i] = rot_noise[gid_to_index_dict[group_ids[i]]]


def set_group_noise_same_v2_(loc_noise, rot_noise, grot_noise, group_ids):
    gid_to_index_dict = {}
    for i, gid in enumerate(group_ids):
        if gid not in gid_to_index_dict:
            gid_to_index_dict[gid] = i
    for i in range(loc_noise.shape[0]):
        loc_noise[i] = loc_noise[gid_to_index_dict[group_ids[i]]]
        rot_noise[i] = rot_noise[gid_to_index_dict[group_ids[i]]]
        grot_noise[i] = grot_noise[gid_to_index_dict[group_ids[i]]]


def get_group_center(locs, group_ids):
    num_groups = 0
    group_centers = np.zeros_like(locs)
    group_centers_ret = np.zeros_like(locs)
    group_id_dict = {}
    group_id_num_dict = OrderedDict()
    for i, gid in enumerate(group_ids):
        if gid >= 0:
            if gid in group_id_dict:
                group_centers[group_id_dict[gid]] += locs[i]
                group_id_num_dict[gid] += 1
            else:
                group_id_dict[gid] = num_groups
                num_groups += 1
                group_id_num_dict[gid] = 1
                group_centers[group_id_dict[gid]] = locs[i]
    for i, gid in enumerate(group_ids):
        group_centers_ret[i] = (
            group_centers[group_id_dict[gid]] / group_id_num_dict[gid]
        )
    return group_centers_ret, group_id_num_dict


def noise_per_object_v3_(
    gt_boxes,
    points=None,
    valid_mask=None,
    rotation_perturb=np.pi / 4,
    center_noise_std=1.0,
    global_random_rot_range=np.pi / 4,
    num_try=5,
    group_ids=None,
):
    """random rotate or remove each groundtrutn independently.
    use kitti viewer to test this function points_transform_

    Args:
        gt_boxes: [N, 7], gt box in lidar.points_transform_
        points: [M, 4], point cloud in lidar.
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [-global_random_rot_range, global_random_rot_range]
    enable_grot = (
        np.abs(global_random_rot_range[0] - global_random_rot_range[1]) >= 1e-3
    )
    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [center_noise_std, center_noise_std, center_noise_std]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes,), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(scale=center_noise_std, size=[num_boxes, num_try, 3])
    # loc_noises = np.random.uniform(
    #     -center_noise_std, center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(
        rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try]
    )
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(
        grot_lowers[..., np.newaxis],
        grot_uppers[..., np.newaxis],
        size=[num_boxes, num_try],
    )
    if group_ids is not None:
        if enable_grot:
            set_group_noise_same_v2_(
                loc_noises, rot_noises, global_rot_noises, group_ids
            )
        else:
            set_group_noise_same_(loc_noises, rot_noises, group_ids)
        group_centers, group_id_num_dict = get_group_center(gt_boxes[:, :3], group_ids)
        if enable_grot:
            group_transform_v2_(
                loc_noises,
                rot_noises,
                gt_boxes[:, :3],
                gt_boxes[:, 6],
                group_centers,
                global_rot_noises,
                valid_mask,
            )
        else:
            group_transform_(
                loc_noises,
                rot_noises,
                gt_boxes[:, :3],
                gt_boxes[:, 6],
                group_centers,
                valid_mask,
            )
        group_nums = np.array(list(group_id_num_dict.values()), dtype=np.int64)

    origin = [0.5, 0.5, 0.5]
    gt_box_corners = box_np_ops.center_to_corner_box3d(
        gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6], origin=origin, axis=2
    )
    if group_ids is not None:
        if not enable_grot:
            selected_noise = noise_per_box_group(
                gt_boxes[:, [0, 1, 3, 4, 6]],
                valid_mask,
                loc_noises,
                rot_noises,
                group_nums,
            )
        else:
            selected_noise = noise_per_box_group_v2_(
                gt_boxes[:, [0, 1, 3, 4, 6]],
                valid_mask,
                loc_noises,
                rot_noises,
                group_nums,
                global_rot_noises,
            )
    else:
        if not enable_grot:
            selected_noise = noise_per_box(
                gt_boxes[:, [0, 1, 3, 4, 6]], valid_mask, loc_noises, rot_noises
            )
        else:
            selected_noise = noise_per_box_v2_(
                gt_boxes[:, [0, 1, 3, 4, 6]],
                valid_mask,
                loc_noises,
                rot_noises,
                global_rot_noises,
            )
    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
    if points is not None:
        point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(
            points,
            gt_boxes[:, :3],
            point_masks,
            loc_transforms,
            rot_transforms,
            valid_mask,
        )

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)


def noise_per_object_v2_(
    gt_boxes,
    points=None,
    valid_mask=None,
    rotation_perturb=np.pi / 4,
    center_noise_std=1.0,
    global_random_rot_range=np.pi / 4,
    num_try=100,
):
    """random rotate or remove each groundtrutn independently.
    use kitti viewer to test this function points_transform_

    Args:
        gt_boxes: [N, 7], gt box in lidar.points_transform_
        points: [M, 4], point cloud in lidar.
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [-global_random_rot_range, global_random_rot_range]

    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [center_noise_std, center_noise_std, center_noise_std]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes,), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(scale=center_noise_std, size=[num_boxes, num_try, 3])
    # loc_noises = np.random.uniform(
    #     -center_noise_std, center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(
        rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try]
    )
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(
        grot_lowers[..., np.newaxis],
        grot_uppers[..., np.newaxis],
        size=[num_boxes, num_try],
    )

    origin = [0.5, 0.5, 0]
    gt_box_corners = box_np_ops.center_to_corner_box3d(
        gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6], origin=origin, axis=2
    )
    if np.abs(global_random_rot_range[0] - global_random_rot_range[1]) < 1e-3:
        selected_noise = noise_per_box(
            gt_boxes[:, [0, 1, 3, 4, 6]], valid_mask, loc_noises, rot_noises
        )
    else:
        selected_noise = noise_per_box_v2_(
            gt_boxes[:, [0, 1, 3, 4, 6]],
            valid_mask,
            loc_noises,
            rot_noises,
            global_rot_noises,
        )
    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    if points is not None:
        surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
        point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(
            points,
            gt_boxes[:, :3],
            point_masks,
            loc_transforms,
            rot_transforms,
            valid_mask,
        )

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)


def global_scaling(gt_boxes, points, scale=0.05):
    if not isinstance(scale, list):
        scale = [-scale, scale]
    noise_scale = np.random.uniform(scale[0] + 1, scale[1] + 1)
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points


def global_rotation(gt_boxes, points, rotation=np.pi / 4):
    if not isinstance(rotation, list):
        rotation = [-rotation, rotation]
    noise_rotation = np.random.uniform(rotation[0], rotation[1])
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2
    )
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2
    )
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 6:8] = box_np_ops.rotation_points_single_angle(
            np.hstack([gt_boxes[:, 6:8], np.zeros((gt_boxes.shape[0], 1))]),
            noise_rotation,
            axis=2,
        )[:, :2]
    gt_boxes[:, -1] += noise_rotation
    return gt_boxes, points


def random_flip(gt_boxes, points, probability=0.5):
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, -1] = -gt_boxes[:, -1] + np.pi
        points[:, 1] = -points[:, 1]
        if gt_boxes.shape[1] > 7:  # y axis: x, y, z, w, h, l, vx, vy, r
            gt_boxes[:, 7] = -gt_boxes[:, 7]
    return gt_boxes, points

def random_flip_both(gt_boxes, points, probability=0.5, flip_coor=None):
    # x flip 
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, -1] = -gt_boxes[:, -1] + np.pi
        points[:, 1] = -points[:, 1]
        if gt_boxes.shape[1] > 7:  # y axis: x, y, z, w, h, l, vx, vy, r
            gt_boxes[:, 7] = -gt_boxes[:, 7]
    
    # y flip 
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        if flip_coor is None:
            gt_boxes[:, 0] = -gt_boxes[:, 0]
            points[:, 0] = -points[:, 0]
        else:
            gt_boxes[:, 0] = flip_coor * 2 - gt_boxes[:, 0]
            points[:, 0] = flip_coor * 2 - points[:, 0]

        gt_boxes[:, -1] = -gt_boxes[:, -1] + 2*np.pi  # TODO: CHECK THIS 
        
        if gt_boxes.shape[1] > 7:  # y axis: x, y, z, w, h, l, vx, vy, r
            gt_boxes[:, 6] = -gt_boxes[:, 6]
    
    return gt_boxes, points


def global_scaling_v2(gt_boxes, points, min_scale=0.95, max_scale=1.05):
    noise_scale = np.random.uniform(min_scale, max_scale)
    points[:, :3] *= noise_scale
    gt_boxes[:, :-1] *= noise_scale
    return gt_boxes, points


def global_rotation_v2(gt_boxes, points, min_rad=-np.pi / 4, max_rad=np.pi / 4):
    noise_rotation = np.random.uniform(min_rad, max_rad)
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2
    )
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2
    )
    gt_boxes[:, -1] += noise_rotation
    return gt_boxes, points


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack(
        (boxes, boxes[:, slices, :]), axis=2
    )  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = box_np_ops.corner_to_standup_nd_jit(boxes)
    qboxes_standup = box_np_ops.corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]
            )
            if iw > 0:
                ih = min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1]
                )
                if ih > 0:
                    for k in range(4):
                        for l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, l, 0]
                            D = lines_qboxes[j, l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]) * (
                                D[0] - A[0]
                            )
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (
                                D[0] - B[0]
                            )
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (
                                    C[0] - A[0]
                                )
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (
                                    D[0] - A[0]
                                )
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (boxes[i, k, 0] - qboxes[j, l, 0])
                                cross -= vec[0] * (boxes[i, k, 1] - qboxes[j, l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for l in range(4):  # point l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (qboxes[j, k, 0] - boxes[i, l, 0])
                                    cross -= vec[0] * (qboxes[j, k, 1] - boxes[i, l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret


def global_translate_(gt_boxes, points, noise_translate_std):
    """
    Apply global translation to gt_boxes and points.
    """

    if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
        noise_translate_std = np.array(
            [noise_translate_std, noise_translate_std, noise_translate_std]
        )
    if all([e == 0 for e in noise_translate_std]):
        return gt_boxes, points
    noise_translate = np.array(
        [
            np.random.normal(0, noise_translate_std[0], 1),
            np.random.normal(0, noise_translate_std[1], 1),
            np.random.normal(0, noise_translate_std[0], 1),
        ]
    ).T

    points[:, :3] += noise_translate
    gt_boxes[:, :3] += noise_translate

    return gt_boxes, points


if __name__ == "__main__":
    bboxes = np.array(
        [
            [0.0, 0.0, 0.5, 0.5],
            [0.2, 0.2, 0.6, 0.6],
            [0.7, 0.7, 0.9, 0.9],
            [0.55, 0.55, 0.8, 0.8],
        ]
    )
    bbox_corners = box_np_ops.minmax_to_corner_2d(bboxes)
    print(bbox_corners.shape)
    print(box_collision_test(bbox_corners, bbox_corners))

import cv2
import numba
import numpy as np
from det3d.core import box_np_ops


@numba.jit(nopython=True)
def _points_to_bevmap_reverse_kernel(
    points,
    voxel_size,
    coors_range,
    coor_to_voxelidx,
    # coors_2d,
    bev_map,
    height_lowers,
    # density_norm_num=16,
    with_reflectivity=False,
    max_voxels=40000,
):
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    height_slice_size = voxel_size[-1]
    coor = np.zeros(shape=(3,), dtype=np.int32)  # DHW
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            # coors_2d[voxelidx] = coor[1:]
        bev_map[-1, coor[1], coor[2]] += 1
        height_norm = bev_map[coor[0], coor[1], coor[2]]
        incomimg_height_norm = (
            points[i, 2] - height_lowers[coor[0]]
        ) / height_slice_size
        if incomimg_height_norm > height_norm:
            bev_map[coor[0], coor[1], coor[2]] = incomimg_height_norm
            if with_reflectivity:
                bev_map[-2, coor[1], coor[2]] = points[i, 3]
    # return voxel_num


def points_to_bev(
    points,
    voxel_size,
    coors_range,
    with_reflectivity=False,
    density_norm_num=16,
    max_voxels=40000,
):
    """convert kitti points(N, 4) to a bev map. return [C, H, W] map.
    this function based on algorithm in points_to_voxel.
    takes 5ms in a reduced pointcloud with voxel_size=[0.1, 0.1, 0.8]

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3] contain reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        with_reflectivity: bool. if True, will add a intensity map to bev map.
    Returns:
        bev_map: [num_height_maps + 1(2), H, W] float tensor.
            `WARNING`: bev_map[-1] is num_points map, NOT density map,
            because calculate density map need more time in cpu rather than gpu.
            if with_reflectivity is True, bev_map[-2] is intensity map.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]  # DHW format
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    # coors_2d = np.zeros(shape=(max_voxels, 2), dtype=np.int32)
    bev_map_shape = list(voxelmap_shape)
    bev_map_shape[0] += 1
    height_lowers = np.linspace(
        coors_range[2], coors_range[5], voxelmap_shape[0], endpoint=False
    )
    if with_reflectivity:
        bev_map_shape[0] += 1
    bev_map = np.zeros(shape=bev_map_shape, dtype=points.dtype)
    _points_to_bevmap_reverse_kernel(
        points,
        voxel_size,
        coors_range,
        coor_to_voxelidx,
        bev_map,
        height_lowers,
        with_reflectivity,
        max_voxels,
    )
    # print(voxel_num)
    return bev_map


def point_to_vis_bev(points, voxel_size=None, coors_range=None, max_voxels=80000):
    if voxel_size is None:
        voxel_size = [0.1, 0.1, 0.1]
    if coors_range is None:
        coors_range = [-50, -50, -3, 50, 50, 1]
    voxel_size[2] = coors_range[5] - coors_range[2]
    bev_map = points_to_bev(points, voxel_size, coors_range, max_voxels=max_voxels)
    height_map = (bev_map[0] * 255).astype(np.uint8)
    return cv2.cvtColor(height_map, cv2.COLOR_GRAY2RGB)


def cv2_draw_lines(img, lines, colors, thickness, line_type=cv2.LINE_8):
    lines = lines.astype(np.int32)
    for line, color in zip(lines, colors):
        color = list(int(c) for c in color)
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)
    return img


def cv2_draw_text(img, locs, labels, colors, thickness, line_type=cv2.LINE_8):
    locs = locs.astype(np.int32)
    font_line_type = cv2.LINE_8
    font = cv2.FONT_ITALIC
    font = cv2.FONT_HERSHEY_DUPLEX
    font = cv2.FONT_HERSHEY_PLAIN
    font = cv2.FONT_HERSHEY_SIMPLEX
    for loc, label, color in zip(locs, labels, colors):
        color = list(int(c) for c in color)
        cv2.putText(
            img, label, tuple(loc), font, 0.7, color, thickness, font_line_type, False
        )
    return img


def draw_box_in_bev(
    img, coors_range, boxes, color, thickness=1, labels=None, label_color=None
):
    """
    Args:
        boxes: center format.
    """
    coors_range = np.array(coors_range)
    bev_corners = box_np_ops.center_to_corner_box2d(
        boxes[:, [0, 1]], boxes[:, [3, 4]], boxes[:, 6]
    )
    bev_corners -= coors_range[:2]
    bev_corners *= np.array(img.shape[:2])[::-1] / (coors_range[3:5] - coors_range[:2])
    standup = box_np_ops.corner_to_standup_nd(bev_corners)
    text_center = standup[:, 2:]
    text_center[:, 1] -= (standup[:, 3] - standup[:, 1]) / 2

    bev_lines = np.concatenate(
        [bev_corners[:, [0, 2, 3]], bev_corners[:, [1, 3, 0]]], axis=2
    )
    bev_lines = bev_lines.reshape(-1, 4)
    colors = np.tile(np.array(color).reshape(1, 3), [bev_lines.shape[0], 1])
    colors = colors.astype(np.int32)
    img = cv2_draw_lines(img, bev_lines, colors, thickness)
    if labels is not None:
        if label_color is None:
            label_color = colors
        else:
            label_color = np.tile(
                np.array(label_color).reshape(1, 3), [bev_lines.shape[0], 1]
            )
            label_color = label_color.astype(np.int32)

        img = cv2_draw_text(img, text_center, labels, label_color, thickness * 2)
    return img


def kitti_vis(points, boxes, labels=None):
    vis_voxel_size = [0.1, 0.1, 0.1]
    vis_point_range = [-10, -30, -3, 54, 30, 1]
    bev_map = point_to_vis_bev(points, vis_voxel_size, vis_point_range)
    bev_map = draw_box_in_bev(bev_map, vis_point_range, boxes, [0, 255, 0], 2, labels)

    return bev_map


def nuscene_vis(points, boxes, labels=None):
    vis_voxel_size = [0.1, 0.1, 0.1]
    vis_point_range = [-50, -50, -3, 50, 50, 1]
    bev_map = point_to_vis_bev(points, vis_voxel_size, vis_point_range)
    bev_map = draw_box_in_bev(bev_map, vis_point_range, boxes, [0, 255, 0], 2, labels)

    return bev_map

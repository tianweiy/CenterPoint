from pathlib import Path

import numba
import numpy as np
from det3d.core.bbox.geometry import (
    points_count_convex_polygon_3d_jit,
    points_in_convex_polygon_3d_jit,
)
try:
    from spconv.utils import rbbox_intersection, rbbox_iou
except:
    print("Import spconv fail, no support for sparse convolution!")


def points_count_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0.5)):
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin, axis=z_axis
    )
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    return points_count_convex_polygon_3d_jit(points[:, :3], surfaces)


def riou_cc(rbboxes, qrbboxes, standup_thresh=0.0):
    # less than 50ms when used in second one thread. 10x slower than gpu
    boxes_corners = center_to_corner_box2d(
        rbboxes[:, :2], rbboxes[:, 2:4], rbboxes[:, 4]
    )
    boxes_standup = corner_to_standup_nd(boxes_corners)
    qboxes_corners = center_to_corner_box2d(
        qrbboxes[:, :2], qrbboxes[:, 2:4], qrbboxes[:, 4]
    )
    qboxes_standup = corner_to_standup_nd(qboxes_corners)
    # if standup box not overlapped, rbbox not overlapped too.
    standup_iou = iou_jit(boxes_standup, qboxes_standup, eps=0.0)
    return rbbox_iou(boxes_corners, qboxes_corners, standup_iou, standup_thresh)


def rinter_cc(rbboxes, qrbboxes, standup_thresh=0.0):
    # less than 50ms when used in second one thread. 10x slower than gpu
    boxes_corners = center_to_corner_box2d(
        rbboxes[:, :2], rbboxes[:, 2:4], rbboxes[:, 4]
    )
    boxes_standup = corner_to_standup_nd(boxes_corners)
    qboxes_corners = center_to_corner_box2d(
        qrbboxes[:, :2], qrbboxes[:, 2:4], qrbboxes[:, 4]
    )
    qboxes_standup = corner_to_standup_nd(qboxes_corners)
    # if standup box not overlapped, rbbox not overlapped too.
    standup_iou = iou_jit(boxes_standup, qboxes_standup, eps=0.0)
    return rbbox_intersection(
        boxes_corners, qboxes_corners, standup_iou, standup_thresh
    )


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners


@numba.njit
def corners_2d_jit(dims, origin=0.5):
    ndim = 2
    corners_norm = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=dims.dtype)
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape((-1, 1, ndim)) * corners_norm.reshape((1, 2 ** ndim, ndim))
    return corners


@numba.njit
def corners_3d_jit(dims, origin=0.5):
    ndim = 3
    corners_norm = np.array(
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        dtype=dims.dtype,
    ).reshape((8, 3))
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape((-1, 1, ndim)) * corners_norm.reshape((1, 2 ** ndim, ndim))
    return corners


@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


def corner_to_standup_nd(boxes_corner):
    assert len(boxes_corner.shape) == 3
    standup_boxes = []
    standup_boxes.append(np.min(boxes_corner, axis=1))
    standup_boxes.append(np.max(boxes_corner, axis=1))
    return np.concatenate(standup_boxes, -1)


def rbbox2d_to_near_bbox(rbboxes):
    """convert rotated bbox to nearest 'standing' or 'lying' bbox.
    Args:
        rbboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        bboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = np.abs(limit_period(rots, 0.5, np.pi))
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    bboxes_center = np.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes


def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                [rot_cos, zeros, -rot_sin],
                [zeros, ones, zeros],
                [rot_sin, zeros, rot_cos],
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                [rot_cos, -rot_sin, zeros],
                [rot_sin, rot_cos, zeros],
                [zeros, zeros, ones],
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                [zeros, rot_cos, -rot_sin],
                [zeros, rot_sin, rot_cos],
                [ones, zeros, zeros],
            ]
        )
    else:
        raise ValueError("axis should in range")

    return np.einsum("aij,jka->aik", points, rot_mat_T)


def rotation_points_single_angle(points, angle, axis=0):
    # points: [N, 3]
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.array(
            [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
            dtype=points.dtype,
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.array(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
            dtype=points.dtype,
        )
    elif axis == 0:
        rot_mat_T = np.array(
            [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
            dtype=points.dtype,
        )
    else:
        raise ValueError("axis should in range")

    return points @ rot_mat_T


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def rotation_box(box_corners, angle):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angle (float): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T = np.array(
        [[rot_cos, -rot_sin], [rot_sin, rot_cos]], dtype=box_corners.dtype
    )
    return box_corners @ rot_mat_T


def center_to_corner_box3d(centers, dims, angles=None, origin=(0.5, 0.5, 0.5), axis=2):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


@numba.jit(nopython=True)
def box2d_to_corner_jit(boxes):
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


def rbbox3d_to_corners(rbboxes, origin=[0.5, 0.5, 0.5], axis=2):
    return center_to_corner_box3d(
        rbboxes[..., :3], rbboxes[..., 3:6], rbboxes[..., 6], origin, axis=axis
    )


def rbbox3d_to_bev_corners(rbboxes, origin=0.5):
    return center_to_corner_box2d(
        rbboxes[..., :2], rbboxes[..., 3:5], rbboxes[..., 6], origin
    )


def minmax_to_corner_2d(minmax_box):
    ndim = minmax_box.shape[-1] // 2
    center = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center
    return center_to_corner_box2d(center, dims, origin=0.0)


def minmax_to_corner_2d_v2(minmax_box):
    # N, 4 -> N 4 2
    return minmax_box[..., [0, 1, 0, 3, 2, 3, 2, 1]].reshape(-1, 4, 2)


def minmax_to_corner_3d(minmax_box):
    ndim = minmax_box.shape[-1] // 2
    center = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center
    return center_to_corner_box3d(center, dims, origin=0.0)


def minmax_to_center_2d(minmax_box):
    ndim = minmax_box.shape[-1] // 2
    center_min = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center_min
    center = center_min + 0.5 * dims
    return np.concatenate([center, dims], axis=-1)


def center_to_minmax_2d_0_5(centers, dims):
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)


def center_to_minmax_2d(centers, dims, origin=0.5):
    if origin == 0.5:
        return center_to_minmax_2d_0_5(centers, dims)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, [0, 2]].reshape([-1, 4])


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


def projection_matrix_to_CRT_kitti(proj):
    # P = C @ [R|T]
    # C is upper triangular matrix, so we need to inverse CR and use QR
    # stable for all kitti camera projection matrix
    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T


def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array(
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]], dtype=C.dtype
    )
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype
    )
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype
    )
    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz


def get_frustum_v2(bboxes, C, near_clip=0.001, far_clip=100):
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    num_box = bboxes.shape[0]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[
        np.newaxis, :, np.newaxis
    ]
    z_points = np.tile(z_points, [num_box, 1, 1])
    box_corners = minmax_to_corner_2d_v2(bboxes)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype
    )
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype
    )
    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=1)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=-1)
    return ret_xyz


@numba.njit
def _add_rgb_to_points_kernel(points_2d, image, points_rgb):
    num_points = points_2d.shape[0]
    image_h, image_w = image.shape[:2]
    for i in range(num_points):
        img_pos = np.floor(points_2d[i]).astype(np.int32)
        if img_pos[0] >= 0 and img_pos[0] < image_w:
            if img_pos[1] >= 0 and img_pos[1] < image_h:
                points_rgb[i, :] = image[img_pos[1], img_pos[0], :]
                # image[img_pos[1], img_pos[0]] = 0


def add_rgb_to_points(points, image, rect, Trv2c, P2, mean_size=[5, 5]):
    kernel = np.ones(mean_size, np.float32) / np.prod(mean_size)
    # image = cv2.filter2D(image, -1, kernel)
    points_cam = lidar_to_camera(points[:, :3], rect, Trv2c)
    points_2d = project_to_image(points_cam, P2)
    points_rgb = np.zeros([points_cam.shape[0], 3], dtype=points.dtype)
    _add_rgb_to_points_kernel(points_2d, image, points_rgb)
    return points_rgb


def project_to_image(points_3d, proj_mat):
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    points_4 = np.concatenate([points_3d, np.ones(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def camera_to_lidar(points, r_rect, velo2cam):
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


def lidar_to_camera(points, r_rect, velo2cam):
    points_shape = list(points.shape[:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    camera_points = points @ (r_rect @ velo2cam).T
    return camera_points[..., :3]


def box_camera_to_lidar(data, r_rect, velo2cam):
    xyz = data[:, 0:3]
    l, h, w = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return np.concatenate([xyz_lidar, w, l, h, r], axis=1)


def box_lidar_to_camera(data, r_rect, velo2cam):
    xyz_lidar = data[:, 0:3]
    w, l, h = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return np.concatenate([xyz, l, h, w, r], axis=1)


def remove_outside_points(points, rect, Trv2c, P2, image_shape):
    # 5x faster than remove_outside_points_v1(2ms vs 10ms)
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(image_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = camera_to_lidar(frustum.T, rect, Trv2c)
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
    indices = points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)
    points = points[indices.reshape([-1])]
    return points


@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, eps=1.0):
    """calculate box iou. note that jit version runs 2x faster than cython in
    my machine!
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = (query_boxes[k, 2] - query_boxes[k, 0] + eps) * (
            query_boxes[k, 3] - query_boxes[k, 1] + eps
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2])
                - max(boxes[n, 0], query_boxes[k, 0])
                + eps
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3])
                    - max(boxes[n, 1], query_boxes[k, 1])
                    + eps
                )
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + eps)
                        * (boxes[n, 3] - boxes[n, 1] + eps)
                        + box_area
                        - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


@numba.jit(nopython=True)
def iou_3d_jit(boxes, query_boxes, add1=True):
    """calculate box iou3d,
    ----------
    boxes: (N, 6) ndarray of float
    query_boxes: (K, 6) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    if add1:
        add1 = 1.0
    else:
        add1 = 0.0
    for k in range(K):
        box_area = (
            (query_boxes[k, 3] - query_boxes[k, 0] + add1)
            * (query_boxes[k, 4] - query_boxes[k, 1] + add1)
            * (query_boxes[k, 5] - query_boxes[k, 2] + add1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 3], query_boxes[k, 3])
                - max(boxes[n, 0], query_boxes[k, 0])
                + add1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 4], query_boxes[k, 4])
                    - max(boxes[n, 1], query_boxes[k, 1])
                    + add1
                )
                if ih > 0:
                    il = (
                        min(boxes[n, 5], query_boxes[k, 5])
                        - max(boxes[n, 2], query_boxes[k, 2])
                        + add1
                    )
                    if il > 0:
                        ua = float(
                            (boxes[n, 3] - boxes[n, 0] + add1)
                            * (boxes[n, 4] - boxes[n, 1] + add1)
                            * (boxes[n, 5] - boxes[n, 2] + add1)
                            + box_area
                            - iw * ih * il
                        )
                        overlaps[n, k] = iw * ih * il / ua
    return overlaps


@numba.jit(nopython=True)
def iou_nd_jit(boxes, query_boxes, add1=True):
    """calculate box iou nd, 2x slower than iou_jit.
    ----------
    boxes: (N, ndim * 2) ndarray of float
    query_boxes: (K, ndim * 2) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    ndim = boxes.shape[1] // 2
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    side_lengths = np.zeros((ndim,), dtype=boxes.dtype)
    if add1:
        add1 = 1.0
    else:
        add1 = 0.0
    invalid = False
    for k in range(K):
        qbox_area = query_boxes[k, ndim] - query_boxes[k, 0] + add1
        for i in range(1, ndim):
            qbox_area *= query_boxes[k, ndim + i] - query_boxes[k, i] + add1
        for n in range(N):
            invalid = False
            for i in range(ndim):
                side_length = (
                    min(boxes[n, i + ndim], query_boxes[k, i + ndim])
                    - max(boxes[n, i], query_boxes[k, i])
                    + add1
                )
                if side_length <= 0:
                    invalid = True
                    break
                side_lengths[i] = side_length
            if not invalid:
                box_area = boxes[n, ndim] - boxes[n, 0] + add1
                for i in range(1, ndim):
                    box_area *= boxes[n, ndim + i] - boxes[n, i] + add1
                inter = side_lengths[0]
                for i in range(1, ndim):
                    inter *= side_lengths[i]
                # inter = np.prod(side_lengths)
                ua = float(box_area + qbox_area - inter)
                overlaps[n, k] = inter / ua

    return overlaps


def points_in_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0.5)):
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin, axis=z_axis
    )
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array(
        [
            [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
            [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
            [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
            [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
            [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
            [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
        ]
    ).transpose([2, 0, 1, 3])
    return surfaces


@numba.jit(nopython=True)
def corner_to_surfaces_3d_jit(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array(
        [0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7]
    ).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces


def assign_label_to_voxel(gt_boxes, coors, voxel_size, coors_range):
    """assign a 0/1 label to each voxel based on whether
    the center of voxel is in gt_box. LIDAR.
    """
    voxel_size = np.array(voxel_size, dtype=gt_boxes.dtype)
    coors_range = np.array(coors_range, dtype=gt_boxes.dtype)
    shift = coors_range[:3]
    voxel_origins = coors[:, ::-1] * voxel_size + shift
    voxel_centers = voxel_origins + voxel_size * 0.5
    gt_box_corners = center_to_corner_box3d(
        gt_boxes[:, :3] - voxel_size * 0.5,
        gt_boxes[:, 3:6] + voxel_size,
        gt_boxes[:, 6],
        origin=[0.5, 0.5, 0.5],
        axis=2,
    )
    gt_surfaces = corner_to_surfaces_3d(gt_box_corners)
    ret = points_in_convex_polygon_3d_jit(voxel_centers, gt_surfaces)
    return np.any(ret, axis=1).astype(np.int64)


def assign_label_to_voxel_v3(gt_boxes, coors, voxel_size, coors_range):
    """assign a 0/1 label to each voxel based on whether
    the center of voxel is in gt_box. LIDAR.
    """
    voxel_size = np.array(voxel_size, dtype=gt_boxes.dtype)
    coors_range = np.array(coors_range, dtype=gt_boxes.dtype)
    shift = coors_range[:3]
    voxel_origins = coors[:, ::-1] * voxel_size + shift
    voxel_maxes = voxel_origins + voxel_size
    voxel_minmax = np.concatenate([voxel_origins, voxel_maxes], axis=-1)
    voxel_corners = minmax_to_corner_3d(voxel_minmax)
    gt_box_corners = center_to_corner_box3d(
        gt_boxes[:, :3],
        gt_boxes[:, 3:6],
        gt_boxes[:, 6],
        origin=[0.5, 0.5, 0.5],
        axis=2,
    )
    gt_surfaces = corner_to_surfaces_3d(gt_box_corners)
    voxel_corners_flat = voxel_corners.reshape([-1, 3])
    ret = points_in_convex_polygon_3d_jit(voxel_corners_flat, gt_surfaces)
    ret = ret.reshape([-1, 8, ret.shape[-1]])
    return ret.any(-1).any(-1).astype(np.int64)


def image_box_region_area(img_cumsum, bbox):
    """check a 2d voxel is contained by a box. used to filter empty
    anchors.
    Summed-area table algorithm:
    ==> W
    ------------------
    |      |         |
    |------A---------B
    |      |         |
    |      |         |
    |----- C---------D
    Iabcd = ID-IB-IC+IA
    Args:
        img_cumsum: [M, H, W](yx) cumsumed image.
        bbox: [N, 4](xyxy) bounding box,
    """
    N = bbox.shape[0]
    M = img_cumsum.shape[0]
    ret = np.zeros([N, M], dtype=img_cumsum.dtype)
    ID = img_cumsum[:, bbox[:, 3], bbox[:, 2]]
    IA = img_cumsum[:, bbox[:, 1], bbox[:, 0]]
    IB = img_cumsum[:, bbox[:, 3], bbox[:, 0]]
    IC = img_cumsum[:, bbox[:, 1], bbox[:, 2]]
    ret = ID - IB - IC + IA
    return ret


def get_minimum_bounding_box_bv(points, voxel_size, bound, downsample=8, margin=1.6):
    x_vsize = voxel_size[0]
    y_vsize = voxel_size[1]
    max_x = points[:, 0].max()
    max_y = points[:, 1].max()
    min_x = points[:, 0].min()
    min_y = points[:, 1].min()
    max_x = np.floor(max_x / (x_vsize * downsample) + 1) * (x_vsize * downsample)
    max_y = np.floor(max_y / (y_vsize * downsample) + 1) * (y_vsize * downsample)
    min_x = np.floor(min_x / (x_vsize * downsample)) * (x_vsize * downsample)
    min_y = np.floor(min_y / (y_vsize * downsample)) * (y_vsize * downsample)
    max_x = np.minimum(max_x + margin, bound[2])
    max_y = np.minimum(max_y + margin, bound[3])
    min_x = np.maximum(min_x - margin, bound[0])
    min_y = np.maximum(min_y - margin, bound[1])
    return np.array([min_x, min_y, max_x, max_y])
    

def box3d_to_bbox(box3d, rect, Trv2c, P2):
    box3d_to_cam = box_lidar_to_camera(box3d, rect, Trv2c)
    box_corners = center_to_corner_box3d(
        box3d[:, :3], box3d[:, 3:6], box3d[:, 6], [0.5, 1.0, 0.5], axis=1
    )
    box_corners_in_image = project_to_image(box_corners, P2)
    # box_corners_in_image: [N, 8, 2]
    minxy = np.min(box_corners_in_image, axis=1)
    maxxy = np.max(box_corners_in_image, axis=1)
    bbox = np.concatenate([minxy, maxxy], axis=1)
    return bbox


def change_box3d_center_(box3d, src, dst):
    dst = np.array(dst, dtype=box3d.dtype)
    src = np.array(src, dtype=box3d.dtype)
    box3d[..., :3] += box3d[..., 3:6] * (dst - src)

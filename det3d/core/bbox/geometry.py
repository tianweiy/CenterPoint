import numba
import numpy as np


@numba.njit
def _points_count_convex_polygon_3d_jit(
    points, polygon_surfaces, normal_vec, d, num_surfaces=None
):
    """count points in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_polygon] array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.full((num_polygons,), num_points, dtype=np.int64)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[j] -= 1
                    break
    return ret


def points_count_convex_polygon_3d_jit(points, polygon_surfaces, num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_polygon] array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_count_convex_polygon_3d_jit(
        points, polygon_surfaces, normal_vec, d, num_surfaces
    )


@numba.njit
def is_line_segment_intersection_jit(lines1, lines2):
    """check if line segments1 and line segments2 have cross point

    Args:
        lines1 (float, [N, 2, 2]): [description]
        lines2 (float, [M, 2, 2]): [description]

    Returns:
        [type]: [description]
    """

    # Return true if line segments AB and CD intersect
    N = lines1.shape[0]
    M = lines2.shape[0]
    ret = np.zeros((N, M), dtype=np.bool_)
    for i in range(N):
        for j in range(M):
            A = lines1[i, 0]
            B = lines1[i, 1]
            C = lines2[j, 0]
            D = lines2[j, 1]
            acd = (D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]) * (D[0] - A[0])
            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
            if acd != bcd:
                abc = (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
                abd = (D[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (D[0] - A[0])
                if abc != abd:
                    ret[i, j] = True
    return ret


@numba.njit
def line_segment_intersection(line1, line2, intersection):
    A = line1[0]
    B = line1[1]
    C = line2[0]
    D = line2[1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            intersection[0] = (ABBA * DC0 - BA0 * CDDC) / DH
            intersection[1] = (ABBA * DC1 - BA1 * CDDC) / DH
            return True
    return False


def _ccw(A, B, C):
    return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) > (
        B[..., 1] - A[..., 1]
    ) * (C[..., 0] - A[..., 0])


def is_line_segment_cross(lines1, lines2):
    # 10x slower than jit version with 1000-1000 random lines input.
    # lines1, [N, 2, 2]
    # lines2, [M, 2, 2]
    A = lines1[:, 0, :][:, np.newaxis, :]
    B = lines1[:, 1, :][:, np.newaxis, :]
    C = lines2[:, 0, :][np.newaxis, :, :]
    D = lines2[:, 1, :][np.newaxis, :, :]
    return np.logical_and(
        _ccw(A, C, D) != _ccw(B, C, D), _ccw(A, B, C) != _ccw(A, B, D)
    )


@numba.jit(nopython=False)
def surface_equ_3d_jit(polygon_surfaces):
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_v = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3]
    normal_v = np.cross(surface_v[:, :, 0, :], surface_v[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :])
    d = np.einsum("aij, aij->ai", normal_v, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d


@numba.jit(nopython=False)
def points_in_convex_polygon_3d_jit_v1(points, polygon_surfaces, num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jit(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def surface_equ_3d(polygon_surfaces):
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_v = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3]
    normal_v = np.cross(surface_v[:, :, 0, :], surface_v[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :])
    d = np.einsum("aij, aij->ai", normal_v, polygon_surfaces[:, :, 0, :])
    return normal_v, -d


def points_in_convex_polygon_3d_jit(points, polygon_surfaces, num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit(
        points, polygon_surfaces, normal_vec, d, num_surfaces
    )


@numba.njit
def _points_in_convex_polygon_3d_jit(
    points, polygon_surfaces, normal_vec, d, num_surfaces=None
):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


@numba.jit
def points_in_convex_polygon_jit(points, polygon, clockwise=True):
    """check points is in 2d convex polygons. True when point in polygon
    Args:
        points: [num_points, 2] array.
        polygon: [num_polygon, num_points_of_polygon, 2] array.
        clockwise: bool. indicate polygon is clockwise.
    Returns:
        [num_points, num_polygon] bool array.
    """
    # first convert polygon to directed lines
    num_points_of_polygon = polygon.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon.shape[0]
    if clockwise:
        vec1 = (
            polygon
            - polygon[
                :,
                [num_points_of_polygon - 1] + list(range(num_points_of_polygon - 1)),
                :,
            ]
        )
    else:
        vec1 = (
            polygon[
                :,
                [num_points_of_polygon - 1] + list(range(num_points_of_polygon - 1)),
                :,
            ]
            - polygon
        )
    # vec1: [num_polygon, num_points_of_polygon, 2]
    ret = np.zeros((num_points, num_polygons), dtype=np.bool_)
    success = True
    cross = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            success = True
            for k in range(num_points_of_polygon):
                cross = vec1[j, k, 1] * (polygon[j, k, 0] - points[i, 0])
                cross -= vec1[j, k, 0] * (polygon[j, k, 1] - points[i, 1])
                if cross >= 0:
                    success = False
                    break
            ret[i, j] = success
    return ret


def points_in_convex_polygon(points, polygon, clockwise=True):
    """check points is in convex polygons. may run 2x faster when write in
    cython(don't need to calculate all cross-product between edge and point)
    Args:
        points: [num_points, 2] array.
        polygon: [num_polygon, num_points_of_polygon, 2] array.
        clockwise: bool. indicate polygon is clockwise.
    Returns:
        [num_points, num_polygon] bool array.
    """
    # first convert polygon to directed lines
    num_lines = polygon.shape[1]
    polygon_next = polygon[:, [num_lines - 1] + list(range(num_lines - 1)), :]
    if clockwise:
        vec1 = (polygon - polygon_next)[np.newaxis, ...]
    else:
        vec1 = (polygon_next - polygon)[np.newaxis, ...]
    vec2 = polygon[np.newaxis, ...] - points[:, np.newaxis, np.newaxis, :]
    # [num_points, num_polygon, num_points_of_polygon, 2]
    cross = np.cross(vec1, vec2)
    return np.all(cross > 0, axis=2)


@numba.njit
def surface_equ_3d_jitv2(surfaces):
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    num_polygon = surfaces.shape[0]
    max_num_surfaces = surfaces.shape[1]
    normal_vec = np.zeros((num_polygon, max_num_surfaces, 3), dtype=surfaces.dtype)
    d = np.zeros((num_polygon, max_num_surfaces), dtype=surfaces.dtype)
    sv0 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    sv1 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    for i in range(num_polygon):
        for j in range(max_num_surfaces):
            sv0[0] = surfaces[i, j, 0, 0] - surfaces[i, j, 1, 0]
            sv0[1] = surfaces[i, j, 0, 1] - surfaces[i, j, 1, 1]
            sv0[2] = surfaces[i, j, 0, 2] - surfaces[i, j, 1, 2]
            sv1[0] = surfaces[i, j, 1, 0] - surfaces[i, j, 2, 0]
            sv1[1] = surfaces[i, j, 1, 1] - surfaces[i, j, 2, 1]
            sv1[2] = surfaces[i, j, 1, 2] - surfaces[i, j, 2, 2]
            normal_vec[i, j, 0] = sv0[1] * sv1[2] - sv0[2] * sv1[1]
            normal_vec[i, j, 1] = sv0[2] * sv1[0] - sv0[0] * sv1[2]
            normal_vec[i, j, 2] = sv0[0] * sv1[1] - sv0[1] * sv1[0]

            d[i, j] = (
                -surfaces[i, j, 0, 0] * normal_vec[i, j, 0]
                - surfaces[i, j, 0, 1] * normal_vec[i, j, 1]
                - surfaces[i, j, 0, 2] * normal_vec[i, j, 2]
            )
    return normal_vec, d


@numba.njit
def _points_in_convex_polygon_3d_jit_v2(points, surfaces):
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


@numba.njit
def points_in_convex_polygon_3d_jit_v2(points, surfaces, num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    num_polygon = surfaces.shape[0]
    max_num_surfaces = surfaces.shape[1]
    num_points = points.shape[0]
    normal_vec = np.zeros((num_polygon, max_num_surfaces, 3), dtype=surfaces.dtype)
    d = np.zeros((num_polygon, max_num_surfaces), dtype=surfaces.dtype)
    sv0 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    sv1 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    ret = np.ones((num_points, num_polygon), dtype=np.bool_)
    for i in range(num_polygon):
        for j in range(max_num_surfaces):
            sv0[0] = surfaces[i, j, 0, 0] - surfaces[i, j, 1, 0]
            sv0[1] = surfaces[i, j, 0, 1] - surfaces[i, j, 1, 1]
            sv0[2] = surfaces[i, j, 0, 2] - surfaces[i, j, 1, 2]
            sv1[0] = surfaces[i, j, 1, 0] - surfaces[i, j, 2, 0]
            sv1[1] = surfaces[i, j, 1, 1] - surfaces[i, j, 2, 1]
            sv1[2] = surfaces[i, j, 1, 2] - surfaces[i, j, 2, 2]
            normal_vec[i, j, 0] = sv0[1] * sv1[2] - sv0[2] * sv1[1]
            normal_vec[i, j, 1] = sv0[2] * sv1[0] - sv0[0] * sv1[2]
            normal_vec[i, j, 2] = sv0[0] * sv1[1] - sv0[1] * sv1[0]

            d[i, j] = (
                -surfaces[i, j, 0, 0] * normal_vec[i, j, 0]
                - surfaces[i, j, 0, 1] * normal_vec[i, j, 1]
                - surfaces[i, j, 0, 2] * normal_vec[i, j, 2]
            )

    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygon):
            for k in range(max_num_surfaces):
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret

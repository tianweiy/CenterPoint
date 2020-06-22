# from open3d import *
import numpy as np
import numpy.linalg as la

eps = 0.00001


def svd(A):
    u, s, vh = la.svd(A)
    S = np.zeros(A.shape)
    S[: s.shape[0], : s.shape[0]] = np.diag(s)
    return u, S, vh


def inverse_sigma(S):
    inv_S = S.copy().transpose()
    for i in range(min(S.shape)):
        if abs(inv_S[i, i]) > eps:
            inv_S[i, i] = 1.0 / inv_S[i, i]
    return inv_S


def svd_solve(A, b):
    U, S, Vt = svd(A)
    inv_S = inverse_sigma(S)
    svd_solution = Vt.transpose() @ inv_S @ U.transpose() @ b

    print("U:")
    print(U)
    print("Sigma:")
    print(S)
    print("V_transpose:")
    print(Vt)
    print("--------------")
    print("SVD solution:")
    print(svd_solution)
    print("A multiplies SVD solution:")
    print(A @ svd_solution)

    return svd_solution


def fit_plane_LSE(points):
    # points: Nx4 homogeneous 3d points
    # return: 1d array of four elements [a, b, c, d] of
    # ax+by+cz+d = 0
    assert points.shape[0] >= 3  # at least 3 points needed
    U, S, Vt = svd(points)
    null_space = Vt[-1, :]
    return null_space


def get_point_dist(points, plane):
    # return: 1d array of size N (number of points)
    dists = np.abs(points @ plane) / np.sqrt(
        plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2
    )
    return dists


def fit_plane_LSE_RANSAC(
    points, iters=1000, inlier_thresh=0.05, return_outlier_list=False
):
    # points: Nx4 homogeneous 3d points
    # return:
    #   plane: 1d array of four elements [a, b, c, d] of ax+by+cz+d = 0
    #   inlier_list: 1d array of size N of inlier points
    max_inlier_num = -1
    max_inlier_list = None

    N = points.shape[0]
    assert N >= 3

    for i in range(iters):
        chose_id = np.random.choice(N, 3, replace=False)
        chose_points = points[chose_id, :]
        tmp_plane = fit_plane_LSE(chose_points)

        dists = get_point_dist(points, tmp_plane)
        tmp_inlier_list = np.where(dists < inlier_thresh)[0]
        tmp_inliers = points[tmp_inlier_list, :]
        num_inliers = tmp_inliers.shape[0]
        if num_inliers > max_inlier_num:
            max_inlier_num = num_inliers
            max_inlier_list = tmp_inlier_list

    final_points = points[max_inlier_list, :]
    plane = fit_plane_LSE(final_points)

    fit_variance = np.var(get_point_dist(final_points, plane))
    # print('RANSAC fit variance: %f' % fit_variance)
    # print(plane)

    dists = get_point_dist(points, plane)

    select_thresh = inlier_thresh * 1

    inlier_list = np.where(dists < select_thresh)[0]
    if not return_outlier_list:
        return plane, inlier_list
    else:
        outlier_list = np.where(dists >= select_thresh)[0]
        return plane, inlier_list, outlier_list


def display_inlier_outlier(cloud, ind):
    inlier_cloud = select_down_sample(cloud, ind)
    outlier_cloud = select_down_sample(cloud, ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    draw_geometries([inlier_cloud, outlier_cloud])


def create_pcd(points):
    pcd = PointCloud()
    pcd.points = Vector3dVector(points[:, :3])
    return pcd


if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    points = np.fromfile("path/to/point", dtype=np.float32).reshape(-1, 5)
    points = np.concatenate((points[:, :3], np.ones((points.shape[0], 1))), axis=1)

    gp_height = -1.78

    p_set_1 = points
    p1, inlier_list1, outlier_list1 = fit_plane_LSE_RANSAC(
        p_set_1, return_outlier_list=True
    )
    p_set_2 = p_set_1[outlier_list1, :]
    p2, inlier_list2, outlier_list2 = fit_plane_LSE_RANSAC(
        p_set_2, return_outlier_list=True
    )
    p_set_3 = p_set_2[outlier_list2, :]
    p3, inlier_list3, outlier_list3 = fit_plane_LSE_RANSAC(
        p_set_3, return_outlier_list=True
    )
    p_set_4 = p_set_3[outlier_list3, :]
    p4, inlier_list4, outlier_list4 = fit_plane_LSE_RANSAC(
        p_set_4, return_outlier_list=True
    )

    ps = [p1, p2, p3, p4]

    for p in [p1]:
        xx = points[:, 0]
        yy = points[:, 1]

        zz = (-p[0] * xx - p[1] * yy - p[3]) / p[2]

        print(f"Current point cloud's gp height is: {np.mean(zz)}")
        points_up = points[np.where(points[:, 2] >= zz)]
        points_down = points[np.where(points[:, 2] < zz)]
        print(f"Current point cloud's gp height is: {np.mean(points_down[:, 2])}")
        ptup = create_pcd(points_up)
        ptdn = create_pcd(points_down)
        ptup.paint_uniform_color([0.5, 0.5, 0.5])
        draw_geometries([ptup, ptdn])

    # print("Downsample the point cloud with a voxel of 0.02")
    # voxel_down_pcd = voxel_down_sample(pcd, voxel_size = 0.02)
    # draw_geometries([voxel_down_pcd])
    #
    # print("Every 5th points are selected")
    # uni_down_pcd = uniform_down_sample(pcd, every_k_points = 5)
    # draw_geometries([uni_down_pcd])
    #
    # print("Statistical oulier removal")
    # cl,ind = statistical_outlier_removal(voxel_down_pcd,
    #         nb_neighbors=20, std_ratio=2.0)
    # display_inlier_outlier(voxel_down_pcd, ind)
    #
    # print("Radius oulier removal")
    # cl,ind = radius_outlier_removal(voxel_down_pcd,
    #         nb_points=16, radius=0.05)
    # display_inlier_outlier(voxel_down_pcd, ind)

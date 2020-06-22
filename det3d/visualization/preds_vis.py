import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")

matplotlib.use("Agg")

colors = {
    "Car": "b",
    "Tram": "r",
    "Cyclist": "g",
    "Van": "c",
    "Truck": "m",
    "Pedestrian": "y",
    "Sitter": "k",
}
axes_limits = [
    [-80, 80],  # X axis range
    [-50, 50],  # Y axis range
    [-5, 5],  # Z axis range
]
axes_str = ["X", "Y", "Z"]


def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color="black"):
    """
    Draws a bounding 3D box in a pyplot axis.
    
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = np.transpose(vertices)[axes, :]
    connections = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)


def display_single_lidar(
    data, gt_boxes=None, dt_boxes=None, points=0.2, view=False, idx=0
):
    # points          : Fraction of lidar points to use. Defaults to `0.2`, e.g. 20%.

    points_step = int(1.0 / points)
    point_size = 0.01 * (1.0 / points)
    velo_range = range(0, data.shape[0], points_step)

    # print(points_step)
    # print(point_size)
    # print(velo_range)

    def draw_point_cloud(
        ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None
    ):
        """
        Convenient method for drawing various point cloud projections as a part of frame statistics.
        """
        ax.scatter(
            *np.transpose(data[:, axes]), s=point_size, c=data[:, 3], cmap="gray"
        )

        ax.set_title(title)
        ax.set_xlabel("{} axis".format(axes_str[axes[0]]))
        ax.set_ylabel("{} axis".format(axes_str[axes[1]]))
        if len(axes) > 2:
            ax.set_xlim3d(*axes_limits[axes[0]])
            ax.set_ylim3d(*axes_limits[axes[1]])
            ax.set_zlim3d(*axes_limits[axes[2]])
            ax.set_zlabel("{} axis".format(axes_str[axes[2]]))
        else:
            ax.set_xlim(*axes_limits[axes[0]])
            ax.set_ylim(*axes_limits[axes[1]])

        # User specified limits
        if xlim3d != None:
            ax.set_xlim3d(xlim3d)
        if ylim3d != None:
            ax.set_ylim3d(ylim3d)
        if zlim3d != None:
            ax.set_zlim3d(zlim3d)

        for i in range(gt_boxes.shape[0]):
            draw_box(ax, gt_boxes[i], axes=axes, color="green")
        for i in range(dt_boxes.shape[0]):
            draw_box(ax, dt_boxes[i], axes=axes, color="red")

    # Draw point cloud data as 3D plot
    f2 = plt.figure(figsize=(10, 5))
    ax2 = f2.add_subplot(111, projection="3d")
    ax2.view_init(45, 45)

    # Hide grid lines
    ax2.grid(False)
    #     plt.axis('off')

    draw_point_cloud(ax2, "Velodyne scan", xlim3d=axes_limits[0])
    # plt.show()
    plt.savefig("/home/zhubenjin/data/Outputs/figs/" + str(idx) + ".png")

    if view:
        # Draw point cloud data as plane projections
        f, ax3 = plt.subplots(3, 1, figsize=(15, 25))
        #         axe3.view_init(35,100)
        draw_point_cloud(
            ax3[0],
            "Velodyne scan, XZ projection (Y = 0), the car is moving in direction left to right",
            axes=[0, 2],  # X and Z axes
        )
        draw_point_cloud(
            ax3[1],
            "Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right",
            axes=[0, 1],  # X and Y axes
        )
        draw_point_cloud(
            ax3[2],
            "Velodyne scan, YZ projection (X = 0), the car is moving towards the graph plane",
            axes=[1, 2],  # Y and Z axes
        )
        # plt.show()
        plt.savefig("/home/zhubenjin/data/Outputs/figs/" + str(idx) + "_3V.png")


def project_velo2camera(vel_data, Trv2c):
    homo_vel_data = np.hstack(
        (vel_data[:, :3], np.ones((vel_data.shape[0], 1), dtype="float32"))
    )
    vel_data_c = np.dot(homo_vel_data, Trv2c.T)
    vel_data_c /= vel_data_c[:, -1].reshape((-1, 1))
    vel_data_c = np.hstack((vel_data_c[:, :3], vel_data[:, -1].reshape((-1, 1))))
    return vel_data_c


def camera_to_lidar(points, rect, velo2cam):
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((rect @ velo2cam).T)
    return lidar_points[..., :3]


def lidar_to_camera(points, r_rect, velo2cam):
    points_shape = list(points.shape[:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    camera_points = points @ (r_rect @ velo2cam).T
    return camera_points[..., :3]


def show_single_pc_vel(raw_lidar, gts, dts, Trv2c, rect, idx):
    """
    raw_lidar: N x 3
    gt_labels: N x 7 in camera
    dts: N x 7 in camera
    rect: 
    Trv2c:
    P2:
    """

    bbox3d_lidar_gt = []
    for box in gts:
        box = camera_to_lidar(box, Trv2c, rect)
        bbox3d_lidar_gt.append(box)
    gts_lidar = np.array(bbox3d_lidar_gt)

    bbox3d_lidar_dt = []
    for box in dts:
        box = camera_to_lidar(box, Trv2c, rect)
        bbox3d_lidar_dt.append(box)
    dts_lidar = np.array(bbox3d_lidar_dt)

    # display_single_lidar(lidar, gts, dts, points=.01, view=True, idx=idx)
    display_single_lidar(
        raw_lidar, gts_lidar, dts_lidar, points=0.01, view=True, idx=idx
    )

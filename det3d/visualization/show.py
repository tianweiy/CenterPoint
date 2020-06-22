import sys

import numpy as np
import pylab as plt


def draw_box(ax, vertices, axes=[0, 1, 2], color="black"):
    vertices = vertices[axes, :]
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
        ax.plot(*vertices[:, connection], c=color, lw=2)


def visualize_feature_maps(
    X, anno=[], keypoints=[], axes=[2, 0], save_filename=None, divide=False
):
    nc = np.ceil(np.sqrt(X.shape[2]))  # column
    nr = np.ceil(X.shape[2] / nc)  # row
    nc = int(nc)
    nr = int(nr)
    plt.figure(figsize=(64, 64))
    for i in range(X.shape[2]):
        ax = plt.subplot(nr, nc, i + 1)
        ax.imshow(X[:, :, i], cmap="jet")
        # plt.colorbar()
        for obj in anno:
            draw_box(ax, obj, axes=axes, color="r")

            # plot head orientation
            center_bottom_forward = np.mean(obj.T[[0, 1, 5, 4]], axis=0)
            center_bottom = np.mean(obj.T[[1, 2, 6, 5]], axis=0)
            ax.plot(
                [center_bottom[0], center_bottom_forward[0]],
                [center_bottom[1], center_bottom_forward[1]],
                c="y",
                lw=3,
            )

        for pts_score in keypoints:
            pts = pts_score[:8]
            if divide:
                pts = pts / 8.0
            for i in range(4):
                ax.plot(pts[2 * i + 1], pts[2 * i + 0], "r*")
            ax.plot([pts[1], pts[3]], [pts[0], pts[2]], c="y", lw=5)
            ax.plot([pts[3], pts[5]], [pts[2], pts[4]], c="g", lw=5)
            ax.plot([pts[5], pts[7]], [pts[4], pts[6]], c="b", lw=5)
            ax.plot([pts[7], pts[1]], [pts[6], pts[0]], c="r", lw=5)

            # ax.plot([pts[7], pts[7]+10*pts_score[11]], [pts[6], pts[6]+10*pts_score[12]], c='c', lw=2)
            # annotations = pts_score[13]
            # ax.annotate(str(annotations), xy=(pts[7]+10*pts_score[11], pts[6]+10*pts_score[12]), xytext=(pts[7]+10*pts_score[11], pts[6]+10*pts_score[12]), arrowprops=dict(facecolor='black', shrink=0.05),)

        ax.axis("off")
    if save_filename:
        plt.savefig(save_filename)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    d = np.load(sys.argv[1])
    print(d["X"].shape, d["annos"])
    X = np.squeeze(d["X"])
    visualize_feature_maps(X, d["annos"], axes=[2, 0], save_filename="/tmp/lidar.png")

import os
import os.path as osp
import nuscenes
from nuscenes.eval.detection.data_classes import EvalBoxes, EvalBox
from nuscenes.utils.data_classes import LidarPointCloud
import json
import random
from det3d.deps.nuscenes.eval.detection.render import visualize_sample
import numpy as np
from matplotlib import pyplot as plt
from nuscenes import NuScenes
from nuscenes.eval.detection.constants import (
    TP_METRICS,
    DETECTION_NAMES,
    DETECTION_COLORS,
    TP_METRICS_UNITS,
    PRETTY_DETECTION_NAMES,
    PRETTY_TP_METRICS,
)
from nuscenes.eval.detection.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import MetricDataList, DetectionMetrics
from nuscenes.eval.detection.utils import boxes_to_sensor
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from collections import defaultdict


def from_file_multisweep(
    nusc: "NuScenes", sample_data_token: str,
):
    """
    Return a point cloud that aggregates multiple sweeps.
    As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
    As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
    :param nusc: A NuScenes instance.
    :param sample_rec: The current sample.
    :param chan: The radar channel from which we track back n sweeps to aggregate the point cloud.
    :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
    :param nsweeps: Number of sweeps to aggregated.
    :param min_distance: Distance below which points are discarded.
    :return: (all_pc, all_times). The aggregated point cloud and timestamps.
    """

    # Init
    # Aggregate current and previous sweeps.
    current_sd_rec = nusc.get("sample_data", sample_data_token)
    current_pc = LidarPointCloud.from_file(
        osp.join(nusc.dataroot, current_sd_rec["filename"])
    )

    return current_pc


def visualize_sample_data(
    nusc: NuScenes,
    sample_data_token: str,
    pred_boxes: EvalBoxes,
    nsweeps: int = 1,
    conf_th: float = 0.15,
    eval_range: float = 50,
    verbose: bool = True,
    savepath: str = None,
) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Retrieve sensor & pose records.

    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    # Get boxes.
    boxes_est_global = pred_boxes[sample_data_token]

    # Map EST boxes to lidar.
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    # Add scores to EST boxes.
    for box_est, box_est_global in zip(boxes_est, boxes_est_global):
        box_est.score = box_est_global.detection_score

    # Get point cloud in lidar frame.
    pc = from_file_multisweep(nusc, sample_data_token)
    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show point cloud.
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, "x", color="black")

    # Show EST boxes.
    for box in boxes_est:
        # Show only predictions with a high score.
        assert not np.isnan(box.score), "Error: Box score cannot be NaN!"
        if box.score >= conf_th:
            box.render(ax, view=np.eye(4), colors=("b", "b", "b"), linewidth=1)

    # Limit visible range.
    axes_limit = eval_range + 3
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    # Show / save plot.
    if verbose:
        print("Rendering sample token %s" % sample_data_token)
    plt.title(sample_data_token)
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()


nusc = nuscenes.NuScenes(
    version="v1.0-trainval", dataroot="/data/Datasets/nuScenes", verbose=True
)

with open(
    "/data/NUSC_SECOND__20190531-193048/results_2/9088db17416043e5880a53178bfa461c.json"
) as f:
    nusc_annos = json.load(f)

print("Plot some examples")
results = nusc_annos["results"]

pred_boxes = defaultdict(list)
for sample_data_token, boxes in results.items():
    pred_boxes[sample_data_token].extend(
        [
            EvalBox(
                sample_token=box["sample_token"],
                translation=tuple(box["translation"]),
                size=tuple(box["size"]),
                rotation=tuple(box["rotation"]),
                velocity=tuple(box["velocity"]),
                detection_name=box["detection_name"],
                attribute_name=box["attribute_name"],
                ego_dist=0.0 if "ego_dist" not in box else float(box["ego_dist"]),
                detection_score=-1.0
                if "detection_score" not in box
                else float(box["detection_score"]),
                num_pts=-1 if "num_pts" not in box else int(box["num_pts"]),
            )
            for box in boxes
        ]
    )


def add_center_dist(nusc, eval_boxes: EvalBoxes):
    """ Adds the cylindrical (xy) center distance from ego vehicle to each box. """

    for sample_token in eval_boxes.keys():
        sd_record = nusc.get("sample_data", sample_token)
        pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

        for box in eval_boxes[sample_token]:
            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            diff = np.array(pose_record["translation"][:2]) - np.array(
                box.translation[:2]
            )
            box.ego_dist = np.sqrt(np.sum(diff ** 2))

    return eval_boxes


pred_boxes = add_center_dist(nusc, pred_boxes)

tokens = list(pred_boxes.keys())

scene_token = "9088db17416043e5880a53178bfa461c"
scene = nusc.get("scene", scene_token)

token2id = {}

frame_id = 1
first_sample = nusc.get("sample", scene["first_sample_token"])
first_sample_data = nusc.get("sample_data", first_sample["data"]["LIDAR_TOP"])
token2id[first_sample_data["token"]] = frame_id

nxt = first_sample_data["next"]
while nxt != "":
    frame_id += 1
    token2id[nxt] = frame_id
    nxt = nusc.get("sample_data", nxt)["next"]

# random.shuffle(pred_boxes.sample_tokens)
sample_data_tokens = list(token2id.keys())

for sample_data_token in sample_data_tokens:
    visualize_sample_data(
        nusc,
        sample_data_token,
        pred_boxes,
        eval_range=50,
        savepath=os.path.join(".", "{}.png".format(token2id[sample_data_token])),
    )

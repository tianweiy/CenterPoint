from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import time
import argparse
import numpy as np
from pub_tracker import PubTracker as Tracker
from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.eval.common.config import config_factory as track_configs
from nuscenes.eval.tracking.evaluate import TrackingEval 
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument("--assigner", type=str, default="greedy")
    parser.add_argument("--root", type=str, default="data/nuScenes")
    parser.add_argument("--version", type=str, default='v1.0-trainval')
    parser.add_argument("--max_age", type=int, default=3)
    parser.add_argument("--vis", action='store_true', default=False)

    args = parser.parse_args()

    return args


def save_first_frame():
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.root, verbose=True)
    if args.version == 'v1.0-trainval':
        scenes = splits.val
    elif args.version == 'v1.0-test':
        scenes = splits.test 
    else:
        raise ValueError("unknown")

    frames = []
    for sample in nusc.sample:
        scene_name = nusc.get("scene", sample['scene_token'])['name']
        if scene_name not in scenes:
            continue

        timestamp = sample["timestamp"] * 1e-6
        token = sample["token"]
        frame = {}
        frame['token'] = token
        frame['timestamp'] = timestamp

        # start of a sequence
        if sample['prev'] == '':
            frame['first'] = True
        else:
            frame['first'] = False
        frames.append(frame)

    # del nusc

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(args.work_dir, 'frames_meta.json'), "w") as f:
        json.dump({'frames': frames}, f)

    return nusc


def main(nusc):
    args = parse_args()
    print('Deploy OK')

    tracker = Tracker(args.assigner, args.max_age, args.work_dir)

    with open(args.checkpoint, 'rb') as f:
        predictions=json.load(f)['results']

    with open(os.path.join(args.work_dir, 'frames_meta.json'), 'rb') as f:
        frames=json.load(f)['frames']

    nusc_annos = {
        "results": {},
        "meta": None,
    }
    size = len(frames)

    print("Begin Tracking\n")
    start = time.time()
    for i in range(size):
        token = frames[i]['token']

        # reset tracking after one video sequence
        if frames[i]['first']:
            # use this for sanity check to ensure your token order is correct
            # print("reset ", i)
            tracker.reset()
            last_time_stamp = frames[i]['timestamp']

        time_lag = frames[i]['timestamp'] - last_time_stamp
        last_time_stamp = frames[i]['timestamp']

        preds = predictions[token]

        outputs = tracker.step_centertrack(preds, time_lag)
        annos = []

        for item in outputs:
            if item['active'] == 0:
                continue
            nusc_anno = {
                "sample_token": token,
                "translation": item['translation'],
                "size": item['size'],
                "rotation": item['rotation'],
                "velocity": item['velocity'],
                "tracking_id": str(item['tracking_id']),
                "tracking_name": item['detection_name'],
                "tracking_score": item['detection_score'],
            }
            annos.append(nusc_anno)
        nusc_annos["results"].update({token: annos})

        if args.vis:
            vis_path = os.path.join(args.work_dir, 'viz',
                                    str(i)+'-'+token+'.png')
            render_boxes(nusc, token, preds, outputs, vis_path)

    del nusc
    end = time.time()

    second = end-start

    speed=size / second
    print("The speed is {} FPS".format(speed))

    nusc_annos["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(args.work_dir, 'tracking_result.json'), "w") as f:
        json.dump(nusc_annos, f)
    return speed

def eval_tracking():
    args = parse_args()
    eval(os.path.join(args.work_dir, 'tracking_result.json'),
        "val",
        args.work_dir,
        args.root
    )

def render_boxes(nusc, token, dets, tracks, out_path, axes_limit=50, nsweeps=1, 
                 eval_filter=True, with_gt_boxes=True, with_map=True,
                 with_point_cloud=False, with_track_boxes=True,
                 with_det_boxes=False, conf_thresh=0.1, verbose=False):
    # Styles
    style = 'debug'
    if style == 'paper':
        text_fontsize = 10
        legend_fontsize = 'large'
        plot_linewidth = 1.5
        plot_ms = 3.0
    elif style == 'debug':
        text_fontsize = 3
        legend_fontsize = 'small'
        plot_linewidth = 0.3
        plot_ms = 0.5
    else:
        raise ValueError('unknown style')

    # Get data records
    lidar_sample_token = nusc.get('sample', token)['data']['LIDAR_TOP']
    sd_record = nusc.get('sample_data', lidar_sample_token)
    sample_rec = nusc.get('sample', sd_record['sample_token'])
    chan = sd_record['channel']
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_record = nusc.get('sample_data', ref_sd_token)

    # Get ego pose
    ego_pose = nusc.get('ego_pose', sd_record['ego_pose_token'])
    ego_translation = ego_pose['translation']
    ego_yaw = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]
    ego_quat = Quaternion(scalar=np.cos(ego_yaw / 2),
                          vector=[0, 0, np.sin(ego_yaw / 2)])

    # Get filter distance
    if eval_filter:
        cfg = track_configs("tracking_nips_2019")
        max_dist = cfg.class_range

    # Create plots
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Render map
    if with_map:
        nusc.explorer.render_ego_centric_map(
            sample_data_token=lidar_sample_token, axes_limit=axes_limit, ax=ax)

    # Render ego vehicle location
    ax.plot(0, 0, 'x', color='red')

    # Render point cloud
    if with_point_cloud:
        pc, _ = LidarPointCloud.from_file_multisweep(
            nusc, sample_rec, chan,'LIDAR_TOP', nsweeps=nsweeps)

        # Retrieve transformation matrices for reference point cloud
        cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
        ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                        rotation=Quaternion(cs_record["rotation"]))

        # Compute rotation between 3D vehicle pose and "flat" vehicle pose 
        # (parallel to global z plane)
        ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
        rotation_vehicle_flat_from_vehicle = np.dot(
            Quaternion(scalar=np.cos(ego_yaw / 2), 
                       vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
            Quaternion(pose_record['rotation']).inverse.rotation_matrix)
        vehicle_flat_from_vehicle = np.eye(4)
        vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
        viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)

        # Plot point cloud
        points = view_points(pc.points[:3, :], viewpoint, normalize=False)
        # dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
        # colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
        colors = 'lightgray'
        point_scale = 0.01
        ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

    # Render gt boxes in lidar frame
    if with_gt_boxes:
        _, gt_boxes, _ = nusc.get_sample_data(ref_sd_token,
                                            use_flat_vehicle_coordinates=True)
        num_pts = [nusc.get('sample_annotation', box.token)['num_lidar_pts'] for box in gt_boxes]
        if eval_filter:
            gt_boxes, _ = filter_boxes(gt_boxes, max_dist, None, num_pts)
        c = 'black' # np.array(nusc.colormap[box.name]) / 255.0
        for box in gt_boxes:
            box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=plot_linewidth)
            ann_record = nusc.get('sample_annotation', box.token)
            inst_id = ann_record['instance_token']
            ax.text(box.center[0], box.center[1]-0.8, inst_id[:4], c='k', fontsize=text_fontsize)
            hist_ann_record = nusc.get('instance', inst_id)
            hist_ann_token = hist_ann_record['first_annotation_token']
            gt_history = []
            while True:
                hist_ann_record = nusc.get('sample_annotation', hist_ann_token)
                gt_history.append(np.dot(ego_quat.inverse.rotation_matrix,
                    np.array(hist_ann_record['translation']) - ego_translation))
                if hist_ann_token == ann_record['token']:
                    break
                hist_ann_token = hist_ann_record['next']
            gt_history = np.array(gt_history)
            ax.plot(gt_history[:,0], gt_history[:,1], 'o-', c=c, ms=plot_ms, 
                    linewidth=plot_linewidth)

    # Render detections in lidar frame
    if with_det_boxes:
        for det in dets:
            if det['detection_score'] < conf_thresh:
                continue
            c = 'lightgray'
            box = Box(det['translation'], det['size'],
                        Quaternion(det['rotation']),
                        name=det['detection_name'])
            box.translate(-np.array(ego_translation))
            box.rotate(ego_quat.inverse)
            box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=plot_linewidth)
            # ax.scatter(box.center[0], box.center[1], 5, c, 'x')

    # Render tracks in ego frame
    if with_track_boxes:
        pred_boxes = []
        track_histories =[]
        for item in tracks:
            if item['active'] == 0 or item['detection_score'] < conf_thresh:
                continue

            # Tranform box to ego frame
            box = Box(item['translation'], item['size'],
                        Quaternion(item['rotation']),
                        label = item['tracking_id'],
                        score = item['detection_score'],
                        velocity = (item['velocity'][0], item['velocity'][1], 0),
                        name = item['detection_name'],
                        token = token)
            box.translate(-np.array(ego_translation))
            box.rotate(ego_quat.inverse)
            pred_boxes.append(box)
            track_histories.append(item['translation_history'])

        # Filter boxes according to evaluation metrics
        if eval_filter:
            pred_boxes, track_histories = filter_boxes(
                pred_boxes, max_dist, track_histories)

        # Plot track boxes
        for box in pred_boxes:
            c = cfg.tracking_colors[box.name]
            box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=plot_linewidth)
            ax.text(box.center[0], box.center[1],
                    str(box.label),
                    c='k', fontsize=text_fontsize)

        for i, track_history in enumerate(track_histories):
            c = cfg.tracking_colors[pred_boxes[i].name]
            track_history = np.array(track_history)
            track_history = track_history.T - np.array(ego_translation).reshape(3,1)
            for i in range(track_history.shape[1]):
                track_history[:,i] = np.dot(ego_quat.inverse.rotation_matrix,
                                            track_history[:,i])

            ax.plot(track_history[0,:], track_history[1,:], 'o-', c=c, 
                    ms=plot_ms, linewidth=plot_linewidth)
            # ax.scatter(track_history[0,-1], track_history[1,-1], c=c, s=2)

    # Plot format and save
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    # ax.axis('off')
    # ax.set_title('{} {labels_type}'.format(sd_record['channel'], labels_type=''))
    ax.set_aspect('equal')
    handles = [mpatches.Patch(color='grey', label='driveable area'),
        # Line2D([0], [0], label='annotation', color='black'),
        Line2D([0], [0], label='car', color=cfg.tracking_colors['car']),
        Line2D([0], [0], label='truck', color=cfg.tracking_colors['truck']),
        Line2D([0], [0], label='bus', color=cfg.tracking_colors['bus']),
        Line2D([0], [0], label='trailer', color=cfg.tracking_colors['trailer']),
        Line2D([0], [0], label='pedestrian', color=cfg.tracking_colors['pedestrian']),
        Line2D([0], [0], label='motorcycle', color=cfg.tracking_colors['motorcycle']),
        Line2D([0], [0], label='bicycle', color=cfg.tracking_colors['bicycle'])]
    ax.legend(handles=handles, fontsize=legend_fontsize, loc='upper right')

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    if verbose:
        plt.show()
    plt.close()


def filter_boxes(boxes, max_dist, track_histories=None, num_pts=None):
    gt_names_to_tracking_names = {
        'animal': 'ignore',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.wheelchair': 'ignore',
        'human.pedestrian.stroller': 'ignore',
        'human.pedestrian.personal_mobility': 'ignore',
        'human.pedestrian.police_officer': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'vehicle.car': 'car',
        'vehicle.motorcycle': 'motorcycle',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.truck': 'truck',
        'vehicle.construction': 'ignore',
        'vehicle.emergency.ambulance': 'ignore',
        'vehicle.emergency.police': 'ignore',
        'vehicle.trailer': 'trailer',
        'movable_object.barrier': 'ignore',
        'movable_object.trafficcone': 'ignore',
        'movable_object.pushable_pullable': 'ignore',
        'movable_object.debris': 'ignore',
        'static_object.bicycle_rack': 'ignore'}

    filtered_boxes = []
    filtered_histories = []
    for i, box in enumerate(boxes):
        # map gt names to tracking names
        name = box.name
        if box.name in gt_names_to_tracking_names.keys():
            name = gt_names_to_tracking_names[box.name]
        if name == 'ignore':
            continue

        # Filter boxes based on distance and number of lidar points
        if num_pts is not None:
            if num_pts[i] == 0:
                continue
        if np.sqrt(np.sum(box.center[:2] ** 2)) >= max_dist[name]:
            continue

        # TODO: Filter bike racks
        filtered_boxes.append(box)
        if track_histories is not None:
            filtered_histories.append(track_histories[i])
    return filtered_boxes, filtered_histories


def eval(res_path, eval_set="val", output_dir=None, root_path=None):
    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=root_path,
    )
    metrics_summary = nusc_eval.main()


def test_time(nusc):
    speeds = []
    for i in range(3):
        speeds.append(main(nusc))

    print("Speed is {} FPS".format( max(speeds)  ))

if __name__ == '__main__':
    nusc = save_first_frame()
    main(nusc)
    # test_time()
    eval_tracking()

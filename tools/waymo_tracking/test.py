from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 
import json
import numpy as np
import time
import copy
import argparse
import copy
import json
import os
import numpy as np
from tools.waymo_tracking.tracker import PubTracker as Tracker
from tqdm import tqdm
import json 
import time
from nuscenes.utils.geometry_utils import transform_matrix
import pickle 
from pyquaternion import Quaternion
from det3d.datasets.waymo.waymo_common import _create_pd_detection

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--info_path", type=str
    )
    parser.add_argument("--max_age", type=int, default=3)
    parser.add_argument("--vehicle", type=float, default=0.8) 
    parser.add_argument("--pedestrian", type=float, default=0.4)  
    parser.add_argument("--cyclist", type=float, default=0.6)  
    parser.add_argument("--score_thresh", type=float, default=0.75)

    args = parser.parse_args()

    return args

def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 

def veh_pos_to_transform(veh_pos):
    "convert vehicle pose to two transformation matrix"
    rotation = veh_pos[:3, :3] 
    tran = veh_pos[:3, 3]

    global_from_car = transform_matrix(
        tran, Quaternion(matrix=rotation), inverse=False
    )

    car_from_global = transform_matrix(
        tran, Quaternion(matrix=rotation), inverse=True
    )

    return global_from_car, car_from_global

def reorganize_info(infos):
    new_info = {}

    for info in infos:
        token = info['token']
        new_info[token] = info

    return new_info 

def main():
    args = parse_args()
    print('Deploy OK')

    max_dist = {
        'VEHICLE': args.vehicle,
        'PEDESTRIAN': args.pedestrian,
        'CYCLIST': args.cyclist
    }

    tracker = Tracker(max_age=args.max_age, max_dist=max_dist, score_thresh=args.score_thresh)

    with open(args.checkpoint, 'rb') as f:
        predictions=pickle.load(f)

    with open(args.info_path, 'rb') as f:
        infos=pickle.load(f)
        infos = reorganize_info(infos)

    global_preds, detection_results = convert_detection_to_global_box(predictions, infos)
    size = len(global_preds)

    print("Begin Tracking {} frames\n".format(size))

    predictions = {} 

    for i in tqdm(range(size)):
        pred = global_preds[i]
        token = pred['token']

        # reset tracking after one video sequence
        if pred['frame_id'] == 0:
            tracker.reset()
            last_time_stamp = pred['timestamp']

        time_lag = (pred['timestamp'] - last_time_stamp) 
        last_time_stamp = pred['timestamp']

        current_det = pred['global_boxs']

        outputs = tracker.step_centertrack(current_det, time_lag)
        tracking_ids = []
        box_ids = [] 

        for item in outputs:
            if item['active'] == 0:
                continue 
            
            box_ids.append(item['box_id'])
            tracking_ids.append(item['tracking_id'])

        # now reorder 
        detection = detection_results[token]

        remained_box_ids = np.array(box_ids)

        track_result = {} 

        # store box id 
        track_result['tracking_ids']= np.array(tracking_ids)   

        # store box parameter 
        track_result['box3d_lidar'] = detection['box3d_lidar'][remained_box_ids]

        # store box label 
        track_result['label_preds'] = detection['label_preds'][remained_box_ids]

        # store box score 
        track_result['scores'] = detection['scores'][remained_box_ids]

        predictions[token] = track_result 

    os.makedirs(args.work_dir, exist_ok=True)
    # save prediction files to args.work_dir 
    _create_pd_detection(predictions, infos, args.work_dir, tracking=True)

    result_path = os.path.join(args.work_dir, 'tracking_pred.bin')
    gt_path = os.path.join(args.work_dir, '../gt_preds.bin')

    print("Use Waymo devkit or online server to evaluate the result")
    print("After building the devkit, you can use the following command")
    print("waymo-open-dataset/bazel-bin/waymo_open_dataset/metrics/tools/compute_tracking_metrics_main \
           {}  {} ".format(result_path, gt_path))
    
    # os.system("waymo_open_dataset/metrics/tools/compute_tracking_metrics_main \
    #       {}  {} ".format(result_path, gt_path))

def transform_box(box, pose):
    """Transforms 3d upright boxes from one frame to another.
    Args:
    box: [..., N, 7] boxes.
    from_frame_pose: [...,4, 4] origin frame poses.
    to_frame_pose: [...,4, 4] target frame poses.
    Returns:
    Transformed boxes of shape [..., N, 7] with the same type as box.
    """
    transform = pose 
    heading = box[..., -1] + np.arctan2(transform[..., 1, 0], transform[..., 0,
                                                                    0])
    center = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    box[..., 0:3]) + np.expand_dims(
                        transform[..., 0:3, 3], axis=-2)

    velocity = box[..., [6, 7]] 

    velocity = np.concatenate([velocity, np.zeros((velocity.shape[0], 1))], axis=-1) # add z velocity

    velocity = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    velocity)[..., [0, 1]] # remove z axis 

    return np.concatenate([center, box[..., 3:6], velocity, heading[..., np.newaxis]], axis=-1)

def label_to_name(label):
    if label == 0:
        return "VEHICLE"
    elif label == 1 :
        return "PEDESTRIAN"
    elif label == 2:
        return "CYCLIST"
    else:
        raise NotImplemented()

def sort_detections(detections):
    indices = [] 

    for det in detections:
        f = det['token']
        seq_id = int(f.split("_")[1])
        frame_id= int(f.split("_")[3][:-4])

        idx = seq_id * 1000 + frame_id
        indices.append(idx)

    rank = list(np.argsort(np.array(indices)))

    detections = [detections[r] for r in rank]

    return detections

def convert_detection_to_global_box(detections, infos):
    ret_list = [] 

    detection_results = {} # copy.deepcopy(detections)

    for token in tqdm(infos.keys()):
        detection = detections[token]
        detection_results[token] = copy.deepcopy(detection)

        info = infos[token]
        # pose = get_transform(info)
        anno_path = info['anno_path']
        ref_obj = get_obj(anno_path)
        pose = np.reshape(ref_obj['veh_to_global'], [4, 4])

        box3d = detection["box3d_lidar"].detach().clone().cpu().numpy() 
        labels = detection["label_preds"].detach().clone().cpu().numpy()
        scores = detection['scores'].detach().clone().cpu().numpy()
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2
        box3d[:, [3, 4]] = box3d[:, [4, 3]]

        box3d = transform_box(box3d, pose)

        frame_id = token.split('_')[3][:-4]

        num_box = len(box3d)

        anno_list =[]
        for i in range(num_box):
            anno = {
                'translation': box3d[i, :3],
                'velocity': box3d[i, [6, 7]],
                'detection_name': label_to_name(labels[i]),
                'score': scores[i], 
                'box_id': i 
            }

            anno_list.append(anno)

        ret_list.append({
            'token': token, 
            'frame_id':int(frame_id),
            'global_boxs': anno_list,
            'timestamp': info['timestamp'] 
        })

    sorted_ret_list = sort_detections(ret_list)

    return sorted_ret_list, detection_results 

if __name__ == '__main__':
    main()

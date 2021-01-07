import argparse
import copy
import json
import os
import sys

import numpy as np
import pickle 
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, Box, RadarPointCloud
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.geometry_utils import points_in_box
from functools import reduce
from tqdm import tqdm
from det3d.core import box_torch_ops
from collections import defaultdict 
import torch 
import glob 

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Models")
    parser.add_argument("ensemble_dir", help="path to a dir that contains all prediction file")
    parser.add_argument("--output_path", help="the path to save ensemble output")    
    parser.add_argument("--data_root", type=str, default="data/nuScenes/v1.0-trainval") 
    
    args = parser.parse_args()

    return args


def get_sample_data(pred):
    box_list = [] 
    score_list = [] 
    pred = pred.copy() 

    for item in pred:    
        box =  Box(item['translation'], item['size'], Quaternion(item['rotation']),
                name=item['detection_name'])
        score_list.append(item['detection_score'])
        box_list.append(box)

    top_boxes = reorganize_boxes(box_list)
    top_scores = np.array(score_list).reshape(-1)

    return top_boxes, top_scores

def reorganize_boxes(box_lidar_nusc):
    rots = []
    centers = []
    wlhs = []
    for i, box_lidar in enumerate(box_lidar_nusc):
        v = np.dot(box_lidar.rotation_matrix, np.array([1, 0, 0]))
        rot = np.arctan2(v[1], v[0])

        rots.append(-rot- np.pi / 2)
        centers.append(box_lidar.center)
        wlhs.append(box_lidar.wlh)

    rots = np.asarray(rots)
    centers = np.asarray(centers)
    wlhs = np.asarray(wlhs)
    gt_boxes_lidar = np.concatenate([centers.reshape(-1,3), wlhs.reshape(-1,3), rots[..., np.newaxis].reshape(-1,1) ], axis=1)
    
    return gt_boxes_lidar

def reorganize_pred_by_class(pred):
    ret_dicts = defaultdict(list)
    for item in pred: 
        ret_dicts[item['detection_name']].append(item) 

    return ret_dicts

def concatenate_list(lists):
    ret = []
    for l in lists:
        ret += l 

    return ret 

ENS_CLASS = ['car', 'truck', 'bus', 'construction_vehicle', 'bicycle']
SMALL_CLASS = ['pedestrian', 'barrier', 'traffic_cone', 'motorcycle']
LARGE_CLASS = ['trailer']
ALL_CLASS = ['car', 'truck', 'bus', 'construction_vehicle', 'bicycle', 'pedestrian', 'barrier', 'traffic_cone', 'motorcycle', 'trailer']

def filter_pred_by_class(preds, small=False, large=False):
    ret_dict = {} 
    for token, pred in preds.items():
        filtered = []

        for item in pred:
            assert item['detection_name'] in ALL_CLASS

            if small:
                if item['detection_name'] not in LARGE_CLASS:
                    filtered.append(item)
            elif large:
                if item['detection_name'] not in SMALL_CLASS:
                    filtered.append(item)

        ret_dict[token] = filtered

    return ret_dict 

def get_pred(path):
    with open(path, 'rb') as f:
        pred=pickle.load(f)

    return pred

def main():
    args = parse_args()

    pred_paths = glob.glob(os.path.join(args.ensemble_dir, '*.pkl'))
    print(pred_paths)

    preds = []
    for path in pred_paths:
        preds.append(get_pred(path))

    merged_predictions = {}
    for token in preds[0].keys():
        annos = [pred[token] for pred in preds]

        merged_predictions[token] = concatenate_list(annos) 

    predictions = merged_predictions
    
    print("Finish Merging")

    nusc_annos = {
        "results": {},
        "meta": None,
    }

    for sample_token, prediction in tqdm(predictions.items()):
        annos = []

        # reorganize pred by class 
        pred_dicts = reorganize_pred_by_class(prediction)

        for name, pred in pred_dicts.items():
            # in global coordinate 
            top_boxes, top_scores = get_sample_data(pred)

            with torch.no_grad():
                top_boxes_tensor = torch.from_numpy(top_boxes)
                boxes_for_nms = top_boxes_tensor[:, [0, 1, 2, 4, 3, 5, -1]]
                boxes_for_nms[:, -1] = boxes_for_nms[:, -1] + np.pi /2 
                top_scores_tensor = torch.from_numpy(top_scores)

                selected = box_torch_ops.rotate_nms(boxes_for_nms, top_scores_tensor, 
                    pre_max_size=None,
                    post_max_size=50,  
                    iou_threshold=0.2,
                ).numpy()
        
            pred = [pred[s] for s in selected]

            annos.extend(pred)

        nusc_annos["results"].update({sample_token: annos})

    nusc_annos["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": True,
        "use_map": False,
        "use_external": False,
    }

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(args.work_dir, 'result.json'), "w") as f:
        json.dump(nusc_annos, f)
    
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.data_root, verbose=True)
    cfg = config_factory("cvpr_2019")
    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=os.path.join(args.work_dir, 'result.json'),
        eval_set='val',
        output_dir=args.work_dir,
        verbose=True,
    )
    metrics_summary = nusc_eval.main(plot_examples=0,)


if __name__ == "__main__":
    main()

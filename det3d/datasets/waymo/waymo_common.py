import os.path as osp
import numpy as np
import pickle
import random

from pathlib import Path
from functools import reduce
from typing import Tuple, List
import os 
import json 
from tqdm import tqdm
import argparse

from tqdm import tqdm
try:
    import tensorflow as tf
    tf.enable_eager_execution()
except:
    print("No Tensorflow")


INDEX_LENGTH = 15
MAX_FRAME = 1000000
CAT_NAME_MAP = {
    1: 'VEHICLE',
    2: 'PEDESTRIAN',
    3: 'SIGN',
    4: 'CYCLIST',
}
CAT_NAME_TO_ID = {
    'VEHICLE': 1,
    'PEDESTRIAN': 2,
    'SIGN': 3,
    'CYCLIST': 4,
}
TYPE_LIST = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 

def label_to_type(label):
    if label <= 1:
        return int(label) + 1
    else:
        return 4

def _create_pd_detection(detections, infos, result_path, tracking=False):
    """Creates a prediction objects file."""
    assert tracking is False, "Not Supported Yet"
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
    from waymo_open_dataset.utils import box_utils

    objects = metrics_pb2.Objects()

    for token, detection in tqdm(detections.items()):
        info = infos[token]
        obj = get_obj(info['path'])

        box3d = detection["box3d_lidar"].detach().cpu().numpy()
        scores = detection["scores"].detach().cpu().numpy()
        labels = detection["label_preds"].detach().cpu().numpy()
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2

        if box3d.shape[1] > 7:
            # drop velocity
            box3d = box3d[:, [0, 1, 2, 3, 4, 5, -1]]

        for i in range(box3d.shape[0]):
            det  = box3d[i]
            score = scores[i]

            label = labels[i]

            o = metrics_pb2.Object()
            o.context_name = obj['scene_name']
            o.frame_timestamp_micros = int(obj['frame_name'].split("_")[-1])

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = det[0]
            box.center_y = det[1]
            box.center_z = det[2]
            box.length = det[3]
            box.width = det[4]
            box.height = det[5]
            box.heading = det[-1]
            o.object.box.CopyFrom(box)
            o.score = score
            # Use correct type.
            o.object.type = label_to_type(label) # int(label)+1

            objects.objects.append(o)

    # Write objects to a file.
    f = open(os.path.join(result_path, 'my_preds.bin'), 'wb')
    f.write(objects.SerializeToString())
    f.close()


def _test_create_pd_detection(infos, tracking=False):
    """Creates a prediction objects file."""
    assert tracking is False, "Not Supported Yet" 
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
    from waymo_open_dataset.utils import box_utils
    
    objects = metrics_pb2.Objects()

    for idx in tqdm(range(len(infos))): 
        info = infos[idx]

        obj = get_obj(info['path'])
        annos = obj['objects']
        num_points_in_gt = np.array([ann['num_points'] for ann in annos])
        box3d = np.array([ann['box'] for ann in annos])

        if len(box3d) == 0:
            continue 

        names = np.array([TYPE_LIST[ann['label']] for ann in annos])
        
        if box3d.shape[1] > 7:
            # drop velocity
            box3d = box3d[:, [0, 1, 2, 3, 4, 5, -1]]


        for i in range(box3d.shape[0]):
            if num_points_in_gt[i] == 0:
                continue 
            if names[i] == 'UNKNOWN':
                continue 

            det  = box3d[i]
            score = 1.0
            label = names[i]

            o = metrics_pb2.Object()
            o.context_name = obj['scene_name']
            o.frame_timestamp_micros = int(obj['frame_name'].split("_")[-1])

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = det[0]
            box.center_y = det[1]
            box.center_z = det[2]
            box.length = det[3]
            box.width = det[4]
            box.height = det[5]
            box.heading = det[-1]
            o.object.box.CopyFrom(box)
            o.score = score
            # Use correct type.
            o.object.type = CAT_NAME_TO_ID[label]
            o.object.num_lidar_points_in_box = num_points_in_gt[i]

            objects.objects.append(o)
        
    # Write objects to a file.
    f = open(os.path.join(args.result_path, 'gt_preds.bin'), 'wb')
    f.write(objects.SerializeToString())
    f.close()
    


def _fill_infos(root_path, frames, split='train', nsweeps=1):
    # load all train infos
    infos = []
    for frame_name in tqdm(frames):  # global id
        path = os.path.join(root_path, split, frame_name)

        info = {
            "path": path,
            "token": frame_name,
        }

        infos.append(info)
    return infos

def get_available_frames(root, split):
    dir_path = os.path.join(root, split)
    available_frames = list(os.listdir(dir_path))

    print(split, " split ", "exist frame num:", len(available_frames))
    return available_frames


def create_waymo_infos(root_path, split='train', nsweeps=1):
    assert split != 'test', "Not Supported Yet"

    frames = get_available_frames(root_path, split)

    waymo_infos = _fill_infos(
        root_path, frames, split, nsweeps
    )

    print(
        f"sample: {len(waymo_infos)}"
    )
    with open(
        os.path.join(root_path, "infos_"+split+"_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps)), "wb"
    ) as f:
        pickle.dump(waymo_infos, f)

def parse_args():
    parser = argparse.ArgumentParser(description="Waymo 3D Extractor")
    parser.add_argument("--path", type=str, default="data/Waymo/tfrecord_training")
    parser.add_argument("--info_path", type=str)
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--gt", action='store_true' )
    args = parser.parse_args()
    return args


def reorganize_info(infos):
    new_info = {}

    for info in infos:
        token = info['token']
        new_info[token] = info

    return new_info 

if __name__ == "__main__":
    args = parse_args()

    with open(args.info_path, 'rb') as f:
        infos = pickle.load(f)
    
    if args.gt:
        _test_create_pd_detection(infos)
        assert 0  

    infos = reorganize_info(infos)
    with open(args.path, 'rb') as f:
        preds = pickle.load(f)
    _create_pd_detection(preds, infos, args.result_path)

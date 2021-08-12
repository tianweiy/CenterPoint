import argparse
import copy
import json
import os
import sys

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import load_checkpoint
import pickle 
import time 
from matplotlib import pyplot as plt 
from det3d.torchie.parallel import collate, collate_kitti
from torch.utils.data import DataLoader
import matplotlib.cm as cm
import subprocess
import cv2
from tools.demo_utils import visual 
from collections import defaultdict

def convert_box(info):
    boxes =  info["gt_boxes"].astype(np.float32)
    names = info["gt_names"]

    assert len(boxes) == len(names)

    detection = {}

    detection['box3d_lidar'] = boxes

    # dummy value 
    detection['label_preds'] = np.zeros(len(boxes)) 
    detection['scores'] = np.ones(len(boxes))

    return detection 

def main():
    cfg = Config.fromfile('configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo.py')
    
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    dataset = build_dataset(cfg.data.val)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_kitti,
        pin_memory=False,
    )

    checkpoint = load_checkpoint(model, 'work_dirs/centerpoint_pillar_512_demo/latest.pth', map_location="cpu")
    model.eval()

    model = model.cuda()

    cpu_device = torch.device("cpu")

    points_list = [] 
    gt_annos = [] 
    detections  = [] 

    for i, data_batch in enumerate(data_loader):
        info = dataset._nusc_infos[i]
        gt_annos.append(convert_box(info))

        points = data_batch['points'][:, 1:4].cpu().numpy()
        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=0,
            )
        for output in outputs:
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.append(output)

        points_list.append(points.T)
    
    print('Done model inference. Please wait a minute, the matplotlib is a little slow...')
    
    for i in range(len(points_list)):
        visual(points_list[i], gt_annos[i], detections[i], i)
        print("Rendered Image {}".format(i))
    
    image_folder = 'demo'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda img_name: int(img_name.split('.')[0][4:]))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    cv2_images = [] 

    for image in images:
        cv2_images.append(cv2.imread(os.path.join(image_folder, image)))

    for img in cv2_images:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    print("Successfully save video in the main folder")

if __name__ == "__main__":
    main()

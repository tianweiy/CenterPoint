#!/bin/bash

python tools/tracking/pub_test.py --work_dir work_dirs/nusc_centerpoint_voxelnet_01voxel_track_circle_nms  --checkpoint work_dirs/nusc_centerpoint_voxelnet_01voxel_circle_nms/infos_val_10sweeps_withvelo_filter_True.json  --max_age 3

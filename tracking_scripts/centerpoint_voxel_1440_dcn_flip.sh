#!/bin/bash

python tools/tracking/pub_test.py --work_dir work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_track  --checkpoint work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip/infos_val_10sweeps_withvelo_filter_True.json  --max_age 3

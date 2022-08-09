# MODEL ZOO 

### Common settings and notes

- The experiments are run with PyTorch 1.1, CUDA 10.0, and CUDNN 7.5.
- The training is conducted on 4 V100 GPUs in a DGX server. 
- Testing times are measured on a TITAN RTX GPU with batch size 1. 
 
## Waymo 3D Detection 

We provide training / validation configurations, pretrained models, and prediction files for all models in the paper. To access these pretrained models, please send us an [email](mailto:yintianwei@utexas.edu) with your name, institute, a screenshot of the the Waymo dataset registration confirmation mail, and your intended usage. Please send a second email if we don't get back to you in two days. Please note that Waymo open dataset is under strict non-commercial license so we are not allowed to share the model with you if it will used for any profit-oriented activities.     

### One-stage VoxelNet 
| Model   | Veh_L2 | Ped_L2 | Cyc_L2  | MAPH   | FPS  |
|---------|--------|--------|---------|--------|------------|
| [VoxelNet](voxelnet/waymo_centerpoint_voxelnet_3x.py) | 66.2 | 62.6 | 67.6 | 65.5 | 13 | 

In the paper, our models only detect Vehicle and Pedestrian. Here, we provide the three classes config that also enables cyclist detection (and perform similarly). We encourage the community to also report three class performance in the future.

### Ablations for training schedule 

CenterPoint is fast to train and converge in as little as 3~6 epochs. We tried a few training schedules for CenterPoint-Voxel and list their performance below.

| Schedule   | Veh_L2 | Ped_L2 | Cyc_L2  | MAPH   | Training Time  |
|------------|--------|--------|---------|--------|----------------|
| [36 epoch](voxelnet/waymo_centerpoint_voxelnet_3x.py) | 66.2 | 62.6 | 67.6 | 65.5 | 84hr |
| [12 epoch](voxelnet/waymo_centerpoint_voxelnet_1x.py) | 65.6 | 61.3 | 67.1 | 64.7 | 28hr | 
| [6 epoch](voxelnet/waymo_centerpoint_voxelnet_6epoch.py) | 65.5 | 59.5 | 66.4 | 63.4 | 14hr | 
| [3 epoch](voxelnet/waymo_centerpoint_voxelnet_3epoch.py) | 61.5 | 56.2 | 64.5 | 60.7 | 7hr | 

### Two-stage VoxelNet

By default, we finetune a pretrained [one stage model](voxelnet/waymo_centerpoint_voxelnet_3x.py) for 6 epochs. To save GPU memory, we also freeze the backbone weight.  

| Model   | Split | Veh_L2 | Ped_L2 | Cyc_L2  | MAPH   | FPS  |
|------------|----|----|--------|---------|--------|----------------|
| [VoxelNet](voxelnet/two_stage/waymo_centerpoint_voxelnet_two_stage_bev_5point_ft_6epoch_freeze.py) | Val | 67.9 | 65.6 | 68.6 | 67.4 | 13 | 
| [VoxelNet](voxelnet/two_stage/waymo_centerpoint_voxelnet_two_stage_bev_5point_ft_6epoch_freeze.py) | Test| 71.9 | 67.0 |  68.2| 69.0 | 13 | 


### Two frame model

To provide richer input information and enable a more reasonable velocity estimation, we transform and merge the Lidar points of previous frame into current frame. This two frame model significanty boosts the detection performance.  

| Model   | Split | Veh_L2 | Ped_L2 | Cyc_L2  | MAPH   | FPS  |
|------------|----|----|--------|---------|--------|----------------|
| [One-stage](voxelnet/waymo_centerpoint_voxelnet_two_sweeps_3x_with_velo.py) | Val | 67.3 | 67.5 | 69.9 | 68.2 | 11 |  
| [Two-stage](voxelnet/two_stage/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel.py) | Val | 69.7 | 70.3 | 70.9 | 70.3 | 11 | 
| [Two-stage](voxelnet/two_stage/waymo_centerpoint_voxelnet_two_sweep_two_stage_bev_5point_ft_6epoch_freeze_with_vel.py) | Test | 73.0 | 71.5 | 71.3 | 71.9 | 11 |  


### PointPillars 

| Model   | Veh_L2 | Ped_L2 | Cyc_L2  | MAPH   | FPS  |
|---------|--------|--------|---------|--------|------------|
| [centerpoint_pillar](pp/waymo_centerpoint_pp_two_pfn_stride1_3x.py) | 65.5 | 55.1 | 60.2 | 60.3 | 19 | 
| [centerpoint_pillar_two_stage](pp/two_stage/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch.py) | 66.7 | 55.9 | 61.7 | 61.4 | 16 | 

For PointPillars, we notice a 1.5 mAPH drop when converting from two class model to three class model. You can refer to [ONE_STAGE](pp/waymo_centerpoint_pp_two_cls_two_pfn_stride1_3x.py) and [TWO_STAGE](pp/two_stage/waymo_centerpoint_pp_two_cls_two_pfn_stride1_two_stage_bev_6epoch.py) configs to reproduce the two class result.

## Waymo 3D Tracking 

For 3D Tracking, we apply our center-based tracking on top of our two frame model's detection result.  

|         | Split | Veh_L2 | Ped_L2 | Cyc_L2  | MOTA   |  FPS  |
|---------|---------|--------|--------|---------|--------|-------|
| [centerpoint_voxel_two_sweep](../../tracking_scripts/centerpoint_voxel_two_sweep_val.sh)| Val |  55.0   | 55.0      | 57.4  | 55.8 |  11    | 
| [centerpoint_voxel_two_sweep](../../tracking_scripts/centerpoint_voxel_two_sweep_test.sh)| Test | 59.4     |  56.6      |   60.0      | 58.7       |  11    | 

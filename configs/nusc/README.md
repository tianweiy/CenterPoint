# MODEL ZOO 

### Common settings and notes

- The experiments are run with PyTorch 1.1, CUDA 10.0, and CUDNN 7.5.
- The training is conducted on 4 V100 GPUs in a DGX server. 
- Testing times are measured on a TITAN RTX GPU with batch size 1. 
 
## nuScenes 3D Detection 

**We provide training / validation configurations, logs, pretrained models, and prediction files for all models in the paper**

### VoxelNet 
| Model                 | Validation MAP  | Validation NDS  | Link          |
|-----------------------|-----------------|-----------------|---------------|
| [centerpoint_voxel_1440](voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py) |59.6  | 66.8 | [URL](https://mitprod-my.sharepoint.com/:f:/g/personal/tianweiy_mit_edu/EhgzjwV2EghOnHFKyRgSadoBr2kUo7yPu52N-I3dG3c5dA?e=a9MdhX)  |

Please refer to [LINK](https://github.com/tianweiy/CenterPoint/issues/249) for centerpoint detection predicitons on nuScenes train/val/test sets.

### VoxelNet(depreacted) 

These results are obtained before the sync bn bug fix + z axis augmentation . 

| Model                 | FPS              | Validation MAP  | Validation NDS  | Link          |
|-----------------------|------------------|-----------------|-----------------|---------------|
| [centerpoint_voxel_1024](voxelnet/nusc_centerpoint_voxelnet_01voxel.py) | 16 | 56.4 | 64.8 | [URL](https://mitprod-my.sharepoint.com/:f:/g/personal/tianweiy_mit_edu/EhT7DKpbj6VDin12xN42PYYB8UqkFTha-qb1F5srEE5UXQ?e=mVaJkC) |


### PointPillars 

| Model                 | FPS       | Validation MAP  | Validation NDS  | Link          |
|-----------------------|-----------------|-----------------|-----------------|---------------|
| [centerpoint_pillar](pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py) | 31 | 50.3 | 60.2 | [URL](https://mitprod-my.sharepoint.com/:f:/g/personal/tianweiy_mit_edu/EkN9vbDmXMJCtSn6dgBLE4wBA1PL96U6MbGhh3lME_G6wA?e=vjhpd2) |


## nuScenes 3D Tracking 

| Model                 | Tracking time | Total time   | Validation AMOTA ↑ | Validation AMOTP ↓ | Link          |
|-----------------------|-----------|------------------|------------------|-------------------|---------------|
| centerpoint_voxel_1024 | 1ms | 64ms | 63.7* | 0.606  | [URL](https://mitprod-my.sharepoint.com/:f:/g/personal/tianweiy_mit_edu/Epy78yQMnZlCuMBWPlUtQ3oBWqQQ2fArTs637DlBHdaHIw?e=q6a2bA) |


*The numbers are from the centerpoint_voxel_1024 config (before the sync bn bug fix + z axis augmentation). Current detection [models](voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py) should perform slightly better.

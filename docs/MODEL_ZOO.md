# MODEL ZOO 

### Common settings and notes

- The experiments are run with PyTorch 1.1, CUDA 10.0, and CUDNN 7.5.
- The training is conducted on 4 V100 GPUs in a DGX server. 
- Testing times are measured on a TITAN Xp GPU with batch size 1. 
 
## nuScenes 3D Detection 

**We provide training / validation configurations, logs, pretrained models, and prediction files for all models in the paper**

### VoxelNet / SECOND / CBGS 
| Model                 | Test time        | Validation MAP  | Validation NDS  | Link          |
|-----------------------|------------------|-----------------|-----------------|---------------|
| [cbgs_reimplemented](../configs/cbgs/nusc_cbgs_01voxel.py) | 78ms | 51.9 | 62.2 | [URL](https://drive.google.com/drive/folders/1BVyTp_RhLDnVMTm7m-qIQYfBRpjtBKqV?usp=sharing) | 
| [centerpoint_voxel_1024](../configs/centerpoint/nusc_centerpoint_voxelnet_01voxel.py) | 76ms | 55.6 | 64.0 | [URL](https://drive.google.com/drive/folders/13yeETy4jbMRTKupMZklW5dmBVoscQ4iO?usp=sharing) |
| [centerpoint_voxel_1024_circle_nms](../configs/centerpoint/nusc_centerpoint_voxelnet_01voxel_circle_nms.py) | 69ms | 55.4 | 63.8 | [URL](https://drive.google.com/drive/folders/1h4v0m6b-8sTKkBc184arT2SqP6adBD9l?usp=sharing) |
| [centerpoint_voxel_1024_dcn_circle_nms](../configs/centerpoint/nusc_centerpoint_voxelnet_dcn_01voxel_circle_nms.py) | 76ms | 55.4 | 63.4 | [URL](https://drive.google.com/drive/folders/1Ig5uzr58cZidkuxcdmx86hTKsZgBbE72?usp=sharing) |
| [centerpoint_voxel_1440_circle_nms](../configs/centerpoint/nusc_centerpoint_voxelnet_0075voxel_circle_nms.py) | 101ms | 56.1 | 64.5 | [URL](https://drive.google.com/drive/folders/12U8yzNMgb64tz85cI_d697FCAEvoZdyF?usp=sharing)|
| [centerpoint_voxel_1440_dcn_circle_nms](../configs/centerpoint/nusc_centerpoint_voxelnet_dcn_0075voxel_circle_nms.py) | 118ms | 56.5 | 65.0 | [URL](https://drive.google.com/drive/folders/1KC2NMU4orxlpCSCZXxLNBpXZfA8PL6ZS?usp=sharing) |
| [centerpoint_voxel_1440_dcn_flip_circle_nms](../configs/centerpoint/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_circle_nms.py) | 440ms | 58.8 | 66.9 | [URL](https://drive.google.com/drive/folders/1PcFHG7mJ57xKDtMMeKC5p4MKShOcoFHh?usp=sharing) 
| [centerpoint_voxel_1440_dcn_flip_rotated_nms](../configs/centerpoint/nusc_centerpoint_voxelnet_dcn_0075voxel_flip.py) | 449ms | 59.1 | 67.1| [URL](https://drive.google.com/drive/folders/1-0o-QhLM9J6wFP3BRCWD5bxJN5zDKAre?usp=sharing) 


### PointPillars 

| Model                 | Test time       | Validation MAP  | Validation NDS  | Link          |
|-----------------------|-----------------|-----------------|-----------------|---------------|
| [pointpillars_reimplemented](../configs/point_pillars/nusc_pp_02voxel.py) | 42ms | 45.5 | 58.4 | [URL](https://drive.google.com/drive/folders/1kozrn-hWaC8lxsJ8hCUyg5bV3kxATdoE?usp=sharing) |
| [centerpoint_pillar_512](../configs/centerpoint/nusc_centerpoint_pp_02voxel.py) | 41ms | 48.3 | 59.1 | [URL](https://drive.google.com/drive/folders/1iZiez0XCxN0ptrM-TR2eIWXRSMXmyOyd?usp=sharing) |
| [centerpoint_pillar_512_circle_nms](../configs/centerpoint/nusc_centerpoint_pp_02voxel_circle_nms.py) | 33ms | 48.3 | 59.1 | [URL](https://drive.google.com/drive/folders/1ueOyXeEP2LWd9E1zTdxQHbDxN3y6DMuQ?usp=sharing) |
| [centerpoint_pillar_512_dcn_circle_nms](../configs/centerpoint/nusc_centerpoint_pp_dcn_02voxel_circle_nms.py) | 41ms | 48.6 | 59.4 | [URL](https://drive.google.com/drive/folders/1jlZ5d6Hmag5_aOZ8VvmK0ai_y1q5utJ1?usp=sharing)|


## nuScenes 3D Tracking 

| Model                 | Tracking time | Total time   | Validation AMOTA ↑ | Validation AMOTP ↓ | Link          |
|-----------------------|-----------|------------------|------------------|-------------------|---------------|
| [centerpoint_megvii_detection](../tracking_scripts/megvii.sh) | 1ms | 192ms | 59.8 | 0.682 | [URL](https://drive.google.com/drive/folders/1s0PxY2ar6FMm8ZTIAVajMdudtneLuvxh?usp=sharing) |
| [centerpoint_pillar_512_circle_nms](../tracking_scripts/centerpoint_pillar_512_circle_nms.sh) | 1ms | 34ms | 54.2 | 0.657 | [URL](https://drive.google.com/drive/folders/1lcEoZxD_3R3Kd_8SLxX3R2wKSoMVXQKt?usp=sharing) |
| [centerpoint_voxel_1024_circle_nms](../tracking_scripts/centerpoint_voxel_1024_circle_nms.sh) | 1ms | 70ms | 62.6 | 0.630 | [URL](https://drive.google.com/drive/folders/1pKq4ic9oBAAx6MjQZAzeV7Fy9Iyv8NoV?usp=sharing) |
| [centerpoint_voxel_1440_dcn_flip_circle_nms](../tracking_scripts/centerpoint_voxel_1440_dcn_flip_circle_nms.sh) | 1ms | 441ms | 65.5 | 0.586 | [URL](https://drive.google.com/drive/folders/1WitTr4BoBFWgaCqS41QCOL1B8DU0sbwO?usp=sharing) |
| [centerpoint_voxel_1440_dcn_flip_rotated_nms](../tracking_scripts/centerpoint_voxel_1440_dcn_flip.sh) | 1ms | 451ms | 65.9 | 0.567 | [URL](https://drive.google.com/drive/folders/1HChuDGc_jBmI1mKPy3FpwQItqLUU5t9Z?usp=sharing) |


- `centerpoint_megvii_detection` is our tracking algorithm with the public detection obtaiend from the [nuScenes website](https://www.nuscenes.org/tracking?externalData=all&mapData=all&modalities=Any)

## nuScenes test set Detection/Tracking
**Our testset submission uses a single CenterPoint-Voxel model with flip test. It is an earlier version of our model that didn't use the Circle-NMS. Feel free to adapt our models to your framework and beat us on the leaderboard.**
### Detection

| Model                 | Test MAP  | Test NDS  | Link          |
|-----------------------|-----------|-----------|---------------|
| [centerpoint_voxel_1440_dcn_flip(Rotated NMS)](../configs/centerpoint/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset.py) | 60.3 | 67.3 | [Detection](https://drive.google.com/file/d/1GJzIBJKxg4NVFXF0SeBzmrL87ALuIEx0/view?usp=sharing) |

### Tracking
| Model                 | Test AMOTA |  Test AMOTP   | Link  |
|-----------------------|------------|---------------|-------|
| [centerpoint_voxel_1440_dcn_flip(Rotated NMS)](../tracking_scripts/centerpoint_voxel_1440_dcn_flip_testset.sh) | 63.8 | 0.555 | [Tracking](https://drive.google.com/file/d/1evPKLwzlJB5QeECCjDWyla-CXzK0F255/view?usp=sharing)|  


#### Notes
- Training on 8 GPUs is OK, if the [linear learning rate rule](https://arxiv.org/abs/1706.02677) is applied. 
- We don't have result for 2 GPU / smaller batch size training. Though it should also work with linear learning rate rule.  
- If you face the out of cpu memory error for training, please reduce the num_worker
- The io/CPU, and num_worker for the dataloader are crucial for the training speed. By default, we use 8 worker. 
- We observe up to 0.2 nuScenes map jittering due to the randomness of cudnn and fixed voxelization. It seems that turning on cudnn batchnorm gives slightly better testing speed and accuracy.    



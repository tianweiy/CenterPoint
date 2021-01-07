## Getting Started with CenterPoint on nuScenes
Modified from [det3d](https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)'s original document.

### Prepare data

#### Download data and organise as follows

```
# For nuScenes Dataset         
└── NUSCENES_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       ├── v1.0-trainval <-- metadata
```

Create a symlink to the dataset root 
```bash
mkdir data && cd data
ln -s DATA_ROOT 
mv DATA_ROOT nuScenes # rename to nuScenes
```
Remember to change the DATA_ROOT to the actual path in your system. 


#### Create data

Data creation should be under the gpu environment.

```
# nuScenes
python tools/create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10
```

In the end, the data and info files should be organized as follows

```
# For nuScenes Dataset 
└── CenterPoint
       └── data    
              └── nuScenes 
                     ├── samples       <-- key frames
                     ├── sweeps        <-- frames without annotation
                     ├── maps          <-- unused
                     |── v1.0-trainval <-- metadata and annotations
                     |── infos_train_10sweeps_withvelo_filter_True.pkl <-- train annotations
                     |── infos_val_10sweeps_withvelo_filter_True.pkl <-- val annotations
                     |── dbinfos_train_10sweeps_withvelo.pkl <-- GT database info files
                     |── gt_database_10sweeps_withvelo <-- GT database 
```

### Train & Evaluate in Command Line

**Now we only support training and evaluation with gpu. Cpu only mode is not supported.**

Use the following command to start a distributed training using 4 GPUs. The models and logs will be saved to ```work_dirs/CONFIG_NAME``` 

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py CONFIG_PATH
```

For distributed testing with 4 gpus,

```bash
python -m torch.distributed.launch --nproc_per_node=4 ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth 
```

For testing with one gpu and see the inference time,

```bash
python ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth --speed_test 
```

The pretrained models and configurations are in [MODEL ZOO](../configs/nusc/README.md).

### Tracking

Please refer to [tracking_scripts](../tracking_scripts) to reproduce all tracking results. The detection files are provided in the [MODEL ZOO](../configs/nusc/README.md)

### Test Set 

Organize your dataset as follows 

```
# For nuScenes Dataset 
└── CenterPoint
       └── data    
              └── nuScenes 
                     ├── samples       <-- key frames
                     ├── sweeps        <-- frames without annotation
                     ├── maps          <-- unused
                     |── v1.0-trainval <-- metadata and annotations
                     |── infos_train_10sweeps_withvelo_filter_True.pkl <-- train annotations
                     |── infos_val_10sweeps_withvelo_filter_True.pkl <-- val annotations
                     |── dbinfos_train_10sweeps_withvelo.pkl <-- GT database info files
                     |── gt_database_10sweeps_withvelo <-- GT database 
                     └── v1.0-test <-- main test folder 
                            ├── samples       <-- key frames
                            ├── sweeps        <-- frames without annotation
                            ├── maps          <-- unused
                            |── v1.0-test <-- metadata and annotations
                            |── infos_test_10sweeps_withvelo.pkl <-- test info
```

Download the ```centerpoint_voxel_1440_dcn_flip``` [here](https://drive.google.com/drive/folders/1uU_wXuNikmRorf_rPBbM0UTrW54ztvMs?usp=sharing), save it into ```work_dirs/nusc_0075_dcn_flip_track```, then run the following commands in the main folder to get detection prediction 

```bash
python tools/dist_test.py configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_dcn_flip.py --work_dir work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset  --checkpoint work_dirs/nusc_0075_dcn_flip_track/voxelnet_converted.pth  --testset --speed_test 
```

With the generated detection files, you can create the tracking prediction by running

```bash 
bash tracking_scripts/centerpoint_voxel_1440_dcn_flip_testset.sh
```

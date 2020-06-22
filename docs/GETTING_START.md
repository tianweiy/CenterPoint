## Getting Started with CenterPoint
Modified from [det3d](https://github.com/poodarchu/det3d)'s original document.

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

**We have created all info files and GT database which can be downloaded [here](https://drive.google.com/file/d/1ySFU1Ikph45cGZZaa0y6GdwBWiXyfgOd/view?usp=sharing) and [here](https://drive.google.com/file/d/1Kz1uvNCIWPzbW_sRZejkScdBG6aFdQQn/view?usp=sharing)(download both files). If you prefer to generate the files locally, please use the following script. The only difference is that our old GT database still contains annotated boxes without any radar/lidar points. However, this should not affect the training as we ensure to remove all GT objects with less than 5 lidar points during the GT-Sampling augmentation.**
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
                     |── v1.0-trainval <-- metadata and annotations
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

The pretrained models and configurations are in [MODEL ZOO](MODEL_ZOO.md).

### Tracking

Please refer to [tracking_scripts](../tracking_scripts) to reproduce all tracking results. The detection files are provided in the [MODEL ZOO](MODEL_ZOO.md)

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
                     |── v1.0-trainval <-- metadata and annotations
                     └── v1.0-test <-- main test folder 
                            ├── samples       <-- key frames
                            ├── sweeps        <-- frames without annotation
                            ├── maps          <-- unused
                            |── v1.0-test <-- metadata and annotations
                            |── infos_test_10sweeps_withvelo.pkl <-- test info
```

Download the ```centerpoint_voxel_1440_dcn_flip``` [here](https://drive.google.com/file/d/1N9P3pzNy0hLEwP-kqsbVgr5xsvkNJHYi/view?usp=sharing), save it into ```work_dirs/centerpoint_voxel_1440_dcn_flip_testset```, then run the following commands in the main folder to get detection prediction 

```bash
python tools/dist_test.py configs/centerpoint/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset.py --work_dir work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset  --checkpoint work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset/epoch_20.pth  --speed_test 
```

With the generated detection files, you can create the tracking prediction by running

```bash 
bash tracking_scripts/centerpoint_voxel_1440_dcn_flip_testset.sh
```

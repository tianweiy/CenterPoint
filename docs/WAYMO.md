## Getting Started with CenterPoint on Waymo

### Prerequisite 

- Follow [INSTALL.md](INSTALL.md) to install all required libraries. 
- Tensorflow 
- Waymo-open-dataset devkit

```bash
conda activate centerpoint 
pip install waymo-open-dataset-tf-1-15-0==1.2.0 
```

### Prepare data

#### Download data and organise as follows

```
# For Waymo Dataset         
└── WAYMO_DATASET_ROOT
       ├── tfrecord_training       
       ├── tfrecord_validation   
       ├── tfrecord_testing 
```

Convert the tfrecord data to pickle files.

```bash
# train set 
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path 'WAYMO_DATASET_ROOT/tfrecord_training/*.tfrecord'  --root_path 'WAYMO_DATASET_ROOT/train/'

# validation set 
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path 'WAYMO_DATASET_ROOT/tfrecord_validation/*.tfrecord'  --root_path 'WAYMO_DATASET_ROOT/val/'

# testing set 
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py --record_path 'WAYMO_DATASET_ROOT/tfrecord_testing/*.tfrecord'  --root_path 'WAYMO_DATASET_ROOT/test/'
```

Create a symlink to the dataset root 
```bash
mkdir data && cd data
ln -s WAYMO_DATASET_ROOT Waymo
```
Remember to change the WAYMO_DATASET_ROOT to the actual path in your system. 


#### Create info files

```bash
# One Sweep Infos 
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --nsweeps=1

python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split val --nsweeps=1

python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split test --nsweeps=1

# Two Sweep Infos (for two sweep detection and tracking models)
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --nsweeps=2

python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split val --nsweeps=2

python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split test --nsweeps=2
```

In the end, the data and info files should be organized as follows

```
└── CenterPoint
       └── data    
              └── Waymo 
                     ├── tfrecord_training       
                     ├── tfrecord_validation
                     ├── train <-- all training frames and annotations 
                     ├── val   <-- all validation frames and annotations 
                     ├── test   <-- all testing frames and annotations 
                     ├── infos_train_01sweeps_filter_zero_gt.pkl
                     ├── infos_train_02sweeps_filter_zero_gt.pkl
                     ├── infos_val_01sweeps_filter_zero_gt.pkl
                     ├── infos_val_02sweeps_filter_zero_gt.pkl
                     ├── infos_test_01sweeps_filter_zero_gt.pkl
                     ├── infos_test_02sweeps_filter_zero_gt.pkl
```

### Train & Evaluate in Command Line

Use the following command to start a distributed training using 4 GPUs. The models and logs will be saved to ```work_dirs/CONFIG_NAME```. 

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

This will generate a `my_preds.bin` file in the work_dir. You can create submission to Waymo server using waymo-open-dataset code by following the instructions [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md).  

If you want to do local evaluation (e.g. for a subset), generate the gt prediction bin files using the script below and follow the waymo instructions [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md).

```bash
python det3d/datasets/waymo/waymo_common.py --info_path data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl --result_path data/Waymo/ --gt 
```

All pretrained models and configurations are in [MODEL ZOO](../configs/waymo/README.md).

### Second-stage Training 

Our final model follows a two-stage training process. For example, to train the two-stage CenterPoint-Voxel model, you first need to train the one stage model using [ONE_STAGE](../configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py) and then train the second stage module using [TWO_STAGE](../configs/waymo/voxelnet/two_stage/waymo_centerpoint_voxelnet_two_stage_bev_5point_ft_6epoch_freeze.py). You can also contact us to access the pretrained models, see details [here](../configs/waymo/README.md). 

### Tracking 

Please refer to options in [test.py](../tools/waymo_tracking/test.py). The prediction file is an intermediate file generated using [dist_test.py](../tools/dist_test.py) that stores predictions in KITTI lidar format. 

### Visualization 

Please refer to [visual.py](../tools/visual.py). It will take a prediction file generated by [simple_inference_waymo.py](../tools/simple_inference_waymo.py) and visualize the point cloud and detections.  

### Test Set 

Add the ```--testset``` flag to the end. 

```bash
python ./tools/dist_test.py CONFIG_PATH --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth --testset 
```

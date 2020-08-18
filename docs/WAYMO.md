## Getting Started with CenterPoint on Waymo

### PointPillars model 

We now support two models using PointPillars architecture. 

|         |  Split  |  mAPH@L1 |mAPH@L2   | 
|---------|---------|---------|--------|
| [PointPillars-512](../configs/point_pillars/waymo_pp_car_large.py)  |  Val    |  71.2   |  62.6|  
| [CenterPoint-Pillar-512](../configs/centerpoint/waymo_centerpoint_pp_car_large.py)  |  Val    |  72.7   | 64.2 |

**We have pretrained models available for download which can be requested by filling out this [form](https://forms.gle/2q2APSAtfmaFJme79).**

### Prerequisite 

- Follow [INSTALL.md](INSTALL.md) to install all required libraries. **spconv is optional**
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
```

Convert the tfrecord data to pickle files.

```bash
# train set 
python generate_waymo_dataset.py --input_file_pattern='WAYMO_DATASET_ROOT/tfrecord_training/segment-*.tfrecord'  --output_filebase='WAYMO_DATASET_ROOT/train/'

# validation set 
python generate_waymo_dataset.py --input_file_pattern='WAYMO_DATASET_ROOT/tfrecord_validation/segment-*.tfrecord'  --output_filebase='WAYMO_DATASET_ROOT/val/'
```

Create a symlink to the dataset root 
```bash
mkdir data && cd data
ln -s WAYMO_DATASET_ROOT 
mv WAYMO_DATASET_ROOT Waymo # rename to Waymo
```
Remember to change the WAYMO_DATASET_ROOT to the actual path in your system. 


#### Create info files

```bash
# train set 
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --nsweeps=1

# val set 
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split val --nsweeps=1
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
                     ├── infos_train_01sweeps_filter_zero_gt.pkl
                     ├── infos_val_01sweeps_filter_zero_gt.pkl
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

This will generate a `my_preds.bin` file in the work_dir. You can create submission to Waymo server using waymo-open-dataset code by following the instructions [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md).  

If you want to do local evaluation (e.g. for a subset), generate the gt prediction bin files using the script below and follow the waymo instructions [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md).

```bash
python det3d/datasets/waymo/waymo_common.py --info_path data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl  --gt 
```

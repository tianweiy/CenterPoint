import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

tasks = [
    dict(num_class=3, class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST']),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        num_input_features=5,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=5, ds_factor=8),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='waymo',
        weight=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2)}, # (output_channel, num_conv)
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)


train_cfg = dict(assigner=assigner)


test_cfg = dict(
    post_center_limit_range=[-80, -80, -10.0, 80, 80, 10.0],
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=4096,
        nms_post_max_size=500,
        nms_iou_threshold=0.7,
    ),
    score_threshold=0.1,
    pc_range=[-75.2, -75.2],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.1, 0.1],
)


# dataset settings
dataset_type = "WaymoDataset"
nsweeps = 1
data_root = "data/Waymo"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path="data/Waymo/dbinfos_train_1sweeps_withvelo.pkl",
    sample_groups=[
        dict(VEHICLE=15),
        dict(PEDESTRIAN=10),
        dict(CYCLIST=10),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                VEHICLE=5,
                PEDESTRIAN=5,
                CYCLIST=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
) 

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.95, 1.05],
    db_sampler=db_sampler,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

voxel_generator = dict(
    range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
    voxel_size=[0.1, 0.1, 0.15],
    max_points_in_voxel=5,
    max_voxel_num=[150000, 200000],
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "data/Waymo/infos_train_01sweeps_filter_zero_gt.pkl"
val_anno = "data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl"
test_anno = None

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)



optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 36
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None 
resume_from = None  
workflow = [('train', 1)]

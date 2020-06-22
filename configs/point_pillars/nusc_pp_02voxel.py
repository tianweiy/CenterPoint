import itertools
import logging

from det3d.builder import build_box_coder
from det3d.utils.config_tool import get_downsample_factor

norm_cfg = None

tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

target_assigner = dict(
    type="iou",
    anchor_generators=[
        dict(
            type="anchor_generator_range",
            sizes=[1.97, 4.63, 1.74],
            anchor_ranges=[-51.2, -51.2, -0.95, 51.2, 51.2, -0.95],
            rotations=[0, 1.57],
            velocities=[0, 0],
            matched_threshold=0.6,
            unmatched_threshold=0.45,
            class_name="car",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[2.51, 6.93, 2.84],
            anchor_ranges=[-51.2, -51.2, -0.40, 51.2, 51.2, -0.40],
            rotations=[0, 1.57],
            velocities=[0, 0],
            matched_threshold=0.55,
            unmatched_threshold=0.4,
            class_name="truck",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[2.85, 6.37, 3.19],
            anchor_ranges=[-51.2, -51.2, -0.225, 51.2, 51.2, -0.225],
            rotations=[0, 1.57],
            velocities=[0, 0],
            matched_threshold=0.5,
            unmatched_threshold=0.35,
            class_name="construction_vehicle",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[2.94, 10.5, 3.47],
            anchor_ranges=[-51.2, -51.2, -0.085, 51.2, 51.2, -0.085],
            rotations=[0, 1.57],
            velocities=[0, 0],
            matched_threshold=0.55,
            unmatched_threshold=0.4,
            class_name="bus",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[2.90, 12.29, 3.87],
            anchor_ranges=[-51.2, -51.2, 0.115, 51.2, 51.2, 0.115],
            rotations=[0, 1.57],
            velocities=[0, 0],
            matched_threshold=0.5,
            unmatched_threshold=0.35,
            class_name="trailer",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[2.53, 0.50, 0.98],
            anchor_ranges=[-51.2, -51.2, -1.33, 51.2, 51.2, -1.33],
            rotations=[0, 1.57],
            velocities=[0, 0],
            matched_threshold=0.55,
            unmatched_threshold=0.4,
            class_name="barrier",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[0.77, 2.11, 1.47],
            anchor_ranges=[-51.2, -51.2, -1.085, 51.2, 51.2, -1.085],
            rotations=[0, 1.57],
            velocities=[0, 0],
            matched_threshold=0.5,
            unmatched_threshold=0.3,
            class_name="motorcycle",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[0.60, 1.70, 1.28],
            anchor_ranges=[-51.2, -51.2, -1.18, 51.2, 51.2, -1.18],
            rotations=[0, 1.57],
            velocities=[0, 0],
            matched_threshold=0.5,
            unmatched_threshold=0.35,
            class_name="bicycle",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[0.67, 0.73, 1.77],
            anchor_ranges=[-51.2, -51.2, -0.935, 51.2, 51.2, -0.935],
            rotations=[0, 1.57],
            velocities=[0, 0],
            matched_threshold=0.6,
            unmatched_threshold=0.4,
            class_name="pedestrian",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[0.41, 0.41, 1.07],
            anchor_ranges=[-51.2, -51.2, -1.285, 51.2, 51.2, -1.285],
            rotations=[0, 1.57],
            velocities=[0, 0],
            matched_threshold=0.6,
            unmatched_threshold=0.4,
            class_name="traffic_cone",
        ),
    ],
    sample_positive_fraction=-1,
    sample_size=512,
    region_similarity_calculator=dict(type="nearest_iou_similarity",),
    pos_area_threshold=-1,
    tasks=tasks,
)

box_coder = dict(
    type="ground_box3d_coder", n_dim=9, linear_dim=False, encode_angle_vector=True,
)

# model settings
model = dict(
    type="PointPillars",
    pretrained=None,
    reader=dict(
        type="PillarFeatureNet",
        num_filters=[64],
        num_input_features=5,
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        norm_cfg=norm_cfg,
    ),
    backbone=dict(type="PointPillarsScatter", ds_factor=1, norm_cfg=norm_cfg,),
    neck=dict(
        type="RPN",
        layer_nums=[3, 5, 5],
        ds_layer_strides=[2, 2, 2],
        ds_num_filters=[64, 128, 256],
        us_layer_strides=[0.5, 1, 2],
        us_num_filters=[128, 128, 128],
        num_input_features=64,
        norm_cfg=norm_cfg,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="MultiGroupHead",
        mode="3d",
        in_channels=sum([128, 128, 128]),  # this is linked to 'neck' us_num_filters
        norm_cfg=norm_cfg,
        tasks=tasks,
        weights=[1,],
        box_coder=build_box_coder(box_coder),
        encode_background_as_zeros=True,
        loss_norm=dict(
            type="NormByNumPositives", pos_cls_weight=1.0, neg_cls_weight=2.0,
        ),
        loss_cls=dict(type="SigmoidFocalLoss", alpha=0.25, gamma=2.0, loss_weight=1.0,),
        use_sigmoid_score=True,
        loss_bbox=dict(
            type="WeightedL1Loss",
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
            codewise=True,
            loss_weight=0.25,
        ),
        encode_rad_error_by_sin=False,
        loss_aux=None,
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)}, # (output_channel, num_conv)
    ),
)

assigner = dict(
    box_coder=box_coder,
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    debug=False,
)

train_cfg = dict(assigner=assigner)

test_cfg = dict(
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
)

# dataset settings
dataset_type = "NuScenesDataset"
nsweeps = 10
data_root = "data/nuScenes"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path="data/nuScenes/dbinfos_train_10sweeps_withvelo.pkl",
    sample_groups=[
        dict(car=2),
        dict(truck=3),
        dict(construction_vehicle=7),
        dict(bus=4),
        dict(trailer=6),
        dict(barrier=2),
        dict(motorcycle=6),
        dict(bicycle=6),
        dict(pedestrian=2),
        dict(traffic_cone=2),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                truck=5,
                bus=5,
                trailer=5,
                construction_vehicle=5,
                traffic_cone=5,
                barrier=5,
                motorcycle=5,
                bicycle=5,
                pedestrian=5,
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
    gt_loc_noise=[0.0, 0.0, 0.0],
    gt_rot_noise=[0.0, 0.0],
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    global_rot_per_obj_range=[0, 0],
    global_trans_noise=[0.2, 0.2, 0.2],
    remove_points_after_sample=False,
    gt_drop_percentage=0.0,
    gt_drop_max_keep_points=15,
    remove_unknown_examples=False,
    remove_environment=False,
    db_sampler=db_sampler,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
)

voxel_generator = dict(
    range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    voxel_size=[0.2, 0.2, 8],
    max_points_in_voxel=20,
    max_voxel_num=30000,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl"
val_anno = "data/nuScenes/infos_val_10sweeps_withvelo_filter_True.pkl"
test_anno = None

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
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

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)

"""training hooks """
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy in training hooks
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
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
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None
workflow = [('train', 1)]

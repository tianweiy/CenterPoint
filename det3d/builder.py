import logging
import pickle
from functools import partial

import det3d.core.sampler.preprocess as prep
import numpy as np
import torch
from det3d.core.anchor.anchor_generator import (
    AnchorGeneratorRange,
    AnchorGeneratorStride,
    BevAnchorGeneratorRange,
)
from det3d.core.bbox import region_similarity
from det3d.core.bbox.box_coders import BevBoxCoderTorch, GroundBox3dCoderTorch
from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.sampler.preprocess import DataBasePreprocessor
from det3d.core.sampler.sample_ops import DataBaseSamplerV2
from det3d.models.losses import GHMCLoss, GHMRLoss, losses
from det3d.solver import learning_schedules
from det3d.solver import learning_schedules_fastai as lsf
from det3d.solver import optim
from det3d.solver.fastai_optim import FastAIMixedOptim, OptimWrapper
from torch import nn


def build_voxel_generator(voxel_config):

    voxel_generator = VoxelGenerator(
        voxel_size=voxel_config.VOXEL_SIZE,
        point_cloud_range=voxel_config.RANGE,
        max_num_points=voxel_config.MAX_POINTS_NUM_PER_VOXEL,
        max_voxels=20000,
    )

    return voxel_generator


def build_similarity_metric(similarity_config):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    similarity_type = similarity_config.type

    if similarity_type == "rotate_iou_similarity":
        return region_similarity.RotateIouSimilarity()
    elif similarity_type == "nearest_iou_similarity":
        return region_similarity.NearestIouSimilarity()
    elif similarity_type == "distance_similarity":
        cfg = similarity_config.distance_similarity
        return region_similarity.DistanceSimilarity(
            distance_norm=cfg.distance_norm,
            with_rotation=cfg.with_rotation,
            rotation_alpha=cfg.rotation_alpha,
        )
    else:
        raise ValueError("unknown similarity type")


def build_db_preprocess(db_prep_config, logger=None):
    logger = logging.getLogger("build_db_preprocess")
    cfg = db_prep_config
    if "filter_by_difficulty" in cfg:
        v = cfg["filter_by_difficulty"]
        return prep.DBFilterByDifficulty(v, logger=logger)
    elif "filter_by_min_num_points" in cfg:
        v = cfg["filter_by_min_num_points"]
        return prep.DBFilterByMinNumPoint(v, logger=logger)
    else:
        raise ValueError("unknown database prep type")


def children(m: nn.Module):
    "Get children of `m`."
    return list(m.children())


def num_children(m: nn.Module) -> int:
    "Get number of children modules in `m`."
    return len(children(m))


def flatten_model(m: nn.Module):
    return sum(map(flatten_model, m.children()), []) if num_children(m) else [m]


def get_layer_groups(m: nn.Module):
    return [nn.Sequential(*flatten_model(m))]


def build_optimizer(optimizer_config, net, name=None, mixed=False, loss_scale=512.0):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    optimizer_type = optimizer_config.TYPE
    config = optimizer_config.VALUE

    if optimizer_type == "rms_prop_optimizer":
        optimizer_func = partial(
            torch.optim.RMSprop,
            alpha=config.decay,
            momentum=config.momentum_optimizer_value,
            eps=config.epsilon,
        )
    elif optimizer_type == "momentum_optimizer":
        optimizer_func = partial(
            torch.optim.SGD,
            momentum=config.momentum_optimizer_value,
            eps=config.epsilon,
        )
    elif optimizer_type == "adam":
        if optimizer_config.FIXED_WD:
            optimizer_func = partial(
                torch.optim.Adam, betas=(0.9, 0.99), amsgrad=config.amsgrad
            )
        else:
            # regular adam
            optimizer_func = partial(torch.optim.Adam, amsgrad=config.amsgrad)

    optimizer = OptimWrapper.create(
        optimizer_func,
        3e-3,
        get_layer_groups(net),
        wd=config.WD,
        true_wd=optimizer_config.FIXED_WD,
        bn_wd=True,
    )

    if optimizer is None:
        raise ValueError("Optimizer %s not supported." % optimizer_type)

    if optimizer_config.MOVING_AVERAGE:
        raise ValueError("torch don't support moving average")

    if name is None:
        # assign a name to optimizer for checkpoint system
        optimizer.name = optimizer_type
    else:
        optimizer.name = name

    return optimizer


def build_lr_scheduler(optimizer, optimizer_config, total_step):
    """Create lr scheduler based on config. note that
    lr_scheduler must accept a optimizer that has been restored.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    optimizer_type = optimizer_config.type
    config = optimizer_config

    if optimizer_type == "rms_prop_optimizer":
        lr_scheduler = _create_learning_rate_scheduler(
            config, optimizer, total_step=total_step
        )
    elif optimizer_type == "momentum_optimizer":
        lr_scheduler = _create_learning_rate_scheduler(
            config, optimizer, total_step=total_step
        )
    elif optimizer_type == "adam":
        lr_scheduler = _create_learning_rate_scheduler(
            config, optimizer, total_step=total_step
        )

    return lr_scheduler


def _create_learning_rate_scheduler(optimizer, learning_rate_config, total_step):
    """Create optimizer learning rate scheduler based on config.

    Args:
        learning_rate_config: A LearningRate proto message.

    Returns:
        A learning rate.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    lr_scheduler = None
    learning_rate_type = learning_rate_config.type
    config = learning_rate_config

    if learning_rate_type == "multi_phase":
        lr_phases = []
        mom_phases = []
        for phase_cfg in config.phases:
            lr_phases.append((phase_cfg.start, phase_cfg.lambda_func))
            mom_phases.append((phase_cfg.start, phase_cfg.momentum_lambda_func))
        lr_scheduler = lsf.LRSchedulerStep(optimizer, total_step, lr_phases, mom_phases)
    elif learning_rate_type == "one_cycle":
        lr_scheduler = lsf.OneCycle(
            optimizer,
            total_step,
            config.lr_max,
            config.moms,
            config.div_factor,
            config.pct_start,
        )
    elif learning_rate_type == "exponential_decay":
        lr_scheduler = lsf.ExponentialDecay(
            optimizer,
            total_step,
            config.initial_learning_rate,
            config.decay_length,
            config.decay_factor,
            config.staircase,
        )
    elif learning_rate_type == "manual_stepping":
        lr_scheduler = lsf.ManualStepping(
            optimizer, total_step, config.boundaries, config.rates
        )
    elif lr_scheduler is None:
        raise ValueError("Learning_rate %s not supported." % learning_rate_type)

    return lr_scheduler


def build_loss(loss_config):
    """Build losses based on the config.

    Builds classification, localization losses and optionally a hard example miner
    based on the config.

    Args:
        loss_config: A losses_pb2.Loss object.

    Returns:
        classification_loss: Classification loss object.
        localization_loss: Localization loss object.
        classification_weight: Classification loss weight.
        localization_weight: Localization loss weight.
        hard_example_miner: Hard example miner object.

    Raises:
        ValueError: If hard_example_miner is used with sigmoid_focal_loss.
    """
    classification_loss = _build_classification_loss(loss_config.classification_loss)
    localization_loss = _build_localization_loss(loss_config.localization_loss)

    classification_weight = loss_config.classification_weight
    localization_weight = loss_config.localization_weight

    hard_example_miner = None  # 'Pytorch don\'t support HardExampleMiner'

    return (
        classification_loss,
        localization_loss,
        classification_weight,
        localization_weight,
        hard_example_miner,
    )


def build_faster_rcnn_classification_loss(loss_config):
    """Builds a classification loss for Faster RCNN based on the loss config.

    Args:
        loss_config: A losses_pb2.ClassificationLoss object.

    Returns:
        Loss based on the config.

    Raises:
        ValueError: On invalid loss_config.
    """
    loss_type = loss_config.TYPE
    config = loss_config.VALUE

    # By default, Faster RCNN second stage classifier uses Softmax loss
    # with anchor-wise outputs.
    return losses.WeightedSoftmaxClassificationLoss(logit_scale=config.logit_scale)


def _build_localization_loss(loss_config):
    """Builds a localization loss based on the loss config.

    Args:
        loss_config: A losses_pb2.LocalizationLoss object.

    Returns:
        Loss based on the config.

    Raises:
        ValueError: On invalid loss_config.
    """
    loss_type = loss_config.type
    config = loss_config

    if loss_type == "weighted_l2":
        if len(config.code_weight) == 0:
            code_weight = None
        else:
            code_weight = config.code_weight
        return losses.WeightedL2LocalizationLoss(code_weight)

    if loss_type == "weighted_smooth_l1":
        if len(config.code_weight) == 0:
            code_weight = None
        else:
            code_weight = config.code_weight
        return losses.WeightedSmoothL1LocalizationLoss(config.sigma, code_weight)
    if loss_type == "weighted_ghm":
        if len(config.code_weight) == 0:
            code_weight = None
        else:
            code_weight = config.code_weight
        return GHMRLoss(config.mu, config.bins, config.momentum, code_weight)

    raise ValueError("Empty loss config.")


def _build_classification_loss(loss_config):
    """Builds a classification loss based on the loss config.

    Args:
        loss_config: A losses_pb2.ClassificationLoss object.

    Returns:
        Loss based on the config.

    Raises:
        ValueError: On invalid loss_config.
    """
    loss_type = loss_config.TYPE
    config = loss_config.VALUE

    if loss_type == "weighted_sigmoid":
        return losses.WeightedSigmoidClassificationLoss()
    elif loss_type == "weighted_sigmoid_focal":
        if config.alpha > 0:
            alpha = config.alpha
        else:
            alpha = None
        return losses.SigmoidFocalClassificationLoss(gamma=config.gamma, alpha=alpha)
    elif loss_type == "weighted_softmax_focal":
        if config.alpha > 0:
            alpha = config.alpha
        else:
            alpha = None
        return losses.SoftmaxFocalClassificationLoss(gamma=config.gamma, alpha=alpha)
    elif loss_type == "weighted_ghm":
        return GHMCLoss(bins=config.bins, momentum=config.momentum)
    elif loss_type == "weighted_softmax":
        return losses.WeightedSoftmaxClassificationLoss(logit_scale=config.logit_scale)
    elif loss_type == "bootstrapped_sigmoid":
        return losses.BootstrappedSigmoidClassificationLoss(
            alpha=config.alpha,
            bootstrap_type=("hard" if config.hard_bootstrap else "soft"),
        )

    raise ValueError("Empty loss config.")


def build_dbsampler(cfg, logger=None):
    logger = logging.getLogger("build_dbsampler")
    prepors = [build_db_preprocess(c, logger=logger) for c in cfg.db_prep_steps]
    db_prepor = DataBasePreprocessor(prepors)
    rate = cfg.rate
    grot_range = cfg.global_random_rotation_range_per_object
    groups = cfg.sample_groups
    # groups = [dict(g.name_to_max_num) for g in groups]
    info_path = cfg.db_info_path
    with open(info_path, "rb") as f:
        db_infos = pickle.load(f)
    grot_range = list(grot_range)
    if len(grot_range) == 0:
        grot_range = None
    sampler = DataBaseSamplerV2(
        db_infos, groups, db_prepor, rate, grot_range, logger=logger
    )

    return sampler


def build_box_coder(box_coder_config):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    box_coder_type = box_coder_config["type"]
    cfg = box_coder_config

    n_dim = cfg.get("n_dim", 9)
    norm_velo = cfg.get("norm_velo", False)

    if box_coder_type == "ground_box3d_coder":
        return GroundBox3dCoderTorch(
            cfg["linear_dim"],
            cfg["encode_angle_vector"],
            n_dim=n_dim,
            norm_velo=norm_velo,
        )
    elif box_coder_type == "bev_box_coder":
        cfg = box_coder_config
        return BevBoxCoderTorch(
            cfg["linear_dim"],
            cfg["encode_angle_vector"],
            cfg["z_fixed"],
            cfg["h_fixed"],
        )
    else:
        raise ValueError("unknown box_coder type")


def build_anchor_generator(anchor_config):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    ag_type = anchor_config.type
    config = anchor_config

    if "velocities" not in config:
        velocities = None
    else:
        velocities = config.velocities

    if ag_type == "anchor_generator_stride":
        ag = AnchorGeneratorStride(
            sizes=config.sizes,
            anchor_strides=config.strides,
            anchor_offsets=config.offsets,
            rotations=config.rotations,
            velocities=velocities,
            match_threshold=config.matched_threshold,
            unmatch_threshold=config.unmatched_threshold,
            class_name=config.class_name,
        )
        return ag
    elif ag_type == "anchor_generator_range":
        ag = AnchorGeneratorRange(
            sizes=config.sizes,
            anchor_ranges=config.anchor_ranges,
            rotations=config.rotations,
            velocities=velocities,
            match_threshold=config.matched_threshold,
            unmatch_threshold=config.unmatched_threshold,
            class_name=config.class_name,
        )
        return ag
    elif ag_type == "bev_anchor_generator_range":
        ag = BevAnchorGeneratorRange(
            sizes=config.sizes,
            anchor_ranges=config.anchor_ranges,
            rotations=config.rotations,
            velocities=velocities,
            match_threshold=config.matched_threshold,
            unmatch_threshold=config.unmatched_threshold,
            class_name=config.class_name,
        )
        return ag
    else:
        raise ValueError(" unknown anchor generator type")

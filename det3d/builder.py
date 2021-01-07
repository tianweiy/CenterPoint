import logging
import pickle
from functools import partial

import det3d.core.sampler.preprocess as prep
import numpy as np
import torch
from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.sampler.preprocess import DataBasePreprocessor
from det3d.core.sampler.sample_ops import DataBaseSamplerV2
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

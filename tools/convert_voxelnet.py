import argparse
import copy
from io import UnsupportedOperation
import json
import os
import sys
import os.path as osp
from collections import OrderedDict

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import __version__, torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time 

def convert_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict into a module
    """
    unexpected_keys = []
    shape_mismatch_pairs = []

    own_state = module.state_dict()
    for name, param in state_dict.items():
        # a hacky fixed to load a new voxelnet 
        if name not in own_state:
            if name[:20] == 'backbone.middle_conv':
                index = int(name[20:].split('.')[1])

                if index in [0, 1, 2]:
                    new_name = 'backbone.conv_input.{}.{}'.format(str(index), name[23:])
                elif index in [3, 4]:
                    new_name = 'backbone.conv1.{}.{}'.format(str(index-3), name[23:]) 
                elif index in [5, 6, 7, 8, 9]:
                    new_name = 'backbone.conv2.{}.{}'.format(str(index-5), name[23:]) 
                elif index in [10, 11, 12, 13, 14]:
                    new_name = 'backbone.conv3.{}.{}'.format(str(index-10), name[24:])
                elif index in [15, 16, 17, 18, 19]:
                    new_name = 'backbone.conv4.{}.{}'.format(str(index-15), name[24:])
                elif index in [20, 21, 22]:
                    new_name = 'backbone.extra_conv.{}.{}'.format(str(index-20), name[24:])
                else:
                    raise NotImplementedError(index)

                if param.size() != own_state[new_name].size():
                    shape_mismatch_pairs.append([name, own_state[name].size(), param.size()])
                    continue

                own_state[new_name].copy_(param)
                print("load {}'s param from {}".format(new_name, name))
                continue 

            unexpected_keys.append(name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if param.size() != own_state[name].size():
            shape_mismatch_pairs.append([name, own_state[name].size(), param.size()])
            continue
        own_state[name].copy_(param)

    all_missing_keys = set(own_state.keys()) - set(state_dict.keys())
    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    err_msg = []
    if unexpected_keys:
        err_msg.append(
            "unexpected key in source state_dict: {}\n".format(
                ", ".join(unexpected_keys)
            )
        )
    if missing_keys:
        err_msg.append(
            "missing keys in source state_dict: {}\n".format(", ".join(missing_keys))
        )
    if shape_mismatch_pairs:
        mismatch_info = "these keys have mismatched shape:\n"
        header = ["key", "expected shape", "loaded shape"]
        table_data = [header] + shape_mismatch_pairs
        table = AsciiTable(table_data)
        err_msg.append(mismatch_info + table.table)

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    args = parser.parse_args()

    return args

def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def save_checkpoint(model, filename, meta=None):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError("meta must be a dict or None, but got {}".format(type(meta)))

    torchie.mkdir_or_exist(osp.dirname(filename))
    if hasattr(model, "module"):
        model = model.module

    checkpoint = {"meta": meta, "state_dict": weights_to_cpu(model.state_dict())}

    torch.save(checkpoint, filename)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['state_dict']

    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}

    convert_state_dict(model, state_dict)

    save_checkpoint(model, osp.join(args.work_dir, 'voxelnet_converted.pth'))

main()
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path

import torch
from tensorboardX import SummaryWriter


def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + str(k))
        else:
            flatted[start + sep + str(k)] = v


def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """flat a nested json-like dict. this function make shadow copy.
    """
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, str(k))
        else:
            flatted[str(k)] = v
    return flatted


def metric_to_str(metrics, sep="."):
    flatted_metrics = flat_nested_json_dict(metrics, sep)
    metrics_str_list = []
    for k, v in flatted_metrics.items():
        if isinstance(v, float):
            metrics_str_list.append(f"{k}={v:.4}")
        elif isinstance(v, (list, tuple)):
            if v and isinstance(v[0], float):
                v_str = ", ".join([f"{e:.4}" for e in v])
                metrics_str_list.append(f"{k}=[{v_str}]")
            else:
                metrics_str_list.append(f"{k}={v}")
        else:
            metrics_str_list.append(f"{k}={v}")
    return ", ".join(metrics_str_list)


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, logger=None):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    if logger is None:
        logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict, logger=None):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, logger=logger)

    # use strict loading
    model.load_state_dict(model_state_dict)


def finetune_load_state_dict(model, loaded_state_dict, logger=None):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    loaded_state_dict = {
        k: v for k, v in loaded_state_dict.items() if not k.startswith("rpn.tasks")
    }
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, logger=logger)

    # use strict loading
    model.load_state_dict(model_state_dict)


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        ckpt_path=None,
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pretrained_path = ckpt_path  # whether pretrained
        self.finetune = False
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        self.logger.info(name)
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            print(dir(self.scheduler))
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        if f is not None:
            f = self.get_checkpoint_file(f)
        elif self.has_checkpoint(self.save_dir):
            # override argument with existing checkpoint
            f = self.get_checkpoint_file(self.save_dir)

        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def finetune_load(self, ckpt_path=None, f=None):
        if ckpt_path is not None:
            self.pretrained_path = ckpt_path
            self.finetune = True
            f = self.get_checkpoint_file(ckpt_path)
        assert f is not None, "Finetune should provide a valid ckpt path"
        self.logger.info("Loading pretrained model from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)

    def has_checkpoint(self, save_dir):
        save_file = os.path.join(save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self, save_dir):
        save_file = os.path.join(save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        if self.finetune:
            finetune_load_state_dict(
                self.model, checkpoint.pop("model"), logger=self.logger
            )
        else:
            load_state_dict(self.model, checkpoint.pop("model"), logger=self.logger)


class det3dCheckpointer(Checkpointer):
    def __init__(
        self,
        # cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(det3dCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        # self.cfg = cfg.clone()
        # self.writer = Writer(save_dir)
        self.logger = logger

    def _load_file(self, f):
        # load native detectron.pytorch checkpoint
        loaded = super(det3dCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded


class Writer:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.log_mjson_file = None
        self.summary_writter = None
        self.metrics = []
        self._text_current_gstep = -1
        self._tb_texts = []

    def open(self):
        save_dir = self.save_dir
        assert save_dir.exists()
        summary_dir = save_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        self.summary_writter = SummaryWriter(str(summary_dir))
        return self

    def close(self):
        assert self.summary_writter is not None
        tb_json_path = str(self.save_dir / "tensorboard_scalars.json")
        self.summary_writter.export_scalars_to_json(tb_json_path)
        self.summary_writter.close()
        self.summary_writter = None

    def log_text(self, text, step, tag="regular log"):
        """This function only add text to log.txt and tensorboard texts
        """
        if step > self._text_current_gstep and self._text_current_gstep != -1:
            total_text = "\n".join(self._tb_texts)
            self.summary_writter.add_text(tag, total_text, global_step=step)
            self._tb_texts = []
            self._text_current_gstep = step
        else:
            self._tb_texts.append(text)

        if self._text_current_gstep == -1:
            self._text_current_gstep = step

    def log_metrics(self, metrics: dict, step):
        flatted_summarys = flat_nested_json_dict(metrics, "/")
        for k, v in flatted_summarys.items():
            if isinstance(v, (list, tuple)):
                if any([isinstance(e, str) for e in v]):
                    continue
                v_dict = {str(i): e for i, e in enumerate(v)}
                for k1, v1 in v_dict.items():
                    self.summary_writter.add_scalar(k + "/" + k1, v1, step)
            else:
                if isinstance(v, str):
                    continue
                self.summary_writter.add_scalar(k, v, step)

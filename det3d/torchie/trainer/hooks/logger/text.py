import datetime
import os.path as osp
from collections import OrderedDict

import torch
import torch.distributed as dist
from det3d import torchie

from .base import LoggerHook


class TextLoggerHook(LoggerHook):
    def __init__(self, interval=10, ignore_last=True, reset_flag=False):
        super(TextLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.time_sec_tot = 0

    def before_run(self, trainer):
        super(TextLoggerHook, self).before_run(trainer)
        self.start_iter = trainer.iter
        self.json_log_path = osp.join(
            trainer.work_dir, "{}.log.json".format(trainer.timestamp)
        )

    def _get_max_memory(self, trainer):
        mem = torch.cuda.max_memory_allocated()
        mem_mb = torch.tensor(
            [mem / (1024 * 1024)], dtype=torch.int, device=torch.device("cuda")
        )
        if trainer.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    def _convert_to_precision4(self, val):
        if isinstance(val, float):
            val = "{:.4f}".format(val)
        elif isinstance(val, list):
            val = [self._convert_to_precision4(v) for v in val]

        return val

    def _log_info(self, log_dict, trainer):
        if trainer.mode == "train":
            log_str = "Epoch [{}/{}][{}/{}]\tlr: {:.5f}, ".format(
                log_dict["epoch"],
                trainer._max_epochs,
                log_dict["iter"],
                len(trainer.data_loader),
                log_dict["lr"],
            )
            if "time" in log_dict.keys():
                self.time_sec_tot += log_dict["time"] * self.interval
                time_sec_avg = self.time_sec_tot / (trainer.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (trainer.max_iters - trainer.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += "eta: {}, ".format(eta_str)
                log_str += "time: {:.3f}, data_time: {:.3f}, transfer_time: {:.3f}, forward_time: {:.3f}, loss_parse_time: {:.3f} ".format(
                    log_dict["time"],
                    log_dict["data_time"],
                    log_dict["transfer_time"] - log_dict["data_time"],
                    log_dict["forward_time"] - log_dict["transfer_time"],
                    log_dict["loss_parse_time"] - log_dict["forward_time"],
                )
                log_str += "memory: {}, ".format(log_dict["memory"])
        else:
            log_str = "Epoch({}) [{}][{}]\t".format(
                log_dict["mode"], log_dict["epoch"] - 1, log_dict["iter"]
            )

        trainer.logger.info(log_str)

        if trainer.world_size > 1:
            class_names = trainer.model.module.bbox_head.class_names
        else:
            class_names = trainer.model.bbox_head.class_names

        for idx, task_class_names in enumerate(class_names):
            log_items = [f"task : {task_class_names}"]
            log_str = ""
            for name, val in log_dict.items():
                # TODO:
                if name in [
                    "mode",
                    "Epoch",
                    "iter",
                    "lr",
                    "time",
                    "data_time",
                    "memory",
                    "epoch",
                    "transfer_time",
                    "forward_time",
                    "loss_parse_time",
                ]:
                    continue

                if isinstance(val, float):
                    val = "{:.4f}".format(val)

                if isinstance(val, list):
                    log_items.append(
                        "{}: {}".format(name, self._convert_to_precision4(val[idx]))
                    )
                else:
                    log_items.append("{}: {}".format(name, val))

            log_str += ", ".join(log_items)
            if idx == (len(class_names) - 1):
                log_str += "\n"
            trainer.logger.info(log_str)

    def _dump_log(self, log_dict, trainer):
        json_log = OrderedDict()
        for k, v in log_dict.items():
            json_log[k] = self._round_float(v)

        if trainer.rank == 0:
            with open(self.json_log_path, "a+") as f:
                torchie.dump(json_log, f, file_format="json")
                f.write("\n")

    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items

    def log(self, trainer):
        log_dict = OrderedDict()
        # Training mode if the output contains the key time
        mode = "train" if "time" in trainer.log_buffer.output else "val"
        log_dict["mode"] = mode
        log_dict["epoch"] = trainer.epoch + 1
        log_dict["iter"] = trainer.inner_iter + 1
        # Only record lr of the first param group
        log_dict["lr"] = trainer.current_lr()[0]
        if mode == "train":
            log_dict["time"] = trainer.log_buffer.output["time"]
            log_dict["data_time"] = trainer.log_buffer.output["data_time"]
            # statistic memory
            if torch.cuda.is_available():
                log_dict["memory"] = self._get_max_memory(trainer)
        for name, val in trainer.log_buffer.output.items():
            if name in ["time", "data_time"]:
                continue
            log_dict[name] = val

        self._log_info(log_dict, trainer)
        self._dump_log(log_dict, trainer)

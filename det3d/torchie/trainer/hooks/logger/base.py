from abc import ABCMeta, abstractmethod

from ..hook import Hook


class LoggerHook(Hook):
    """Base class for logger hooks

    Args:
        interval (int)
        ignore_last (bool)
        reset_flag (bool)
    """

    __metaclass__ = ABCMeta

    def __init__(self, interval=10, ignore_last=True, reset_flag=False):
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag

    @abstractmethod
    def log(self, trainer):
        pass

    def before_run(self, trainer):
        for hook in trainer.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break

    def before_epoch(self, trainer):
        trainer.log_buffer.clear()

    def after_train_iter(self, trainer):
        if self.every_n_inner_iters(trainer, self.interval):
            trainer.log_buffer.average(self.interval)
        elif self.end_of_epoch(trainer) and not self.ignore_last:
            # not precise but more stable
            trainer.log_buffer.average(self.interval)

        if trainer.log_buffer.ready:
            self.log(trainer)
            if self.reset_flag:
                trainer.log_buffer.clear_output()

    def after_train_epoch(self, trainer):
        if trainer.log_buffer.ready:
            self.log(trainer)
            if self.reset_flag:
                trainer.log_buffer.clear_output()

    def after_val_epoch(self, trainer):
        trainer.log_buffer.average()
        self.log(trainer)
        if self.reset_flag:
            trainer.log_buffer.clear_output()

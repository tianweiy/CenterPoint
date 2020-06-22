import time

from .hook import Hook


class IterTimerHook(Hook):
    def before_epoch(self, runner):
        self.t = time.time()

    def before_iter(self, runner):
        runner.log_buffer.update({"data_time": time.time() - self.t})

    def after_iter(self, runner):
        runner.log_buffer.update({"time": time.time() - self.t})
        self.t = time.time()

    def after_data_to_device(self, runner):
        runner.log_buffer.update({"transfer_time": time.time() - self.t})

    def after_forward(self, runner):
        runner.log_buffer.update({"forward_time": time.time() - self.t})

    def after_parse_loss(self, runner):
        runner.log_buffer.update({"loss_parse_time": time.time() - self.t})

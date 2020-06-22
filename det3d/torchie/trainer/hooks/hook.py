class Hook(object):
    def before_run(self, trainer):
        pass

    def after_run(self, trainer):
        pass

    def before_epoch(self, trainer):
        pass

    def after_epoch(self, trainer):
        pass

    def before_iter(self, trainer):
        pass

    def after_iter(self, trainer):
        pass

    def after_data_to_device(self, trainer):
        pass

    def after_forward(self, trainer):
        pass

    def after_parse_loss(self, trainer):
        pass

    def before_train_epoch(self, trainer):
        self.before_epoch(trainer)

    def before_val_epoch(self, trainer):
        self.before_epoch(trainer)

    def after_train_epoch(self, trainer):
        self.after_epoch(trainer)

    def after_val_epoch(self, trainer):
        self.after_epoch(trainer)

    def before_train_iter(self, trainer):
        self.before_iter(trainer)

    def before_val_iter(self, trainer):
        self.before_iter(trainer)

    def after_train_iter(self, trainer):
        self.after_iter(trainer)

    def after_val_iter(self, trainer):
        self.after_iter(trainer)

    def every_n_epochs(self, trainer, n):
        return (trainer.epoch + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, trainer, n):
        return (trainer.iter + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, trainer, n):
        return (trainer.inner_iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, trainer):
        return trainer.inner_iter + 1 == len(trainer.data_loader)

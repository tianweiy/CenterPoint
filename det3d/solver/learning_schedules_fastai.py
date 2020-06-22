import math
from functools import partial

import numpy as np


class LRSchedulerStep(object):
    def __init__(self, fai_optimizer, total_step, lr_phases, mom_phases):
        self.optimizer = fai_optimizer
        self.total_step = total_step
        self.lr_phases = []

        for i, (start, lambda_func) in enumerate(lr_phases):
            if len(self.lr_phases) != 0:
                assert self.lr_phases[-1][0] < int(start * total_step)
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(lr_phases) - 1:
                self.lr_phases.append(
                    (
                        int(start * total_step),
                        int(lr_phases[i + 1][0] * total_step),
                        lambda_func,
                    )
                )
            else:
                self.lr_phases.append(
                    (int(start * total_step), total_step, lambda_func)
                )
        assert self.lr_phases[0][0] == 0
        self.mom_phases = []
        for i, (start, lambda_func) in enumerate(mom_phases):
            if len(self.mom_phases) != 0:
                assert self.mom_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(mom_phases) - 1:
                self.mom_phases.append(
                    (
                        int(start * total_step),
                        int(mom_phases[i + 1][0] * total_step),
                        lambda_func,
                    )
                )
            else:
                self.mom_phases.append(
                    (int(start * total_step), total_step, lambda_func)
                )
        # assert self.mom_phases[0][0] == 0
        if len(mom_phases) > 0:
            assert self.mom_phases[0][0] == 0

    def step(self, step):
        lrs, moms = [], []

        for start, end, func in self.lr_phases:
            if step >= start:
                # self.optimizer.lr = func((step - start) / (end - start))
                lrs.append(func((step - start) / (end - start)))
        if len(lrs) > 0:
            self.optimizer.lr = lrs[-1]
        for start, end, func in self.mom_phases:
            if step >= start:
                moms.append(func((step - start) / (end - start)))
                self.optimizer.mom = func((step - start) / (end - start))
        if len(moms) > 0:
            self.optimizer.mom = moms[-1]


def annealing_cos(start, end, pct):
    # print(pct, start, end)
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class OneCycle(LRSchedulerStep):
    def __init__(self, fai_optimizer, total_step, lr_max, moms, div_factor, pct_start):
        self.lr_max = lr_max
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        a1 = int(total_step * self.pct_start)
        a2 = total_step - a1
        low_lr = self.lr_max / self.div_factor
        lr_phases = (
            (0, partial(annealing_cos, low_lr, self.lr_max)),
            (self.pct_start, partial(annealing_cos, self.lr_max, low_lr / 1e4)),
        )
        mom_phases = (
            (0, partial(annealing_cos, *self.moms)),
            (self.pct_start, partial(annealing_cos, *self.moms[::-1])),
        )
        fai_optimizer.lr, fai_optimizer.mom = low_lr, self.moms[0]
        super().__init__(fai_optimizer, total_step, lr_phases, mom_phases)


class ExponentialDecay(LRSchedulerStep):
    def __init__(
        self,
        fai_optimizer,
        total_step,
        initial_learning_rate,
        decay_length,
        decay_factor,
        staircase=True,
    ):
        """
        Args:
            decay_length: must in (0, 1)
        """
        assert decay_length > 0
        assert decay_length < 1
        self._decay_steps_unified = decay_length
        self._decay_factor = decay_factor
        self._staircase = staircase
        step = 0
        stage = 1
        lr_phases = []
        if staircase:
            while step <= total_step:
                func = lambda p, _d=initial_learning_rate * stage: _d
                lr_phases.append((step / total_step, func))
                stage *= decay_factor
                step += int(decay_length * total_step)
        else:
            func = lambda p: pow(decay_factor, (p / decay_length))
            lr_phases.append((0, func))
        super().__init__(fai_optimizer, total_step, lr_phases, [])


class ManualStepping(LRSchedulerStep):
    def __init__(self, fai_optimizer, total_step, boundaries, rates):
        assert all([b > 0 and b < 1 for b in boundaries])
        assert len(boundaries) + 1 == len(rates)
        boundaries.insert(0, 0.0)
        lr_phases = []
        for start, rate in zip(boundaries, rates):
            func = lambda p, _d=rate: _d
            lr_phases.append((start, func))
        super().__init__(fai_optimizer, total_step, lr_phases, [])


class FakeOptim:
    def __init__(self):
        self.lr = 0
        self.mom = 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    opt = FakeOptim()  # 3e-3, wd=0.4, div_factor=10
    # schd = OneCycle(opt, 100, 3e-3, (0.95, 0.85), 10.0, 0.1)
    schd = ExponentialDecay(opt, 100, 3e-4, 0.1, 0.8, staircase=True)
    schd = ManualStepping(opt, 100, [0.8, 0.9], [0.001, 0.0001, 0.00005])

    lrs = []
    moms = []
    for i in range(100):
        schd.step(i)
        lrs.append(opt.lr)
        moms.append(opt.mom)
    plt.plot(lrs)
    # plt.plot(moms)
    # plt.show()
    # plt.plot(moms)
    plt.show()

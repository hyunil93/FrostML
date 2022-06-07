from typing import Optional

import numpy as np

from torch.optim import Optimizer


class CosineAnnealingWarmRestarts:

    def __init__(self, optimizer: Optimizer, T_0: int, T_up: int, T_mult: float = 1.,
                 lr_delta: float = 1., eps: float = 1e-8, last_epoch: int = -1) -> None:
        r"""Set the learning rate of each parameter group using a cosine annealing
        schedule.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_0 (int): Initial number of iterations.
            T_up (int): Initial period of warm up.
            T_mult (float): Multiplier for iterations. Defaults: 1.
            lr_delta (float): Multiplier for initial lr. Defaults: 1.
            eps (float): Minimal decay applied to lr. Defaults: 1e-8
            last_epoch (int): The index of last epoch. Default: -1
        """
        self.T = T_0
        self.T_0 = T_0
        self.T_up = T_up
        self.T_rev = 0
        self.T_mult = T_mult

        self.optimizer = optimizer
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.target_lr = self.initial_lr
        self.lr_delta = lr_delta

        self.eps = eps
        self.last_epoch = last_epoch

    def step(self, epoch: Optional[int] = None) -> None:
        r"""Step could be called after every batch update.

        Args:
            epoch (int): The index of current epoch. Default: None
        """
        epoch = self.last_epoch + 1 if epoch is None else epoch

        if (epoch - self.T_rev) % self.T == 0 and epoch != 0:
            self.target_lr *= self.lr_delta
            self.T_rev += self.T
            self.T *= self.T_mult

        term_warmup = float((epoch - self.T_rev) % self.T) / float(max(1.0, self.T_up)) * self.target_lr + self.eps
        term_anneal = 0.5 * (self.target_lr - self.eps) * \
            (1 + np.cos(np.pi * (((epoch - self.T_rev) % self.T - self.T_up) / (self.T - self.T_up)))) + self.eps

        self.last_epoch = np.floor(epoch)
        self.optimizer.param_groups[0]['lr'] = term_warmup if (epoch - self.T_rev) % self.T < self.T_up else term_anneal

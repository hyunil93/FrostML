from typing import Dict, Optional, Union

import warnings
import copy
import os.path
import shutil
import numpy as np
import torch


def check_checkpoint_path_redundancy(path):
    if os.path.exists(path):
        warnings.warn('The checkpoint already exists on the same path.')


def save_checkpoint_sync(path, ckpt, f) -> None:
    torch.save(ckpt, os.path.join(path, f))


class CKPT:

    base_folder = 'checkpoint'

    def __init__(self, dirs: Union[str, os.PathLike], ckpt: Dict, last_epoch: int = -1) -> None:
        r"""

        Args:
            dirs: ...
            ckpt: ...
            last_epoch: ...
        """
        self.path = os.path.join(dirs, self.base_folder)
        self.ckpt = ckpt
        self.last_epoch = last_epoch

        if os.path.exists(self.path):
            warnings.warn('')
            proceed = input('Proceed? [y/N]: ')
            proceed = True if proceed.lower() == 'y' else False
            if proceed:
                shutil.rmtree(self.path)
            else:
                pass

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def step(self, epoch: Optional[int] = None) -> None:
        epoch = self.last_epoch + 1 if epoch is None else epoch
        snapshot = dict(epoch=epoch)
        snapshot.update(copy.deepcopy(self.ckpt))  # make snapshot of checkpoint
        save_checkpoint_sync(self.path, snapshot, f'checkpoint_epoch_{epoch}.pth')
        self.last_epoch = np.floor(epoch)

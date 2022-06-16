from typing import *

import warnings
import os.path
import shutil
import secrets


def _warn_checkpoint_path_redundancy(p: Union[str, os.PathLike]) -> None:
    if os.path.exists(p):
        warnings.warn('Warning :: the checkpoint path already exists\n'
                      'If continue, your previous runs may be overwritten\n'
                      'You can use `use_unique_id=True` to avoid checkpoint crashes\n')


def _initiate_checkpoint_path(p: Union[str, os.PathLike], force_rm: bool = False) -> None:
    if os.path.exists(p):
        if force_rm:
            shutil.rmtree(p)
    if not os.path.exists(p):
        os.makedirs(p)


class CKPT:

    unique_id = secrets.token_hex(4)
    base_folder = 'checkpoint'

    def __init__(
            self, f, ckpt,
            *,
            last_global_step: int = -1,
            use_unique_id: bool = False,
            remove_previous_runs: bool = False,
    ) -> None:
        if use_unique_id:
            self.base_folder = f'{self.base_folder}.{self.unique_id}'

        self.f = os.path.join(f, self.base_folder)
        self.checkpoint = ckpt
        self.last_global_step = last_global_step

        _warn_checkpoint_path_redundancy(f)
        _initiate_checkpoint_path(f, force_rm=remove_previous_runs)

    def step(self, global_step: Optional[int] = None) -> None:
        global_step = self.last_global_step + 1 if global_step is None else global_step

        # save checkpoint...

        self.last_global_step = int(global_step)

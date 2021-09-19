from typing import List
from abc import ABC

import torch


class Callback(ABC):
    def on_init_end(self, trainer):
        pass

    def on_epoch_start(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def on_batch_start(self, trainer, batch: tuple, batch_idx: int):
        pass

    def on_batch_end(
        self,
        trainer,
        batch: tuple,
        batch_idx: int,
        loss: List[torch.Tensor],
        model_output: List[torch.Tensor],
    ):
        pass

    def on_train_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

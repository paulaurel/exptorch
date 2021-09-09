from pathlib import Path
from warnings import warn
from typing import Optional, Union

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ..trainer import Trainer
from .callbacks import Callback
from ..utils.validation import validate_value


try:
    from torchvision.utils import make_grid
except ImportError:
    warn(
        "Failed to import 'make_grid' from 'torchvision.utils'."
        " TensorboardLogger will be unable to log batches of images."
    )


def tensor_to_scalar(tensor):
    try:
        return tensor.item()
    except ValueError:
        raise ValueError(
            "Expected single element tensor."
            f" Received tensor of shape {tensor.shape}."
        )


class TensorboardLogger(Callback):
    def __init__(self):
        self._writer = None
        self._log_dir = None

    def _init_writer(self, exp_dir: Path):
        self._log_dir = Path(exp_dir) / "logs"
        self._log_dir.mkdir()
        self._writer = SummaryWriter(log_dir=str(self._log_dir))

    def on_init_end(self, trainer: Trainer):
        self._init_writer(trainer.exp_dir)

    @property
    def log_dir(self):
        if self._log_dir is None:
            raise AttributeError(
                "The log_dir has not been defined yet."
                "It is defined on completion of Trainer's initialization."
            )
        return self._log_dir

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()

    @staticmethod
    def _make_tag(description: str, prefix: Optional[str] = ""):
        return f"{prefix}/{description}" if len(prefix) != 0 else description

    def log_scalar(
            self,
            scalar: Union[torch.Tensor, float],
            global_step: int,
            description: str,
            prefix: str = "",
    ):
        if not isinstance(scalar, torch.Tensor):
            scalar = torch.Tensor(scalar)

        self._writer.add_scalar(
            tag=self._make_tag(description, prefix),
            scalar_value=tensor_to_scalar(scalar),
            global_step=global_step
        )

    def log_image(
            self,
            img: Union[np.ndarray, torch.Tensor],
            global_step: int,
            description: str,
            prefix: str,
            dataformat: str = "CHW",
            **kwargs,
    ):
        validate_value(img.shape, allowed_values=[3, 4])
        if img.ndim == 4:
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            img = make_grid(img, **kwargs)

        self._writer.add_image(
            tag=self._make_tag(description, prefix),
            img_tensor=img,
            global_step=global_step,
            dataformats=dataformat,
        )

    def on_train_end(self, trainer: Trainer):
        self.flush()
        self.close()

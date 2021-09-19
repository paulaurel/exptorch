from enum import Enum
from pathlib import Path
from inspect import signature
from types import GeneratorType
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Callable

import torch
from torch.utils.data import DataLoader

from .containers import Struct
from .callbacks.progress_bar import ProgressBar
from .utils.validation import validate_type, has_arg, validate_arg


class TrainerStatus(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    FINISHED = "finished"

    @property
    def stopped(self) -> bool:
        return self in (self.INTERRUPTED, self.FINISHED)


@dataclass
class TrainerState:
    status: TrainerStatus = TrainerStatus.INITIALIZING
    epoch: int = 1
    batch: int = 1
    num_train_batches: int = None
    num_val_batches: int = None

    @property
    def num_completed_epochs(self):
        return self.epoch - 1

    def reset_batch(self):
        self.batch = 1

    def increment_batch(self):
        self.batch += 1

    def increment_epoch(self):
        self.epoch += 1


def to_device(data, device: torch.device):
    if isinstance(data, (list, tuple, GeneratorType)):
        data_gen = (to_device(x, device) for x in data)
        if isinstance(data, List):
            return list(data_gen)
        if isinstance(data, Tuple):
            return tuple(data_gen)
        return data_gen
    elif isinstance(data, (torch.Tensor, torch.nn.Module)):
        return data.to(device)
    elif isinstance(data, (Struct, dict)):
        return Struct(
            (key, to_device(value, device))
            for key, value in data.items()
        )
    else:
        return data


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:

    _required_training_step_args = ["batch", "batch", "loss_criterion"]

    def __init__(
            self,
            model: torch.nn.Module,
            optimizers,
            optimizer_params: Union[Struct, dict],
            loss_criterion: Union[torch.nn.Module, Callable, Struct],
            train_data_loader: DataLoader,
            max_epochs: int,
            exp_dir: Path,
            device: Optional[torch.device] = None,
            callbacks: Optional[List] = None,
            val_data_loader: Optional[DataLoader] = None,
    ):
        self._model = model
        self._optimizers = optimizers
        self._optimizer_params = optimizer_params
        self._loss_criterion = loss_criterion
        self._train_data_loader = train_data_loader
        self._val_data_loader = val_data_loader
        self._callbacks = [] if callbacks is None else callbacks
        self._callbacks = callbacks + [ProgressBar()]
        self._exp_dir = exp_dir
        self._max_epochs = max_epochs
        self._device = get_device() if device is None else device

        self.state = TrainerState()

        self._call_hook("on_init_end")

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def optimizers(self) -> List:
        return self._optimizers

    @property
    def max_epochs(self) -> int:
        return self._max_epochs

    @property
    def exp_dir(self) -> Path:
        return self._exp_dir

    @property
    def device(self) -> torch.device:
        return self._device

    def _terminate_training(self) -> bool:
        return self.state.num_completed_epochs == self._max_epochs

    def fit(self):
        self._init_training()
        self.state.status = TrainerStatus.RUNNING
        self._call_hook("on_train_start")

        while not self._terminate_training():
            self._run_epoch()

        self._call_hook("on_train_end")
        self.state.status = TrainerStatus.FINISHED

    def _run_epoch(self):
        self._update_state_on_epoch_start()
        self._call_hook("on_epoch_start")
        for batch_idx, batch in enumerate(self._train_data_loader):
            self._run_batch(batch, batch_idx)
        self._call_hook("on_epoch_end")
        self._update_state_on_epoch_end()

    def _run_batch(self, batch, batch_idx):
        batch = to_device(batch, self.device)
        self._call_hook("on_batch_start", batch, batch_idx)
        result = self._training_step(batch, batch_idx)
        self._call_hook("on_batch_end", **result)
        self._update_state_on_batch_end()

    def _update_state_on_epoch_start(self):
        self.state.num_train_batches = len(self._train_data_loader)
        self.state.num_val_batches = len(self._val_data_loader) if self._val_data_loader is not None else 0

    def _update_state_on_batch_end(self):
        self.state.increment_batch()

    def _update_state_on_epoch_end(self):
        self.state.increment_epoch()
        self.state.reset_batch()

    def _call_hook(self, hook_name: str, *args, **kwargs):
        for callback in self._callbacks:
            callback_fn = getattr(callback, hook_name)
            callback_fn(self, *args, **kwargs)

    def _init_training(self):
        self._init_model_for_training()
        self._init_optimizer()
        self._loss_criterion = to_device(self._loss_criterion, self.device)

    def _init_model_for_training(self):
        self._model.train()
        torch.set_grad_enabled(True)
        self._model.to(self._device)

    def _init_optimizer(self):
        if not hasattr(self._model, "configure_optimizers"):
            raise AttributeError(f"Require that {type(self._model).__name__} has a configure_optimizers method.")

        self._optimizers = self.model.configure_optimizers(self._optimizers, **self._optimizer_params)
        if not isinstance(self._optimizers, (list, tuple)):
            self._optimizers = [self._optimizers]
        for optimizer in self._optimizers:
            validate_type(
                optimizer, required_type=torch.optim.Optimizer, obj_name="optimizer"
            )

    def _check_train_args(self):
        for required_arg in self._required_training_step_args:
            validate_arg(self.model.training_step, required_arg)

    def _build_train_kwargs(self, batch, batch_idx, opt_idx):
        self._check_train_args()
        kwargs = dict(batch=batch, batch_idx=batch_idx, loss_criterion=self._loss_criterion)

        if (num_opts := len(self._optimizers)) > 1:
            has_opt_idx_arg = has_arg(self.model.training_step, "optimizer_idx")
            if not has_opt_idx_arg:
                raise ValueError(
                    f"Given model, {type(self.model).__name__}, instantiates {num_opts} optimizers and therefore,"
                    f" the model's method {self.model.training_step.__name__} requires the argument 'opt_idx'."
                    " This argument is missing."
                    f" Provided arguments are: {signature(self.model.training_step).parameters}"
                )
            kwargs["optimizer_idx"] = opt_idx

        return kwargs

    def _training_step(self, batch: tuple, batch_idx: int) -> Struct:
        result = Struct(batch=batch, batch_idx=batch_idx, loss=[], model_output=[])
        for opt_idx, opt in enumerate(self._optimizers):
            kwargs = self._build_train_kwargs(batch, batch_idx, opt_idx)
            loss, model_output = self._model.training_step(**kwargs)
            loss.backward()
            opt.step()
            opt.zero_grad()
            result.model_output.append(model_output.detach())
            result.loss.append(loss.detach())
        return result

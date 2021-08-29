from enum import Enum
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

    _required_training_step_args = ["batch", "batch_idx", "loss_criterion"]

    def __init__(
            self,
            model: torch.nn.Module,
            optimizers: Union[Struct, torch.optim],
            loss_criterion: Union[torch.nn.Module, Callable, Struct],
            train_data_loader: DataLoader,
            config: Struct,
            device: Optional[torch.device] = None,
            callbacks: Optional[Struct] = None,
            val_data_loader: Optional[DataLoader] = None,
    ):
        self._model = model
        self._optimizers = optimizers
        self._loss_criterion = loss_criterion
        self._train_data_loader = train_data_loader
        self._val_data_loader = val_data_loader
        self._callbacks = [] if callbacks is None else callbacks
        self._callbacks = callbacks + [ProgressBar]
        self._config = config
        self._device = get_device() if device is None else device

        self.num_train_batches = len(train_data_loader)
        self.num_val_batches = 0 if val_data_loader is None else len(val_data_loader)
        self.state = TrainerState()

        self._call_hook("on_init_end", self)

    @property
    def config(self) -> Struct:
        return self._config

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def terminate_training(self) -> bool:
        return self.state.epoch == self._config.train_params.num_epochs

    def fit(self):
        self._init_training()
        self.state.status = TrainerStatus.RUNNING
        self._call_hook("on_train_start", self)

        while not self.terminate_training():
            self._run_epoch()

        self._call_hook("on_train_end", self)
        self.state.status = TrainerStatus.FINISHED

    def _run_epoch(self):
        self._call_hook("on_epoch_start", self)
        for batch_idx, batch in enumerate(self._train_data_loader):
            self._run_batch(batch, batch_idx)
        self._call_hook("on_epoch_end", self)

    def _run_batch(self, batch, batch_idx):
        batch = to_device(batch, self.device)
        self._call_hook("on_batch_start", self, batch, batch_idx)
        result = self._training_step(batch, batch_idx)
        self._call_hook("on_batch_end", self, **result)
        self._update_state_on_batch_end()

    def _update_state_on_epoch_start(self):
        self.state.num_train_batches = len(self._train_data_loader)
        self.state.num_val_batches = len(self._val_data_loader) if self._val_data_loader is not None else None

    def _update_state_on_batch_end(self):
        self.state.increment_batch()

    def _update_state_on_epoch_end(self):
        self.state.increment_epoch()
        self.state.reset_batch()

    def _call_hook(self, hook_name: str, *args, **kwargs):
        for callback in self._callbacks.values():
            callback_fn = getattr(callback, hook_name)
            callback_fn(*args, **kwargs)

    def _on_epoch_end(self):
        for callback in self._callbacks.values():
            callback.on_epoch_end(
                self.state, self._model, self._optimizers, self._val_data_loader
            )

    def _init_training(self):
        self._init_model_for_training()
        self._init_optimizer()
        self._loss_criterion = to_device(self._loss_criterion, self.device)

    def _init_model_for_training(self):
        self._model.train()
        torch.set_grad_enabled(True)
        self._model.to(self._device)

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
                    f"Given model defines {num_opts} optimizers and therefore,"
                    f" the model's method {self.model.training_step.__name__} requires the argument 'opt_idx'."
                    " This argument is missing."
                    f" Provided arguments are: {signature(self.model.training_step).parameters}"
                )
            kwargs["optimizer_idx"] = opt_idx

        return kwargs

    def _init_optimizer(self):
        self._optimizers = self.model.configure_optimizers(self._optimizers, **self._config.optimizer_params)
        if not isinstance(self._optimizers, (list, tuple)):
            self._optimizers = list(self._optimizers)
        for optimizer in self._optimizers:
            validate_type(
                optimizer, required_type=torch.optim.Optimizer, obj_name="optimizer"
            )

    def _training_step(self, batch: tuple, batch_idx: int) -> Struct:
        result = Struct(batch=batch, batch_idx=batch_idx, loss=[], model_output=[])
        for opt_idx, opt in self._optimizers:
            kwargs = self._build_train_kwargs(batch, batch_idx, opt_idx)
            loss, model_output = self._model.training_step(**kwargs)
            loss.backward()
            opt.step()
            opt.zero_grad()
            result.model_output.append(model_output.detach())
            result.loss.append(loss.detach())
        return result

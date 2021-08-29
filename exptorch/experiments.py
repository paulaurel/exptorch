import os
import json
import pickle
import inspect
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Union
from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from .containers import Struct, Params
from .utils.itertools import named_product
from .utils.validation import validate_type


def run_on_local(config):
    model = config.model(**config.model_params)
    optimizer = model.configure_optimizers(config.optimizers, **config.optimizer_params)
    model.train()
    loss_fn = config.losses()
    train_dataset = config.train_dataset(**config.train_dataset_params)
    train_data_loader = DataLoader(train_dataset, batch_size=config.train_params.batch_size)
    running_loss = 0.0
    for epoch in range(config.train_params.epochs):
        for idx, batch in enumerate(train_data_loader):
            loss = model.training_step(batch, loss_fn)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

            if idx % 1000 == 999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f" % (epoch + 1, idx + 1, running_loss / 1000)
                )
                running_loss = 0.0

    return model


def is_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def run_on_cloud(num_workers):
    pass
    # 1. copy experiment configuration to could instance GCP
    # 2. create required miniconda image
    # 3. pull repository
    # 4. checkout branch given in
    # 5. load config

    # 6. start training

    # 7. finished training
    # 8. copy files from instance to local machine
    # 9. shutdown instance


def _serialize_to_pickle(exp_config):
    pass


def _serialize_to_json(exp_config):
    pass


def save_config(exp_config: Struct):
    exp_config_fname = exp_config.exp_dir / "config.pkl"
    with exp_config_fname.open("wb") as file:
        pickle.dump(exp_config, file)


def required_arguments(fn: Callable):
    all_arguments = inspect.signature(fn).parameters
    default_arguments = {(arg_name, arg_desc)
                         for arg_name, arg_desc in all_arguments
                         if arg_desc.default is not inspect.Parameter.empty
                         }
    pass


def _extract_desc(data: Union[Struct, Callable]) -> str:
    if isinstance(data, Callable):
        return data.__name__

    def _to_str(_value):
        if isinstance(_value, Callable):
            return _value.__name__
        return str(_value)

    return "_".join([f"{key}_{_to_str(value)}" for key, value in data.items()])


def label_experiment(exp_config: Struct, exp_idx: int) -> str:
    items_required_in_label = [
        exp_config.model,
        exp_config.train_dataset,
        exp_config.train_params,
        exp_config.optimizers,
        exp_config.optimizer_params,
        exp_config.losses,
    ]
    exp_config_label = "_".join(map(_extract_desc, items_required_in_label)).lower()
    exp_idx_label = f"exp_idx_{exp_idx:03d}"
    return f"{exp_config_label}_{exp_idx_label}"


def make_experiment_dir(exp_config: Struct, exp_idx: int) -> os.PathLike:
    exp_label = label_experiment(exp_config, exp_idx)
    exp_dir = exp_config.exp_base_dir / exp_label
    exp_dir.mkdir()
    return exp_dir


def track_experiment():
    pass
    # create a commit on a branch with the current state


def load_experiment(exp_config: Struct):
    pass


def create_experiments(
    *,
    save_dir: os.PathLike,
    model: Callable,
    model_params: Params,
    train_dataset: Callable,
    train_params: Params,
    optimizers: Struct,
    optimizer_params: Params,
    losses: Struct,
    train_dataset_params: Params = None,
    callbacks: List[Callable] = None,
    val_dataset: Callable = None,
    run: bool = False,
):
    """Create experiments for the specified subset of the hyperparameter space.

    Notes
    -----
    The subset of the hyperparameter space is constructed by cartesian product of all parameters, i.e.:
        CartesianProduct(exp_params, model_params, optimizer_params, losses)

    Parameters
    ----------
    save_dir: PathLike object
        Path to which the experiments are written.
    model: Callable
        Model class to be trained with various parameter settings.
    model_params: Params
        Parameters defining the model instance(s).
    train_dataset: Callable
        Dataset class capturing the training dataset.
    train_params: Params
        Parameters defining the training loop(s).
    optimizers: Struct
        Struct of optimizers.
    optimizer_params: Params
        Parameters defining the optimizer instance(s).
    losses: Struct
        Struct of loss functions.
    train_dataset_params: Struct
        Optional parameters defining the train_dataset.
        If provided the train_dataset is instantiated
        as follows train_dataset(**train_dataset_params),
        instead of train_dataset().
    callbacks: List[Callable]
        List of callbacks called during the model's training.
    val_dataset: Callable
        Dataset class capturing the validation dataset.
    run: bool
        Whether to run the defined experiments.
    """
    base_dir = Path(save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir.mkdir()

    exp_params = Struct(
        base_dir=base_dir,
        train_params=train_params.expand(),
        model=model,
        model_params=model_params.expand(),
        optimizers=optimizers.values(),
        optimizer_params=optimizer_params.expand(),
        train_dataset=train_dataset,
        train_dataset_params=train_dataset_params.expand(),
        losses=losses.values(),
    )

    if callbacks is not None:
        validate_type(callbacks, required_type=list, obj_name="callbacks")
        exp_params.callbacks = [callbacks]

    if val_dataset is not None:
        validate_type(val_dataset, required_type=Callable, obj_name="val_dataset")
        exp_params.val_dataset = val_dataset

    for exp_idx, exp_config in enumerate(named_product(**exp_params)):
        exp_dir = make_experiment_dir(exp_config, exp_idx)
        exp_config.exp_dir = exp_dir
        save_config(exp_config)

        if run:
            run_on_local(exp_config)

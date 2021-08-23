from json import dumps
from typing import Callable, Union

from .containers import Struct, Params
from .utils.itertools import named_product

import torch


def run_on_local():
    pass


def is_serializable(obj):
    try:
        dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def run_on_cloud(num_workers):
    pass


def save_experiment(model, dataset, optimizers, callbacks, exp_config):
    # - experiment_dir
    #     |- config.json
    #     |- model definition
    torch.save(model)
    pass


def rerun_experiment():
    pass


def load_experiment():
    pass


def create_experiments(
    model: torch.nn.Module,
    optimizers: Struct,
    losses: Struct,
    callbacks: Struct,
    dataset,
    exp_params: Params,
    model_params: Params,
    optimizer_params: Params,
    run: bool = False,
):
    """Create experiments for the specified subset of the hyperparameter space.

    Notes
    -----
    The subset of the hyperparameter space is constructed by cartesian product of all parameters, i.e.:
        CartesianProduct(exp_params, model_params, optimizer_params, losses)

    Parameters
    ----------
    model: torch.nn.Module
        Model to be trained with various parameter settings.
    optimizers: Struct
        Struct of optimizers.
    losses: Struct
        Struct of loss functions.
    callbacks: list
        List of callbacks called during training.
    dataset
    exp_params: Params
        Parameters defining experiment(s).
    model_params: Params
        Parameters defining model instance(s).
    optimizer_params: Params
        Parameters defining optimizer instance(s).
    run: bool
    """
    exp_config = Struct(
        exp_params=exp_params.expand(),
        model_params=model_params.expand(),
        optimizer_params=optimizer_params.expand(),
        loss=losses.values(),
    )

    for exp_config in named_product(**exp_config):
        save_experiment(model, dataset, optimizers, callbacks, **exp_config)

        if run:
            run_on_local()

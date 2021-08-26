from json import dumps
from pathlib import Path
from datetime import datetime
from typing import Callable, Union

import torch
from torch.utils.data import DataLoader

from .containers import Struct, Params
from .utils.itertools import named_product


def run_on_local(
        model: torch.nn.Module,
        optimizer: torch.optim,
        loss: torch.nn.Module,
        dataset,
        config: Struct,
):
    optimizer = model.configure_optimizers([torch.optim.Adam], **config.optimizer_params)
    model.train()
    loss_fn = loss()
    dataset = DataLoader(dataset, batch_size=config.exp_params.batch_size)
    running_loss = 0.0
    for epoch in range(config.exp_params.num_epochs):
        for idx, batch in enumerate(dataset):
            loss = model.training_step(batch, loss_fn)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

            if idx % 1000 == 999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, idx + 1, running_loss / 1000))
                running_loss = 0.0

    return model


def is_serializable(obj):
    try:
        dumps(obj)
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


def save_experiment(model, dataset, loss, optimizers, callbacks, exp_config, config_dir):
    # - experiment_dir described by time
    #   - respective experiment_dir
    #     |- config.json
    #     |- model definition

    # 1. create directory label - use some hash
    # 2. generate config.json --> need to serialize objects
    # 3. pickle callables and dataset
    # 4. save --> model, optimizers, loss

    torch.save(model)
    pass


def track_experiment():
    pass
    # create a commit on a branch with the current state


def load_experiment():
    pass


def get_exp_dir_name() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def create_experiments(
    *,
    model: torch.nn.Module,
    optimizers: Struct,
    losses: Struct,
    dataset,
    train_params: Params,
    model_params: Params,
    optimizer_params: Params,
    save_dir: Path,
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
    dataset:
    train_params: Params
        Parameters defining the training loop(s).
    model_params: Params
        Parameters defining the model instance(s).
    optimizer_params: Params
        Parameters defining the optimizer instance(s).
    save_dir: Path
        Directory path in which experiment and its configuration is saved.
    run: bool
    """

    exp_dir = save_dir / get_exp_dir_name()
    exp_dir.mkdir()

    exp_params = Struct(
        train_params=train_params.expand(),
        model_params=model_params.expand(),
        optimizer_params=optimizer_params.expand(),
        loss=losses.values(),
    )

    for exp_config in named_product(**exp_params):
        save_experiment(model, dataset, exp_config.loss, optimizers, callbacks, config_dir, all_)

        if run:
            run_on_local(
                model=model(**exp_config.model_params),
                loss=exp_config.loss,
                dataset=dataset,
                optimizer=optimizers,
                config=exp_config,
            )

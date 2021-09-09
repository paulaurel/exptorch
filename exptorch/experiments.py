import os
import pickle
from pathlib import Path
from functools import partial
from datetime import datetime
from typing import List, Union, Optional
from collections.abc import Callable

from torch.utils.data import DataLoader

from .trainer import Trainer
from .containers import Struct, Params
from .utils.itertools import named_product
from .utils.validation import validate_type


def _init_config_factory(config: Struct, key: str, param_key: Optional[str] = None):
    if isinstance(config[key], Callable):
        try:
            return config[key](**config.get(param_key, {}))
        except TypeError:
            raise TypeError(

            )
    elif isinstance(config[key], list):
        try:
            return [item() for item in config[key]]
        except TypeError:
            raise TypeError(

            )
    else:
        try:
            return Struct(
                (key, value(**config.get(param_key, {}).get(key, {})))
                for key, value in config.losses.items()
            )
        except TypeError:
            raise TypeError


init_model = partial(_init_config_factory, key="model", param_key="model_params")
init_callbacks = partial(_init_config_factory, key="callbacks")
init_loss = partial(_init_config_factory, key="losses", param_key="loss_params")
init_train_dataset = partial(_init_config_factory, key="train_dataset", param_key="train_dataset_params")


def run_on_local(config):
    trainer = Trainer(
        model=init_model(config),
        loss_criterion=init_loss(config),
        train_data_loader=DataLoader(
            dataset=init_train_dataset(config),
            batch_size=config.train_params.batch_size,
        ),
        max_epochs=config.train_params.epochs,
        exp_dir=config.exp_dir,
        optimizers=config.optimizers,
        optimizer_params=config.optimizer_params,
        callbacks=init_callbacks(config),
    )
    trainer.fit()
    return trainer.model


def run_on_remote(num_workers):
    """
    TODO @paul
        1. copy experiment configuration to could instance GCP
        2. create required miniconda image
        3. pull repository
        4. checkout branch given in
        5. load config
        6. start training
        7. finished training
        8. copy files from instance to local machine
        9. shutdown instance`
    """
    pass


def serialize_to_json(config: Struct, fname: Path):
    # TODO @paul implement a json serialization that is human readable
    raise NotImplementedError


def serialize_to_pickle(config: Struct, fname: Path):
    with fname.open("wb") as file:
        pickle.dump(config, file)


def deserialize_from_pickle(exp_config_fname: Path) -> Struct:
    with exp_config_fname.open("rb") as file:
        exp_config = pickle.load(file)
    return exp_config


def save_experiment(exp_config: Struct):
    if not hasattr(exp_config, "exp_dir"):
        raise AttributeError(
            f"Require that '{type(exp_config).__name__}' object 'exp_config' has attribute 'exp_dir'."
            f" Given 'exp_config' has following attributes: {exp_config.__dict__.keys()}."
            " Ensure that 'exp_dir' attribute is set."
        )
    serialize_to_pickle(exp_config, exp_config.exp_dir / "config.pkl")
    # serialize_to_json(exp_config, exp_config.exp_dir / "config.json")


def load_experiment(exp_config_fname: Union[Path, os.PathLike]) -> Struct:
    return deserialize_from_pickle(Path(exp_config_fname))


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


def make_experiment_dir(exp_config: Struct, exp_idx: int) -> Path:
    exp_label = label_experiment(exp_config, exp_idx)
    exp_dir = exp_config.base_dir / exp_label
    exp_dir.mkdir()
    return exp_dir


def _validate_experiment_params(exp_params: Params):
    raise NotImplementedError


def create_experiments(
    *,
    save_dir: os.PathLike,
    model: Callable,
    model_params: Params,
    train_dataset: Callable,
    train_params: Params,
    optimizers: Union[Callable, List, Struct],
    optimizer_params: Params,
    losses: Union[Callable, List, Struct],
    train_dataset_params: Union[Params, Struct] = None,
    callbacks: List[Callable] = None,
    val_dataset: Callable = None,
    val_dataset_params: Union[Params, Struct] = None,
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
        Struct of loss_criterion functions.
    train_dataset_params: Struct
        Optional parameters defining the train_dataset.
        If provided the train_dataset is instantiated
        as follows train_dataset(**dataset_params),
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
        optimizers=optimizers,
        optimizer_params=optimizer_params.expand(),
        train_dataset=train_dataset,
        train_dataset_params=Params() if train_dataset_params is None else train_dataset_params.expand(),
        losses=losses,
        callbacks=None if callbacks is None else [callbacks],
        val_dataset=val_dataset,
    )

    for exp_idx, exp_config in enumerate(named_product(**exp_params)):
        exp_config.exp_dir = make_experiment_dir(exp_config, exp_idx)
        save_experiment(exp_config)
        if run:
            run_on_local(exp_config)

import os
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Union
from collections.abc import Callable

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
    train_data_loader = DataLoader(
        train_dataset, batch_size=config.train_params.batch_size
    )
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


def run_on_cloud(num_workers):
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


def track_experiment():
    raise NotImplementedError


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
    train_dataset_params: Union[Params, Struct] = None,
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
        exp_config.exp_dir = make_experiment_dir(exp_config, exp_idx)
        save_experiment(exp_config)
        if run:
            run_on_local(exp_config)

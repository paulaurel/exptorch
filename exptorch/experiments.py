import os
import torch
import pickle
import subprocess
from pathlib import Path
from warnings import warn
from datetime import datetime
from types import FunctionType
from collections.abc import Callable
from torch.utils.data import DataLoader
from typing import List, Union, Type, Optional

from .trainer import Trainer
from .containers import Struct, Params
from .utils.constructor import construct
from .utils.itertools import named_product
from .utils.validation import validate_value


def run_on_local(config):
    """Run experiment specified by the experiment configuration on the local machine.

    Parameters
    ----------
    config: Struct
        Experiment configuration specifying the learning experiment.
    """
    trainer = Trainer(
        model=construct(config.model, **config.model_params),
        loss_criterion=construct(config.losses, **config.loss_params),
        train_data_loader=DataLoader(
            dataset=construct(config.train_dataset, **config.train_dataset_params),
            batch_size=config.train_params.batch_size,
        ),
        max_epochs=config.train_params.epochs,
        exp_dir=config.exp_dir,
        optimizers=config.optimizers,
        optimizer_params=config.optimizer_params,
        callbacks=construct(config.callbacks),
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
    raise NotImplementedError


def run_experiments(exp_configs: List[Struct], execution_strategy: str = "local"):
    """Run all of experiment configurations defined within exp_configs.

    Parameters
    ----------
    exp_configs: List[Struct]
        List of experiment configurations.
        Each list entry defines a experiment configuration, which will be executed.
    execution_strategy: str
        Defines whether the experiment is performed on the local machine or on a
        remote machine. Can taken on two values ("local", "remote").
    """
    validate_value(
        execution_strategy,
        allowed_value=("local", "remote"),
        value_name="execution_strategy",
    )

    for exp_idx, exp_config in enumerate(exp_configs):
        exp_config = load_experiment(exp_config.exp_dir / "config.pkl")

        if execution_strategy == "local":
            try:
                run_on_local(exp_config)
            except Exception as err_msg:
                warn(
                    f"Experiment {label_experiment(exp_config, exp_idx)} failed"
                    f" with the following error: \n {err_msg}."
                )

    if execution_strategy == "remote":
        run_on_remote(num_workers=5)


def serialize_to_json(config: Struct, fname: Path):
    # TODO @paul implement a json serialization that is human readable
    raise NotImplementedError


def serialize_to_pickle(config: Struct, config_fname: Path):
    """Serialize configuration via pickle.

    Parameters
    ----------
    config: Struct
        Configuration to be serialized and stored to disk.
    config_fname: Path
        File path to which the serialized configuration is saved.
    """
    with config_fname.open("wb") as fid:
        pickle.dump(config, fid)


def deserialize_from_pickle(exp_config_fname: Path) -> Struct:
    with exp_config_fname.open("rb") as fid:
        exp_config = pickle.load(fid)
    return exp_config


def save_experiments(exp_configs: List[Struct]):
    """Save all experiment configurations."""
    for exp_idx, exp_config in enumerate(exp_configs):
        exp_config.exp_dir = make_experiment_dir(exp_config, exp_idx)
        save_experiment(exp_config)


def save_experiment(exp_config: Struct):
    """Save single experiment configuration by serializing the configuration via pickle and json."""
    if not hasattr(exp_config, "exp_dir"):
        raise AttributeError(
            f"Require that '{type(exp_config).__name__}' object 'exp_config' has attribute 'exp_dir'."
            f" Given 'exp_config' has following attributes: {exp_config.__dict__.keys()}."
            " Ensure that 'exp_dir' attribute is set."
        )
    serialize_to_pickle(exp_config, exp_config.exp_dir / "config.pkl")
    # TODO serialize_to_json(exp_config, exp_config.exp_dir / "config.json")


def load_experiment(exp_config_fname: Union[Path, os.PathLike]) -> Struct:
    """Load experiment from pickled experiment configuration file."""
    exp_config = deserialize_from_pickle(Path(exp_config_fname))
    if (cur_rev_hash := get_git_revision_hash()) != exp_config.git_rev_hash:
        warn(
            "The current commit is different to the commit the experiment was saved on.\n"
            f"The current commit is: {cur_rev_hash}.\n"
            f"The experiment was saved on commit: {exp_config.git_rev_hash}.\n"
            "Consequently, experiment results are not guaranteed to be identical and the experiment can fail."
        )
    return exp_config


def _extract_description(data: Union[Struct, Callable]) -> str:
    if isinstance(data, Callable):
        return data.__name__

    def _to_str(_value):
        if isinstance(_value, Callable):
            return _value.__name__
        return str(_value)

    return "_".join([f"{key}_{_to_str(value)}" for key, value in data.items()])


def label_experiment(exp_config: Struct, exp_idx: int) -> str:
    """Label experiment by defining a unique experiment description."""
    items_required_in_label = [
        exp_config.model,
        exp_config.train_dataset,
        exp_config.train_params,
        exp_config.optimizers,
        exp_config.optimizer_params,
        exp_config.losses,
    ]
    exp_config_label = "_".join(
        map(_extract_description, items_required_in_label)
    ).lower()
    exp_idx_label = f"exp_idx_{exp_idx:03d}"
    return f"{exp_config_label}_{exp_idx_label}"


def make_experiment_dir(exp_config: Struct, exp_idx: int) -> Path:
    """Make unique experiment directory for the given experiment configuration.

    Parameters
    ----------
    exp_config: Struct
        Experiment configuration
    exp_idx: int
        Experiment idx.

    Returns
    -------
    exp_dir: Path
        Path to unique experiment directory.
    """
    exp_label = label_experiment(exp_config, exp_idx)
    exp_dir = exp_config.base_dir / exp_label
    exp_dir.mkdir()
    return exp_dir


def get_git_revision_hash() -> str:
    """Get current git revision hash as a string."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def validate_experiment_config(exp_config: Struct):
    """Validate that given experiment configuration is executable.

    Parameters
    ----------
    exp_config: Struct
        Experiment configuration describing the learning experiment,
        i.e. the class_obj, the class_obj parameters, the training loss, ...

    Raises
    ------
        Raises TypeError if the given experiment configuration is not executable.
    """


def create_experiments(
    *,
    save_dir: os.PathLike,
    model: Type[torch.nn.Module],
    train_dataset: Type[torch.utils.data.Dataset],
    train_params: Params,
    optimizers: Union[Type[torch.optim.Optimizer], list, tuple, Struct],
    losses: Union[FunctionType, list, tuple, Struct, Type[torch.nn.Module]],
    callbacks: List[type] = None,
    val_dataset: Optional[Type[torch.utils.data.Dataset]] = None,
    model_params: Optional[Params] = None,
    optimizer_params: Optional[Params] = None,
    loss_params: Optional[Params] = None,
    train_dataset_params: Optional[Params] = None,
    val_dataset_params: Optional[Params] = None,
    execution_strategy: str = "save",
):
    """Create experiments for the specified subset of the hyperparameter space.

    Notes
    -----
    The subset of the hyperparameter space is constructed by cartesian product of all parameters, i.e.:
        CartesianProduct(train_params, model_params, optimizer_params, losses, ...)

    Parameters
    ----------
    save_dir: PathLike object
        Path to which the experiments are written.
    model: Type[torch.nn.Module]
        Model class to be trained with various parameter settings.
    model_params: Optional[Params]
        Parameters defining the model.
    train_params: Params
        Parameters defining the training loop(s).
    train_dataset: Type[torch.nn.Dataset]
        Dataset class capturing the training dataset.
    train_dataset_params: Optional[Params]
        Optional parameters defining the train_dataset.
    optimizers: Struct
        Struct of optimizers.
    optimizer_params: Optional[Params]
        Parameters defining the optimizer instance(s).
    losses: FunctionType, list, Struct, Type[torch.nn.Module]
        Losses defines the loss criterion, which is used to train the model.
        It can be a function, a list, a struct or a torch.nn.Module class.
        A single function defining the loss criterion.
        A struct defining a single complex loss criterion,
        which might consist of multiple functions.
        A list defining multiple different losses.
        A single loss class of type torch.nn.Module, e.g. torch.nn.MSE.
    loss_params: Optional[Params]
        Optional parameters defining the losses,
        e.g. weight values associated with the loss.
    callbacks: list
        List of callbacks called during the model's training.
    val_dataset: Type[torch.utils.data.Dataset]
        Dataset class capturing the validation dataset.
    val_dataset_params: Optional[Params]
        Optional parameters defining the val_train_dataset.
    execution_strategy: str
        Whether to run the defined experiments.
    """

    def _ensure_params(_params: Params):
        return _params if _params is not None else Params()

    validate_value(
        execution_strategy,
        allowed_value=("save", "remote", "local"),
        value_name="execution_strategy",
    )

    base_dir = Path(save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir.mkdir()

    exp_params = Struct(
        base_dir=base_dir,
        train_params=_ensure_params(train_params).expand(),
        model=model,
        model_params=_ensure_params(model_params).expand(),
        optimizers=optimizers,
        optimizer_params=_ensure_params(optimizer_params).expand(),
        train_dataset=train_dataset,
        train_dataset_params=_ensure_params(train_dataset_params).expand(),
        losses=losses,
        loss_params=_ensure_params(loss_params).expand(),
        callbacks=[[]] if callbacks is None else [callbacks],
        val_dataset=val_dataset,
        val_dataset_params=_ensure_params(val_dataset_params).expand(),
        git_rev_hash=(get_git_revision_hash(),),
    )

    exp_configs = list(named_product(**exp_params))
    save_experiments(exp_configs)
    run_experiments(exp_configs, execution_strategy)

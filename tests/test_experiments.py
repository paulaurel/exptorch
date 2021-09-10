import pytest

from exptorch import Struct
from exptorch.experiments import (
    save_experiment,
    load_experiment,
    label_experiment,
    make_experiment_dir,
    get_git_revision_hash,
)


class EmptyModel:
    pass


class EmptyOptimizer:
    pass


class EmptyTrainDataset:
    pass


class EmptyLoss:
    pass


class Adam:
    pass


SINGLE_OPT_EXP = Struct(
    model=EmptyModel,
    train_params=Struct(epochs=100, batch_size=16),
    model_params=Struct(input_dim=10, output_dim=20),
    optimizers=EmptyOptimizer,
    optimizer_params=Struct(lr=0.001),
    train_dataset=EmptyTrainDataset,
    train_dataset_params=Struct(dataset_dir="./dataset_dir"),
    losses=EmptyLoss,
    git_rev_hash=get_git_revision_hash(),
)
MULTI_OPT_EXP = Struct(
    model=EmptyModel,
    train_params=Struct(epochs=100, batch_size=16),
    model_params=Struct(input_dim=10, output_dim=20),
    optimizers=Struct(
        discriminator_opt=Adam,
        generator_opt=Adam,
    ),
    optimizer_params=Struct(lr_discrim=0.001, lr_gen=0.002),
    train_dataset=EmptyTrainDataset,
    train_dataset_params=Struct(dataset_dir="./dataset_dir"),
    losses=EmptyLoss,
    git_rev_hash=get_git_revision_hash(),
)
EXPECTED_SINGLE_OPT_EXP_LABEL = (
    "emptymodel_emptytraindataset_epochs_100_batch_size_16"
    "_emptyoptimizer_lr_0.001_emptyloss_exp_idx_000"
)
EXPECTED_MULTI_OPT_EXP_LABEL = (
    "emptymodel_emptytraindataset_epochs_100_batch_size_16"
    "_discriminator_opt_adam_generator_opt_adam"
    "_lr_discrim_0.001_lr_gen_0.002_emptyloss_exp_idx_002"
)


@pytest.mark.parametrize(
    "exp_config, exp_idx, expected_label",
    [
        [SINGLE_OPT_EXP, 0, EXPECTED_SINGLE_OPT_EXP_LABEL],
        [MULTI_OPT_EXP, 2, EXPECTED_MULTI_OPT_EXP_LABEL],
    ],
)
class TestExperimentsFunctionality:
    @staticmethod
    def test_label_experiment(exp_config, exp_idx, expected_label):
        assert label_experiment(exp_config, exp_idx) == expected_label

    @staticmethod
    def test_make_experiment(tmpdir, exp_config, exp_idx, expected_label):
        exp_config.base_dir = tmpdir
        make_experiment_dir(exp_config, exp_idx)
        assert (tmpdir / expected_label).exists()

    @staticmethod
    def test_save_and_load_config(tmpdir, exp_config, exp_idx, expected_label):
        exp_config.base_dir = tmpdir
        exp_config.exp_dir = make_experiment_dir(exp_config, exp_idx)
        save_experiment(exp_config)
        exp_config_fname = tmpdir / expected_label / "config.pkl"
        assert exp_config_fname.exists()

        loaded_exp_config = load_experiment(exp_config_fname)
        assert loaded_exp_config == exp_config
        assert load_experiment is not exp_config_fname


@pytest.mark.parametrize("exp_config", [SINGLE_OPT_EXP, MULTI_OPT_EXP])
def test_save_config_error_msg(exp_config):
    if hasattr(exp_config, "exp_dir"):
        delattr(exp_config, "exp_dir")
    with pytest.raises(AttributeError):
        save_experiment(exp_config)

import pytest

from exptorch import Struct, Params
from exptorch.utils.validation import validate_type, validate_value


@pytest.mark.parametrize(
    "invalid_required_type", [[dict, Struct], (dict(a=5), dict), (5, 4), dict(a=5)]
)
def test_validate_type_raises_on_invalid_arg(invalid_required_type):
    with pytest.raises(TypeError) as type_error:
        validate_type(
            obj=dict(a=5), required_type=invalid_required_type, obj_name="obj_name"
        )
    expected_error_msg = (
        "validate_type requires that second argument, required_type, is a type"
        " or a tuple of types."
    )
    assert type_error.value.args[0] == expected_error_msg


@pytest.mark.parametrize(
    "obj, required_type, obj_name, expected_error_msg",
    [
        [
            5,
            float,
            "learning_rate",
            (
                "Require learning_rate to be of type: float."
                " Given object has type int."
            ),
        ],
        [
            dict(a=5, b=8, c=10),
            (Struct, Params),
            "experiment_parameters",
            (
                "Require experiment_parameters to be of type: Struct, Params."
                " Given object has type dict."
            ),
        ],
    ],
)
def test_validate_type_raises_on_type_mismatch(
    obj, required_type, obj_name, expected_error_msg
):
    with pytest.raises(TypeError) as type_error:
        validate_type(obj, required_type=required_type, obj_name=obj_name)
    assert type_error.value.args[0] == expected_error_msg


@pytest.fixture
def allowed_values():
    return tuple(["safe", "remote", "local"])


@pytest.mark.parametrize("value", ["safe", "remote", "local"])
def test_validate_value_no_raise(value, allowed_values):
    validate_value(value, allowed_value=allowed_values, value_name="execution_strategy")


@pytest.mark.parametrize(
    "value, allowed_values, value_name, expected_error_msg",
    [
        [
            5,
            (2, 3),
            "epochs",
            "Require that epochs corresponds to one of the following values: 2, 3."
            " Instead epochs has the value: 5.",
        ],
        [
            (16, 28, 28, 3),
            ((16, 3, 28, 28),),
            "batch_shape",
            "Require that batch_shape corresponds to one of the following values: (16, 3, 28, 28)."
            " Instead batch_shape has the value: (16, 28, 28, 3).",
        ],
    ],
)
def test_validate_value_raises_on_value_mismatch(
    value, allowed_values, value_name, expected_error_msg
):
    with pytest.raises(ValueError) as value_error:
        validate_value(value, allowed_value=allowed_values, value_name=value_name)
    assert value_error.value.args[0] == expected_error_msg

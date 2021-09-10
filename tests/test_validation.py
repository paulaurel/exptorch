import pytest

from exptorch import Struct, Params
from exptorch.utils.validation import validate_type


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
    with pytest.raises(TypeError) as error:
        validate_type(obj, required_type=required_type, obj_name=obj_name)
    assert error.value.args[0] == expected_error_msg

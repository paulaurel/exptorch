import pytest

from exptorch import Struct, Params
from exptorch.utils.validation import validate_type


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
def test_validate_type(obj, required_type, obj_name, expected_error_msg):
    with pytest.raises(TypeError) as error:
        validate_type(obj, required_type=required_type, obj_name=obj_name)
    assert error.value.args[0] == expected_error_msg

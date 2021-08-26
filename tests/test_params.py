import pytest

from exptorch import Params, Struct


FIXED = Struct(a=1, b=2)
FREE = Struct(c=[3, 4], d=[5, 6])
PARAMS = Params(fixed=FIXED, **FREE)
EXPANDED_PARAMS = [
    Struct(a=1, b=2, c=3, d=5),
    Struct(a=1, b=2, c=3, d=6),
    Struct(a=1, b=2, c=4, d=5),
    Struct(a=1, b=2, c=4, d=6),
]


def test_empty():
    empty_params = Params()
    assert empty_params.fixed == Struct() and empty_params.free == Struct()
    assert list(empty_params.expand()) == []


@pytest.mark.parametrize(
    "invalid_fixed_params, error_msg",
    [
        [
            5,
            (
                "Require Params.fixed to be of type: ('Struct',)."
                " Given object has type int."
            ),
        ],
        [
            dict(a=1, b=2),
            (
                "Require Params.fixed to be of type: ('Struct',)."
                " Given object has type dict."
            ),
        ],
    ],
)
def test_raise_type_error_for_invalid_fixed_params(invalid_fixed_params, error_msg):
    with pytest.raises(TypeError) as type_error:
        Params(fixed=invalid_fixed_params, **FREE)
    assert type_error.value.args[0] == error_msg


@pytest.mark.parametrize(
    "invalid_free_params, error_msg",
    [
        [
            Struct(c=3, d=[5, 6]),
            (
                "Require Params.c to be of type: ('list',)."
                " Given object has type int."
            ),
        ],
        [
            Struct(c=3, d=4),
            (
                "Require Params.c to be of type: ('list',)."
                " Given object has type int."
            ),
        ],
        [
            Struct(c=[3, 4], d=5),
            (
                "Require Params.d to be of type: ('list',)."
                " Given object has type int."
            ),
        ],
    ],
)
def test_raise_type_error_for_invalid_free_params(invalid_free_params, error_msg):
    with pytest.raises(TypeError) as type_error:
        Params(fixed=FIXED, **invalid_free_params)
    assert type_error.value.args[0] == error_msg


@pytest.mark.parametrize(
    "invalid_free_params",
    [
        Struct(a=[3, 4], d=[5, 6]),
        Struct(c=[3, 4], b=[5, 6]),
        Struct(a=[3, 4], b=[5, 6]),
    ],
)
def test_raise_value_error_for_invalid_free_params(invalid_free_params):
    with pytest.raises(ValueError):
        Params(fixed=FIXED, **invalid_free_params)


def test_fixed():
    assert PARAMS.fixed == FIXED


def test_free():
    assert PARAMS.free == FREE


def test_expand():
    expected_param_sets = set(
        tuple((parameter_set.items())) for parameter_set in EXPANDED_PARAMS
    )
    computed_param_sets = set(
        tuple(parameter_set.items()) for parameter_set in PARAMS.expand()
    )
    assert expected_param_sets == computed_param_sets

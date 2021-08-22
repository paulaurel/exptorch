import pytest
from copy import deepcopy

from exptorch import Struct

STRUCT = Struct(a=1, b=2, c=3)
KEYS, VALUES = ["a", "b", "c"], [1, 2, 3]

NESTED_STRUCT = Struct(a=1, b=Struct(d=2))


@pytest.mark.parametrize("key", KEYS)
def test_access(key):
    assert getattr(STRUCT, key) == STRUCT[key]


def test_builtin_keys_method():
    assert set(STRUCT.keys()) == set(KEYS)


def test_builtin_items_method():
    for key, value in STRUCT.items():
        assert STRUCT[key] == value


def test_comprehension():
    assert Struct(zip(KEYS, VALUES)) == STRUCT


def test_deepcopy():
    copied_struct = deepcopy(NESTED_STRUCT)
    assert copied_struct is not NESTED_STRUCT
    assert copied_struct == NESTED_STRUCT

    initial_value, new_value = copied_struct.b, None
    copied_struct.b = new_value
    assert NESTED_STRUCT.b == initial_value


def test_nesting():
    assert NESTED_STRUCT.b.d == NESTED_STRUCT["b"]["d"]

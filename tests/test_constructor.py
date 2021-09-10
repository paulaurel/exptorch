import pytest
from inspect import signature
from functools import partial

from exptorch import Struct
from exptorch.utils.constructor import construct


class EmptyClass:
    pass


class FooClass:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class BarClass:
    def __init__(self, a):
        self.a = a


def bar_func(a, b):
    pass


def test_construct_empty_list():
    assert construct([]) == []


def test_construct_empty_dict():
    assert construct(dict()) == dict()


def test_construct_single_class():
    constructed_obj = construct(FooClass, **dict(a=1, b=2))
    assert isinstance(constructed_obj, FooClass)


def test_construct_list_of_classes():
    constructed_objs = construct([FooClass for _ in range(5)], a=1, b=2)
    assert all(isinstance(obj, FooClass) for obj in constructed_objs)


def test_construct_struct_of_classes():
    struct_kwargs = Struct(bar=Struct(a=1), foo=Struct(a=1, b=1))
    constructed_struct = construct(
        Struct(bar=BarClass, foo=FooClass, empty=EmptyClass),
        **struct_kwargs,
    )
    assert isinstance(constructed_struct.bar, BarClass)
    assert isinstance(constructed_struct.foo, FooClass)


def test_construct_func():
    constructed_func = construct(bar_func, a=1)
    assert signature(constructed_func) == signature(partial(bar_func, a=1))


def test_construct_raises():
    with pytest.raises(TypeError) as error_msg:
        construct(5)
    assert error_msg.value.args[0] == "The object, 5, could not be constructed."


def test_construct_raises_with_kwargs():
    with pytest.raises(TypeError) as error_msg:
        construct(5, **dict(a=1, b=2))
    expected_error_msg = (
        "The object, 5, could not be constructed from the following kwargs:"
        " a=1, b=2."
    )
    assert error_msg.value.args[0] == expected_error_msg

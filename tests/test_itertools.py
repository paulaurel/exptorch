import pytest

from exptorch import Struct, Params
from exptorch.utils.itertools import named_product, pairwise


PARAMS = Struct(
    a=range(2),
    b=tuple(range(3)),
    c=1,
    d=[tuple(range(2))],
    e=Params(fixed=Struct(f=1), g=[0, 1]).expand(),
)
EXPECTED_NAMED_PRODUCT = [
    Struct(a=0, b=0, c=1, d=(0, 1), e=Struct(f=1, g=0)),
    Struct(a=0, b=1, c=1, d=(0, 1), e=Struct(f=1, g=0)),
    Struct(a=0, b=2, c=1, d=(0, 1), e=Struct(f=1, g=0)),
    Struct(a=1, b=0, c=1, d=(0, 1), e=Struct(f=1, g=0)),
    Struct(a=1, b=1, c=1, d=(0, 1), e=Struct(f=1, g=0)),
    Struct(a=1, b=2, c=1, d=(0, 1), e=Struct(f=1, g=0)),
    Struct(a=0, b=0, c=1, d=(0, 1), e=Struct(f=1, g=1)),
    Struct(a=0, b=1, c=1, d=(0, 1), e=Struct(f=1, g=1)),
    Struct(a=0, b=2, c=1, d=(0, 1), e=Struct(f=1, g=1)),
    Struct(a=1, b=0, c=1, d=(0, 1), e=Struct(f=1, g=1)),
    Struct(a=1, b=1, c=1, d=(0, 1), e=Struct(f=1, g=1)),
    Struct(a=1, b=2, c=1, d=(0, 1), e=Struct(f=1, g=1)),
]


def test_named_product():
    def _make_hashable(_items):
        return tuple(sorted(f"{key}, {value}" for key, value in _items))

    computed_named_product_sets = set(
        tuple(_make_hashable(param_set.items()))
        for param_set in named_product(**PARAMS)
    )
    expected_named_product_sets = set(
        tuple(_make_hashable(param_set.items())) for param_set in EXPECTED_NAMED_PRODUCT
    )
    assert computed_named_product_sets == expected_named_product_sets


@pytest.mark.parametrize(
    "iterable, expected_output",
    [[range(4), [(0, 1), (1, 2), (2, 3)]], ["abc", [("a", "b"), ("b", "c")]]],
)
def test_pairwise(iterable, expected_output):
    assert list(pairwise(iterable)) == expected_output

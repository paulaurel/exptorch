from typing import List
from itertools import tee, product
from collections.abc import Iterable

from exptorch.containers import Struct


def pairwise(iterable) -> Iterable:
    """Iterate over all pairs of consecutive items in a list.

    Notes
    -----
        [s0, s1, s2, s3, ...] -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def named_product(**kwargs) -> Iterable:
    """Iterate over the cartesian product between all input kwargs.values().
    Assign a name to each value within the cartesian product.
    The name corresponds to the key associated with the value.

    Notes
    -----
        named_product(a=A, b=B) returns the same as: (Struct(a=x, b=y) for x in A for y in B).

    Examples
    --------
        named_product(a=1, b=[2, 3]) --> Struct(a=1, b=2), Struct(a=1, b=3)
    """

    def _ensure_iterable_values(values: Iterable) -> List[Iterable]:
        """Ensure that all elements within values are iterable."""
        return [value if isinstance(value, Iterable) else [value] for value in values]

    for config in product(*_ensure_iterable_values(kwargs.values())):
        yield Struct(zip(kwargs.keys(), config))

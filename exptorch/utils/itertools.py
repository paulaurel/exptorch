from itertools import tee, product

from exptorch.containers import Struct


def pairwise(iterable):
    """Iterate over all pairs of consecutive items in a list.

    Notes
    -----
        [s0, s1, s2, s3, ...] -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def named_product(**kwargs):
    keys = kwargs.keys()
    for config in product(*kwargs.values()):
        yield Struct(zip(keys, config))

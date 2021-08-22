from itertools import product
from copy import deepcopy

from .utils.validation import _validate_type


class Struct(dict):
    """Dictionary like container object exposing keys as attributes.

    The Struct container enables values to be accessed both via __getitem__,
    i.e. by key, and via __getattr__, i.e. by attribute.

    Examples
    --------
    >>> struct = Struct(a=1, b=2)
    >>> struct["a"] == struct.a
    True
    >>> struct["b"]
    2
    >>> struct.b
    2
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class Params(Struct):
    __fixed_name = "fixed"

    def __init__(self, fixed=None, **kwargs):
        fixed = fixed if fixed is not None else Struct()
        _validate_type(fixed, required_type=Struct, obj_name=self.__fixed_name)
        self.fixed = fixed
        self._validate_kwargs(kwargs)
        super().__init__(**kwargs)

    @property
    def free(self):
        return Struct(
            {key: self[key] for key in self.keys() if key != self.__fixed_name}
        )

    def _validate_kwargs(self, kwargs):
        self._validate_kwargs_type(kwargs)

    @staticmethod
    def _validate_kwargs_type(kwargs):
        for key, value in kwargs.items():
            _validate_type(value, required_type=list, obj_name=key)

    def expand(self):
        _fixed_param_set = deepcopy(self.fixed)
        for _free_param_set in product(*self.free.values()):
            _fixed_param_set.update(zip(self.free.keys(), _free_param_set))
            yield Struct(_fixed_param_set)

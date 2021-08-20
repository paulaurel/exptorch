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

    __base_label = "base"

    def __init__(self, base=None, **kwargs):
        base = base if base is not None else Struct()
        _validate_type(base, required_type=Struct, obj_name=self.__base_label)
        self.base = base
        self._validate_kwargs(kwargs)
        super().__init__(**kwargs)

    @property
    def free(self):
        return Struct({key: self[key] for key in self.keys() if key != self.__base_label})

    def _validate_kwargs(self, kwargs):
        self._validate_kwargs_type(kwargs)

    @staticmethod
    def _validate_kwargs_type(kwargs):
        for key, value in kwargs.items():
            _validate_type(value, required_type=list, obj_name=key)

    def flatten(self):
        _base = deepcopy(self.base)
        _free_param_keys = self.free.keys()
        for _updated_params in product(*self.free.values()):
            _base.update(zip(_free_param_keys, _updated_params))
            yield Struct(_base)

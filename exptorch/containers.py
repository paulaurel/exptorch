from typing import Generator
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
    """Container object storing fixed and free parameters."""

    __fixed_key_label = "fixed"

    def __init__(self, fixed=None, **free_params):
        fixed = fixed if fixed is not None else Struct()
        _validate_type(fixed, required_type=Struct, obj_name=self.__fixed_key_label)
        self.fixed = fixed

        self._validate_free_params(free_params)
        super().__init__(**free_params)

    @property
    def free(self) -> Struct:
        return Struct(
            {key: self[key] for key in self.keys() if key != self.__fixed_key_label}
        )

    def _validate_free_params(self, free_params):
        self._validate_free_params_type(free_params)
        self._validate_free_params_keys(free_params)

    @staticmethod
    def _validate_free_params_type(free_params):
        for key, value in free_params.items():
            _validate_type(value, required_type=list, obj_name=key)

    def _validate_free_params_keys(self, free_params):
        _intersecting_keys = free_params.keys() & self.fixed.keys()
        if _intersecting_keys:
            raise ValueError(
                "Requires free parameter keys to not intersect with fixed parameter keys."
                f" Intersecting free parameter keys: {_intersecting_keys}"
            )

    def is_empty(self) -> bool:
        _empty_fixed = len(self.fixed) == 0
        _empty_free = len(self.free) == 0
        return _empty_fixed and _empty_free

    def expand(self) -> Generator:
        """Expands the free parameters to define all valid parameter combinations.
        Appends the cartesian product of the free parameters to the fixed parameters.

        Returns
        -------
        Generator[Struct]
            Returns generator containing all valid parameter combinations.
        """
        if self.is_empty():
            return

        _fixed_param_set = deepcopy(self.fixed)
        for _free_param_set in product(*self.free.values()):
            _fixed_param_set.update(zip(self.free.keys(), _free_param_set))
            yield Struct(_fixed_param_set)

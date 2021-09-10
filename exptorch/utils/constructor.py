import inspect
from functools import partial

from ..containers import Struct


def construct(obj, **kwargs):
    """Construct object from provided keyword arguments.

    Raises
    ------
        Raises TypeError if the object cannot be constructed
        from the provided keyword arguments.
    """
    if inspect.isclass(obj):
        return obj(**kwargs)
    elif inspect.isfunction(obj):
        return partial(obj, **kwargs)
    elif isinstance(obj, (list, tuple)):
        return [construct(item, **kwargs) for item in obj]
    elif isinstance(obj, dict):
        return Struct(
            (key, construct(value, **kwargs.get(key, {}))) for key, value in obj.items()
        )
    else:
        if kwargs:
            error_msg = (
                f"The object, {obj}, could not be constructed from the following kwargs:"
                f" {', '.join(f'{key}={value}' for key, value in kwargs.items())}."
            )
        else:
            error_msg = f"The object, {obj}, could not be constructed."
        raise TypeError(error_msg)

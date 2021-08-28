from inspect import signature
from typing import Union, Tuple, Type, Callable


def validate_type(obj, *, required_type: Union[Type, Tuple], obj_name: str):
    """Validate that the object's type corresponds to specified required type.

    Parameters
    ----------
    obj: object
        Validated object.
    required_type: Union[Type, Tuple]
        Type or tuple of types specifying the object's required type(s).
    obj_name: str
        String specifying the object's name.

    Raises
    ------
        Raises TypeError if the object does not correspond to the required type.
    """
    required_type = (
        required_type if isinstance(required_type, tuple) else (required_type,)
    )

    if not isinstance(obj, required_type):
        required_type_name = tuple(_type.__name__ for _type in required_type)
        error_msg = (
            f"Require {obj_name} to be of type: {required_type_name}."
            f" Given object has type {type(obj).__name__}."
        )
        raise TypeError(error_msg)


def has_arg(func: Callable, arg_name: str) -> bool:
    return arg_name in signature(func).parameters


def validate_arg(func: Callable, arg_name: str):
    if not has_arg(func, arg_name):
        raise ValueError(
            f"Require that the function '{func.__name__}'"
            f" has the argument '{arg_name}'. This argument is missing."
            f" Arguments provided: {signature(func).parameters}"
        )


def validate_value(value, *, allowed_values):
    if value not in allowed_values:
        raise ValueError(
            "Expected "
        )
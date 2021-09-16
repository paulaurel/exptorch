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
        Raises TypeError if the provided required_type is not a type or a tuple of types.
        Raises TypeError if the object does not correspond to the required type.
    """
    required_type = (
        required_type if isinstance(required_type, tuple) else (required_type,)
    )
    if not all(isinstance(_type, type) for _type in required_type):
        raise TypeError(
            f"{validate_type.__name__} requires that second argument, required_type, is a type"
            " or a tuple of types."
        )

    if not isinstance(obj, required_type):
        required_type_name = tuple(_type.__name__ for _type in required_type)
        raise TypeError(
            f"Require {obj_name} to be of type: {', '.join(required_type_name)}."
            f" Given object has type {type(obj).__name__}."
        )


def validate_value(value, *, allowed_value, value_name):
    """Validate that given value corresponds to the allowed_value.

    Notes
    -----
    A tuple, as in ``validate_value(x, allowed_value=(A, B, ...), value_name="x")``,
    may be given as the target to check against. This is equivalent to
    ``validate_value(x, allowed_value=A, value_name="x")
    or validate_value(x, allowed_value=B, value_name="x") or ...``

    Raises
    ------
    Raises ValueError if the value does not correspond to the allowed_value.
    """
    allowed_value = (
        allowed_value if isinstance(allowed_value, tuple) else (allowed_value,)
    )

    if value not in allowed_value:
        raise ValueError(
            f"Require that {value_name} corresponds to"
            f" one of the following values: {', '.join(map(str, allowed_value))}."
            f" Instead {value_name} has the value: {value}."
        )


def has_arg(func: Callable, arg_name: str) -> bool:
    return arg_name in signature(func).parameters


def validate_arg(func: Callable, arg_name: str):
    if not has_arg(func, arg_name):
        raise ValueError(
            f"Require that the function '{func.__name__}'"
            f" has the argument '{arg_name}'. This argument is missing."
            f" Arguments provided: {signature(func).parameters}"
        )
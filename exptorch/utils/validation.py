from typing import Union, Tuple, Type


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

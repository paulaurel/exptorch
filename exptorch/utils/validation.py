def _validate_type(obj, *, required_type, obj_name):
    if not isinstance(obj, required_type):
        raise TypeError()

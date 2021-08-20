class Struct(dict):
    """Container object exposing keys as attributes.

    The Struct enables values to be accessed both via __getitem__,
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

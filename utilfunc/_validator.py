from typing import Tuple

def _figsize_validator(figsize: Tuple[int or float]):
    assert isinstance(figsize, tuple), \
        f"dtype of figsize expected tuple, got {type(figsize)}"
    assert len(figsize) == 2, \
        f"length of figsize expected 2, got {len(figsize)}"
    for i, v in enumerate(figsize):
        assert isinstance(v, int) or isinstance(v, float), \
            f"figsize expected tuple of int or float, got {type(v)} in figsize[{i}]"
    return None

def _bool_validator(arg: bool):
    assert isinstance(arg, bool), \
        f"dtype of arg expected bool, got {type(arg)}"
    return None

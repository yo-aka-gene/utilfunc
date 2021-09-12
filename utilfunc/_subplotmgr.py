import pandas as pd

from numpy import ceil, floor, sqrt
from typing import Tuple, List
from ._validator import _figsize_validator as fv
from ._validator import _bool_validator as bv

def subplotmgr(
    df: pd.core.frame.DataFrame,
    figsize: Tuple[int or float],
    landscape: bool = False):
    assert isinstance(df, pd.core.frame.DataFrame), \
        f"df expected pandas.core.frame.DataFrame, got {type(df)}"

    fv(figsize)
    bv(landscape)

    n_v = int(floor(sqrt(df.shape[1]))) if landscape \
        else int(ceil(df.shape[1]/floor(sqrt(df.shape[1]))))
    n_h = int(ceil(df.shape[1]/floor(sqrt(df.shape[1])))) if landscape\
        else int(floor(sqrt(df.shape[1])))

    ret = {
        "nrows": n_v,
        "ncols": n_h,
        "figsize": (figsize[0] * n_h, figsize[1] * n_v)
    }

    return ret

def list2subplot(
    l_df: List[pd.core.frame.DataFrame],
    figsize: Tuple[int or float],
    landscape: bool = False):
    assert isinstance(l_df, list), \
        f"df expected list, got {type(l_df)}"
    for i, v in enumerate(l_df):
        assert isinstance(v, pd.core.frame.DataFrame), \
            f"l_df expected list of pandas.core.frame.DataFrame, got {type(v)} in l_df[{i}]"

    fv(figsize)
    bv(landscape)

    n_v = int(floor(sqrt(len(l_df)))) if landscape \
        else int(ceil(len(l_df)/floor(sqrt(len(l_df)))))
    n_h = int(ceil(len(l_df)/floor(sqrt(len(l_df))))) if landscape \
        else int(floor(sqrt(len(l_df))))
    
    ret = {
        "nrows": n_v,
        "ncols": n_h,
        "figsize": (figsize[0] * n_h, figsize[1] * n_v)
    }

    return ret

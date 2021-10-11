import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import norm
from ._subplotmgr import subplotmgr as spm

def qqplot(
    df: pd.core.frame.DataFrame,
    ax: plt.axes = None,
    figsize: tuple = (5, 5),
    wspace: float = 0.3,
    hspace: float = 0.3,
    landscape: bool = False
    ):
    assert isinstance(df, pd.core.frame.DataFrame), \
        f"df expected pandas.core.frame.DataFrame, got {type(df)}"
    
    quantile_rate = np.arange(0.5, df.shape[0]+0.5, 1)/df.shape[0]
    quantile_val = np.array(
        [
            [
                df.iloc[:, j].mean() + (df.iloc[:, j].std(ddof=1) * norm.ppf(i)) for i in quantile_rate
            ] for j in range(df.shape[1])
        ]
    )

    data_qq = [
        pd.DataFrame(
            df.iloc[:, i].sort_values(ascending=True)
            ).assign(quantile_val = quantile_val[i]) for i in range(df.shape[1])
            ]
    if ax is None:
        fig, ax = plt.subplots(**spm(df, figsize, landscape))
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    ncols = spm(df, figsize)["ncols"]

    for i, v in enumerate(data_qq):
        ax[i//ncols, i%ncols].plot(
            np.linspace(v.min().min(), v.max().max(), 1000),
            np.linspace(v.min().min(), v.max().max(), 1000),
            c="k",
            linestyle="dashed"
        )
        v.plot.scatter(
            0,
            1,
            ax=ax[i//ncols, i%ncols],
            xlabel="Theoretical Quantiles",
            ylabel="Sample Quantiles",
            title=f"{v.iloc[:, 0].name}"
        )

    return ax

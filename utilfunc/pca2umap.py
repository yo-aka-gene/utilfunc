#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

from scipy.sparse.csgraph import connected_components
from ._subplotmgr import list2subplot
from ._defacto_val import _umap_val
from typing import List, Union, Tuple


# In[2]:


def pre_pca(data:pd.core.frame.DataFrame,
            random_state:int = 0,
            req_cont:int = 80) -> pd.core.frame.DataFrame:
    
    model = PCA(random_state=random_state)
    
    _df = pd.DataFrame(
        model.fit_transform(data),
        index = data.index,
        columns = [f"PC{i+1}" for i in range(min(data.shape))]
    )
    
    cont = pd.DataFrame(
        {
            "cont. [%]": model.explained_variance_ratio_*100,
            "cum. cont. [%]": model.explained_variance_ratio_.cumsum()*100
        },
        index = _df.columns
    )
    
    accepted_dim = cont[cont.iloc[:, 1]<req_cont].shape[0]+1
    
    df_pca = _df.iloc[:, :accepted_dim]
    
    return df_pca


# In[3]:


def umap_sep(data:pd.core.frame.DataFrame,
             random_state:int = 0,
             n_components:int = 2,
             min_dist:List[float] = _umap_val().min_dist,
             metric:str = "euclidean",
             n_nbr:List[int] = _umap_val().n_nbr) -> List[pd.core.frame.DataFrame]:
    
    umap_arg = {
        'random_state': random_state,
        'n_components': n_components,
        'metric': metric
    }
    len_min_dist = len(min_dist)
    len_n_nbr = len(n_nbr)
    n_combi = len_min_dist * len_n_nbr
    
    model_umap = [umap.UMAP(
        min_dist = min_dist[i % len_min_dist],
        n_neighbors = n_nbr[i // len_min_dist],
        **umap_arg
    ) for i in range(n_combi)]
    
    df_umap = [model.fit_transform(data) for model in model_umap]
    
    df_umap_sep = [
        pd.DataFrame(
            v,
            index = data.index,
            columns = [
                f'UMAP{e+1} \
                    (min_dist={min_dist[i % len_min_dist]}, n_nbr={n_nbr[i // len_min_dist]})' for e in range(n_components)
                ]
        )
        for i, v in enumerate(df_umap)
    ]
    return df_umap_sep 


# In[4]:


def umap_viz(
    target: str or np.ndarray or pd.core.indexes.base.Index,
    data:list,
    cmap: str = "plasma",
    uni_cbar: bool = True,
    uni_scale: bool = True,
    data_scale: pd.core.frame.DataFrame = pd.DataFrame(),
    cax: list = [0.925, 0.15, 0.02, 0.1],
    alpha: int or float = 1,
    aspect: Union[str, Tuple[int]] = _umap_val().subplots,
    figsize: tuple = (3, 3),
    mini: bool = False,
    title: str = "",
    cbar_label: str = ""
):
    data_scale = data_scale if type(target)==str or type(target)==type(None) else target
    _kwarg = {
        "c": data_scale.loc[:, target] if type(target)==str else target,
        "cmap": cmap,
        "alpha": alpha,
        "vmin": data_scale.min().min() if uni_scale else data.min().min(),
        "vmax": data_scale.max().max() if uni_scale else data.max().max()
    }
    if mini:
        fig, ax = plt.subplots(figsize = figsize)
        data[0].plot.scatter(x=0, y=1, ax=ax,
                             title = title,
                             colorbar = False,
                             **_kwarg)
        if type(target) != type(None):
            cbar = plt.colorbar(
                    plt.cm.ScalarMappable(
                        plt.Normalize(_kwarg["vmin"], _kwarg["vmax"]),
                        cmap = cmap
                    ),
                    cax = plt.axes([0.925, 0.15, 0.05, 0.3]),
                    label = f"{target} $[\log_2(TPM+1)]$" if type(target)==str else cbar_label
                )
    else:
        if isinstance(aspect, str):
            assert aspect == "landscape" or aspect == "portrait", \
                f"aspect expected either of 'landscape', 'portrait', or tuple of designated values; got{aspect}"
            landscape = True if aspect == "landscape" else False
            ret = list2subplot(l_df=data, figsize=figsize, landscape=landscape)
            n_v, n_h = ret["n_rows"], ret["n_cols"]
        else:
            assert isinstance(aspect, tuple), \
                f"aspect expected either of 'landscape', 'portrait', or tuple of designated values; got{aspect}"
            n_v, n_h = aspect
        
        fig, ax = plt.subplots(
            n_v, n_h,
            figsize = (figsize[0]*n_h, figsize[1]*n_v)
        )
    
        for i, v in enumerate(data):
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            v.plot.scatter(x=0, y=1,
                           ax = ax[i//n_h, i%n_h],
                           colorbar = not uni_cbar,
                           **_kwarg)
        
            if uni_cbar and type(target) != type(None):
                cbar = plt.colorbar(
                    plt.cm.ScalarMappable(
                        plt.Normalize(_kwarg["vmin"], _kwarg["vmax"]),
                        cmap = cmap
                    ),
                    cax = plt.axes(cax),
                    label = f"{target} $[\log_2(TPM+1)]$" if type(target)==str else cbar_label
                )
        
    return None                      


# In[5]:


def umap_compare(
    target: list,
    ref: pd.core.frame.DataFrame,
    data:list,
    cmap: str = "plasma",
    cax: list = [0.925, 0.15, 0.02, 0.1],
    alpha: int or float = 0.5, 
    figsize: tuple = (3, 3),
    unit: str = "$\log_2(TPM+1)$"
):
    data_scale = [ref.loc[:, i] for i in target]
    _kwarg = {
        "cmap": cmap,
        "colorbar": False,
        "alpha": alpha,
        "vmin": ref.min().min(),
        "vmax": ref.max().max()
    }
    n_v, n_h = (
        int(np.ceil(len(target)/np.floor(np.sqrt(len(target))))),
        int(np.floor(np.sqrt(len(target))))
    )
        
    fig, ax = plt.subplots(
        n_v, n_h,
        figsize = (figsize[0]*n_h, figsize[1]*n_v)
    )
    
    for i, v in enumerate(data_scale):
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        data[0].plot.scatter(x=0, y=1,
                             ax = ax[i//n_h, i%n_h],
                             c = v,
                             title = target[i],
                             **_kwarg)
    
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(
                plt.Normalize(_kwarg["vmin"], _kwarg["vmax"]),
                cmap = cmap
            ),
            cax = plt.axes(cax),
            label = f"{unit}"
        )
        
    return None                     


# In[ ]:





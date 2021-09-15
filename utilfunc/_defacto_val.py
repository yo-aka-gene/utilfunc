from typing import NamedTuple, List, Tuple

class _umap_val(NamedTuple):
    min_dist: List[float] = [0, 0.01, 0.05, 0.1, 0.5, 1]
    n_nbr: List[int] = [5, 15, 30, 50, 100]
    subplots: Tuple[int] = (5, 6)

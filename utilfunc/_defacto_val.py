from typing import NamedTuple, List, Tuple

class _umap_val(NamedTuple):
    min_dist: List[float] = [0, 0.01, 0.05, 0.1, 0.5, 1]
    n_nbr: List[int] = [5, 15, 30, 50, 100]
    subplots: Tuple[int] = (5, 6)
    title: List[str] = [
        f"min_dist:{[0, 0.01, 0.05, 0.1, 0.5, 1][i % 6]}, n_nbr:{[5, 15, 30, 50, 100][i // 6]}" for i in range(30)
    ]

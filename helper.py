from typing import Iterable, Tuple

import numpy as np
import numpy.typing as npt


def indices_with_neighbor(arr: npt.NDArray, equal: np.generic, neighbor: np.generic) -> npt.NDArray[np.intp]:
    """
    Return an (N,2) array of (row, col) indices where arr == equal and at least one of the 8 neighbors equals neighbor.
    """
    mask_a = (arr == equal)
    mask_b = np.pad((arr == neighbor).astype(np.uint8), pad_width=1, mode='constant', constant_values=0)
    offsets: Iterable[Tuple[int,int]] = [(-1, -1), (-1, 0), (-1, 1),
                                         (0,  -1),          (0,  1),
                                         (1,  -1), (1,  0), (1,  1)]
    neighbor_count = np.zeros_like(arr, dtype=np.uint8)
    for dr, dc in offsets:
        neighbor_count += mask_b[1+dr : 1+dr+arr.shape[0], 1+dc : 1+dc+arr.shape[1]]

    return np.argwhere(mask_a & (neighbor_count > 0))

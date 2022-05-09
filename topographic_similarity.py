from typing import Sequence, Hashable, List
import editdistance
import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr
import itertools


def compute_topsim(
    messages: Sequence[Sequence[Hashable]],
    meanings: Sequence[Sequence[Hashable]],
    eos_id: int = 0,
):
    assert len(messages) > 0
    assert len(messages) == len(meanings)
    assert all(len(meanings[0]) == len(meanings[i]) for i in range(len(meanings)))
    messages = [
        (tuple(x)[:tuple(x).index(eos_id)] if eos_id in x else x) for x in messages
    ]
    msg_dist: List[int] = []
    mng_dist: List[int] = []
    for i in range(len(messages)):
        for j in range(i + 1, len(messages)):
            msg_dist.append(editdistance.eval(messages[i], messages[j]))
            mng_dist.append(distance.hamming(meanings[i], meanings[j]))
    topsim: float = spearmanr(msg_dist, mng_dist).correlation
    if np.isnan(topsim):
        topsim = 0
    return topsim


if __name__ == "__main__":
    colors = ["red", "blue", "green", "purple"]
    shapes = ["triangle", "circle", "square", "star"]
    dummy_data = list(itertools.product(colors, shapes))
    print("dummy_data:", dummy_data)
    print("segment-base TopSim:", compute_topsim(dummy_data, dummy_data))
    print("character-base TopSim:", compute_topsim([list("".join(x)) for x in dummy_data], dummy_data))

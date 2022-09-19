import itertools
from typing import Dict, List, Tuple, Sequence
from collections import defaultdict

import numpy as np

from topographic_similarity import compute_topsim
from boundary_detection import EntropyCalculator


def generate_random_synthetic_language(
    n_attributes: int,
    n_values: int,
    max_len: int = 36,
    vocab_size: int = 8,
    random_seed: int = 0,
):
    attval_to_segment: Dict[Tuple[int, int], Tuple[int, ...]] = {}
    assert max_len >= n_attributes
    random_state = np.random.RandomState(random_seed)
    segment_len = max_len // n_attributes
    for a in range(n_attributes):
        for v in range(n_values):
            segment = tuple(random_state.choice(vocab_size, segment_len))
            attval_to_segment[a, v] = segment
    inputs: List[Tuple[int, ...]] = []
    synthetic_messages: List[Tuple[int, ...]] = []
    for input in itertools.product(range(n_values), repeat=n_attributes):
        message = sum((attval_to_segment[a, v] for a, v in enumerate(input)), start=())
        synthetic_messages.append(message)
        inputs.append(input)
    return synthetic_messages, inputs


def is_monotonically_increasing(sequence: Sequence[float]):
    prev = sequence[0]
    for next in sequence[1:]:
        if next < prev:
            return False
        prev = next
    return True


def standard_error(sequence: Sequence[float]):
    return np.std(sequence, ddof=1) / np.sqrt(len(sequence))


def main():
    random_seeds = list(range(8))
    configurations = [
        (1, 4096),
        (2, 64),
        (3, 16),
        (4, 8),
        (6, 4),
        (12, 2),
    ]

    attval_config_to_list_mean_n_boundaries: defaultdict[Tuple[int, int], List[float]] = defaultdict(list)
    attval_config_to_list_vocab_size: defaultdict[Tuple[int, int], List[int]] = defaultdict(list)
    attval_config_to_list_cTopSim: defaultdict[Tuple[int, int], List[float]] = defaultdict(list)
    attval_config_to_list_wTopSim: defaultdict[Tuple[int, int], List[float]] = defaultdict(list)

    for n_attributes, n_values in configurations:
        for seed in random_seeds:
            print("(n_att,n_val)=({},{})".format(n_attributes, n_values))
            messages, inputs = generate_random_synthetic_language(
                n_attributes=n_attributes,
                n_values=n_values,
                random_seed=seed,
            )
            scheme = EntropyCalculator(messages, threshold=0.5)

            attval_config_to_list_mean_n_boundaries[n_attributes, n_values].append(scheme.mean_n_boundaries)
            print("(n_att,n_val)=({},{}): mean_n_boundaries={}".format(n_attributes, n_values, scheme.mean_n_boundaries))

            attval_config_to_list_vocab_size[n_attributes, n_values].append(scheme.vocab_size)
            print("(n_att,n_val)=({},{}): vocab_size={}".format(n_attributes, n_values, scheme.vocab_size))

            if n_attributes > 1:  # Otherwise TopSim is not well-defined.
                c_topsim = compute_topsim(messages, inputs)
                w_topsim = compute_topsim(scheme.hashed_segments, inputs)
                attval_config_to_list_cTopSim[n_attributes, n_values].append(c_topsim)
                attval_config_to_list_wTopSim[n_attributes, n_values].append(w_topsim)
                print("(n_att,n_val)=({},{}): C-TopSim={}, W-TopSim={}".format(n_attributes, n_values, c_topsim, w_topsim))

    for n_attributes, n_values in configurations:
        print("(n_att,n_val)=({},{}):".format(n_attributes, n_values))
        print("\t# boundaries: mean={}, SEM={}".format(
            np.mean(attval_config_to_list_mean_n_boundaries[n_attributes, n_values]),
            standard_error(attval_config_to_list_mean_n_boundaries[n_attributes, n_values]),
        ))
        print("\tvocabulary size: mean={}, SEM={}".format(
            np.mean(attval_config_to_list_vocab_size[n_attributes, n_values]),
            standard_error(attval_config_to_list_vocab_size[n_attributes, n_values]),
        ))
        print("\tC-TopSim: mean={}, SEM={}".format(
            np.mean(attval_config_to_list_cTopSim[n_attributes, n_values]),
            standard_error(attval_config_to_list_cTopSim[n_attributes, n_values]),
        ))
        print("\tW-TopSim: mean={}, SEM={}".format(
            np.mean(attval_config_to_list_wTopSim[n_attributes, n_values]),
            standard_error(attval_config_to_list_wTopSim[n_attributes, n_values]),
        ))


if __name__ == '__main__':
    main()

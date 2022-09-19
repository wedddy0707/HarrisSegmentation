from typing import (
    Callable,
    List,
    Tuple,
)
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot  # type: ignore

# Type Hint of one_hot is not given in advance.
one_hot: Callable[..., torch.LongTensor]


def one_hotify(
    data: List[Tuple[int, ...]],
    n_values: int,
) -> List[torch.Tensor]:
    return [
        one_hot(
            torch.as_tensor(d, dtype=torch.long),
            num_classes=n_values,
        )
        .view(-1)
        .float()
        for d in data
    ]


class ScaledDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    examples: List[torch.Tensor]
    scaling_factor: float

    @classmethod
    def create_train_test_data(
        cls,
        n_attributes: int,
        n_values: int,
        p_hold_out: float,
        *,
        n_guessable_attributes: int = 0,
        scaling_factor: float = 1,
        random_seed: int = 0,
    ):
        assert n_attributes >= 0, n_attributes
        assert n_values > 0, n_values
        assert 0 <= n_guessable_attributes <= n_attributes, (n_attributes, n_guessable_attributes)
        assert p_hold_out >= 0

        all_list = list(itertools.product(range(n_values), repeat=n_attributes))

        random_state = np.random.RandomState(seed=random_seed)
        permutation = random_state.permutation(len(all_list))  # type: ignore
        test_size = int(p_hold_out * len(all_list))

        trn_list = [all_list[i] for i in permutation[test_size:]]
        tst_list = [all_list[i] for i in permutation[:test_size]]
        assert len(trn_list) + len(tst_list) == len(all_list)

        trn_guessable_list = [e[:n_guessable_attributes] for e in trn_list]
        tst_guessable_list = [e[:n_guessable_attributes] for e in tst_list]
        all_guessable_list = [e[:n_guessable_attributes] for e in all_list]

        trn_dataset = cls(
            one_hotify(trn_list, n_values),
            one_hotify(trn_guessable_list, n_values),
            scaling_factor=scaling_factor,
        )
        dev_dataset = cls(
            one_hotify(trn_list, n_values),
            one_hotify(trn_guessable_list, n_values),
            scaling_factor=1,
        )
        tst_dataset = cls(
            one_hotify(tst_list, n_values),
            one_hotify(tst_guessable_list, n_values),
            scaling_factor=1,
        )
        all_dataset = cls(
            one_hotify(all_list, n_values),
            one_hotify(all_guessable_list, n_values),
            scaling_factor=1,
        )
        return trn_dataset, dev_dataset, tst_dataset, all_dataset

    def __init__(
        self,
        examples: List[torch.Tensor],
        guessable_examples: List[torch.Tensor],
        scaling_factor: float = 1,
    ) -> None:
        assert len(examples) == len(guessable_examples)
        self.examples = examples
        self.guessable_examples = guessable_examples
        self.scaling_factor = scaling_factor

    def __len__(self) -> int:
        return int(len(self.examples) * self.scaling_factor)

    def __getitem__(self, k: int):
        k = k % len(self.examples)
        return self.examples[k], torch.zeros(1), self.guessable_examples[k]

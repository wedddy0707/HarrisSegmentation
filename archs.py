from typing import (
    Tuple,
    Optional,
)
import torch
import torch.nn as nn


class Receiver(nn.Module):
    def __init__(
        self,
        n_outputs: int,
        n_hidden: int,
        n_inputs: Optional[int] = None,
    ):
        super().__init__()
        self.fc1 = nn.Linear(n_hidden, n_outputs)
        if n_inputs is None or n_inputs == 0:
            self.fc2 = None
        else:
            self.fc2 = nn.Linear(n_inputs, n_outputs)

    def forward(
        self,
        x: torch.Tensor,
        input: Optional[torch.Tensor],
        _aux_input: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if input is None or self.fc2 is None:
            return self.fc1(x)
        else:
            return self.fc1(x) + self.fc2(input)


class Sender(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
    ):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)

    def forward(
        self,
        x: torch.Tensor,
        _aux_input: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.fc1(x)
        return x


class Freezer(nn.Module):
    def __init__(self, wrapped: nn.Module):
        super().__init__()
        self.wrapped = wrapped
        self.wrapped.eval()

    def train(self, mode: bool):
        return self

    def forward(  # type: ignore
        self,
        *input: Tuple[Optional[torch.Tensor], ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            r = self.wrapped(*input)
        return r


class PlusOneWrapper(torch.nn.Module):
    def __init__(self, wrapped: torch.nn.Module):
        super().__init__()
        self.wrapped = wrapped

    def forward(
        self,
        *input: Tuple[Optional[torch.Tensor], ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r1, r2, r3 = self.wrapped(*input)
        r1 = r1[:, :-1]
        return r1 + 1, r2, r3

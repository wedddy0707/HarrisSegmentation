from typing import (
    Any,
)
import torch.nn as nn
from egg.core import Callback
from egg.core import Interaction


def reset_module_parameters(m: nn.Module):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        reset_parameters()
    for child in m.children():
        reset_module_parameters(child)


class PeriodicAgentResetter(Callback):
    def __init__(
        self,
        sender_life_span: int,
        receiver_life_span: int,
        sender_phase: int = 0,
        receiver_phase: int = 0,
    ) -> None:
        super().__init__()
        self.sender_life_span = sender_life_span
        self.receiver_life_span = receiver_life_span
        self.sender_phase = sender_phase
        self.receiver_phase = receiver_phase

    def on_train_begin(self, trainer_instance: Any) -> None:
        self.trainer = trainer_instance

    def on_epoch_begin(self, epoch: int) -> None:
        # Assume that epoch begins with 1, not 0.
        if self.sender_life_span > 0 and (epoch + self.sender_phase - 1) % self.sender_life_span == 0:
            reset_module_parameters(self.trainer.game.sender)
        if self.receiver_life_span > 0 and (epoch + self.receiver_phase - 1) % self.receiver_life_span == 0:
            reset_module_parameters(self.trainer.game.receiver)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        pass

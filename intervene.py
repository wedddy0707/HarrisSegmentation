import json
import os
from typing import Sequence, Any, List, Literal, Dict, Optional, Tuple, Union
import editdistance
import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.utils.clip_grad import clip_grad_norm_

from egg import core
from data import ScaledDataset
from archs import Sender, Receiver, DiffLoss, UIDGame


_ACC = "acc"
_LOSS = "loss"
_INPUT = "input"
_SEMANTICS = "semantics"
_MESSAGE = "message"
_OUTPUT = "output"
_SPLIT = "split"


class AskSender(core.Callback):
    split_to_dataset: Dict[Literal["train", "dev", "test"], ScaledDataset]
    device: torch.device
    freq: Optional[int]

    def __init__(
        self,
        split_to_dataset: Dict[Literal["train", "dev", "test"], ScaledDataset],
        device: torch.device,
        n_attributes: int,
        n_values: int,
        freq: Optional[int] = 1,
    ):
        self.split_to_dataset = split_to_dataset
        self.device = device
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.freq = None if freq == 0 else freq

    def on_train_begin(self, trainer_instance: core.Trainer):
        self.trainer = trainer_instance
        self.epoch_counter = self.trainer.start_epoch

    def on_train_end(self):
        self.dump()

    def on_epoch_end(self, *stuff: Any):
        self.epoch_counter += 1
        if self.freq is not None and self.epoch_counter % self.freq == 0:
            self.dump()

    def cut_eos(self, x: Sequence[int]):
        return x[:(x.index(0) if 0 in x else None)]

    def ask_sender(self):
        data: defaultdict[str, List[Any]] = defaultdict(list)

        game: Union[core.SenderReceiverRnnReinforce, UIDGame] = self.trainer.game
        sender: Sender = game.sender
        receiver: Receiver = game.receiver
        loss_fn: DiffLoss = game.loss

        game.eval()
        with torch.no_grad():
            for split, dataset in self.split_to_dataset.items():
                if len(dataset) == 0:
                    continue
                batch_size = len(dataset)
                loader = DataLoader(dataset, batch_size=batch_size)
                for input_s, _, _ in loader:
                    input_s: torch.Tensor = input_s.to(self.device)
                    output_s = sender.forward(input_s)[0]
                    output_r = receiver.forward(output_s)[0]
                    loss, rest = loss_fn.forward(
                        input_s, output_s, None, output_r, None
                    )
                    acc: torch.Tensor = rest["acc"]
                    data[_ACC].extend(acc.tolist())
                    data[_LOSS].extend(loss.tolist())
                    data[_INPUT].extend(input_s.view(batch_size, self.n_attributes, self.n_values).argmax(dim=-1).tolist())
                    data[_MESSAGE].extend(output_s.tolist())
                    data[_OUTPUT].extend(output_r.tolist())
                    data[_SPLIT].extend([split] * input_s.shape[0])
        game.train()
        return dict(data)

    def dump(self) -> None:
        raise NotImplementedError


class Metrics(AskSender):
    def dump(self):
        data: pd.DataFrame = pd.DataFrame(self.ask_sender())
        data: pd.DataFrame = data[data[_SPLIT] == "train"]
        msg: List[List[int]] = data[_MESSAGE].tolist()
        ipt: List[List[int]] = data[_INPUT].tolist()
        msg_dist: List[int] = []
        ipt_dist: List[int] = []
        for i in range(len(msg)):
            for j in range(i + 1, len(msg)):
                msg_dist.append(editdistance.eval(self.cut_eos(msg[i]), self.cut_eos(msg[j])))
                ipt_dist.append(editdistance.eval(self.cut_eos(ipt[i]), self.cut_eos(ipt[j])))
        topsim: float = spearmanr(msg_dist, ipt_dist).correlation
        output = {
            "mode": "metric",
            "epoch": self.epoch_counter,
            "topsim": topsim,
        }
        print(json.dumps(output, default=repr), flush=True)


class DumpCorpus(AskSender):
    def on_train_begin(self, trainer_instance: Any):
        super().on_train_begin(trainer_instance)
        self.dump(mode="dataset")

    def dump(
        self,
        mode: Literal["dataset", "language"] = "language",
    ):
        assert mode in {"dataset", "language"}
        data = self.ask_sender()
        if mode == "dataset":
            output = {
                "mode": mode,
                "data": {
                    k: v for k, v in data.items()
                    if k in {_INPUT, _SEMANTICS, _SPLIT}
                },
            }
        else:
            output = {
                "mode": mode,
                "epoch": self.epoch_counter,
                "data": {
                    k: v for k, v in data.items()
                    if k in {_MESSAGE, _OUTPUT, _ACC, _LOSS}
                },
            }
        print(json.dumps(output, default=repr).replace(" ", ""), flush=True)


class AgentResetter(core.Callback):
    epoch: int
    sender: Sender
    receiver: Receiver
    baselines: "defaultdict[str, Any]"

    def on_train_begin(self, trainer_instance: core.Trainer) -> None:
        self.trainer = trainer_instance
        self.epoch = self.trainer.start_epoch

        self.game: Union[core.SenderReceiverRnnReinforce, UIDGame] = self.trainer.game
        assert isinstance(self.game, (core.SenderReceiverRnnReinforce, UIDGame))

        self.sender: Sender = self.game.sender
        self.receiver: Receiver = self.game.receiver
        self.baselines: "defaultdict[str, Any]" = self.game.mechanics.baselines

    def reset_module_parameters(self, m: nn.Module):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()
        for child in m.children():
            self.reset_module_parameters(child)

    def reset_baseline(self):
        for k in self.baselines.keys():
            self.baselines[k] = self.baselines[k].__class__()


class PeriodicAgentResetter(AgentResetter):
    def __init__(
        self,
        sender_life_span: Optional[int],
        receiver_life_span: Optional[int],
    ) -> None:
        super().__init__()
        if sender_life_span == 0:
            sender_life_span = None
        if receiver_life_span == 0:
            receiver_life_span = None
        self.sender_life_span = sender_life_span
        self.receiver_life_span = receiver_life_span

    def on_epoch_begin(self, *_) -> None:
        self.epoch += 1
        if self.sender_life_span is not None and (self.epoch - 1) % self.sender_life_span == 0:
            self.reset_module_parameters(self.sender)
            self.reset_baseline()
        if self.receiver_life_span is not None and (self.epoch - 1) % self.receiver_life_span == 0:
            self.reset_module_parameters(self.receiver)
            self.reset_baseline()


class TransmissionDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    inputs: List[torch.Tensor]
    messages: List[torch.Tensor]

    def __init__(
        self,
        inputs: List[torch.Tensor],
        messages: List[torch.Tensor],
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.messages = messages

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.messages[index]

    def __len__(self):
        return len(self.inputs)


class Transmission(AgentResetter):
    def __init__(
        self,
        n_epochs_for_sender_training: int,
        n_epochs_for_receiver_training: int,
        lr_for_sender_training: float,
        lr_for_receiver_training: float,
        n_samples_per_input: int,
        device: torch.device,
        batch_size: int = 32,
        clip: float = 50.0,
        frequency: Optional[int] = None,
    ) -> None:
        super().__init__()

        if frequency == 0:
            frequency = None
        self.frequency = frequency

        self.n_epochs_for_sender_training = n_epochs_for_sender_training
        self.n_epochs_for_receiver_training = n_epochs_for_receiver_training
        self.lr_for_sender_training = lr_for_sender_training
        self.lr_for_receiver_training = lr_for_receiver_training
        self.batch_size = batch_size
        self.clip = clip

        self.n_samples_per_input = n_samples_per_input

        self.device = device

    def on_train_begin(self, trainer_instance: core.Trainer) -> None:
        super().on_train_begin(trainer_instance)
        self.train_data: DataLoader[Sequence[torch.Tensor]] = trainer_instance.train_data
        self.loss_fn: DiffLoss = self.game.loss
        self.sender_optimizer = Adam(self.sender.parameters(), lr=self.lr_for_sender_training)
        self.receiver_optimizer = Adam(self.receiver.parameters(), lr=self.lr_for_receiver_training)

    def on_epoch_begin(self, *_) -> None:
        self.epoch += 1
        if self.frequency is not None and (self.epoch - 1) % self.frequency == 0:
            loader = self.make_loader()
            self.reset_module_parameters(self.sender)
            self.reset_module_parameters(self.receiver)
            self.reset_baseline()
            self.train_sender(loader)
            self.train_receiver(loader)

    def make_loader(self):
        inputs: List[torch.Tensor] = []
        messages: List[torch.Tensor] = []
        self.sender.train()  # For probabilistic sampling
        with torch.no_grad():
            for batch in self.train_data:
                input: torch.Tensor = batch[0].to(self.device)
                for _ in range(self.n_samples_per_input):
                    message: torch.Tensor = self.sender.forward(input)[0]
                    inputs.append(input.detach().cpu())
                    messages.append(message.detach().cpu())

        num_workers = os.cpu_count()
        if num_workers is None:
            num_workers = 0

        return DataLoader(
            TransmissionDataset(inputs, messages),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def train_sender(
        self,
        loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
        reinforce: bool = False,
    ) -> None:
        self.sender.train()
        for _ in range(self.n_epochs_for_sender_training):
            for input, gold_message in loader:
                self.sender_optimizer.zero_grad()

                input: torch.Tensor = input.to(self.device)
                gold_message: torch.Tensor = gold_message.to(self.device)

                if reinforce:
                    raise NotImplementedError
                else:
                    _, loss = self.sender.forward_teacher_forcing(
                        input, gold_message,
                    )

                loss.mean().backward()  # type: ignore
                clip_grad_norm_(self.sender.parameters(), self.clip)
                self.sender_optimizer.step()

    def train_receiver(
        self,
        loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
        reinforce: bool = False,
    ) -> None:
        self.receiver.train()
        for _ in range(self.n_epochs_for_receiver_training):
            for input, gold_message in loader:
                self.receiver_optimizer.zero_grad()

                input: torch.Tensor = input.to(self.device)
                gold_message: torch.Tensor = gold_message.to(self.device)

                if reinforce:
                    raise NotImplementedError
                else:
                    output, _, _ = self.receiver.forward(gold_message)
                    loss, _ = self.loss_fn.forward(input, None, None, output, None)

                loss.mean().backward()  # type: ignore
                clip_grad_norm_(self.receiver.parameters(), self.clip)
                self.receiver_optimizer.step()

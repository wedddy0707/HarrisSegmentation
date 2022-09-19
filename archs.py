from typing import Tuple, Optional, Dict, Any, Literal, List, NamedTuple
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from egg.core.baselines import MeanBaseline
from egg.core.interaction import LoggingStrategy
from egg.core.rnn import RnnEncoder
from egg.core.util import find_lengths


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
        return r1 + 1, r2, r3


class DiffLoss(torch.nn.Module):
    def __init__(
        self,
        n_attributes: int,
        n_values: int,
    ):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

    def forward(
        self,
        sender_input: torch.Tensor,
        _message: Optional[torch.Tensor],
        _receiver_input: Optional[torch.Tensor],
        receiver_output: torch.Tensor,
        _labels: Optional[torch.Tensor],
        *args: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size = sender_input.size(0)
        sender_input = sender_input.view(batch_size, self.n_attributes, self.n_values)
        receiver_output = receiver_output.view(
            batch_size, self.n_attributes, self.n_values
        )

        acc = (
            torch.sum(
                (
                    receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)
                ).detach(),
                dim=1,
            ) == self.n_attributes
        ).float()
        acc_or = (
            receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)
        ).float()

        receiver_output = receiver_output.view(
            batch_size * self.n_attributes, self.n_values
        )
        labels = sender_input.argmax(dim=-1).view(batch_size * self.n_attributes)
        loss = (
            F.cross_entropy(receiver_output, labels, reduction="none")
            .view(batch_size, self.n_attributes)
            .mean(dim=-1)
        )

        return loss, {"acc": acc, "acc_or": acc_or}


class Sender(nn.Module):
    def __init__(
        self,
        n_features: int,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        max_len: int,
        num_layers: int = 1,
        cell: Literal["rnn", "gru", "lstm"] = "rnn",
        force_eos: bool = False,
    ):
        super().__init__()
        self.force_eos = force_eos

        self.max_len = max_len
        if force_eos:
            assert self.max_len > 1, "Cannot force eos when max_len is below 1"
            self.max_len -= 1

        self.encoder = nn.Linear(n_features, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_embedding = nn.parameter.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        cell_type = {
            'rnn': nn.RNNCell,
            'gru': nn.GRUCell,
            'lstm': nn.LSTMCell,
        }[cell.lower()]
        self.cells = nn.ModuleList(
            cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else
            cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.num_layers)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ):
        prev_h = [self.encoder.forward(x)]
        prev_h.extend([torch.zeros_like(prev_h[0]) for _ in range(self.num_layers - 1)])
        prev_c = [torch.zeros_like(prev_h[0]) for _ in range(self.num_layers)]  # only used for LSTM
        prev_e = self.sos_embedding.unsqueeze(0).expand(x.shape[0], -1)

        list_sequence: List[torch.Tensor] = []
        list_logits: List[torch.Tensor] = []
        list_entropy: List[torch.Tensor] = []

        h = c = torch.zeros_like(prev_h[0])

        for _ in range(self.max_len):
            for i, layer in enumerate(self.cells):
                if isinstance(layer, nn.LSTMCell):
                    h, c = layer.forward(prev_e, (prev_h[i], prev_c[i]))
                    prev_c[i] = c
                elif isinstance(layer, (nn.RNNCell, nn.GRUCell)):
                    h = layer.forward(prev_e, prev_h[i])
                else:
                    raise NotImplementedError
                prev_h[i] = h
                prev_e = h
            output = self.hidden_to_output.forward(h)
            distr = Categorical(logits=output)
            x = distr.sample() if self.training else output.argmax(dim=-1)
            prev_e = self.embedding(x)

            list_entropy.append(distr.entropy())
            list_logits.append(distr.log_prob(x))
            list_sequence.append(x)

        sequence = torch.stack(list_sequence).permute(1, 0)
        logits = torch.stack(list_logits).permute(1, 0)
        entropy = torch.stack(list_entropy).permute(1, 0)

        if self.force_eos:
            zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)
            sequence = torch.cat([sequence, zeros.long()], dim=1)
            logits = torch.cat([logits, zeros], dim=1)
            entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy

    def forward_teacher_forcing(
        self,
        x: torch.Tensor,
        gold_message: torch.Tensor,
    ):
        gold_message_len = find_lengths(gold_message)

        prev_h = [self.encoder.forward(x)]
        prev_h.extend([torch.zeros_like(prev_h[0]) for _ in range(self.num_layers - 1)])
        prev_c = [torch.zeros_like(prev_h[0]) for _ in range(self.num_layers)]  # only used for LSTM
        prev_e = self.sos_embedding.unsqueeze(0).expand(x.shape[0], -1)

        list_loss: List[torch.Tensor] = []
        list_logits: List[torch.Tensor] = []

        h = c = torch.zeros_like(prev_h[0])

        for step in range(gold_message.shape[1]):
            for i, layer in enumerate(self.cells):
                if isinstance(layer, nn.LSTMCell):
                    h, c = layer.forward(prev_e, (prev_h[i], prev_c[i]))
                    prev_c[i] = c
                elif isinstance(layer, (nn.RNNCell, nn.GRUCell)):
                    h = layer.forward(prev_e, prev_h[i])
                else:
                    raise NotImplementedError
                prev_h[i] = h
                prev_e = h
            x = gold_message[:, step]
            prev_e = self.embedding(x)

            gold_decoding_is_finished = gold_message_len <= step

            step_logits = self.hidden_to_output.forward(h)
            step_loss = \
                F.cross_entropy(input=step_logits, target=x, reduction="none") * \
                gold_decoding_is_finished
            list_logits.append(step_logits)
            list_loss.append(step_loss)

        logits = torch.stack(list_logits).permute(1, 0)
        loss = torch.stack(list_loss).permute(1, 0).sum(dim=-1)

        return logits, loss


class Receiver(nn.Module):
    def __init__(
        self,
        n_features: int,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        cell: Literal["rnn", "gru", "lstm"] = "rnn",
        num_layers: int = 1,
        n_inputs: int = 0,
    ):
        super().__init__()
        self.encoder = RnnEncoder(
            vocab_size,
            embed_dim,
            hidden_size,
            cell,
            num_layers,
        )
        self.decoder_1 = nn.Linear(
            hidden_size,
            n_features,
        )
        if n_inputs == 0:
            self.decoder_2 = None
        else:
            self.decoder_2 = nn.Linear(
                n_inputs,
                n_features,
            )

    def forward(
        self,
        message: torch.Tensor,
        input: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ):
        encoded = self.encoder.forward(message, lengths=lengths)
        output = self.decoder_1.forward(encoded)
        if self.decoder_2 is not None and input is not None:
            output = output + self.decoder_2.forward(input)

        logits = torch.zeros(message.shape[0]).to(message.device)
        entropy = logits

        return output, logits, entropy


class DummyMechanics(NamedTuple):
    baselines: "defaultdict[str, MeanBaseline]"


class UIDGame(nn.Module):
    def __init__(
        self,
        sender: Sender,
        receiver: Receiver,
        sender_entropy_coeff: float,
        n_attributes: int,
        n_values: int,
    ):
        super().__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.max_loss = torch.log(torch.as_tensor(n_values ** n_attributes)).item()
        self.loss = DiffLoss(n_attributes, n_values)
        self.baselines: "defaultdict[str, MeanBaseline]" = defaultdict(MeanBaseline)
        self.logging_strategy = LoggingStrategy()

    @property
    def mechanics(self):
        return DummyMechanics(self.baselines)

    def forward(
        self,
        sender_input: torch.Tensor,
        labels: torch.Tensor,
        receiver_input: Optional[torch.Tensor] = None,
        aux_input: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # Assume messages are fixed-length.
        message, log_prob_s, entropy_s = self.sender.forward(sender_input)

        max_len = message.shape[1]

        aux_info: Dict[str, Any] = {}
        total_loss = torch.as_tensor(0.0)
        receiver_output = torch.as_tensor(0.0)

        for i in range(max_len):
            expected_loss = self.max_loss * (max_len - i - 1) / max_len

            receiver_output, _, _ = self.receiver.forward(message[:, :i + 1])
            loss, aux_info = self.loss.forward(sender_input, None, None, receiver_output, None)
            total_loss = total_loss + torch.abs(loss - expected_loss)

        total_loss = total_loss / max_len

        optimized_loss = \
            + total_loss \
            + (total_loss.detach() - self.baselines["loss"].predict(total_loss.detach())) * log_prob_s.sum(dim=-1) \
            + entropy_s.mean(dim=-1) * self.sender_entropy_coeff

        if self.training:
            self.baselines["loss"].update(total_loss.detach())

        interaction = self.logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            aux_input=aux_input,
            message=message.detach(),
            receiver_output=receiver_output.detach(),
            message_length=None,
            aux=aux_info,
        )

        return optimized_loss.mean(), interaction

from typing import (
    Any,
    List,
    Dict,
    Union,
    Optional,
    Tuple,
    Literal,
)
from collections import defaultdict
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.core.batch import Batch
from egg.core.interaction import Interaction
from egg.core.baselines import (
    NoBaseline,
    MeanBaseline,
    BuiltInBaseline,
)
from egg.zoo.compo_vs_generalization.archs import (
    Receiver,
    Sender,
)
from egg.zoo.compo_vs_generalization.data import (
    ScaledDataset,
    enumerate_attribute_value,
    one_hotify,
    split_train_test,
)
from egg.zoo.compo_vs_generalization.intervention import Evaluator


def s2b(s: str):
    EXP_FOR_TRUE = {"true", "t", "yes", "y", "1"}
    EXP_FOR_FALSE = {"false", "f", "no", "n", "0"}
    s = s.lower()
    if s in EXP_FOR_TRUE:
        return True
    elif s in EXP_FOR_FALSE:
        return False
    else:
        raise ValueError(
            f"Unknown Literal {s}. "
            f"If you mean \"True\", then use one of {EXP_FOR_TRUE}. "
            f"Otherwise use one of {EXP_FOR_FALSE}."
        )

def get_params(params: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_attributes", type=int, default=4, help="")
    parser.add_argument("--n_values", type=int, default=4, help="")
    parser.add_argument("--data_scaler", type=int, default=100)
    parser.add_argument("--stats_freq", type=int, default=0)
    parser.add_argument(
        "--baseline", type=str, choices=["no", "mean", "builtin"], default="mean"
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=50,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=50,
        help="Size of the hidden layer of Receiver (default: 10)",
    )

    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-2,
        help="Entropy regularisation coeff for Sender (default: 1e-2)",
    )

    parser.add_argument("--sender_cell", type=str, default="rnn")
    parser.add_argument("--receiver_cell", type=str, default="rnn")
    parser.add_argument(
        "--sender_emb",
        type=int,
        default=10,
        help="Size of the embeddings of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_emb",
        type=int,
        default=10,
        help="Size of the embeddings of Receiver (default: 10)",
    )
    parser.add_argument(
        "--early_stopping_thr",
        type=float,
        default=0.99999,
        help="Early stopping threshold on accuracy (defautl: 0.99999)",
    )
    parser.add_argument(
        "--variable_length",
        type=s2b,
        default=False,
        help="",
    )

    args = core.init(arg_parser=parser, params=params)
    return args


class DiffLoss(torch.nn.Module):
    def __init__(
        self,
        n_attributes: int,
        n_values:     int,
    ):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values

    def forward(
        self,
        sender_input:    torch.Tensor,
        _message:        Optional[torch.Tensor],
        _receiver_input: Optional[torch.Tensor],
        receiver_output: torch.Tensor,
        _labels:         Optional[torch.Tensor],
        _aux_input:      Optional[Dict[str, torch.Tensor]],
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
            )
            == self.n_attributes
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


class DumpCorpus(core.Callback):
    def __init__(
        self,
        loaders_metrics: List[Tuple[str, DataLoader[Tuple[Optional[torch.Tensor], ...]], torch.nn.Module]],
        n_attributes: int,
        n_values: int,
        device: torch.device,
        freq: int = 1,
    ):
        self.loaders_metrics = loaders_metrics
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.device = device
        self.freq = freq

    def on_train_begin(self, trainer_instance: Any):
        self.trainer = trainer_instance
        self.epoch_counter: int = self.trainer.start_epoch
        self.__dump(mode='dataset')
        self.__dump(mode='language')

    def on_epoch_end(self, *stuff: Any):  # type: ignore
        self.epoch_counter += 1
        if self.freq <= 0 or self.epoch_counter % self.freq != 0:
            return
        self.__dump(mode='language')

    def __collect_data(self):
        data: "defaultdict[Literal['split', 'sender_input', 'message', 'acc'], List[Any]]" = defaultdict(list)
        game = self.trainer.game
        game.eval()
        old_loss = game.loss
        for split_name, loader, metric in self.loaders_metrics:
            game.loss = metric
            for batch in loader:
                if not isinstance(batch, Batch):
                    batch = Batch(*batch)
                batch = batch.to(self.device)
                with torch.no_grad():
                    interaction: Interaction = game(*batch)[1]
                assert (
                    interaction.sender_input is not None and
                    interaction.message is not None and
                    interaction.aux is not None and
                    'acc' in interaction.aux
                )
                batch_size = interaction.sender_input.size(0)
                data['split'].extend(
                    [split_name] * batch_size
                )
                data['sender_input'].extend(
                    interaction.sender_input
                    .view(batch_size, self.n_attributes, self.n_values)
                    .argmax(dim=-1)
                    .tolist()
                )
                data['message'].extend(
                    interaction.message.tolist()
                )
        game.loss = old_loss
        game.train()
        return data

    def __dump(self, mode: Literal['dataset', 'language']):
        assert mode in {'dataset', 'language'}, mode
        keys = ('sender_input', 'split') if mode == 'dataset' else ('message', 'acc')
        output: Dict[str, Union[str, int, Dict[str, Any]]] = dict(
            mode=mode,
            data={
                k: v for k, v in self.__collect_data().items() if k in keys
            },
        )
        if mode == 'language':
            output['epoch'] = self.epoch_counter
        print(json.dumps(output).replace(' ', ''), flush=True)


def main(params: List[str]):
    opts = get_params(params)
    assert not hasattr(opts, 'mode')
    print(json.dumps(dict(mode='config', **vars(opts)), default=repr))

    full_list = enumerate_attribute_value(
        opts.n_attributes, opts.n_values
    )
    train_list, test_list = split_train_test(full_list, 0)

    train_tensors, test_tensors, full_tensors = map(
        lambda x: one_hotify(x, opts.n_attributes, opts.n_values),
        (train_list, test_list, full_list),
    )
    train_dataset, validation_dataset, test_dataset, full_dataset = (
        ScaledDataset(train_tensors, opts.data_scaler),
        ScaledDataset(train_tensors, 1),
        ScaledDataset(test_tensors, 1),
        ScaledDataset(full_tensors, 1),
    )
    train_loader, validation_loader, test_loader, _ = (
        DataLoader(train_dataset,      batch_size=opts.batch_size, shuffle=True),
        DataLoader(validation_dataset, batch_size=opts.batch_size),
        DataLoader(test_dataset,       batch_size=opts.batch_size),
        DataLoader(full_dataset,       batch_size=opts.batch_size),
    )

    n_dim = opts.n_attributes * opts.n_values

    if opts.receiver_cell in ["lstm", "rnn", "gru"]:
        receiver = Receiver(n_hidden=opts.receiver_hidden, n_outputs=n_dim)
        receiver = core.RnnReceiverDeterministic(
            receiver,
            opts.vocab_size + 1,
            opts.receiver_emb,
            opts.receiver_hidden,
            cell=opts.receiver_cell,
        )
    else:
        raise ValueError(f"Unknown receiver cell, {opts.receiver_cell}")

    if opts.sender_cell in ["lstm", "rnn", "gru"]:
        sender = Sender(n_inputs=n_dim, n_hidden=opts.sender_hidden)
        sender = core.RnnSenderReinforce(
            agent=sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_emb,
            hidden_size=opts.sender_hidden,
            max_len=opts.max_len,
            cell=opts.sender_cell,
        )
    else:
        raise ValueError(f"Unknown sender cell, {opts.sender_cell}")

    if not opts.variable_length:
        sender = PlusOneWrapper(sender)
    loss = DiffLoss(opts.n_attributes, opts.n_values)

    baseline = {
        "no":      NoBaseline,
        "mean":    MeanBaseline,
        "builtin": BuiltInBaseline,
    }[opts.baseline]

    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0.0,
        length_cost=0.0,
        baseline_type=baseline,
    )
    optimizer = torch.optim.Adam(game.parameters(), lr=opts.lr)

    # holdout_evaluator = Evaluator(
    #     [
    #         (
    #             "test",
    #             test_loader,
    #             DiffLoss(opts.n_attributes, opts.n_values),
    #         ),
    #     ],
    #     opts.device,
    #     freq=0,
    # )
    dump_corpus = DumpCorpus(
        [
            (
                "validation",
                validation_loader,
                DiffLoss(opts.n_attributes, opts.n_values),
            ),
            (
                "test",
                test_loader,
                DiffLoss(opts.n_attributes, opts.n_values),
            ),
        ],
        opts.n_attributes,
        opts.n_values,
        device=opts.device,
        freq=1,
    )
    early_stopper = EarlyStopperAccuracy(
        opts.early_stopping_thr,
        validation=True
    )

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=[
            core.ConsoleLogger(as_json=True, print_train_loss=False),
            early_stopper,
            # holdout_evaluator,
            dump_corpus,
        ],
    )
    trainer.train(n_epochs=opts.n_epochs)

    print("---End--")

    core.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])

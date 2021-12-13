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
    split_holdout,
    split_train_test,
)
from egg.zoo.compo_vs_generalization.intervention import Evaluator, Metrics


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

    args = core.init(arg_parser=parser, params=params)
    return args


class DiffLoss(torch.nn.Module):
    def __init__(
        self,
        n_attributes: int,
        n_values:     int,
        generalization: bool = False,
    ):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.test_generalization = generalization

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
        device = sender_input.device
        sender_input = sender_input.view(batch_size, self.n_attributes, self.n_values)
        receiver_output = receiver_output.view(
            batch_size, self.n_attributes, self.n_values
        )

        if self.test_generalization:
            acc = torch.as_tensor(0, dtype=torch.float, device=device)
            acc_or = torch.as_tensor(0, dtype=torch.float, device=device)
            loss = torch.as_tensor(0, dtype=torch.float, device=device)

            for attr in range(self.n_attributes):
                zero_index = torch.nonzero(sender_input[:, attr, 0]).squeeze()
                masked_size = zero_index.size(0)
                masked_input = torch.index_select(sender_input, 0, zero_index)
                masked_output = torch.index_select(receiver_output, 0, zero_index)

                no_attribute_input = torch.cat(
                    [masked_input[:, :attr, :], masked_input[:, attr + 1:, :]], dim=1
                )
                no_attribute_output = torch.cat(
                    [masked_output[:, :attr, :], masked_output[:, attr + 1:, :]], dim=1
                )

                n_attributes = self.n_attributes - 1
                attr_acc = (
                    (
                        (
                            no_attribute_output.argmax(dim=-1)
                            == no_attribute_input.argmax(dim=-1)
                        ).sum(dim=1)
                        == n_attributes
                    )
                    .float()
                    .mean()
                )
                acc += attr_acc

                attr_acc_or = (
                    (
                        no_attribute_output.argmax(dim=-1)
                        == no_attribute_input.argmax(dim=-1)
                    )
                    .float()
                    .mean()
                )
                acc_or += attr_acc_or
                labels = no_attribute_input.argmax(dim=-1).view(
                    masked_size * n_attributes
                )
                predictions = no_attribute_output.view(
                    masked_size * n_attributes, self.n_values
                )
                # NB: THIS LOSS IS NOT SUITABLY SHAPED TO BE USED IN REINFORCE TRAINING!
                loss += F.cross_entropy(predictions, labels, reduction="mean")

            acc = acc / self.n_attributes
            acc_or = acc_or / self.n_attributes
        else:
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


class DumpCorpus(core.Callback):
    def __init__(
        self,
        loaders_metrics: List[Tuple[str, DataLoader[Tuple[Optional[torch.Tensor], ...]], torch.nn.Module]],
        device: torch.device,
        freq: int = 1,
    ):
        self.loaders_metrics = loaders_metrics
        self.device = device
        self.freq = freq

    def on_train_begin(self, trainer_instance: Any):
        self.trainer = trainer_instance
        self.epoch_counter: int = self.trainer.start_epoch
        self.__dump(mode='dataset')

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
                    interaction.sender_input.argmax(dim=-1).tolist()
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
        output: Dict[str, Union[str, Dict[str, Any]]] = dict(
            mode=mode,
            data={
                k: v for k, v in self.__collect_data().items() if k in keys
            },
        )
        print(json.dumps(output).replace(' ', ''), flush=True)


def main(params: List[str]):
    opts = get_params(params)
    assert not hasattr(opts, 'mode')
    print(json.dumps(dict(mode='config', **vars(opts)), default=repr))

    full_data = enumerate_attribute_value(
        opts.n_attributes, opts.n_values
    )
    train, generalization_holdout = split_holdout(full_data)
    train, uniform_holdout = split_train_test(train, 0.1)

    generalization_holdout, train, uniform_holdout, full_data = map(
        lambda x: one_hotify(x, opts.n_attributes, opts.n_values),
        (generalization_holdout, train, uniform_holdout, full_data),
    )
    train, validation = (
        ScaledDataset(train, opts.data_scaler),
        ScaledDataset(train, 1),
    )
    generalization_holdout, uniform_holdout, full_data = map(
        ScaledDataset,
        (generalization_holdout, uniform_holdout, full_data),
    )
    generalization_holdout_loader, uniform_holdout_loader, _ = map(
        lambda x: DataLoader(x, batch_size=opts.batch_size),
        (generalization_holdout, uniform_holdout, full_data),
    )
    train_loader = DataLoader(train, batch_size=opts.batch_size)
    validation_loader = DataLoader(validation, batch_size=len(validation))

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

    metrics_evaluator = Metrics(
        validation.examples,
        opts.device,
        opts.n_attributes,
        opts.n_values,
        opts.vocab_size + 1,
        freq=opts.stats_freq,
    )

    loaders = [
        (
            "generalization-holdout",
            generalization_holdout_loader,
            DiffLoss(opts.n_attributes, opts.n_values, generalization=True),
        ),
        (
            "uniform-holdout",
            uniform_holdout_loader,
            DiffLoss(opts.n_attributes, opts.n_values),
        ),
    ]

    holdout_evaluator = Evaluator(loaders, opts.device, freq=0)
    early_stopper = EarlyStopperAccuracy(opts.early_stopping_thr, validation=True)

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=[
            core.ConsoleLogger(as_json=True, print_train_loss=False),
            early_stopper,
            metrics_evaluator,
            holdout_evaluator,
            DumpCorpus(loaders, device=opts.device),
        ],
    )
    trainer.train(n_epochs=opts.n_epochs)

    print("---End--")

    core.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])

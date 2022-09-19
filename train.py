from typing import List, Literal
import argparse
import json
import torch
import sys
from torch.utils.data import DataLoader

import egg.core as core
from egg.core import EarlyStopperAccuracy
from egg.core.baselines import MeanBaseline

from data import ScaledDataset
from archs import Sender, Receiver, PlusOneWrapper, DiffLoss, UIDGame
from intervene import DumpCorpus, PeriodicAgentResetter


class MyNamespace:
    # Customized options
    n_attributes: int
    n_values: int
    data_scaler: int
    stats_freq: int
    baseline: Literal["no", "mean", "builtin"]
    sender_hidden: int
    receiver_hidden: int
    sender_entropy_coeff: float
    sender_cell: Literal["rnn", "gru", "lstm"]
    receiver_cell: Literal["rnn", "gru", "lstm"]
    sender_emb: int
    receiver_emb: int
    early_stopping_thr: float
    variable_length: bool
    sender_life_span: int
    receiver_life_span: int
    n_guessable_attributes: int
    run_uid_game: bool
    # Common options
    batch_size: int
    max_len: int
    vocab_size: int
    lr: float
    device: torch.device
    n_epochs: int

    def __init__(self) -> None:
        self.n_attributes = 4
        self.n_values = 4
        self.data_scaler = 100
        self.stats_freq = 0
        self.baseline = "mean"
        self.sender_hidden = 100
        self.receiver_hidden = 100
        self.sender_entropy_coeff = 1e-2
        self.sender_cell = "gru"
        self.receiver_cell = "gru"
        self.sender_emb = 10
        self.receiver_emb = 10
        self.early_stopping_thr = 0.99999
        self.variable_length = False
        self.sender_life_span = 0
        self.receiver_life_span = 0
        self.n_guessable_attributes = 0
        self.run_uid_game = False


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
    namespace = MyNamespace()

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_attributes", type=int, default=namespace.n_attributes)
    parser.add_argument("--n_values", type=int, default=namespace.n_values)
    parser.add_argument("--data_scaler", type=int, default=namespace.data_scaler)
    parser.add_argument("--stats_freq", type=int, default=namespace.stats_freq)
    parser.add_argument("--baseline", choices=["no", "mean", "builtin"], default=namespace.baseline)
    parser.add_argument("--sender_hidden", type=int, default=namespace.sender_hidden)
    parser.add_argument("--receiver_hidden", type=int, default=namespace.receiver_hidden)
    parser.add_argument("--sender_entropy_coeff", type=float, default=namespace.sender_entropy_coeff)
    parser.add_argument("--sender_cell", choices=["rnn", "gru", "lstm"], default=namespace.sender_cell)
    parser.add_argument("--receiver_cell", choices=["rnn", "gru", "lstm"], default=namespace.receiver_cell)
    parser.add_argument("--sender_emb", type=int, default=namespace.sender_emb)
    parser.add_argument("--receiver_emb", type=int, default=namespace.receiver_emb)
    parser.add_argument("--early_stopping_thr", type=float, default=namespace.early_stopping_thr)
    parser.add_argument("--variable_length", type=s2b, default=namespace.variable_length)
    parser.add_argument("--sender_life_span", type=int, default=namespace.sender_life_span)
    parser.add_argument("--receiver_life_span", type=int, default=namespace.receiver_life_span)
    parser.add_argument("--n_guessable_attributes", type=int, default=namespace.n_guessable_attributes)
    parser.add_argument("--run_uid_game", type=s2b, default=namespace.run_uid_game)

    args = core.init(arg_parser=parser, params=params)

    for k, v in vars(args).items():
        setattr(namespace, k, v)

    return namespace


def main(params: List[str]):
    opts = get_params(params)
    assert not hasattr(opts, 'mode')
    print(json.dumps(dict(mode='config', **vars(opts)), default=repr))

    trn_dataset, dev_dataset, tst_dataset, all_dataset = ScaledDataset.create_train_test_data(
        n_attributes=opts.n_attributes,
        n_values=opts.n_values,
        p_hold_out=0.0,
        n_guessable_attributes=opts.n_guessable_attributes,
        scaling_factor=opts.data_scaler,
    )
    trn_loader, dev_loader, _, _ = (
        DataLoader(trn_dataset, batch_size=opts.batch_size, shuffle=True),
        DataLoader(dev_dataset, batch_size=opts.batch_size),
        DataLoader(tst_dataset, batch_size=opts.batch_size),
        DataLoader(all_dataset, batch_size=opts.batch_size),
    )

    n_dim = opts.n_attributes * opts.n_values

    receiver = Receiver(
        n_features=n_dim,
        vocab_size=opts.vocab_size + 1,
        embed_dim=opts.receiver_emb,
        hidden_size=opts.receiver_hidden,
        cell=opts.receiver_cell,
    )
    sender = Sender(
        n_features=n_dim,
        vocab_size=opts.vocab_size,
        embed_dim=opts.sender_emb,
        hidden_size=opts.sender_hidden,
        max_len=opts.max_len,
        cell=opts.sender_cell,
    )

    if not opts.variable_length:
        sender = PlusOneWrapper(sender)
    loss = DiffLoss(opts.n_attributes, opts.n_values)

    if opts.run_uid_game:
        pass
        game = UIDGame(
            sender,
            receiver,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            n_attributes=opts.n_attributes,
            n_values=opts.n_values,
        )
    else:
        game = core.SenderReceiverRnnReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=0.0,
            length_cost=0.0,
            baseline_type=MeanBaseline,
        )
    optimizer = torch.optim.Adam(game.parameters(), lr=opts.lr)

    dump_corpus = DumpCorpus(
        {
            # "train": trn_dataset,
            "dev": dev_dataset,
            "test": tst_dataset,
        },
        device=opts.device,
        n_attributes=opts.n_attributes,
        n_values=opts.n_values,
        freq=0,
    )
    early_stopper = EarlyStopperAccuracy(
        opts.early_stopping_thr,
        validation=True
    )
    agent_resetter = PeriodicAgentResetter(
        opts.sender_life_span,
        opts.receiver_life_span,
    )

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=trn_loader,
        validation_data=dev_loader,
        callbacks=[
            core.ConsoleLogger(as_json=True, print_train_loss=False),
            early_stopper,
            dump_corpus,
            agent_resetter,
        ],
    )
    trainer.train(n_epochs=opts.n_epochs)  # type: ignore
    core.close()


if __name__ == "__main__":
    main(sys.argv[1:])

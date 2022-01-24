import sys
import argparse
import itertools
import json
import math
import pathlib
import yaml
from collections import Counter, defaultdict
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    NamedTuple,
    Hashable,
    Optional,
    Tuple,
    TypeVar,
    Sequence,
    Set,
    Union,
)

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
from logfile_reader import LogFile, get_logfiles


T_EntrCalc = TypeVar('T_EntrCalc', bound=Hashable)

class EntropyCalculator(Generic[T_EntrCalc]):
    __data: List[Tuple[T_EntrCalc, ...]]
    __alph: Optional[Set[T_EntrCalc]]
    __freq: "Optional[Counter[Tuple[T_EntrCalc, ...]]]"
    __entropy_conditioned_on_sequence: Optional[Dict[Tuple[T_EntrCalc, ...], float]]
    __entropy_conditioned_on_length: Optional[Dict[int, float]]
    __boundaries: Optional[List[Set[int]]]
    __segmentations: Optional[List[Tuple[Tuple[T_EntrCalc, ...], ...]]]
    __kempe_entropy: Optional[List[Tuple[float, ...]]]

    def __init__(
        self,
        data: List[Tuple[T_EntrCalc, ...]],
        attach_bos: bool = False,
        attach_eos: bool = False,
        bos_id: T_EntrCalc = -1,
        eos_id: T_EntrCalc = -2,
        threshold: float = 0,
        kempe_context_length: int = 3,
        reverse: bool = False,
        verbose: bool = False,
    ):
        self.attach_bos = attach_bos
        self.attach_eos = attach_eos
        self.threshold = threshold
        self.kempe_context_length = kempe_context_length
        self.reverse = reverse
        self.verbose = verbose
        if self.reverse:
            self.__data = [tuple(reversed(d)) for d in data]
        else:
            self.__data = [tuple(d) for d in data]
        if attach_bos:
            self.__data = [(bos_id,) + x for x in self.__data]
        if attach_eos:
            self.__data = [x + (eos_id,) for x in self.__data]
        self.reset()

    def reset(self):
        self.__alph = None
        self.__freq = None
        self.__entropy_conditioned_on_sequence = None
        self.__entropy_conditioned_on_length = None
        self.__boundaries = None
        self.__segmentations = None
        self.__kempe_entropy = None

    @property
    def data(self) -> List[Tuple[T_EntrCalc, ...]]:
        return self.__data

    @property
    def alph(self) -> Set[T_EntrCalc]:
        if self.__alph is None:
            self.__alph = set(itertools.chain.from_iterable(self.data))
        return self.__alph

    @property
    def freq(self) -> "Counter[Tuple[T_EntrCalc, ...]]":
        if self.__freq is None:
            # get frequencies of non-empty sequences.
            self.__freq = Counter(
                s[i:j]
                for s in self.data
                for i in range(len(s))
                for j in range(i + 1, len(s) + 1)
            )
            # The frequency of empty sequence is defined as follows.
            # This is just for the convenience.
            self.__freq[tuple()] = sum(self.__freq[(a,)] for a in self.alph)
        return self.__freq

    @property
    def entropy_conditioned_on_sequence(
        self,
    ) -> Dict[Tuple[T_EntrCalc, ...], float]:
        if self.__entropy_conditioned_on_sequence is None:
            self.__entropy_conditioned_on_sequence = dict()
            for context, context_freq in self.freq.items():
                successors_freq = list(filter(
                    (0).__lt__,
                    (self.freq[context + (a,)] for a in self.alph)
                ))
                if len(successors_freq) == 0:
                    """
                    This means "context" is of maximum length.
                    Skip in this case.
                    """
                    continue
                self.__entropy_conditioned_on_sequence[context] = (
                    -1 * sum(
                        successor_freq * (
                            math.log2(successor_freq) -
                            math.log2(context_freq)
                        )
                        for successor_freq in successors_freq
                    ) / context_freq
                )
        return self.__entropy_conditioned_on_sequence

    @property
    def entropy_conditioned_on_length(
        self,
    ) -> Dict[int, float]:
        if self.__entropy_conditioned_on_length is None:
            entropy_grouped_by_length: 'defaultdict[int, List[Tuple[int, float]]]' = defaultdict(list)
            for seq, ent in self.entropy_conditioned_on_sequence.items():
                entropy_grouped_by_length[len(seq)].append((self.freq[seq], ent))

            self.__entropy_conditioned_on_length = dict()
            for length, v in entropy_grouped_by_length.items():
                total_freq = sum(map(lambda x: x[0], v))
                self.__entropy_conditioned_on_length[length] = sum(f * e for f, e in v) / total_freq
        return self.__entropy_conditioned_on_length

    @property
    def boundaries(
        self,
    ) -> List[Set[int]]:
        if self.__boundaries is None:
            self.__boundaries = []
            for d in self.data:
                self.__boundaries.append(set())
                start: int = 0
                width: int = 2
                while start < len(d):
                    context = d[start:start + width]
                    if (
                        self.entropy_conditioned_on_sequence[context] -
                        self.entropy_conditioned_on_sequence[context[:-1]] >
                        self.threshold
                    ):
                        self.__boundaries[-1].add(start + width)
                    if start + width + 1 < len(d):
                        width = 1 + width
                    else:
                        start = 1 + start
                        width = 1
        return self.__boundaries
    
    @property
    def n_boundaries(
        self,
    ) -> List[int]:
        return [len(b) for b in self.boundaries]

    @property
    def segmentations(self) -> List[Tuple[Tuple[T_EntrCalc, ...], ...]]:
        if self.__segmentations is None:
            segs: List[List[Tuple[T_EntrCalc, ...]]] = []
            for data, boundaries in zip(self.data, self.boundaries):
                segs.append([])
                bot = 0
                for up in sorted(boundaries):
                    word = data[bot:up]
                    bot = up
                    segs[-1].append(word)
            self.__segmentations = [tuple(x) for x in segs]
        return self.__segmentations
    
    @property
    def kempe_entropy(self) -> List[Tuple[float, ...]]:
        if self.__kempe_entropy is None:
            result: List[List[float]] = []
            for d in self.data:
                result.append([])
                for i in range(len(d)):
                    result[-1].append(
                        self.entropy_conditioned_on_sequence[
                            d[max(0, i - self.kempe_context_length):i]
                        ]
                    )
            self.__kempe_entropy = [tuple(x) for x in result]
        return self.__kempe_entropy


def mean(x: Sequence[Union[float, int]]) -> float:
    return float(np.mean(x))  # type: ignore


class Plotter:
    log_files: Dict[Hashable, List[LogFile]]
    least_acc: Optional[float]

    def __init__(
        self,
        log_files: Dict[Hashable, List[LogFile]],
        least_acc: Optional[float] = None,
        img_dir: Union[pathlib.Path, str] = pathlib.Path("./")
    ):
        self.log_files = log_files
        self.least_acc = least_acc
        self.img_dir = pathlib.Path(img_dir)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.markers = [
            "o",
            "s",
            "D",
            "^",
            "v",
            "x",
        ]
        matplotlib.rc('font', family='Noto Sans CJK JP')

    def __get_trained_language(
        self,
        log: LogFile,
    ) -> List[Tuple[int, ...]]:
        if self.least_acc is None:
            epoch = log.max_epoch
        else:
            epoch = log.get_first_epoch_to_reach_acc(float(self.least_acc))
            if epoch is None:
                raise ValueError(f"Language with accuracy >= {self.least_acc} is not found.")
                # print(
                #     f"Language with accuracy >= {self.least_acc} is not found. "
                #     f"Instead, use language with accuracy = {log.max_acc}."
                # )
                # epoch = log.get_first_epoch_to_reach_acc(log.max_acc)
                # assert isinstance(epoch, int)
        return [tuple(m) for m in log.extract_corpus(epoch)["message"]]  # type: ignore
    
    def __get_untrained_language(
        self,
        log: LogFile,
    ) -> List[Tuple[int, ...]]:
        return [tuple(m) for m in log.extract_corpus(0)["message"]]  # type: ignore

    def __hierarchical_entropy_calculator(
        self,
        data: List[Tuple[int, ...]],
        thr: float,
        n: int = 1,
    ) -> EntropyCalculator[Hashable]:
        assert n > 0
        e: EntropyCalculator[Hashable] = EntropyCalculator(data, threshold=thr)
        for _ in range(n - 1):
            e = EntropyCalculator(e.segmentations, threshold=thr)
        return e

    def plot_n_boundaries_over_thresholds(
        self,
        figname: Optional[Union[str, pathlib.Path]] = None,
        verbose: bool = False,
    ) -> None:
        stop_x = 2
        step_x = 0.25
        x_data = [step_x * i for i in range(int(stop_x / step_x) + 1)]
        y_data: 'defaultdict[Hashable, List[List[float]]]' = defaultdict(list)
        for attval, log_files in self.log_files.items():
            if verbose:
                print(f'{self.__class__.__name__}: {attval}')
            for log in log_files:
                try:
                    y_data[attval].append([
                        mean(
                            EntropyCalculator(
                                self.__get_trained_language(log),
                                threshold=x
                            )
                            .n_boundaries
                        )
                        for x in x_data
                    ])
                except Exception as e:
                    print(f"Exception caught: {e}")
        fig, ax = plt.subplots(
            1,
            1,
            sharex=True,
            sharey=True,
        )
        for i, (attval, y_data_lists) in enumerate(y_data.items()):
            ax.plot(
                x_data,
                np.mean(y_data_lists, axis=0),
                marker=self.markers[i],
                fillstyle="none",
                linestyle="--",
                label=(
                    f'$(a,v)={attval}$'
                ),
            )
        ax.legend()
        ax.set_xlabel('threshold')
        ax.set_ylabel('1メッセージあたりの仮説境界の平均数')
        # fig.suptitle('Number of Boundaries With Various Thresholds')
        if figname is None:
            figname = "number_of_hypothetical_boundaries.png"
        plt.savefig(self.img_dir / figname , bbox_inches='tight')

    def plot_zipf(
        self,
        figname: Optional[Union[str, pathlib.Path]] = None,
        threshold: float = 0,
        mode: Literal['zipf', 'zla'] = 'zipf',
        window_size_for_moving_average: int = 1,
        n_repetitions_of_boundary_detection: int = 1,
        remove_hapax_legomena: bool = False,
        verbose: bool = False,
    ) -> None:
        fig, ax = plt.subplots(
            1,
            1,
            sharex=True,
            sharey=True,
        )
        for attval, log_files in self.log_files.items():
            if verbose:
                print(f'{self.__class__.__name__}: {attval}')
            for log in log_files:
                try:
                    len_and_freq = [
                        (len(c[0]), c[1])
                        for c in Counter(
                            itertools.chain.from_iterable(
                                self.__hierarchical_entropy_calculator(
                                    self.__get_trained_language(
                                        log
                                    ),
                                    thr=threshold,
                                    n=n_repetitions_of_boundary_detection,
                                )
                                .segmentations
                            )
                        )
                        .most_common()
                        if not remove_hapax_legomena or c[1] > 1
                    ]
                    lengths_grouped_by_freq: "defaultdict[int, List[int]]" = defaultdict(list)
                    for length, freq in len_and_freq:
                        lengths_grouped_by_freq[freq].append(length)
                    plot_data = [
                        freq
                        if mode == "zipf"
                        else mean(lengths_grouped_by_freq[freq])
                        for _, freq in len_and_freq
                    ]
                    plot_data = (
                        [plot_data[0]] * (window_size_for_moving_average // 2) +
                        plot_data +
                        [plot_data[-1]] * (window_size_for_moving_average // 2)
                    )
                    plot_data = [
                        mean(plot_data[i:i + window_size_for_moving_average])
                        for i in range(len(plot_data) - window_size_for_moving_average)
                    ]
                    ax.plot(
                        range(1, 1 + len(plot_data)),
                        plot_data,
                        label=f'(n_att,n_val)={attval}',
                    )
                except Exception as e:
                    print(f"Exception caught: ({e})")
        ax.legend()
        ax.set_xlabel('Frequency Rank')
        if mode == 'zipf':
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel('Frequency')
        else:
            # ax.set_xscale('log')
            ax.set_ylabel('Word Length')
        fig.suptitle(
            f'{mode.upper()} Plot '
            f'with Treshold {threshold} '
            f'of Moving Average of Length {window_size_for_moving_average}'
        )
        if figname is None:
            figname = f'{mode.lower()}_thr{threshold}_hie{n_repetitions_of_boundary_detection}.png'
        fig.savefig(self.img_dir / figname , bbox_inches='tight')

    def plot_entropy_conditioned_on_length(
        self,
        label_format: str = '$(a,v)={}$',
        figname: Optional[Union[str, pathlib.Path]] = None,
        verbose: bool = False,
    ) -> None:
        fig, ax = plt.subplots(
            1,
            1,
            sharex=True,
            sharey=True,
        )
        for attval, log_files in self.log_files.items():
            if verbose:
                print(f'{self.__class__.__name__}: {attval}')
            for _, log in enumerate(log_files):
                try:
                    #############################
                    # Plot for trained language #
                    #############################
                    length, entropy = zip(*sorted(
                        EntropyCalculator(
                            self.__get_trained_language(log),
                        )
                        .entropy_conditioned_on_length
                        .items()
                    ))
                    ax.plot(
                        length,
                        entropy,
                        linewidth=0.5,
                        label=(
                            None if len(ax.get_lines()) > 0 else "妥当な創発言語"
                        ),
                        color="red",
                        # marker="x",
                    )
                    ###############################
                    # Plot for untrained language #
                    ###############################
                    length, entropy = zip(*sorted(
                        EntropyCalculator(
                            self.__get_untrained_language(log),
                        )
                        .entropy_conditioned_on_length
                        .items()
                    ))
                    ax.plot(
                        length,
                        entropy,
                        linewidth=0.5,
                        label=(
                            None if len(ax.get_lines()) > 1 else "学習前のスピーカから得た創発言語"
                        ),
                        color="blue",
                        # linestyle="dashdot",
                        # marker="+",
                    )
                except Exception as e:
                    print(f"Exception caught: ({e}).")
        ax.legend()
        ax.set_xlabel('$n$')
        ax.set_ylabel('$H(n)$')
        # fig.suptitle('Conditional Entropy $H(n)$')
        if figname is None:
            figname = f'conditional_entropy.png'
        fig.savefig(self.img_dir / figname , bbox_inches='tight')

    def plot_sample_utterance(
        self,
        key: Hashable,
        random_seed: int = 1,
        figname: Optional[Union[str, pathlib.Path]] = None,
        verbose: bool = False,
    ):
        random_state = np.random.RandomState(random_seed)
        log = next(iter(filter(
            lambda log: self.least_acc is None or log.max_acc >= self.least_acc,
            self.log_files[key],
        )))
        entr = EntropyCalculator(
            self.__get_trained_language(log),
            threshold=1.25,
        )
        utter_id: int = random_state.choice(range(len(entr.data)))
        utter = entr.data[utter_id]
        boundaries = entr.boundaries[utter_id]

        fig, ax = plt.subplots(
            1,
            1,
            sharex=True,
            sharey=True,
        )
        for i, boundary in enumerate(sorted(boundaries)):
            ax.plot([boundary - 0.5] * 2, [-0.3, 2.5], linestyle="--", color="black")
        for i, boundary in enumerate(sorted(boundaries)):
            bottom = boundary - 2
            while not (
                entr.entropy_conditioned_on_sequence[utter[bottom:boundary]] -
                entr.entropy_conditioned_on_sequence[utter[bottom:boundary - 1]] > entr.threshold
            ):
                assert bottom >=0
                bottom -= 1
            anno_data = [utter[bottom:up + 1] for up in range(bottom, boundary)]
            x_plot_data = [x + 0.5 for x in range(bottom, boundary)]
            y_plot_data = [entr.entropy_conditioned_on_sequence[x] for x in anno_data]
            ax.plot(
                x_plot_data,
                y_plot_data,
                marker={
                    0: "v",
                    1: "^",
                    2: "<",
                    3: ">",
                }[i % 4],
                label=(
                    f"仮説境界{i+1}に対する推移"
                ),
            )
            for idx in reversed(
                range(len(anno_data)) if len(anno_data) < 3 else (0, -2, -1)
            ):
                anno = anno_data[idx]
                x, y = x_plot_data[idx], y_plot_data[idx]
                anno_str = "$h(" + (
                    ",".join(map(str, anno))
                    if len(anno) < 5
                    else f"{anno[0]},\\ldots,{anno[-2]},{anno[-1]}"
                ) + ")$"
                ax.annotate(
                    anno_str,
                    (x, y),
                    (x + 0.25, y + 0.25),
                    bbox=dict(
                        boxstyle="round",
                        facecolor="white",
                        alpha=0.5,
                    ),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3",
                    ),
                )
            # prev_boundary = boundary
        ax.legend()
        ax.set_xticks(list(range(len(utter))))
        ax.set_xticklabels([str(u) for u in utter])
        ax.set_ylim(-0.1, 2.5)
        ax.set_xlabel("メッセージ")
        ax.set_ylabel("$h$")
        if figname is None:
            figname = f'branching_entropy_sample.png'
        fig.savefig(self.img_dir / figname , bbox_inches='tight')


    def plot_entropy_waves(
        self,
        figname: Optional[Union[str, pathlib.Path]] = None,
        verbose: bool = False,
    ) -> None:
        fig, ax = plt.subplots(
            1,
            1,
            sharex=True,
            sharey=True,
        )
        for attval, log_files in self.log_files.items():
            if verbose:
                print(f'{self.__class__.__name__}: {attval}')
            for log in log_files:
                waves = (
                    EntropyCalculator(
                        self.__get_trained_language(log),
                    )
                    .kempe_entropy
                )
                for i, wave in enumerate(waves[:1]):
                    if i == 0:
                        ax.plot(
                            range(len(wave)),
                            wave,
                            label=f'(n_att,n_val)={attval}',
                        )
                    else:
                        ax.plot(
                            range(len(wave)),
                            wave,
                            color=ax.get_lines()[-1].get_color()
                        )
        ax.legend()
        ax.set_xlabel('Message Index')
        ax.set_ylabel('Kempe Entropy')
        fig.suptitle(f'Entropy Wave')
        if figname is None:
            figname = f'entropy_wave.png'
        fig.savefig(self.img_dir / figname , bbox_inches='tight')


def get_params(params: Sequence[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True, help="")
    args = parser.parse_args(params)

    with open(args.config_yaml, 'r') as fileobj:
        yamlobj: Dict[str, Any] = yaml.safe_load(fileobj)
    for k, v in yamlobj.items():
        setattr(args, k, v)

    return args


def main(params: Sequence[str]):
    opts = get_params(params)
    log_files = get_logfiles(opts.log_dirs)
    plotter = Plotter(
        log_files,
        least_acc=opts.least_acc,
        img_dir=opts.img_dir,
    )
    # plotter.plot_entropy_conditioned_on_length(
    #     verbose=True,
    # )
    # plotter.plot_n_boundaries_over_thresholds(
    #     verbose=True,
    # )
    plotter.plot_sample_utterance(
        key=(2, 64),
        verbose=True,
    )

if __name__ == '__main__':
    main(sys.argv[1:])

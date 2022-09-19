import itertools
import pathlib
import sys
import yaml
from collections import Counter, defaultdict, OrderedDict
from typing import (Dict, Generic, Hashable, List, Literal, Optional, Sequence,
                    Set, Tuple, TypeVar, Union)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from config_reader import get_params
from logfile_reader import LogFile, get_logfiles
from topographic_similarity import compute_topsim

T = TypeVar('T', bound=Hashable)


class EntropyCalculator(Generic[T]):
    __data: List[Tuple[T, ...]]
    __alph: Optional[Set[T]]
    __freq: "Optional[Counter[Tuple[T, ...]]]"
    __branching_entropy: Optional[Dict[Tuple[T, ...], float]]
    __conditional_entropy: Optional[Dict[int, float]]
    __boundaries: Optional[List[Set[int]]]
    __segments: Optional[List[Tuple[Tuple[T, ...], ...]]]
    __segment_ids: Optional[Dict[Tuple[T, ...], int]]
    __hashed_segments: Optional[List[Tuple[int, ...]]]
    __random_boundaries: Optional[List[Set[int]]]
    __random_segments: Optional[List[Tuple[Tuple[T, ...], ...]]]
    __random_segment_ids: Optional[Dict[Tuple[T, ...], int]]
    __hashed_random_segments: Optional[List[Tuple[int, ...]]]

    def __init__(
        self,
        data: List[Tuple[T, ...]],
        threshold: float = 0,
        random_seed: int = 0,
        reverse: bool = False,
        verbose: bool = False,
    ):
        self.reverse = reverse
        self.verbose = verbose
        if self.reverse:
            self.__data = [tuple(reversed(d)) for d in data]
        else:
            self.__data = [tuple(d) for d in data]
        self.__reset_on_init()
        self.threshold = threshold
        self.random_seed = random_seed

    def __reset_on_init(self):
        self.__alph = None
        self.__freq = None
        self.__branching_entropy = None
        self.__conditional_entropy = None
        self.__reset_on_setting_threshold()
        self.__reset_on_setting_random_seed()

    def __reset_on_setting_threshold(self):
        self.__boundaries = None
        self.__segments = None
        self.__segment_ids = None
        self.__hashed_segments = None
        self.__reset_on_setting_random_seed()

    def __reset_on_setting_random_seed(self):
        self.__random_boundaries = None
        self.__random_segments = None
        self.__random_segment_ids = None
        self.__hashed_random_segments = None

    @property
    def threshold(self) -> float:
        return self.__threshold

    @threshold.setter
    def threshold(self, x: float):
        self.__threshold = x
        self.__reset_on_setting_threshold()

    @property
    def random_seed(self) -> int:
        return self.__random_seed

    @random_seed.setter
    def random_seed(self, x: int):
        self.__random_seed = x
        self.__reset_on_setting_random_seed()

    @property
    def data(self) -> List[Tuple[T, ...]]:
        return self.__data

    @property
    def alph(self) -> Set[T]:
        if self.__alph is None:
            self.__alph = set(itertools.chain.from_iterable(self.data))
        return self.__alph

    @property
    def freq(self) -> "Counter[Tuple[T, ...]]":
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
            self.__freq[tuple()] = sum(len(s) for s in self.data)
        return self.__freq

    @property
    def branching_entropy(
        self,
    ) -> Dict[Tuple[T, ...], float]:
        if self.__branching_entropy is None:
            self.__branching_entropy = dict()
            for context, context_freq in self.freq.items():
                succ_freq_list = [self.freq[context + (a,)] for a in self.alph]
                # if sum(succ_freq_list) == 0:
                #     continue
                self.__branching_entropy[context] = (
                    -1 * sum(
                        succ_freq * (np.log2(succ_freq) - np.log2(context_freq))
                        for succ_freq in succ_freq_list if succ_freq > 0
                    ) / context_freq
                )
        return self.__branching_entropy

    @property
    def conditional_entropy(
        self,
    ) -> Dict[int, float]:
        if self.__conditional_entropy is None:
            self.__conditional_entropy = dict()
            length_to_total_freq: Dict[int, int] = dict()
            for seq, ent in self.branching_entropy.items():
                seq_len = len(seq)
                if seq_len not in self.__conditional_entropy:
                    self.__conditional_entropy[seq_len] = 0
                if seq_len not in length_to_total_freq:
                    length_to_total_freq[seq_len] = 0
                self.__conditional_entropy[seq_len] += self.freq[seq] * ent
                length_to_total_freq[seq_len] += self.freq[seq]
            for length, total_freq in length_to_total_freq.items():
                self.__conditional_entropy[length] /= total_freq
        return self.__conditional_entropy

    @property
    def boundaries(self) -> List[Set[int]]:
        if self.__boundaries is None:
            self.__boundaries = []
            for d in self.data:
                self.__boundaries.append(set())
                start: int = 0
                width: int = 2
                """
                We begin with width=2, while the algorithm in the paper begins with width=1.
                It is because this code block assumes that self.branching_entropy is already computed.
                """
                while start < len(d):
                    context = d[start:start + width]
                    if self.branching_entropy[context] - self.branching_entropy[context[:-1]] > self.threshold:
                        self.__boundaries[-1].add(start + width)
                    if start + width + 1 < len(d):
                        width = 1 + width
                    else:
                        start = 1 + start
                        width = 2
        return self.__boundaries

    @property
    def segments(self) -> List[Tuple[Tuple[T, ...], ...]]:
        if self.__segments is None:
            segs: List[List[Tuple[T, ...]]] = []
            for data, boundaries in zip(self.data, self.boundaries):
                segs.append([])
                bot = 0
                for top in sorted(boundaries | {len(data)}):
                    word = data[bot:top]
                    bot = top
                    segs[-1].append(word)
            self.__segments = [tuple(x) for x in segs]
        return self.__segments

    @property
    def segment_ids(self):
        if self.__segment_ids is None:
            self.__segment_ids = {
                s: i + 1 for i, s in
                enumerate(set(itertools.chain.from_iterable(self.segments)))
            }
        return self.__segment_ids

    @property
    def hashed_segments(self):
        if self.__hashed_segments is None:
            self.__hashed_segments = [
                tuple(self.segment_ids[x] for x in s)
                for s in self.segments
            ]
        return self.__hashed_segments

    @property
    def random_boundaries(self) -> List[Set[int]]:
        if self.__random_boundaries is None:
            random_state = np.random.RandomState(seed=self.random_seed)
            self.__random_boundaries = [
                set(random_state.choice(np.arange(1, len(data), dtype=np.int_), size=len(boundaries)))
                for data, boundaries in zip(self.data, self.boundaries)
            ]
        return self.__random_boundaries

    @property
    def random_segments(self) -> List[Tuple[Tuple[T, ...], ...]]:
        if self.__random_segments is None:
            segs: List[List[Tuple[T, ...]]] = []
            for data, boundaries in zip(self.data, self.random_boundaries):
                segs.append([])
                bot = 0
                for top in sorted(boundaries | {len(data)}):
                    word = data[bot:top]
                    bot = top
                    segs[-1].append(word)
            self.__random_segments = [tuple(x) for x in segs]
        return self.__random_segments

    @property
    def random_segment_ids(self):
        if self.__random_segment_ids is None:
            self.__random_segment_ids = {
                s: i + 1 for i, s in
                enumerate(set(itertools.chain.from_iterable(self.random_segments)))
            }
        return self.__random_segment_ids

    @property
    def hashed_random_segments(self):
        if self.__hashed_random_segments is None:
            self.__hashed_random_segments = [
                tuple(self.random_segment_ids[x] for x in s)
                for s in self.random_segments
            ]
        return self.__hashed_random_segments

    @property
    def n_boundaries(self) -> List[int]:
        return [len(b) for b in self.boundaries]

    @property
    def mean_n_boundaries(self) -> float:
        return np.mean(self.n_boundaries)

    @property
    def vocab_size(self) -> int:
        return len(self.segment_ids)


def standard_error_of_mean(
    x: Union[npt.NDArray[np.float_], Sequence[float]]
) -> npt.NDArray[np.float_]:
    x = np.array(x)
    return np.std(x, ddof=1) / np.sqrt(np.size(x))


class Plotter:
    log_files: Dict[Hashable, List[LogFile]]
    least_acc: Optional[float]
    trained_lang_mode: Literal["min_epoch_to_reach_least_acc", "max_epoch"]

    def __init__(
        self,
        log_files: Dict[Hashable, List[LogFile]],
        least_acc: Optional[float] = None,
        trained_lang_mode: Literal[
            "min_epoch_to_reach_least_acc",
            "max_epoch",
        ] = "max_epoch",
        img_dir: Union[pathlib.Path, str] = pathlib.Path("./img_dir"),
        tmp_dir: Union[pathlib.Path, str] = pathlib.Path("./tmp_dir"),
    ):
        self.log_files = log_files
        self.least_acc = least_acc
        self.trained_lang_mode = trained_lang_mode
        self.img_dir = pathlib.Path(img_dir)
        self.tmp_dir = pathlib.Path(tmp_dir)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.markers = ["o", "+", "*", "^", "v", "x"]
        self.hatches = ["\\", "+", "-", "|", "x", "/"]
        matplotlib.rc('font', family='Noto Sans CJK JP')

    def __get_sender_inputs(
        self,
        log: LogFile,
    ) -> List[Tuple[int, ...]]:
        return [tuple(m) for m in log.extract_corpus(log.max_epoch)["sender_input"]]  # type: ignore

    def __get_trained_language(
        self,
        log: LogFile,
    ) -> List[Tuple[int, ...]]:
        if self.least_acc is None:
            epoch = log.max_epoch
        elif self.trained_lang_mode == "min_epoch_to_reach_least_acc":
            epoch = log.get_first_epoch_to_reach_acc(float(self.least_acc))
            if epoch is None:
                raise ValueError(f"Language with accuracy >= {self.least_acc} is not available.")
        elif self.trained_lang_mode == "max_epoch":
            epoch = log.max_epoch
            acc: float = log.extract_learning_history("test")["acc"].tolist()[epoch - 1]
            if acc < self.least_acc:
                raise ValueError(f"Language with accuracy >= {self.least_acc} is not available.")
        else:
            raise NotImplementedError
        return [tuple(m) for m in log.extract_corpus(epoch)["message"]]  # type: ignore

    def __get_untrained_language(
        self,
        log: LogFile,
    ) -> List[Tuple[int, ...]]:
        return [tuple(m) for m in log.extract_corpus(0)["message"]]  # type: ignore

    def report_n_valid_emergent_languages(self) -> None:
        n_valid_langs: Dict[Hashable, int] = dict()
        for attval, log_files in self.log_files.items():
            print(f'{self.__class__.__name__}: {attval}')
            n_valid_langs[attval] = 0
            for log in log_files:
                try:
                    self.__get_trained_language(log)
                    n_valid_langs[attval] += 1
                except Exception as e:
                    print(f"Exception caught: {e}")
                    continue
        print("Report:", n_valid_langs)

    def plot_n_hypothetical_boundaries(
        self,
        thresholds: Sequence[float] = (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2),
        figname: Union[str, pathlib.Path] = "n_hypothetical_boundaries.png",
        attval_format: str = '${}$',
        threshold_format: str = '$threshold={}$',
        xlabel: str = "$(n_{{att}},n_{{val}})$ Configuration",
        ylabel: str = "Mean # Hypo-boundaries per Message",
    ) -> None:
        thr_to_attval_to_data_list: OrderedDict[float, OrderedDict[Hashable, List[float]]] = OrderedDict()

        for attval, log_files in self.log_files.items():
            print(f'{self.__class__.__name__}: {attval}')
            for log in log_files:
                try:
                    lang = self.__get_trained_language(log)
                except Exception as e:
                    print(f"Exception caught: {e}")
                    continue
                entr_calc = EntropyCalculator(lang)
                for thr in thresholds:
                    if thr not in thr_to_attval_to_data_list:
                        thr_to_attval_to_data_list[thr] = OrderedDict()
                    if attval not in thr_to_attval_to_data_list[thr]:
                        thr_to_attval_to_data_list[thr][attval] = []
                    entr_calc.threshold = thr
                    thr_to_attval_to_data_list[thr][attval].append(entr_calc.mean_n_boundaries)
                del entr_calc

        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(111)
        for thr, attval_to_data_list in thr_to_attval_to_data_list.items():
            data_lists = list(attval_to_data_list.values())
            x: npt.NDArray[np.int_] = np.arange(len(data_lists))
            y: npt.NDArray[np.float_] = np.array([np.mean(x) for x in data_lists])
            y_sem: npt.NDArray[np.float_] = np.array([standard_error_of_mean(x) for x in data_lists])
            ax.plot(
                x,
                y,
                marker="x",
                label=threshold_format.format(thr),
            )
            ax.fill_between(
                x,
                y - y_sem,
                y + y_sem,
                color=ax.get_lines()[-1].get_color(),
                alpha=0.3,
            )
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        ax.set_xticks(list(range(len(self.log_files.keys()))))
        ax.set_xticklabels([attval_format.format(attval) for attval in self.log_files.keys()], rotation=45)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(self.img_dir / figname, bbox_inches='tight')

    def plot_vocab_size(
        self,
        thresholds: Sequence[float] = (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2),
        figname: Union[str, pathlib.Path] = "vocab_size.png",
        attval_format: str = '${}$',
        threshold_format: str = '$threshold={}$',
        xlabel: str = "$(n_{{att}},n_{{val}})$ Configuration",
        ylabel: str = "Vocabulary Size",
    ) -> None:
        thr_to_attval_to_data_list: OrderedDict[float, OrderedDict[Hashable, List[float]]] = OrderedDict()

        for attval, log_files in self.log_files.items():
            print(f'{self.__class__.__name__}: {attval}')
            for log in log_files:
                try:
                    lang = self.__get_trained_language(log)
                except Exception as e:
                    print(f"Exception caught: {e}")
                    continue
                entr_calc = EntropyCalculator(lang)
                for thr in thresholds:
                    if thr not in thr_to_attval_to_data_list:
                        thr_to_attval_to_data_list[thr] = OrderedDict()
                    if attval not in thr_to_attval_to_data_list[thr]:
                        thr_to_attval_to_data_list[thr][attval] = []
                    entr_calc.threshold = thr
                    thr_to_attval_to_data_list[thr][attval].append(entr_calc.vocab_size)
                del entr_calc

        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(111)
        for thr, attval_to_data_list in thr_to_attval_to_data_list.items():
            data_lists = list(attval_to_data_list.values())
            x: npt.NDArray[np.int_] = np.arange(len(data_lists))
            y: npt.NDArray[np.float_] = np.array([np.mean(x) for x in data_lists])
            y_sem: npt.NDArray[np.float_] = np.array([standard_error_of_mean(x) for x in data_lists])
            ax.plot(
                x,
                y,
                marker="x",
                label=threshold_format.format(thr),
            )
            ax.fill_between(
                x,
                y - y_sem,
                y + y_sem,
                color=ax.get_lines()[-1].get_color(),
                alpha=0.3,
            )
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        ax.set_xticks(list(range(len(self.log_files.keys()))))
        ax.set_xticklabels([attval_format.format(attval) for attval in self.log_files.keys()], rotation=45)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(self.img_dir / figname, bbox_inches='tight')

    def plot_sigurd_style_zla(
        self,
        threshold: float = 0,
        plot_trained_lang: bool = True,
        plot_untrained_lang: bool = False,
        figname: Optional[Union[str, pathlib.Path]] = None,
        label_format: str = '$(n_{{att}},n_{{val}})={}$, {}.',
        verbose: bool = False,
    ):
        def get_map_from_len_to_freq_percentage(
            lang: List[Tuple[int, ...]],
            max_len: int,
        ):
            len_count = Counter(map(len, itertools.chain.from_iterable(EntropyCalculator(lang, threshold=threshold).segments)))
            total_value = sum(len_count.values())
            return {i + 1: len_count[i + 1] / total_value for i in range(max_len)}

        attval_to_maps_for_trained_lang: defaultdict[Hashable, List[Dict[int, float]]] = defaultdict(list)
        attval_to_maps_for_untrained_lang: defaultdict[Hashable, List[Dict[int, float]]] = defaultdict(list)
        for attval, log_files in self.log_files.items():
            if verbose:
                print(f'{self.__class__}: {attval}')
            for log in log_files:
                try:
                    trained_lang_data = self.__get_trained_language(log)
                    untrained_lang_data = self.__get_untrained_language(log)
                except Exception as e:
                    print(f"Exception caught: {e}")
                    continue
                max_len = int(log.extract_config().max_len)
                attval_to_maps_for_trained_lang[attval].append(get_map_from_len_to_freq_percentage(trained_lang_data, max_len))
                attval_to_maps_for_untrained_lang[attval].append(get_map_from_len_to_freq_percentage(untrained_lang_data, max_len))
        what_to_plot: List[Tuple[str, Dict[Hashable, List[Dict[int, float]]]]] = []
        if plot_trained_lang:
            what_to_plot.append(("trained", attval_to_maps_for_trained_lang))
        if plot_untrained_lang:
            what_to_plot.append(("untrained", attval_to_maps_for_untrained_lang))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for data_type, attval_to_maps in what_to_plot:
            for attval, map_list in attval_to_maps.items():
                for i, m in enumerate(map_list):
                    ax.plot(
                        list(m.keys()),
                        list(m.values()),
                        label=(None if i > 0 else label_format.format(attval, data_type)),
                        color=(None if i == 0 else ax.get_lines()[-1].get_color()),
                    )
        ax.legend()
        if figname is None:
            figname = f'sigurd_style_zla_thr{threshold}.png'
        fig.savefig(self.img_dir / figname, bbox_inches='tight')

    def plot_zla(
        self,
        attval: Hashable,
        figname: Optional[Union[str, pathlib.Path]] = None,
        xlabel: str = "Frequency Rank",
        ylabel: str = "Hypo-segment Length",
        thresholds: Sequence[float] = (0, 0.5, 1.5, 2),
        max_rank: Optional[int] = None,
        attval_format: str = '$(n_{{att}},n_{{val}})={}$',
        threshold_format: str = '$threshold={}$',
        verbose: bool = False,
    ):
        thr_to_trained_data_list: Dict[float, List[List[float]]] = dict()
        for log_file in self.log_files[attval]:
            try:
                lang_data = self.__get_trained_language(log_file)
            except Exception as e:
                print(f"Exception caught: {e}")
                continue
            entr_calc = EntropyCalculator(lang_data)
            for thr in thresholds:
                if thr not in thr_to_trained_data_list:
                    thr_to_trained_data_list[thr] = []

                entr_calc.threshold = thr

                freqs: List[int] = []
                freq_to_lens: defaultdict[int, List[int]] = defaultdict(list)

                for word, freq in Counter(itertools.chain.from_iterable(entr_calc.segments)).most_common():
                    freqs.append(freq)
                    freq_to_lens[freq].append(len(word))

                thr_to_trained_data_list[thr].append([np.mean(freq_to_lens[freq]) for freq in freqs])
            del entr_calc

        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(111)
        for thr in thresholds:
            data_list = thr_to_trained_data_list[thr]
            y: npt.NDArray[np.float_] = np.array(
                [
                    np.mean([e for e in x if e is not None]) for x in zip(*data_list)
                    # if len([e for e in x if e is not None]) > 2
                ][:max_rank]
            )
            y_sem: npt.NDArray[np.float_] = np.array(
                [
                    standard_error_of_mean([e for e in x if e is not None]) for x in zip(*data_list)
                    # if len([e for e in x if e is not None]) > 2
                ][:max_rank]
            )
            x: npt.NDArray[np.int_] = np.arange(np.size(y)) + 1
            ax.plot(
                x,
                y,
                label=attval_format.format(attval) + ", " + threshold_format.format(thr),
            )
            ax.fill_between(
                x,
                y - y_sem,
                y + y_sem,
                color=ax.get_lines()[-1].get_color(),
                alpha=0.3,
            )
        ax.legend(bbox_to_anchor=(0.5, -0.25), loc="upper center")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if figname is None:
            figname = f'zla_attval{attval}.png'
        fig.savefig(self.img_dir / figname, bbox_inches='tight')

    def plot_zipf(
        self,
        attval: Hashable,
        figname: Optional[Union[str, pathlib.Path]] = None,
        label_format: str = '$(n_{{att}},n_{{val}})={}$. $threshold={}$.',
    ) -> None:
        thr_to_y_data: Dict[float, List[Sequence[int]]] = dict()

        tmp_name = f"zipfs_law_attval{attval}.yaml"
        tmp_file = self.tmp_dir / tmp_name
        if False and tmp_file.is_file():
            with open(tmp_file, "r") as file_obj:
                loaded = yaml.full_load(file_obj)
            thr_to_y_data.update(loaded["thr_to_y_data"])

        for log in self.log_files[attval]:
            try:
                lang = self.__get_trained_language(log)
            except Exception as e:
                print(f"Exception caught: {e}")
                continue
            thresholds = [0, 0.5, 1, 1.5]
            for thr in thresholds:
                thr_to_y_data[thr] = thr_to_y_data.get(thr, [])
                freqs = [
                    x[1] for x in Counter(itertools.chain.from_iterable(EntropyCalculator(lang, threshold=thr).segments)).most_common()
                ]
                thr_to_y_data[thr].append(freqs)

        with open(tmp_file, "w") as file_obj:
            yaml.dump({"thr_to_y_data": thr_to_y_data}, file_obj)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, (thr, y_data_list) in enumerate(sorted(thr_to_y_data.items())):
            for j, each_y_data in enumerate(y_data_list):
                ax.plot(
                    [k + 1 for k in range(len(each_y_data))],
                    each_y_data,
                    label=(None if j > 0 else label_format.format(attval, thr)),
                    color=(None if j == 0 else ax.get_lines()[-1].get_color()),
                    linestyle=({
                        0: "solid",
                        1: "dashed",
                        2: "dashdot",
                        3: "dotted",
                    }[i % 4]),
                )
        ax.legend()
        ax.set_xlabel('Frequency Rank')
        ax.set_ylabel('Frequency')
        ax.set_xscale('log')
        ax.set_yscale('log')
        if figname is None:
            figname = f'zipf_attval{attval}.png'
        fig.savefig(self.img_dir / figname, bbox_inches='tight')

    def plot_conditional_entropy(
        self,
        figname: Union[str, pathlib.Path] = "conditional_entropy.png",
        xlabel: str = "$n$",
        ylabel: str = "$H(n)$",
    ) -> None:
        entropies_before_training: Dict[Hashable, List[List[float]]] = dict()
        entropies_after_training: Dict[Hashable, List[List[float]]] = dict()

        for attval, log_files in self.log_files.items():
            print(f'{self.__class__.__name__}: {attval}')
            entropies_before_training[attval] = list()
            entropies_after_training[attval] = list()
            for _, log in enumerate(log_files):
                try:
                    lang_before_training = self.__get_untrained_language(log)
                    lang_after_training = self.__get_trained_language(log)
                except Exception as e:
                    print(f"Exception caught: {e}")
                    continue
                entropies_after_training[attval].append(
                    [v for _, v in sorted(EntropyCalculator(lang_after_training).conditional_entropy.items())]
                )
                entropies_before_training[attval].append(
                    [v for _, v in sorted(EntropyCalculator(lang_before_training).conditional_entropy.items())]
                )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for attval, _ in self.log_files.items():
            for entropies_bf, entropies_af in zip(
                entropies_before_training[attval],
                entropies_after_training[attval]
            ):
                ax.plot(
                    list(range(len(entropies_bf))),
                    entropies_bf,
                    linewidth=0.75,
                    label=(
                        None if len(ax.get_lines()) > 0 else "Emergent language before training"
                    ),
                    color="blue",
                    linestyle="dashed"
                )
                ax.plot(
                    list(range(len(entropies_af))),
                    entropies_af,
                    linewidth=0.75,
                    label=(
                        None if len(ax.get_lines()) > 1 else "Successful language after training"
                    ),
                    color="red",
                )
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(self.img_dir / figname, bbox_inches='tight')

    def plot_sample_utterances(
        self,
        key: Hashable,
        threshold: float,
        random_seed: int = 0,
        figname: Optional[Union[str, pathlib.Path]] = None,
    ):
        random_state = np.random.RandomState(random_seed)
        selected_log = None
        for log in self.log_files[key]:
            try:
                self.__get_trained_language(log)
                selected_log = log
                break
            except Exception as e:
                print(f"Exception caught: {e}")
        assert selected_log is not None
        entr = EntropyCalculator(
            self.__get_trained_language(selected_log),
            threshold=threshold,
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
                entr.branching_entropy[utter[bottom:boundary]] - entr.branching_entropy[utter[bottom:boundary - 1]] > entr.threshold
            ):
                assert bottom >= 0
                bottom -= 1
            anno_data = [utter[bottom:up + 1] for up in range(bottom, boundary)]
            x_plot_data = [x + 0.5 for x in range(bottom, boundary)]
            y_plot_data = [entr.branching_entropy[x] for x in anno_data]
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
                    f"transition for boundary {i+1}"
                ),
                linewidth=3,
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
                # ax.annotate(
                #     anno_str,
                #     (x, y),
                #     (x + 0.25, y + 0.25),
                #     bbox=dict(
                #         boxstyle="round",
                #         facecolor="white",
                #         alpha=0.5,
                #     ),
                #     arrowprops=dict(
                #         arrowstyle="->",
                #         connectionstyle="arc3",
                #     ),
                # )
            # prev_boundary = boundary
        # ax.legend(bbox_to_anchor=(0.5, -0.125), loc="upper center", ncol=2)
        ax.set_xticks(list(range(len(utter))))
        ax.set_xticklabels([str(u) for u in utter])
        ax.set_ylim(-0.1, 2.5)
        ax.set_xlabel("メッセージ")  # "message")
        ax.set_ylabel("$h$")
        if figname is None:
            figname = f"branching_entropy_sample_attval{key}_thr{threshold}_seed{random_seed}.png"
        fig.tight_layout()
        fig.savefig(self.img_dir / figname, bbox_inches='tight')

    def plot_topsim(
        self,
        thresholds: Sequence[float] = (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2),
        figname: Union[pathlib.Path, str] = "topsim.png",
        xlabel: str = "$threshold$",
        ylabel: str = "TopSim",
        attval_format: str = '$(n_{{att}},n_{{val}})={}$',
        verbose: bool = True,
    ):
        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(111)
        for i, (attval, log_files) in enumerate(self.log_files.items()):
            if int(next(iter(log_files)).extract_config().n_attributes) < 2:
                continue
            if verbose:
                print(f'{self.__class__}: {attval}')
            plain_topsims: List[float] = []
            thr_to_topsims: defaultdict[float, List[float]] = defaultdict(list)
            for log in log_files:
                try:
                    sender_input_data = self.__get_sender_inputs(log)
                    trained_lang_data = self.__get_trained_language(log)
                except Exception as e:
                    print(f"Exception caught: {e}")
                    continue

                plain_topsims.append(compute_topsim(trained_lang_data, sender_input_data))

                entr_calc = EntropyCalculator(trained_lang_data)
                for thr in thresholds:
                    if verbose:
                        print(f"{self.__class__}: {thr}")
                    entr_calc.threshold = thr
                    thr_to_topsims[thr].append(compute_topsim(entr_calc.hashed_segments, sender_input_data))
                del entr_calc

            data_lists = [plain_topsims] + [thr_to_topsims[thr] for thr in thresholds]
            x: npt.NDArray[np.float_] = np.array([-0.25] + list(thresholds))
            y: npt.NDArray[np.float_] = np.array([np.mean(d) for d in data_lists])
            y_sem: npt.NDArray[np.float_] = np.array([np.std(d, ddof=1) / np.sqrt(np.size(d)) for d in data_lists])
            ax.plot(
                x,
                y,
                label=attval_format.format(attval),
                marker=self.markers[i % len(self.markers)],
            )
            ax.fill_between(
                x,
                y - y_sem,
                y + y_sem,
                color=ax.get_lines()[-1].get_color(),
                alpha=0.3,
            )
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks((-0.25,) + tuple(thresholds))
        ax.set_xticklabels(("$-\\infty$",) + tuple(thresholds))
        fig.savefig(self.img_dir / figname, bbox_inches='tight')

    def plot_topsim_compared_to_random_baseline(
        self,
        attval: Hashable,
        thresholds: Sequence[float] = (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2),
        figname: Optional[Union[pathlib.Path, str]] = None,
        xlabel: str = "$threshold$",
        ylabel: str = "TopSim",
        attval_format: str = '$(n_{{att}},n_{{val}})={}$',
        verbose: bool = True,
    ):
        thr_to_topsims: defaultdict[float, List[float]] = defaultdict(list)
        thr_to_random_seg_topsims: defaultdict[float, List[float]] = defaultdict(list)

        for log_file in self.log_files[attval]:
            if int(log_file.extract_config().n_attributes) < 2:
                raise ValueError("n_attributes must be larger than 1.")
            try:
                sender_input_data = self.__get_sender_inputs(log_file)
                trained_lang_data = self.__get_trained_language(log_file)
            except Exception as e:
                print(f"Exception caught: {e}")
                continue
            entr_calc = EntropyCalculator(trained_lang_data)
            for thr in thresholds:
                if verbose:
                    print(f"{self.__class__}: {thr}")
                entr_calc.threshold = thr
                thr_to_topsims[thr].append(compute_topsim(entr_calc.hashed_segments, sender_input_data))
                thr_to_random_seg_topsims[thr].append(compute_topsim(entr_calc.hashed_random_segments, sender_input_data))
            del entr_calc
        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(111)
        data_lists = [thr_to_topsims[thr] for thr in thresholds]
        random_seg_data_lists = [thr_to_random_seg_topsims[thr] for thr in thresholds]
        x: npt.NDArray[np.float_] = np.array(list(thresholds))
        y: npt.NDArray[np.float_] = np.array([np.mean(d) for d in data_lists])
        y_sem: npt.NDArray[np.float_] = np.array([np.std(d, ddof=1) / np.sqrt(np.size(d)) for d in data_lists])
        y_random: npt.NDArray[np.float_] = np.array([np.mean(d) for d in random_seg_data_lists])
        y_random_sem: npt.NDArray[np.float_] = np.array([np.std(d, ddof=1) / np.sqrt(np.size(d)) for d in random_seg_data_lists])
        ax.plot(
            x,
            y,
            label=attval_format.format(attval),
            marker="o",
        )
        ax.fill_between(
            x,
            y - y_sem,
            y + y_sem,
            color=ax.get_lines()[-1].get_color(),
            alpha=0.3,
        )
        ax.plot(
            x,
            y_random,
            label=attval_format.format(attval) + " (random boundary)",
            marker="D",
        )
        ax.fill_between(
            x,
            y_random - y_random_sem,
            y_random + y_random_sem,
            color=ax.get_lines()[-1].get_color(),
            alpha=0.3,
        )
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if figname is None:
            figname = f"topsim_attval{attval}_vs_random_baseline.png"
        fig.savefig(self.img_dir / figname, bbox_inches='tight')


def generate_random_synthetic_language(
    n_attributes: int,
    n_values: int,
    max_len: int,
    vocab_size: int,
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
    synthetic_language: List[Tuple[int, ...]] = []
    for values in itertools.product(range(n_values), repeat=n_attributes):
        message = sum((attval_to_segment[a, v] for a, v in enumerate(values)), start=())
        synthetic_language.append(message)


def main(params: Sequence[str]):
    opts = get_params(params)
    log_files = get_logfiles(opts.log_dirs)
    plotter = Plotter(
        log_files,
        least_acc=opts.least_acc,
        trained_lang_mode=opts.trained_lang_mode,
        img_dir=opts.img_dir,
        tmp_dir=opts.tmp_dir,
    )
    plotter.report_n_valid_emergent_languages()
    print("")
    print("")
    print("")
    print(f"Please uncomment function calls you want in the last code block in {__file__}")
    print("")
    print("")
    print("")
    plt.rcParams["font.size"] = 14  # 18
    plotter.plot_conditional_entropy()
    # plotter.plot_sample_utterances(key=(2, 64), threshold=1.25, random_seed=1)
    # plotter.plot_n_hypothetical_boundaries()
    # plotter.plot_vocab_size()
    # plotter.plot_topsim()
    # plotter.plot_topsim_compared_to_random_baseline(attval=(2, 64))
    # plotter.plot_topsim_compared_to_random_baseline(attval=(3, 16))
    # plotter.plot_topsim_compared_to_random_baseline(attval=(4, 8))
    # plotter.plot_topsim_compared_to_random_baseline(attval=(6, 4))
    # plotter.plot_topsim_compared_to_random_baseline(attval=(12, 2))
    # plotter.plot_zla(attval=(1, 4096))
    # plotter.plot_zla(attval=(2, 64))
    # plotter.plot_zla(attval=(3, 16))
    # plotter.plot_zla(attval=(4, 8))
    # plotter.plot_zla(attval=(6, 4))
    # plotter.plot_zla(attval=(12, 2))
    # for thr in [1.0, 1.25]:
    #     for seed in [0, 1, 2, 3, 4, 5]:
    #         plotter.plot_sample_utterances(key=(2, 64), threshold=thr, random_seed=seed)


if __name__ == '__main__':
    main(sys.argv[1:])

import json
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Tuple, Union

import pandas as pd


class LogFile:
    def __init__(
        self,
        log_path: pathlib.Path,
    ) -> None:
        assert log_path.is_file()
        self.log_path = log_path
        self.read()

    def read(self):
        with self.log_path.open() as fileobj:
            self.lines = fileobj.readlines()
        self.jsons:    'defaultdict[str, Dict[Any, Any]]' = defaultdict(dict)
        self.line_idx: 'defaultdict[str, Dict[int, int]]' = defaultdict(dict)
        self.max_epoch = 0
        self.max_acc = 0.0
        self.line_no_of_config:  int = -1
        self.line_no_of_dataset: int = -1
        self.line_no_of_language: Dict[Optional[int], int] = {}
        self.line_no_of_train:    Dict[Optional[int], int] = {}
        self.line_no_of_test:     Dict[Optional[int], int] = {}
        self.line_no_of_metric:   Dict[Optional[int], int] = {}

        for i, line in enumerate(self.lines):
            try:
                info = json.loads(line)
            except ValueError:
                continue
            mode:  Optional[str] = info.pop('mode',  None)
            epoch: Optional[int] = info.pop('epoch', None)
            if mode == 'config':
                self.line_no_of_config = i
            elif mode == 'dataset':
                self.line_no_of_dataset = i
            elif mode == 'language':
                self.line_no_of_language[epoch] = i
            elif mode == 'train':
                self.line_no_of_train[epoch] = i
            elif mode == 'test':
                self.line_no_of_test[epoch] = i
                self.max_acc = max(self.max_acc, info["acc"])
            elif mode == 'metric':
                self.line_no_of_metric[epoch] = i
            if epoch is not None:
                self.max_epoch = max(self.max_epoch, epoch)
        return self.lines

    def write(self, path: Optional[pathlib.Path] = None):
        if path is None:
            path = self.log_path
        with path.open(mode="w") as fileobj:
            fileobj.writelines(self.lines)

    def get_first_epoch_to_reach_acc(
        self,
        acc: float,
    ) -> Union[int, None]:
        for epoch in range(1, self.max_epoch + 1):
            info = json.loads(self.lines[self.line_no_of_test[epoch]])
            if info["acc"] >= acc:
                return epoch
        return None

    def extract_corpus(self, epoch: int) -> pd.DataFrame:
        data: Dict[str, Any] = dict()
        data.update(json.loads(self.lines[self.line_no_of_dataset])['data'])
        data.update(json.loads(self.lines[self.line_no_of_language[epoch]])['data'])
        return pd.DataFrame(data=data)

    def extract_config(self):
        config = json.loads(self.lines[self.line_no_of_config])
        names = [(str(k), type(v)) for k, v in config.items()]
        return NamedTuple('Config', names)(**config)

    def extract_learning_history(
        self,
        mode: Literal['train', 'test', 'metric'],
    ) -> pd.DataFrame:
        assert mode in {'train', 'test', 'metric'}, mode
        data: 'defaultdict[str, Any]' = defaultdict(list)
        for epoch in range(1, self.max_epoch + 1):
            info = json.loads(
                self.lines[
                    self.line_no_of_train[epoch] if mode == 'train'
                    else self.line_no_of_test[epoch] if mode == 'test'
                    else self.line_no_of_metric[epoch]
                ]
            )
            for k, v in info.items():
                data[k].append(v)
        return pd.DataFrame(data=data)


def get_logfiles(
    log_dirs: List[str]
) -> Dict[Tuple[int, int], List[LogFile]]:
    log_files: 'defaultdict[Tuple[int, ...], List[LogFile]]' = defaultdict(list)
    for log_dir in map(pathlib.Path, log_dirs):
        assert log_dir.is_dir()
        for log_file in map(LogFile, log_dir.glob('*.log')):
            config = log_file.extract_config()
            key = (
                int(config.n_attributes),
                int(config.n_values),
                # int(config.n_guessable_attributes),
            )
            log_files[key].append(log_file)
            print(f"{key}, {config.random_seed}, {log_file.max_epoch}")
    return log_files
        

import argparse
from typing import Any, Dict, Sequence

import yaml


def get_params(params: Sequence[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True, help="")
    args = parser.parse_args(params)

    with open(args.config_yaml, 'r') as fileobj:
        yamlobj: Dict[str, Any] = yaml.safe_load(fileobj)
    for k, v in yamlobj.items():
        setattr(args, k, v)

    return args

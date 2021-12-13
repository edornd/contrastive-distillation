import collections
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict
from uuid import uuid4

from torch.utils.data import DataLoader

import yaml
from pydantic import BaseSettings
from saticl.logging.console import DistributedLogger
from tqdm import tqdm


def current_timestamp() -> str:
    return datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M")


def git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def generate_id() -> str:
    return str(uuid4())


def makedirs(path: str) -> None:
    try:
        return os.makedirs(path)
    except OSError:
        # most likely race conditions among ranks
        pass


def prepare_folder(root_folder: Path, experiment_id: str = ""):
    if isinstance(root_folder, str):
        root_folder = Path(root_folder)
    full_path = root_folder / experiment_id
    if not full_path.exists():
        makedirs(str(full_path))
    return full_path


def prepare_base_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)-24s: %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M',
    )


def get_logger(name: str) -> logging.Logger:
    return DistributedLogger(logging.getLogger(name))


def prepare_file_logging(experiment_folder: Path, filename: str = "output.log") -> None:
    logger = logging.getLogger()
    handler = logging.FileHandler(experiment_folder / filename)
    handler.setLevel(logging.INFO)
    # get the handler from the base handler
    handler.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(handler)


def progressbar(dataloder: DataLoader, epoch: int = 0, stage: str = "train", disable: bool = False):
    pbar = tqdm(dataloder, file=sys.stdout, unit="batch", postfix={"loss": "--"}, disable=disable)
    pbar.set_description(f"Epoch {epoch:<3d} - {stage}")
    return pbar


def store_config(config: BaseSettings, path: Path) -> None:
    with open(str(path), "w") as file:
        yaml.dump(config.dict(), file)


def load_config(path: Path, config_class: Callable) -> BaseSettings:
    assert path.exists(), f"Missing training configuration for path: {path.parent}"
    # load the training configuration
    with open(str(path), "r", encoding="utf-8") as file:
        train_params = yaml.load(file, Loader=yaml.Loader)
    return config_class(**train_params)


def flatten_config(config: dict, parent_key: str = "", separator: str = "/") -> Dict[str, Any]:
    items = []
    for k, v in config.items():
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_config(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))
    return dict(items)

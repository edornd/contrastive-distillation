import logging

import click
from saticl import experiments
from saticl.cli import command
from saticl.config import Configuration
from tqdm.contrib.logging import logging_redirect_tqdm

INFO_FMT = "%(asctime)s - %(name)s  [%(levelname)s]: %(message)s"
DEBUG_FMT = "%(asctime)s - %(pathname)s (%(funcName)s) [%(levelname)s]: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M"


@click.group()
def cli():
    pass


@command(config=Configuration)
def train(config: Configuration):
    log_level = logging.DEBUG if config.debug else logging.INFO
    log_format = DEBUG_FMT if config.debug else INFO_FMT
    logging.basicConfig(level=log_level, format=log_format, datefmt=DATE_FMT)
    logging.root.setLevel(log_level)
    return experiments.train(config)


if __name__ == "__main__":
    cli.add_command(train)
    with logging_redirect_tqdm():
        cli()

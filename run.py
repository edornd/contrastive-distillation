import logging

import click
from saticl import testing, training
from saticl.cli import command
from saticl.config import Configuration, SSLConfiguration, TestConfiguration
from saticl.preproc import isaid, isprs
from saticl.preproc.config import ISAIDPreprocConfig, ISPRSPreprocConfig
from tqdm.contrib.logging import logging_redirect_tqdm

INFO_FMT = "%(asctime)s - %(name)s  [%(levelname)s]: %(message)s"
DEBUG_FMT = "%(asctime)s - %(pathname)s (%(funcName)s) [%(levelname)s]: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M"


def init_logging(log_level: int, log_format: str, date_format: str) -> None:
    logging.basicConfig(level=log_level, format=log_format, datefmt=date_format)
    logging.root.setLevel(log_level)


@click.group()
def cli():
    pass


@command(config=ISPRSPreprocConfig)
def prepare_isprs(config: ISPRSPreprocConfig):
    init_logging(logging.INFO, INFO_FMT, DATE_FMT)
    isprs.main(config)


@command(config=ISAIDPreprocConfig)
def prepare_isaid(config: ISAIDPreprocConfig):
    init_logging(logging.INFO, INFO_FMT, DATE_FMT)
    isaid.main(config)


@command(config=Configuration)
def train(config: Configuration):
    log_level = logging.DEBUG if config.debug else logging.INFO
    log_format = DEBUG_FMT if config.debug else INFO_FMT
    init_logging(log_level, log_format, DATE_FMT)
    return training.train(config)


@command(config=TestConfiguration)
def test(config: TestConfiguration):
    log_level = logging.DEBUG if config.debug else logging.INFO
    log_format = DEBUG_FMT if config.debug else INFO_FMT
    init_logging(log_level, log_format, DATE_FMT)
    return testing.test(config)


if __name__ == "__main__":
    cli.add_command(prepare_isprs)
    cli.add_command(prepare_isaid)
    cli.add_command(train)
    cli.add_command(test)
    with logging_redirect_tqdm():
        cli()

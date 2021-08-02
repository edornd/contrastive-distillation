import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_env(name: str) -> str:
    if (result := os.getenv(name)) is None:
        raise ValueError(f"Missing env variable '{name}'")
    return result


@pytest.fixture(scope="session")
def potsdam_path():
    return Path(get_env("DATA_ROOT_POTSDAM"))

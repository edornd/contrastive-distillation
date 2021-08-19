from abc import ABC
from enum import Enum


class DatasetSplits(str, Enum):
    train = "train"
    valid = "valid"
    test = "test"


class DatasetInfo(ABC):
    """
    Generic class containing information about the dataset.
    """

    def __init__(self) -> None:
        super().__init__()
        self.num_classes = None
        self.label2index = None
        self.index2label = None
        self.image_dims = (512, 512)

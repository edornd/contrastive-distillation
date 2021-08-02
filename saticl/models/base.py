from abc import abstractclassmethod, abstractmethod
from typing import List

from torch import nn

from timm.models.features import FeatureInfo


class Encoder(nn.Module):

    @property
    @abstractmethod
    def feature_info(self) -> FeatureInfo:
        ...


class Decoder(nn.Module):

    @abstractclassmethod
    def required_indices(cls, encoder: str) -> List[int]:
        ...

    # @abstractmethod
    # def output_channels(self) -> List[int]:
    #     ...

    @abstractmethod
    def output(self) -> int:
        ...


class Head(nn.Module):

    @abstractmethod
    def __init__(self, in_channels: int, num_classes: int):
        ...

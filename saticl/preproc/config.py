from enum import Enum
from typing import List, Optional

from pydantic import BaseSettings, Field


class ISPRSDatasets(str, Enum):
    potsdam = "potsdam"
    vaihingen = "vaihingen"


class ISPRSChannels(str, Enum):
    RGB = "rgb"
    RGBIR = "rgbir"
    IRRG = "irrg"


class ISPRSPreprocConfig(BaseSettings):
    dataset: ISPRSDatasets = Field(ISPRSDatasets.potsdam, description="ISPRS dataset to be processed")
    src: str = Field(None, required=True, description="Path to the root folder of Potsdam or Vaihingen")
    dst: str = Field(None, required=True, description="Destination folder where results will be stored")
    channels: ISPRSChannels = Field(ISPRSChannels.RGBIR, description="Which channel combination to process")
    target_size: int = Field(512, description="Output size for the smaller tiles")
    overlap: int = Field(12, description="How many pixels should be overlapped between strides")
    ignore_index: int = Field(0, description="Value to be placed on pixels outside boundaries (to be ignored)")
    use_boundary: bool = Field(True, description="Whether to use labels with or without boundaries (with=no erosion)")
    normalize: bool = Field(False, description="Also normalize images using min-max")
    stats_only: bool = Field(False, description="Just compute dataset statistics, without processing")


class ISAIDPreprocConfig(BaseSettings):
    src: str = Field(None, required=True, description="Path to the root folder of iSAID")
    dst: str = Field(None, required=True, description="Destination folder where results will be stored")
    patch_size: int = Field(512, description="Size of a single tile")
    overlap: int = Field(64, description="How many pixels should be overlapped between strides")
    subsets: List[str] = Field(["train", "valid"], description="Which subset to parse")
    subdir: Optional[str] = Field("images", description="Optional extra subdirectory to include images")

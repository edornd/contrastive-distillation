import glob
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

import tifffile as tif
from saticl.datasets.base import DatasetBase


class ISPRSDataset(DatasetBase):

    _categories = {0: "impervious_surfaces", 1: "building", 2: "low_vegetation", 3: "tree", 4: "car", 5: "clutter"}

    def __init__(self,
                 path: Path,
                 city: str = "potsdam",
                 subset: str = "train",
                 postfix: str = "rgb",
                 channels: int = 3,
                 ignore_index: int = 255,
                 include_dsm: bool = False,
                 transform: Callable = None) -> None:
        super().__init__()
        self._postfix = postfix
        self._channels = channels
        self._ignore_index = ignore_index
        self._include_dsm = include_dsm
        self._name = city
        self._subset = subset
        self.transform = transform
        # gather files to build the list of available pairs
        path = path / city / subset
        self.image_files = sorted(glob.glob(os.path.join(path, self.image_naming())))
        self.label_files = sorted(glob.glob(os.path.join(path, self.label_naming())))
        assert len(self.image_files) > 0, f"No images found, is the given path correct? ({str(path)})"
        assert len(self.image_files) == len(self.label_files), \
            f"Length mismatch between tiles and masks: {len(self.image_files)} != {len(self.label_files)}"
        # check matching sub-tiles
        for image, mask in zip(self.image_files, self.label_files):
            image_tile = "_".join(os.path.basename(image).split("_")[:-1])
            mask_tile = "_".join(os.path.basename(mask).split("_")[:-1])
            assert image_tile == mask_tile, f"image: {image_tile} != mask: {mask_tile}"
        # add the optional digital surface map
        if include_dsm:
            self.dsm_files = sorted(glob.glob(os.path.join(path, self.dsm_naming())))
            assert len(self.image_files) == len(self.dsm_files), "Length mismatch between tiles and DSMs"
            for image, dsm in zip(self.image_files, self.dsm_files):
                image_tile = "_".join(os.path.basename(image).split("_")[:-1])
                dsm_tile = "_".join(os.path.basename(dsm).split("_")[:-1])
                assert image_tile == dsm_tile, f"image: {image_tile} != mask: {dsm_tile}"

    def has_background(self) -> bool:
        return False

    def name(self) -> str:
        return self._name

    def stage(self) -> str:
        return self._subset

    def categories(self) -> Dict[int, str]:
        return self._categories

    def ignore_index(self) -> int:
        return self._ignore_index

    def add_mask(self, mask: List[bool], stage: str = None) -> None:
        assert len(mask) == len(self.image_files), \
            f"Mask is the wrong size! Expected {len(self.image_files)}, got {len(mask)}"
        self.image_files = [x for include, x in zip(mask, self.image_files) if include]
        self.label_files = [x for include, x in zip(mask, self.label_files) if include]
        if self._include_dsm:
            self.dsm_files = [x for include, x in zip(mask, self.dsm_files) if include]
        if stage:
            self._subset = stage

    def image_naming(self) -> str:
        return f"*_{self._postfix}.tif"

    def label_naming(self) -> str:
        return "*_mask.tif"

    def dsm_naming(self) -> str:
        return "*_dsm.tif"

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the image/label pair, with optional augmentations and preprocessing steps.
        Augmentations should be provided for a training dataset, while preprocessing should contain
        the transforms required in both cases (normalizations, ToTensor, ...)

        :param index:   integer pointing to the tile
        :type index:    int
        :return:        image, mask tuple
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        image = tif.imread(self.image_files[index]).astype(np.uint8)
        mask = tif.imread(self.label_files[index]).astype(np.uint8)
        # cut out IR if required
        image = image[:, :, :self._channels]
        # add Digital surface map as extra channel to the image
        if self._include_dsm:
            dsm = tif.imread(self.dsm_files[index]).astype(np.float32)
            image = np.dstack((image, dsm))
        # preprocess if required
        if self.transform is not None:
            pair = self.transform(image=image, mask=mask)
            image = pair.get("image")
            mask = pair.get("mask")
        return image, mask

    def __len__(self) -> int:
        return len(self.image_files)


class PotsdamDataset(ISPRSDataset):

    def __init__(self,
                 path: Path,
                 subset: str,
                 include_dsm: bool = False,
                 transform: Callable = None,
                 channels: int = 3) -> None:
        super().__init__(path,
                         city="potsdam",
                         subset=subset,
                         postfix="rgbir",
                         channels=channels,
                         include_dsm=include_dsm,
                         transform=transform)


class VaihingenDataset(ISPRSDataset):

    def __init__(self,
                 path: Path,
                 subset: str,
                 include_dsm: bool = False,
                 transform: Callable = None,
                 channels: int = 3) -> None:
        super().__init__(path,
                         city="vaihingen",
                         subset=subset,
                         postfix="rgb",
                         channels=channels,
                         include_dsm=include_dsm,
                         transform=transform)

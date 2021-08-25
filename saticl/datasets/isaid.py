import glob
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from PIL import Image
from saticl.datasets.base import DatasetBase


class ISAIDDataset(DatasetBase):

    _categories = {
        0: "background",
        1: "ship",
        2: "storage_tank",
        3: "baseball_diamond",
        4: "tennis_court",
        5: "basketball_court",
        6: "ground_track_field",
        7: "bridge",
        8: "large_vehicle",
        9: "small_vehicle",
        10: "helicopter",
        11: "swimming_pool",
        12: "roundabout",
        13: "soccer_ball_field",
        14: "plane",
        15: "harbor"
    }
    _palette = {
        0: (0, 0, 0),
        1: (0, 0, 63),
        2: (0, 191, 127),
        3: (0, 63, 0),
        4: (0, 63, 127),
        5: (0, 63, 191),
        6: (0, 63, 255),
        7: (0, 127, 63),
        8: (0, 127, 127),
        9: (0, 0, 127),
        10: (0, 0, 191),
        11: (0, 0, 255),
        12: (0, 63, 63),
        13: (0, 127, 191),
        14: (0, 127, 255),
        15: (0, 100, 155)
    }

    def __init__(self,
                 path: Path,
                 subset: str = "train",
                 transform: Callable = None,
                 channels: int = 3,
                 ignore_index: int = 255) -> None:
        super().__init__()
        assert channels == 3, "iSAID only supports RGB"
        self._channels = channels
        self._ignore_index = ignore_index
        self._subset = subset
        self.transform = transform
        # gather files to build the list of available pairs
        path = path / subset
        self.image_files = [f for f in sorted(glob.glob(str(path / "*.png"))) if not f.endswith("_mask.png")]
        self.label_files = sorted(glob.glob(str(path / "*_mask.png")))
        assert len(self.image_files) > 0, f"No images found, is the given path correct? ({str(path)})"
        assert len(self.image_files) == len(self.label_files), \
            f"Length mismatch between tiles and masks: {len(self.image_files)} != {len(self.label_files)}"
        # check matching sub-tiles
        for image, mask in zip(self.image_files, self.label_files):
            image_tile = os.path.basename(image).replace(".png", "")
            mask_tile = os.path.basename(mask).replace("_mask.png", "")
            assert image_tile == mask_tile, f"image: {image_tile} != mask: {mask_tile}"

    def has_background(self) -> bool:
        return True

    def name(self) -> str:
        return self._name

    def stage(self) -> str:
        return self._subset

    def categories(self) -> Dict[int, str]:
        return self._categories

    def palette(self) -> Dict[int, tuple]:
        return self._palette

    def ignore_index(self) -> int:
        return self._ignore_index

    def add_mask(self, mask: List[bool], stage: str = None) -> None:
        assert len(mask) == len(self.image_files), \
            f"Mask is the wrong size! Expected {len(self.image_files)}, got {len(mask)}"
        self.image_files = [x for include, x in zip(mask, self.image_files) if include]
        self.label_files = [x for include, x in zip(mask, self.label_files) if include]
        if stage:
            self._subset = stage

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the image/label pair, with optional augmentations and preprocessing steps.
        Augmentations should be provided for a training dataset, while preprocessing should contain
        the transforms required in both cases (normalizations, ToTensor, ...)

        :param index:   integer pointing to the tile
        :type index:    int
        :return:        image, mask tuple
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        image = np.array(Image.open(self.image_files[index])).astype(np.uint8)
        label = np.array(Image.open(self.label_files[index])).astype(np.uint8)
        # preprocess if required, cast mask to Long for torch's CE
        if self.transform is not None:
            pair = self.transform(image=image, mask=label)
            image = pair.get("image")
            label = pair.get("mask")
        else:
            label = torch.tensor(label)
        return image, label

    def __len__(self) -> int:
        return len(self.image_files)

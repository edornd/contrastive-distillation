import os
from glob import glob
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from PIL import Image
from saticl.datasets.base import DatasetBase


class AgriVisionDataset(DatasetBase):

    _categories = {
        0: "background",
        1: "double_plant",
        2: "drydown",
        3: "endrow",
        4: "nutrient_deficiency",
        5: "planter_skip",
        6: "water",
        7: "waterway",
        8: "weed_cluster",
        9: "storm_damage",    # not evaluated in the actual challenge, discarded
        255: "ignored",    # ignored areas around the fields
    }
    _palette = {
        0: (0, 0, 0),    # background
        1: (23, 190, 207),    # double_plant
        2: (32, 119, 180),    # drydown
        3: (148, 103, 189),    # endrow
        4: (43, 160, 44),    # nutrient_deficiency
        5: (127, 127, 127),    # planter_skip
        6: (214, 39, 40),    # water
        7: (140, 86, 75),    # waterway
        8: (255, 127, 14),    # weed cluster
        255: (0, 0, 0)
    }

    def __init__(self,
                 path: Path,
                 subset: str = "train",
                 transform: Callable = None,
                 channels: int = 3,
                 ignore_index: int = 255):
        assert channels in (3, 4), f"Channel count not supported: {channels}"
        self.subset = subset
        self.transform = transform
        self.channels = channels
        self._ignore_index = ignore_index

        self.rgb_files = sorted(glob(str(path / subset / "images" / "rgb" / "*.jpg")))
        self.nir_files = sorted(glob(str(path / subset / "images" / "nir" / "*.jpg")))

        assert len(self.rgb_files) > 0, "No files found!"
        assert len(self.rgb_files) == len(self.nir_files), \
            f"Length mismatch: RGB: {len(self.rgb_files)} - NIR: {len(self.nir_files)}"
        self.image_names = list()
        for rgb, nir in zip(self.rgb_files, self.nir_files):
            rgb_name = os.path.basename(rgb).replace(".jpg", "")
            nir_name = os.path.basename(nir).replace(".jpg", "")
            assert rgb_name == nir_name, f"ID mismatch - RGB: {rgb_name}, NIR: {nir_name}"
            self.image_names.append(rgb_name)

        self.mask_files = sorted(glob(str(path / subset / "gt" / "*.png")))
        assert len(self.rgb_files) == len(self.mask_files), \
            f"Length mismatch: RGB: {len(self.rgb_files)} - GT: {len(self.mask_files)}"
        for rgb, gt in zip(self.rgb_files, self.mask_files):
            rgb_name = os.path.basename(rgb).replace(".jpg", "")
            gt_name = os.path.basename(gt).replace(".png", "")
            assert rgb_name == gt_name, f"ID mismatch - RGB: {rgb_name}, NIR: {gt_name}"

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, index):
        # read RGB, if also NIR is required, stack it at the bottom
        image = np.array(Image.open(self.rgb_files[index]))
        if self.channels == 4:
            nir = np.array(Image.open(self.nir_files[index]))
            image = np.dstack((image, nir))
        # read the mask, remove any 'storm damage' and put background instead
        label = np.array(Image.open(self.mask_files[index]))
        label[label == 9] = 0
        # if self.size != label.size:
        #     label = F.resize(label, self.size, torchvision.transforms.InterpolationMode.NEAREST)
        #     label = np.array(label)
        if self.transform is not None:
            pair = self.transform(image=image, mask=label)
            image = pair.get("image")
            label = pair.get("mask")
        return image, label

    def add_mask(self, mask: List[bool], stage: str = None) -> None:
        assert len(mask) == len(self.rgb_files), \
            f"Mask is the wrong size! Expected {len(self.rgb_files)}, got {len(mask)}"
        self.rgb_files = [x for include, x in zip(mask, self.rgb_files) if include]
        self.mask_files = [x for include, x in zip(mask, self.mask_files) if include]
        if self.nir_files:
            self.nir_files = [x for include, x in zip(mask, self.nir_files) if include]
        if stage:
            self.subset = stage

    def name(self) -> str:
        return "agrivision"

    def stage(self) -> str:
        return self.subset

    def categories(self) -> Dict[int, str]:
        return self._categories

    def palette(self) -> Dict[int, tuple]:
        return self._palette

    def ignore_index(self) -> int:
        return self._ignore_index

    def has_background(self) -> bool:
        return True

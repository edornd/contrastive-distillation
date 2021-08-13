from typing import Any, Callable, Dict, List

import torch

from saticl.datasets.base import DatasetBase
from saticl.datasets.transforms import SSLTransform


class SSLDataset(DatasetBase):

    def __init__(self, dataset: DatasetBase, transform: SSLTransform) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        image, mask = self.dataset[index]
        # we already passed from albumentations, image is now a tensor channels-first
        # the IR slice needs to be brought back to channels-last, numpy format
        rgb, ir = image[:-1], image[-1]
        rgb_rot, j = self.transform(image=rgb.numpy().transpose(1, 2, 0))
        ir_rot, k = self.transform(image=ir.unsqueeze(-1).numpy())
        z = (k - j) % 4
        y_rot = torch.tensor(z, dtype=torch.long)
        # we wrap everything but the mask into a tuple because of ICL dataset wrapper
        # [3, 512, 512], [1, 521, 512], [3, 512, 512], [1, 512, 512], scalar, [512, 512]
        return (rgb, ir.unsqueeze(0), rgb_rot, ir_rot, y_rot.long()), mask

    def name(self) -> str:
        return self.dataset.name()

    def stage(self) -> str:
        return self.dataset.stage()

    def categories(self) -> Dict[int, str]:
        return self.dataset.categories()

    def palette(self) -> Dict[int, tuple]:
        return self.dataset.palette()

    def add_mask(self, mask: List[bool], stage: str = None) -> None:
        return self.dataset.add_mask(mask, stage)

    def ignore_index(self) -> int:
        return self.dataset.ignore_index()

    def has_background(self) -> bool:
        return self.dataset.has_background()


class RotationDataset(SSLDataset):

    def __init__(self, dataset: DatasetBase, transform: Callable) -> None:
        super().__init__(dataset=dataset, transform=transform)

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return self.dataset.__getattribute__(name)

    def __getitem__(self, index) -> Any:
        image, label = self.dataset[index]
        image_rot = image.permute(1, 2, 0).numpy()
        label_rot = label.numpy()
        pair = self.transform(image=image_rot, mask=label_rot)
        image_rot = pair.get("image")
        label_rot = pair.get("mask")
        return image, image_rot, label, label_rot

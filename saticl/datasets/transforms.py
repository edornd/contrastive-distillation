from typing import Iterable

import torch

import albumentations as alb
from albumentations.pytorch import ToTensorV2


class Denormalize:

    def __init__(self, mean: Iterable[float] = (0.485, 0.456, 0.406), std: Iterable[float] = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        single_image = tensor.ndim == 3
        tensor = tensor.unsqueeze(0) if single_image else tensor
        for t, m, s in zip(tensor, self.mean, self.std):
            t[:3].mul_(s).add_(m)
        # swap from [B, C, H, W] to [B, H, W, C]
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = tensor[0] if single_image else tensor
        return tensor.detach().cpu().numpy()


def adapt_channels(mean: tuple, std: tuple, in_channels: int = 3):
    assert mean is not None and std is not None, "Non-null means and stds required"
    if in_channels > len(mean):
        mean += (mean[0],)
        std += (std[0],)
    return mean, std


def train_transforms(image_size: int,
                     in_channels: int,
                     mean: tuple = (0.485, 0.456, 0.406),
                     std: tuple = (0.229, 0.224, 0.225)):
    # alb.ChannelDropout(p=0.5, fill_value=0),
    # if input channels are 4 and mean and std are for RGB only, copy red for IR
    mean, std = adapt_channels(mean, std, in_channels=in_channels)
    min_crop = image_size // 4 * 3
    max_crop = image_size
    return alb.Compose([
        alb.RandomSizedCrop(min_max_height=(min_crop, max_crop), height=image_size, width=image_size, p=0.8),
        alb.Flip(p=0.5),
        alb.RandomRotate90(p=0.5),
        alb.OneOf([alb.GaussNoise(var_limit=(20, 60), p=0.5),
                   alb.Blur(blur_limit=(3, 5), p=0.5)]),
        alb.RandomBrightnessContrast(p=0.8),
        alb.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def test_transforms(in_channels: int = 3,
                    mean: tuple = (0.485, 0.456, 0.406),
                    std: tuple = (0.229, 0.224, 0.225)) -> alb.Compose:
    mean, std = adapt_channels(mean, std, in_channels=in_channels)
    return alb.Compose([alb.Normalize(mean=mean, std=std), ToTensorV2()])


def inverse_transform(mean: tuple = (0.485, 0.456, 0.406), std: tuple = (0.229, 0.224, 0.225)):
    return Denormalize(mean=mean, std=std)

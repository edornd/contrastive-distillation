from typing import Any, Dict, Iterable

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
        channels = tensor.size(1)
        # slice to support 1 to 3 channels
        means = self.mean[:channels]
        stds = self.std[:channels]
        for t, mean, std in zip(tensor, means, stds):
            t[:3].mul_(std).add_(mean)
        # swap from [B, C, H, W] to [B, H, W, C]
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = tensor[0] if single_image else tensor
        return tensor.detach().cpu().numpy()


class SSLTransform:
    """Wrapper around Albumentations' ReplayCompose, that allows to retrieve the transform
    parameters after applying it.
    """

    def __init__(self, transform: alb.ReplayCompose, track_params: Dict[int, str]) -> None:
        self.track_params = track_params
        self.transform = transform

    def __call__(self, *args, force_apply=False, **data) -> Any:
        data = self.transform(*args, force_apply=force_apply, **data)
        image = data["image"]
        info = data["replay"]["transforms"]
        params = list()
        # iterate target parameters, append what's available
        for trf_index, param_name in self.track_params.items():
            trf_params = info[trf_index].get("params")
            if trf_params and param_name in trf_params:
                params.append(trf_params[param_name])
        return image, *params


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


def ssl_transforms():
    return SSLTransform(alb.ReplayCompose([alb.RandomRotate90(always_apply=True),
                                           ToTensorV2()]),
                        track_params={0: "factor"})

import logging
import random
from typing import Any, Dict, Iterable, List

import torch

import albumentations as alb
from albumentations import functional as func
from albumentations.pytorch import ToTensorV2
from saticl.logging.console import DistributedLogger


LOG = DistributedLogger(logging.getLogger(__name__))


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


class ModalityDropout(alb.ImageOnlyTransform):
    """Randomly Drop RGB or IR in the input Image.

    Args:
        fill_value (int, float): pixel value for the dropped channel.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16, unit32, float32
    """

    def __init__(self,
                 rgb_channels: tuple = (0, 1, 2),
                 ir_channel: int = 3,
                 fill_value: int = 0,
                 always_apply: bool = False,
                 p: float = 0.5):
        super().__init__(always_apply, p)
        self.rgb_channels = rgb_channels
        self.ir_channel = (ir_channel,)
        self.fill_value = fill_value

    def apply(self, img, channels_to_drop=(0,), **params):
        return func.channel_dropout(img, channels_to_drop, self.fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        num_channels = img.shape[-1]
        if len(img.shape) == 2 or num_channels != 4:
            raise NotImplementedError("Images has one channel, mod. dropout requires 4")
        which_modality = random.randint(0, 1)
        channels_to_drop = self.rgb_channels if which_modality else self.ir_channel
        return {"channels_to_drop": channels_to_drop}

    def get_transform_init_args_names(self):
        return ("rgb_channels", "ir_channel", "fill_value")

    @property
    def targets_as_params(self):
        return ["image"]


class ContrastiveTransform:

    def __init__(self, transform_a: alb.Compose, transform_b: alb.Compose = None) -> None:
        if transform_b is None:
            LOG.warn("Second transform is missing, using the first for both images")
            transform_b = transform_a
        self.transform_a = transform_a
        self.transform_b = transform_b

    def __call__(self, image: Any, mask: Any) -> Any:
        pair1 = self.transform_a(image=image, mask=mask)
        pair2 = self.transform_b(image=image, mask=mask)
        x1, y1 = (pair1[k] for k in ("image", "mask"))
        x2, y2 = (pair2[k] for k in ("image", "mask"))
        return x1, x2, y1.long(), y2.long()


def adapt_channels(mean: tuple, std: tuple, in_channels: int = 3, copy_channel: int = 0):
    assert mean is not None and std is not None, "Missing required means and stds"
    replicas = in_channels - len(mean)
    for _ in range(replicas):
        mean += (mean[copy_channel],)
        std += (std[copy_channel],)
    return mean, std


def train_transforms(image_size: int,
                     in_channels: int,
                     mean: tuple = (0.485, 0.456, 0.406),
                     std: tuple = (0.229, 0.224, 0.225),
                     channel_dropout: float = 0.0,
                     modality_dropout: float = 0.0,
                     normalize: bool = True,
                     tensorize: bool = True,
                     compose: bool = True):
    # alb.ChannelDropout(p=0.5, fill_value=0),
    min_crop = image_size // 2
    max_crop = image_size
    transforms = [
        alb.ElasticTransform(alpha=1, sigma=30, alpha_affine=30),
        alb.RandomSizedCrop(min_max_height=(min_crop, max_crop), height=image_size, width=image_size, p=0.8),
        alb.Flip(p=0.5),
        alb.RandomRotate90(p=0.5),
        alb.RandomBrightnessContrast(p=0.5),
    ]
    if channel_dropout > 0:
        transforms.append(alb.ChannelDropout(p=channel_dropout))
    if modality_dropout > 0:
        transforms.append(ModalityDropout(p=modality_dropout))
    if normalize:
        # if input channels are 4 and mean and std are for RGB only, copy red for IR
        mean, std = adapt_channels(mean, std, in_channels=in_channels)
        transforms.append(alb.Normalize(mean=mean, std=std))
    if tensorize:
        transforms.append(ToTensorV2())
    return alb.Compose(transforms) if compose else transforms


def test_transforms(in_channels: int = 3,
                    mean: tuple = (0.485, 0.456, 0.406),
                    std: tuple = (0.229, 0.224, 0.225)) -> alb.Compose:
    mean, std = adapt_channels(mean, std, in_channels=in_channels)
    return alb.Compose([alb.Normalize(mean=mean, std=std), ToTensorV2()])


def inverse_transform(mean: tuple = (0.485, 0.456, 0.406), std: tuple = (0.229, 0.224, 0.225)):
    return Denormalize(mean=mean, std=std)


def geom_transforms(base: List[alb.BasicTransform] = None,
                    in_channels: int = 3,
                    mean: tuple = (0.485, 0.456, 0.406),
                    std: tuple = (0.229, 0.224, 0.225),
                    normalize: bool = True,
                    tensorize: bool = True,
                    compose: bool = True):
    transforms = base or []
    transforms.extend([alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.25, rotate_limit=90, always_apply=True)])
    if normalize:
        mean, std = adapt_channels(mean, std, in_channels=in_channels)
        transforms.append(alb.Normalize(mean=mean, std=std))
    if tensorize:
        transforms.append(ToTensorV2())
    return alb.Compose(transforms) if compose else transforms


def ssl_transforms():
    return SSLTransform(alb.ReplayCompose([alb.RandomRotate90(always_apply=True),
                                           ToTensorV2()]),
                        track_params={0: "factor"})

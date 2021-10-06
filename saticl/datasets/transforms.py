from typing import List

import albumentations as alb
from albumentations.pytorch import ToTensorV2
from saticl.config import AugInvarianceConfig
from saticl.transforms import (
    ColorJitter,
    Compose,
    Denormalize,
    FixedRotation,
    ModalityDropout,
    RandomFlip,
    RandomRotation,
    SSLTransform,
)


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
    # Remove for actual training
    # return alb.Compose([ToTensorV2()])
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


def invariance_transforms(config: AugInvarianceConfig):
    transforms = []
    if config.flip:
        transforms.append(RandomFlip(p=0.5))
    if config.fixed_angles:
        transforms.append(FixedRotation(p=1.0))
    else:
        transforms.append(RandomRotation(p=1.0, angles=360))
    if config.color:
        transforms.append(ColorJitter(p=1.0, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0))
    return Compose(transforms)


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

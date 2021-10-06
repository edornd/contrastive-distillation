import logging
import numbers
import random
from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

import albumentations as alb
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as func


LOG = logging.getLogger(__name__)


class Transform:
    """Generic transform class, base for all others.
    Why this? Because I needed to apply transforms to tensors, both image and label.
    """

    def __init__(self, p: float) -> None:
        self.p = p

    @abstractmethod
    def __call__(self,
                 images: Sequence[torch.Tensor],
                 label: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError("Subclass this!")

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomFlip(Transform):
    """Horizontally flip the given images randomly with a given probability.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p)

    def __call__(self, *images: torch.Tensor, label: Optional[torch.Tensor] = None):
        if random.random() < self.p:
            flip_fn = func.hflip if random.randint(0, 1) else func.vflip
            images = [flip_fn(img) for img in images]
            if label is not None:
                return images, flip_fn(label)
            return images
        if label is not None:
            return images, label
        return images


class FixedRotation(Transform):
    """Rotates the given images and optional label by multiples of 90 degrees.
    Useful to avoid empty pixels.
    """

    def __init__(self,
                 p: float = 0.5,
                 angles: Sequence[int] = (0, 90, 180, 270),
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR,
                 mirror: bool = True):
        # always applied,
        super().__init__(p)
        reverse = [-a for a in angles] if mirror else []
        self.angles = list(angles) + reverse
        self.interp = interpolation

    def __call__(self, *images: torch.Tensor, label: torch.Tensor = None):
        if random.random() < self.p:
            angle = random.choice(self.angles)
            images = [func.rotate(img, angle, interpolation=self.interp) for img in images]
            if label is not None:
                label = func.rotate(label, angle, interpolation=InterpolationMode.NEAREST)
                return images, label
            return images
        else:
            if label is not None:
                return images, label
            return images


class RandomRotation(Transform):
    """Rotates the given tensors by a random angle, in the given range.
    """

    def __init__(self,
                 p: float = 0.5,
                 angles: Union[int, tuple] = 360,
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR):
        super().__init__(p)
        if isinstance(angles, int):
            angles = (-angles, angles)
        self.min, self.max = angles
        self.interp = interpolation

    def __call__(self, *images: torch.Tensor, label: torch.Tensor = None):
        if random.random() < self.p:
            angle = random.randint(self.min, self.max)
            images = [func.rotate(img, angle, interpolation=self.interp, fill=0) for img in images]
            if label is not None:
                label = func.rotate(label, angle, interpolation=InterpolationMode.NEAREST, fill=255)
                return images, label
            return images
        else:
            if label is not None:
                return images, label
            return images


class ColorJitter(Transform):
    """Randomly change the brightness, contrast and saturation of an image.
    """
    def __init__(self,
                 p: float = 0.5,
                 brightness: Union[float, tuple] = 0,
                 contrast: Union[float, tuple] = 0,
                 saturation: Union[float, tuple] = 0,
                 hue: Union[float, tuple] = 0):
        super().__init__(p)
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self,
                     value: Any,
                     name: str,
                     center: float = 1,
                     bound: tuple = (0, float('inf')),
                     clip_first_on_zero: bool = True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))
        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def get_params(self):
        """Get a randomized transform to be applied on image.
        Arguments are the same as of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if self.brightness is not None:
            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            transforms.append(lambda img: func.adjust_brightness(img, brightness_factor))

        if self.contrast is not None:
            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            transforms.append(lambda img: func.adjust_contrast(img, contrast_factor))

        if self.saturation is not None:
            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            transforms.append(lambda img: func.adjust_saturation(img, saturation_factor))

        if self.hue is not None:
            hue_factor = random.uniform(self.hue[0], self.hue[1])
            transforms.append(lambda img: func.adjust_hue(img, hue_factor))

        random.shuffle(transforms)
        return transforms

    def __call__(self, *images: torch.Tensor, label: torch.Tensor = None):
        if random.random() < self.p:
            transforms = self.get_params()
            transformed = []
            for image in images:
                # only applies to RGB images, the rest leave it as it is
                if image.shape[1] != 3:
                    transformed.append(image)
                    continue
                # apply list of color transformations to RGB
                for transform in transforms:
                    image = transform(image)
                transformed.append(image)
        else:
            transformed = images
        if label is not None:
            return transformed, label
        return transformed


class Compose(Transform):
    """Composes several transforms together.
    """

    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = transforms

    def __call__(self, *images: torch.Tensor, label: Optional[torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if label is not None:
            for t in self.transforms:
                images, label = t(*images, label=label)
            return images, label
        else:
            for t in self.transforms:
                images = t(*images)
            return images

    def __repr__(self):
        return "Compose[" + ",".join([str(t) for t in self.transforms]) + "]"


class Denormalize:

    def __init__(self, mean: Sequence[float] = (0.485, 0.456, 0.406), std: Sequence[float] = (0.229, 0.224, 0.225)):
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
            t[:3].mul_(std - 1e-6).add_(mean)
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

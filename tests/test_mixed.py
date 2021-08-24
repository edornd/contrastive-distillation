import numpy as np
import torch

from PIL import Image
from saticl.datasets.transforms import train_transforms
from saticl.transforms import Denormalize
from saticl.utils.ml import make_grid


def test_denormalize():
    random = np.random.randint(0, 255, size=(512, 512, 3)).astype(np.uint8)
    transform = train_transforms(image_size=512, in_channels=3)
    img = transform(image=random)["image"]
    assert type(img) == torch.Tensor
    assert img.shape == (3, 512, 512)
    # now denormalize
    img = Denormalize()(img)
    assert type(img) == np.ndarray
    assert img.shape == (512, 512, 3)
    assert img.min() >= 0
    assert img.max() <= 255


def test_make_grid():
    img = np.random.randint(0, 255, size=(512, 512, 3)).astype(np.uint8)
    true_mask = np.random.randint(0, 255, size=(512, 512, 3)).astype(np.uint8)
    pred_mask = np.random.randint(0, 255, size=(512, 512, 3)).astype(np.uint8)
    merged = make_grid(img, true_mask, pred_mask)
    image = Image.fromarray(merged)
    assert image.size == (512 * 3, 512)

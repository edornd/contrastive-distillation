import logging
from pathlib import Path

import numpy as np

import albumentations as alb
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from saticl.datasets import create_dataset
from saticl.datasets.transforms import (
    ContrastiveTransform,
    Denormalize,
    ModalityDropout,
    geom_transforms,
    ssl_transforms,
    train_transforms,
)
from saticl.datasets.wrappers import ContrastiveDataset, SSLDataset
from saticl.utils.ml import seed_everything

LOG = logging.getLogger(__name__)


def test_modality_dropout(potsdam_path: Path):
    # instantiate transforms for training
    seed_everything(1337)
    mean = (0.485, 0.456, 0.406, 0.485)
    std = (0.229, 0.224, 0.225, 0.229),
    train_transform = alb.Compose([ModalityDropout(p=0.5), alb.Normalize(mean=mean, std=std), ToTensorV2()])
    dataset = create_dataset("potsdam", path=potsdam_path, subset="train", transform=train_transform, channels=4)

    denorm = Denormalize()
    n_images = 16
    values = np.random.choice(len(dataset), size=n_images, replace=False)
    samples = [dataset.__getitem__(i) for i in values]
    nrows = int(np.sqrt(n_images))
    ncols = nrows * 2    # account for two IR, same and rotated
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))

    for r in range(nrows):
        for c in range(ncols // 2):
            index = r * nrows + c
            # retrieve images and denormalize them, channels IRRG
            image, mask = samples[index]
            img = denorm(image[[3, 0, 1]])
            # plot images
            axes[r, c * 2].imshow(img)
            axes[r, c * 2].set_title("IRRG")
            axes[r, c * 2 + 1].imshow(mask)
            axes[r, c * 2 + 1].set_title("mask")

    plt.tight_layout()
    plt.savefig("data/modality_dropout.png")
    plt.close(fig)


def test_ssl_augmentations(potsdam_path: Path):
    # instantiate transforms for training
    seed_everything(1337)
    train_transform = train_transforms(image_size=512, in_channels=4)
    # eval_transform = test_transforms(in_channels=4)
    # create the train dataset, then split or create the ad hoc validation set
    ssl_transform = ssl_transforms()
    dataset = create_dataset("potsdam", path=potsdam_path, subset="train", transform=train_transform, channels=4)
    dataset = SSLDataset(dataset, transform=ssl_transform)

    denorm = Denormalize()
    n_images = 16
    values = np.random.choice(len(dataset), size=n_images, replace=False)
    samples = [dataset.__getitem__(i) for i in values]

    nrows = int(np.sqrt(n_images))
    ncols = nrows * 3    # account for two IR, same and rotated
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 10))

    for r in range(nrows):
        for c in range(ncols // 3):
            index = r * nrows + c
            # retrieve images and denormalize them
            (rgb, ir, ir_rot, y_rot), _ = samples[index]
            img = denorm(rgb)
            ir = denorm(ir)
            ir_rot = denorm(ir_rot)
            # plot images
            axes[r, c * 3].imshow(img)
            axes[r, c * 3].set_title("RGB")
            axes[r, c * 3 + 1].imshow(ir)
            axes[r, c * 3 + 1].set_title("IR")
            axes[r, c * 3 + 2].imshow(ir_rot)
            axes[r, c * 3 + 2].set_title(f"IR rot({y_rot.item()})")
            # set titles

    plt.tight_layout()
    plt.savefig("data/ssl_augs.png")
    plt.close(fig)


def test_contrastive_augmentations_potsdam(potsdam_path: Path):
    # instantiate transforms for training
    seed_everything(1337)
    # create the train dataset, then split or create the ad hoc validation set
    train_transform = train_transforms(image_size=512,
                                       in_channels=4,
                                       channel_dropout=0.5,
                                       normalize=False,
                                       tensorize=False)
    train_dataset = create_dataset("potsdam", path=potsdam_path, subset="train", transform=train_transform, channels=4)
    extra_trf = geom_transforms(in_channels=4, normalize=True, tensorize=True)
    dataset = ContrastiveDataset(train_dataset, transform=ContrastiveTransform(extra_trf, extra_trf))

    denorm = Denormalize()
    n_images = 16
    values = np.random.choice(len(dataset), size=n_images, replace=False)
    samples = [dataset.__getitem__(i) for i in values]
    nrows = int(np.sqrt(n_images))
    width = 4
    ncols = nrows * width    # account for two IR, same and rotated
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * width, 10))

    for r in range(nrows):
        for c in range(ncols // width):
            index = r * nrows + c
            # retrieve images and denormalize them, channels IRRG
            img1, img2, mask1, mask2 = samples[index]
            img1 = denorm(img1[[3, 0, 1]])
            img2 = denorm(img2[[3, 0, 1]])
            # plot images
            axes[r, c * width].imshow(img1)
            axes[r, c * width].set_title("x1")
            axes[r, c * width + 1].imshow(mask1)
            axes[r, c * width + 1].set_title("y1")
            axes[r, c * width + 2].imshow(img2)
            axes[r, c * width + 2].set_title("x2")
            axes[r, c * width + 3].imshow(mask2)
            axes[r, c * width + 3].set_title("y2")

    plt.tight_layout()
    plt.savefig("data/contrastive_set.png")
    plt.close(fig)

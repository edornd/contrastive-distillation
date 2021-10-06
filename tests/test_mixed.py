import logging
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import albumentations as alb
from albumentations.pytorch import ToTensorV2
from inplace_abn import InPlaceABNSync
from matplotlib import pyplot as plt
from PIL import Image
from saticl.config import AugInvarianceConfig
from saticl.datasets import create_dataset
from saticl.datasets.transforms import invariance_transforms, train_transforms
from saticl.losses.regularization import AugmentationInvariance
from saticl.models import create_decoder, create_encoder
from saticl.models.icl import ICLSegmenter
from saticl.tasks import Task
from saticl.transforms import Denormalize
from saticl.utils.ml import make_grid, seed_everything, seed_worker

LOG = logging.getLogger(__name__)


def prepare_model(task: Task):
    act_layer = torch.nn.Identity
    norm_layer = partial(InPlaceABNSync, activation="leaky_relu", activation_param=0.01)

    encoder = create_encoder(name="tresnet_m",
                             decoder="unet",
                             pretrained=False,
                             freeze=False,
                             output_stride=None,
                             act_layer=act_layer,
                             norm_layer=norm_layer,
                             channels=3)
    decoder = create_decoder(name="unet", feature_info=encoder.feature_info, act_layer=act_layer, norm_layer=norm_layer)
    return ICLSegmenter(encoder, decoder, classes=task.num_classes_per_task(), return_features=True)


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


def test_rot_invariance(potsdam_path: Path, checkpoint_path: Path):
    # instantiate transforms for training
    seed_everything(1337)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225),
    train_transform = alb.Compose([alb.Normalize(mean=mean, std=std), ToTensorV2()])
    dataset = create_dataset("potsdam", path=potsdam_path, subset="train", transform=train_transform, channels=3)
    task = Task(dataset="potsdam", name="6s", step=2, add_background=True)
    ckpt_path = checkpoint_path / "models" / f"task6s_step-{task.step}.pth"

    train_loader = DataLoader(dataset=dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=1,
                              worker_init_fn=seed_worker,
                              drop_last=True)
    model = prepare_model(task)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    x, y = next(iter(train_loader))
    aug_config = AugInvarianceConfig(factor=0.1, factor_icl=0.1, flip=True, fixed_angles=True)
    auginv = AugmentationInvariance(transform=invariance_transforms(aug_config))

    with torch.no_grad():
        _, features = model(x)
        (x_rot, f_rot), y_rot = auginv.apply_transform(x, features, label=y)
        _, r_features = model(x_rot)
        LOG.info("%s - %s", str(x.shape), str(x_rot.shape))
        LOG.info("%s - %s - %s", str(features.shape), str(r_features.shape), str(f_rot.shape))

        tensors = {"features": features, "features_rot1": r_features, "features_rot2": f_rot}

        for name, features in tensors.items():
            nr, nc = 8, 8
            f, axes = plt.subplots(nrows=nr, ncols=nc, sharex=True, sharey=True, figsize=(12, 12))
            for x in range(nr):
                for y in range(nc):
                    index = x * nr + y
                    axes[x, y].imshow(features.squeeze(0)[index].cpu().numpy())
                    axes[x, y].axis("off")
            plt.tight_layout()
            plt.subplots_adjust(wspace=.05, hspace=.05)
            plt.savefig(f"data/{name}.png")
            plt.close(f)

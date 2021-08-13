import logging

import numpy as np
import torch

import albumentations as alb
from albumentations.pytorch import ToTensorV2
from saticl.config import ModelConfig
from saticl.losses import FocalTverskyLoss, UnbiasedFTLoss
from saticl.losses.regularization import AugmentationInvariance, MultiModalScaling
from saticl.prepare import create_multi_encoder

LOG = logging.getLogger(__name__)


def test_tversky_loss():
    # emulate a logits output
    torch.manual_seed(42)
    y_pred = torch.rand((2, 5, 256, 256)) * 5
    y_true = torch.randint(0, 4, (2, 256, 256))
    # compute loss
    criterion = FocalTverskyLoss(ignore_index=255)
    loss = criterion(y_pred, y_true)
    LOG.info(loss)
    assert loss >= 0 and loss <= 1


def test_tversky_loss_unbiased():
    # emulate a logits output
    torch.manual_seed(42)
    y_pred = torch.rand((2, 5, 256, 256)) * 5
    y_true = torch.randint(0, 4, (2, 256, 256))
    # compute loss
    criterion = UnbiasedFTLoss(old_class_count=3, ignore_index=255)
    loss = criterion(y_pred, y_true)
    LOG.info(loss)
    assert loss >= 0 and loss <= 1


def test_multimodal_scaling():
    cfg = ModelConfig(encoder="resnet34", decoder="unet", pretrained=True)
    encoder = create_multi_encoder("resnet34", "resnet34", config=cfg, return_features=True)
    inputs = torch.rand((2, 4, 256, 256))
    (rgb, ir), out = encoder(inputs)
    # create the regularizer and compute the value
    reg = MultiModalScaling(reduction="mean")
    result = reg(rgb, ir)
    LOG.info("Regularization: %s", str(result))


def test_rotation_invariance():
    cfg = ModelConfig(encoder="resnet34", decoder="unet", pretrained=True)
    encoder = create_multi_encoder("resnet34", "resnet34", config=cfg, return_features=True)

    trf = alb.Compose(
        [alb.Normalize(mean=(0.485, 0.456, 0.406, 0.485), std=(0.229, 0.224, 0.225, 0.229)),
         ToTensorV2()])
    trf_rot = alb.Compose([
        alb.Normalize(mean=(0.485, 0.456, 0.406, 0.485), std=(0.229, 0.224, 0.225, 0.229)),
        alb.RandomRotate90(always_apply=True),
        ToTensorV2()
    ])
    # create a sample image in uint8 format, simulate batch with unsqueeze
    image = np.random.randint(0, 255, (256, 256, 4))
    inputs = trf(image=image)["image"].unsqueeze(0)
    rotated = trf_rot(image=image)["image"].unsqueeze(0)
    _, out1 = encoder(inputs)
    _, out2 = encoder(rotated)
    # create the regularizer and compute the value
    reg = AugmentationInvariance()
    result = reg(out1[-1], out2[-1])
    LOG.info("Rot invariance: %s", str(result))

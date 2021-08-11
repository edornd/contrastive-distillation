from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn

from saticl.config import Configuration, Metrics, ModelConfig, SSLConfiguration
from saticl.datasets import create_dataset
from saticl.datasets.icl import ICLDataset
from saticl.datasets.transforms import test_transforms, train_transforms
from saticl.metrics import F1Score, IoU, Metric, lenient_argmax
from saticl.models import create_decoder, create_encoder
from saticl.models.encoders import MultiEncoder
from saticl.models.icl import ICLSegmenter
from saticl.models.ssl import PretextClassifier
from saticl.tasks import Task
from saticl.utils.common import get_logger
from saticl.utils.ml import mask_set


LOG = get_logger(__name__)


def prepare_dataset(config: Configuration) -> ICLDataset:
    # instantiate transforms for training and evaluation
    data_root = Path(config.data_root)
    train_transform = train_transforms(image_size=config.image_size,
                                       in_channels=config.in_channels,
                                       channel_transforms=config.channel_drop,
                                       modality_transforms=config.modality_drop)
    eval_transform = test_transforms(in_channels=config.in_channels)
    LOG.debug("Train transforms: %s", str(train_transform))
    LOG.debug("Eval. transforms: %s", str(eval_transform))
    # create the train dataset, then split or create the ad hoc validation set
    train_dataset = create_dataset(config.dataset,
                                   path=data_root,
                                   subset="train",
                                   transform=train_transform,
                                   channels=config.in_channels)
    if not config.has_val:
        # Many datasets do not have a validation set or a test set with annotations.
        # In the first case (like this one), a portion of training is reserved for validation
        # In the second case, the validation becomes testing and the training is again split randomly
        train_mask, val_mask, _ = mask_set(len(train_dataset), val_size=config.val_size, test_size=0.0)
        LOG.debug("Creating val. set from training, split: %d - %d", len(train_mask), len(val_mask))
        val_dataset = create_dataset(config.dataset,
                                     path=data_root,
                                     subset="train",
                                     transform=eval_transform,
                                     channels=config.in_channels)
        train_dataset.add_mask(train_mask)
        val_dataset.add_mask(val_mask, stage="valid")
    else:
        # When the validation set is present, by all means use it
        val_dataset = create_dataset(config.dataset,
                                     path=data_root,
                                     subset="valid",
                                     transform=eval_transform,
                                     channels=config.in_channels)
    return train_dataset, val_dataset


def create_multi_encoder(name_rgb: str, name_ir: str, config: ModelConfig) -> MultiEncoder:
    encoder_a = create_encoder(name=name_rgb,
                               decoder=config.decoder,
                               pretrained=config.pretrained,
                               freeze=config.freeze,
                               output_stride=config.output_stride,
                               act_layer=config.act,
                               norm_layer=config.norm,
                               channels=3)
    encoder_b = create_encoder(name=name_ir,
                               decoder=config.decoder,
                               pretrained=config.pretrained,
                               freeze=config.freeze,
                               output_stride=config.output_stride,
                               act_layer=config.act,
                               norm_layer=config.norm,
                               channels=1)
    return MultiEncoder(encoder_a, encoder_b, act_layer=config.act, norm_layer=config.norm)


def prepare_model(config: Configuration, task: Task) -> nn.Module:
    cfg = config.model
    # instead of creating a new var, encoder is exploited for different purposes
    # we expect a single encoder name, or a comma-separated list of names, one for RGB and one for IR
    # e.g. valid examples: 'tresnet_m' - 'resnet34,resnet34'
    enc_names = cfg.encoder.split(",")
    if len(enc_names) == 1:
        encoder = create_encoder(name=enc_names[0],
                                 decoder=cfg.decoder,
                                 pretrained=cfg.pretrained,
                                 freeze=cfg.freeze,
                                 output_stride=cfg.output_stride,
                                 act_layer=cfg.act,
                                 norm_layer=cfg.norm,
                                 channels=config.in_channels)
    elif len(enc_names) == 2:
        encoder = create_multi_encoder(name_rgb=enc_names[0], name_ir=enc_names[1], config=cfg)
    else:
        NotImplementedError(f"The provided list of encoders is not supported: {cfg.encoder}")
    # create decoder: always uses the RGB encoder as reference
    # SSMA blocks in the multiencoder merge the features into a number of channels equal to RGB for each layer.
    decoder = create_decoder(name=cfg.decoder,
                             feature_info=encoder.feature_info,
                             act_layer=cfg.act,
                             norm_layer=cfg.norm)
    # extract intermediate features when encoder KD is required
    extract_features = config.kd.encoder_factor > 0
    icl_model = ICLSegmenter(encoder, decoder, classes=task.num_classes_per_task(), return_features=extract_features)
    return icl_model


def prepare_model_ssl(config: SSLConfiguration, task: Task) -> Tuple[nn.Module, nn.Module]:
    cfg = config.model
    encoder = create_multi_encoder(name_rgb=cfg.encoder, name_ir=cfg.encoder_ir, config=cfg)
    decoder = create_decoder(name=cfg.decoder, feature_info=encoder.feature_info, act_layer=cfg.act, norm_layer=cfg.norm)
    # extract intermediate features when encoder KD is required
    extract_features = config.kd.encoder_factor > 0
    icl_model = ICLSegmenter(encoder, decoder, classes=task.num_classes_per_task(), return_features=extract_features)
    # create also a classifier for the pretext class, using the same encoders
    ssl_model = PretextClassifier(encoder.encoder_rgb, encoder.encoder_ir,
                                  act_layer=cfg.act,
                                  norm_layer=cfg.norm,
                                  num_classes=cfg.pretext_classes)
    return icl_model, ssl_model


def prepare_metrics(task: Task, device: torch.device) -> Tuple[dict, dict]:
    # prepare metrics
    num_classes = task.old_class_count() + task.current_class_count()
    t_metrics = (Metrics.f1, Metrics.iou)
    v_metrics = (m for m in Metrics)
    train_metrics = {e.name: e.value(num_classes=num_classes, device=device) for e in t_metrics}
    valid_metrics = {e.name: e.value(num_classes=num_classes, device=device) for e in v_metrics}
    valid_metrics.update(dict(class_iou=IoU(num_classes=num_classes, reduction=None, device=device),
                              class_f1=F1Score(num_classes=num_classes, reduction=None, device=device)))
    LOG.debug("Train metrics: %s", str(list(train_metrics.keys())))
    LOG.debug("Eval. metrics: %s", str(list(valid_metrics.keys())))
    return train_metrics, valid_metrics


def prepare_metrics_ssl(num_classes: int, device: torch.device) -> Dict[str, Metric]:
    metrics = (Metrics.f1,)
    transform = partial(lenient_argmax, ndims=1)
    ssl_metrics = {m.name: m.value(num_classes=num_classes, device=device, transform=transform) for m in metrics}
    LOG.debug("Pretext task metrics: %s", str(list(ssl_metrics.keys())))
    return ssl_metrics

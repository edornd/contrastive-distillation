import logging
from functools import partial

import torch
from torch import nn

from inplace_abn.abn import InPlaceABN
from saticl.models import create_decoder, create_encoder
from saticl.models.icl import ICLSegmenter
from saticl.tasks import Task

LOG = logging.getLogger(__name__)


def test_encoder_resnet_unet():
    model = create_encoder(name="resnet50",
                           decoder="unet",
                           pretrained=False,
                           output_stride=16,
                           freeze=False,
                           act_layer=nn.ReLU,
                           norm_layer=nn.BatchNorm2d)
    channels = model.feature_info.channels()
    reduction = model.feature_info.reduction()
    LOG.debug("channels: %s", str(channels))
    LOG.debug("reduct.:  %s", str(reduction))
    out = model(torch.rand(1, 3, 256, 256))
    # check that the output is indeed a list of 5 tensors
    assert len(out) == 5
    # check that the channels and reductions are consistent
    for y, c, r in zip(out, channels, reduction):
        LOG.debug("%s", str(y.shape))
        assert y.shape[1] == c
        assert r == (256 // y.shape[-1])


def test_encoder_resnet_deeplab():
    model = create_encoder(name="resnet50",
                           decoder="deeplabv3",
                           pretrained=False,
                           output_stride=16,
                           freeze=False,
                           act_layer=nn.ReLU,
                           norm_layer=nn.BatchNorm2d)
    channels = model.feature_info.channels()
    reduction = model.feature_info.reduction()
    LOG.debug("channels: %s", str(channels))
    LOG.debug("reduct.:  %s", str(reduction))
    out = model(torch.rand(1, 3, 256, 256))
    # check that the output is indeed a list of 1 tensor
    assert len(out) == 1
    # check that the channels and reductions are consistent
    for y, c, r in zip(out, channels, reduction):
        LOG.debug("%s", str(y.shape))
        assert y.shape[1] == c
        assert r == (256 // y.shape[-1])


def test_encoder_resnet_deeplabv3plus():
    model = create_encoder(name="resnet50",
                           decoder="deeplabv3p",
                           pretrained=False,
                           output_stride=16,
                           freeze=False,
                           act_layer=nn.ReLU,
                           norm_layer=nn.BatchNorm2d)
    channels = model.feature_info.channels()
    reduction = model.feature_info.reduction()
    LOG.debug("channels: %s", str(channels))
    LOG.debug("reduct.:  %s", str(reduction))
    out = model(torch.rand(1, 3, 256, 256))
    # check that the output is indeed a list of 2 tensors
    assert len(out) == 2
    # check that the channels and reductions are consistent
    for y, c, r in zip(out, channels, reduction):
        LOG.debug("%s", str(y.shape))
        assert y.shape[1] == c
        assert r == (256 // y.shape[-1])


def test_encoder_tresnet_unet():
    model = create_encoder(name="tresnet_m",
                           decoder="unet",
                           pretrained=False,
                           output_stride=16,
                           freeze=False,
                           act_layer=nn.ReLU,
                           norm_layer=nn.BatchNorm2d)
    channels = model.feature_info.channels()
    reduction = model.feature_info.reduction()
    LOG.debug("channels: %s", str(channels))
    LOG.debug("reduct.:  %s", str(reduction))
    out = model(torch.rand(1, 3, 256, 256))
    # check that the output is indeed a list of 5 tensors
    assert len(out) == 4
    # check that the channels and reductions are consistent
    for y, c, r in zip(out, channels, reduction):
        LOG.debug("%s", str(y.shape))
        assert y.shape[1] == c
        assert r == (256 // y.shape[-1])


def test_decoder_resnet_unet():
    enc = create_encoder(name="resnet50",
                         decoder="unet",
                         pretrained=False,
                         output_stride=16,
                         freeze=False,
                         act_layer=nn.ReLU,
                         norm_layer=nn.BatchNorm2d)
    dec = create_decoder(name="unet", feature_info=enc.feature_info, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d)
    inp = torch.rand(1, 3, 256, 256)
    out = enc(inp)
    assert len(out) == 5
    out = dec(out)
    assert out.shape == (1, dec.output(), 256, 256)
    LOG.debug("shape:  %s", str(out.shape))


def test_decoder_tresnet_unet():
    enc = create_encoder(name="tresnet_m",
                         decoder="unet",
                         pretrained=False,
                         output_stride=16,
                         freeze=False,
                         act_layer=nn.ReLU,
                         norm_layer=nn.BatchNorm2d)
    dec = create_decoder(name="unet", feature_info=enc.feature_info, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d)
    inp = torch.rand(1, 3, 256, 256)
    out = enc(inp)
    assert len(out) == 4
    out = dec(out)
    assert out.shape == (1, dec.output(), 256, 256)
    LOG.debug("shape:  %s", str(out.shape))


def test_decoder_tresnet_unet_iabn():
    enc = create_encoder(name="tresnet_m",
                         decoder="unet",
                         pretrained=False,
                         output_stride=16,
                         freeze=False,
                         act_layer=nn.ReLU,
                         norm_layer=nn.BatchNorm2d)
    iabn = partial(InPlaceABN, activation="leaky_relu", activation_param=0.01)
    dec = create_decoder(name="unet", feature_info=enc.feature_info, act_layer=nn.Identity, norm_layer=iabn)
    inp = torch.rand(1, 3, 256, 256)
    out = enc(inp)
    assert len(out) == 4
    out = dec(out)
    assert out.shape == (1, dec.output(), 256, 256)
    LOG.debug("shape:  %s", str(out.shape))


def test_icl_model_resnet_unet_step_0():
    enc = create_encoder(name="resnet50",
                         decoder="unet",
                         pretrained=False,
                         output_stride=16,
                         freeze=False,
                         act_layer=nn.ReLU,
                         norm_layer=nn.BatchNorm2d)
    dec = create_decoder(name="unet", feature_info=enc.feature_info, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d)

    # extract intermediate features when encoder KD is required
    extract_features = True
    task = Task(name="222a", dataset="potsdam", step=0)
    icl_model = ICLSegmenter(enc, dec, classes=task.num_classes_per_task(), return_features=extract_features)
    # we do not expect old classifiers at step 1
    assert len(icl_model.classifiers) == 1
    inp = torch.rand(1, 3, 256, 256)
    # should be composed of two pieces given extract features = True
    out = icl_model(inp)
    assert len(out) == 2
    out, (enc_features, dec_features) = out
    assert out.shape == (1, 2, 256, 256)
    assert dec_features.shape == (1, dec.output(), 256, 256)
    assert len(enc_features) == 5


def test_icl_model_resnet_unet_step_1():
    enc = create_encoder(name="resnet50",
                         decoder="unet",
                         pretrained=False,
                         output_stride=16,
                         freeze=False,
                         act_layer=nn.ReLU,
                         norm_layer=nn.BatchNorm2d)
    dec = create_decoder(name="unet", feature_info=enc.feature_info, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d)

    # extract intermediate features when encoder KD is required
    extract_features = True
    task = Task(name="222a", dataset="potsdam", step=1, add_background=True)
    icl_model = ICLSegmenter(enc, dec, classes=task.num_classes_per_task(), return_features=extract_features)
    # we do not expect old classifiers at step 1
    assert len(icl_model.classifiers) == 2
    icl_model.init_classifier()
    # assert that the init worked
    LOG.info("old weight shape: %s", str(icl_model.classifiers[0].out.weight.shape))
    LOG.info("new weight shape: %s", str(icl_model.classifiers[1].out.weight.shape))
    # assert that both new classes got the new fancy weights
    assert torch.all(icl_model.classifiers[0].out.weight[0] == icl_model.classifiers[1].out.weight[0])
    assert torch.all(icl_model.classifiers[0].out.weight[0] == icl_model.classifiers[1].out.weight[1])
    inp = torch.rand(1, 3, 256, 256)
    # should be composed of two pieces given extract features = True
    out = icl_model(inp)
    assert len(out) == 2
    out, (enc_features, dec_features) = out
    # output dimension should include bkgr + 2 classes + 2 classes
    assert out.shape == (1, 5, 256, 256)
    assert dec_features.shape == (1, dec.output(), 256, 256)
    assert len(enc_features) == 5

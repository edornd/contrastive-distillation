from typing import Type

from torch import nn

import timm
from saticl.models.base import Decoder, Encoder
from saticl.models.decoders import available_decoders
from saticl.utils.ml import expand_input
from timm.models.features import FeatureInfo


def filter_encoder_args(encoder: str, **kwargs: dict) -> dict:
    exclude = list()
    if encoder.startswith("tresnet"):
        exclude = ["norm_layer", "act_layer", "output_stride"]
    if encoder.startswith("efficientnet"):
        exclude = ["norm_layer", "act_layer"]
    for arg in exclude:
        kwargs.pop(arg, None)
    return kwargs


def create_encoder(name: str,
                   decoder: str,
                   pretrained: bool,
                   freeze: bool,
                   output_stride: int,
                   act_layer: Type[nn.Module],
                   norm_layer: Type[nn.Module],
                   channels: int = 3) -> Encoder:
    # assert that the encoder exists or is among custom ones
    assert timm.list_models(name), f"Encoder '{name}' not implemented"
    assert decoder in available_decoders, f"Decoder '{name}' not implemented"
    # build a dictionary of additional arguments, not every model has them
    additional_args = dict(act_layer=act_layer, norm_layer=norm_layer, output_stride=output_stride)
    additional_args = filter_encoder_args(encoder=name, **additional_args)
    # create the encoder
    indices = available_decoders[decoder].func.required_indices(encoder=name)
    model = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=indices, **additional_args)
    if channels > 3:
        model = expand_input(model)
    # freeze layers in the encoder if required
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    # return the encoder
    return model


def create_decoder(name: str, feature_info: FeatureInfo, act_layer: Type[nn.Module],
                   norm_layer: Type[nn.Module]) -> Decoder:
    # sanity check to keep going with no worries
    assert name in available_decoders, f"Decoder '{name}' not implemented"
    # retrieve the partial object and instantiate with the common params
    decoder_class = available_decoders.get(name)
    decoder = decoder_class(feature_channels=feature_info.channels(),
                            feature_reductions=feature_info.reduction(),
                            act_layer=act_layer,
                            norm_layer=norm_layer)
    return decoder

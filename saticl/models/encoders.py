from typing import Type

from torch import Tensor, nn

import timm
from saticl.models.base import Encoder
from saticl.models.modules import SSMA
from timm.models.features import FeatureInfo

# just a simple wrapper to include custom encoders into the list, if required
available_encoders = {name: timm.create_model for name in timm.list_models()}


class MultiEncoder(Encoder):

    def __init__(self, encoder_rgb: Encoder, encoder_ir: Encoder, act_layer: Type[nn.Module],
                 norm_layer: Type[nn.Module]):
        super().__init__()
        self.encoder_rgb = encoder_rgb
        self.encoder_ir = encoder_ir
        self.ssmas = nn.ModuleList()
        for rgb_chs, ir_chs in zip(self.encoder_rgb.feature_info.channels(), self.encoder_ir.feature_info.channels()):
            self.ssmas.append(SSMA(rgb_channels=rgb_chs, ir_channels=ir_chs, act_layer=act_layer,
                                   norm_layer=norm_layer))

    @property
    def feature_info(self) -> FeatureInfo:
        return self.encoder_rgb.feature_info

    def forward(self, inputs: Tensor) -> Tensor:
        # expecting x to be [batch, 4, h, w]
        # we pass the first 3 to the RGB enc., the last one to the IR enc.
        rgb, ir = inputs
        rgb_features = self.encoder_rgb(rgb)
        ir_features = self.encoder_ir(ir)
        out_features = []
        for module, rgb, ir in zip(self.ssmas, rgb_features, ir_features):
            out_features.append(module(rgb, ir))
        return out_features

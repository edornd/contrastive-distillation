from functools import partial
from typing import List, Type

from torch import Tensor, nn

import timm
from saticl.models.base import Encoder
from saticl.models.modules import SSMA
from timm.models.features import FeatureInfo


class MultiEncoder(Encoder):

    def __init__(self,
                 encoder_rgb: Encoder,
                 encoder_ir: Encoder,
                 act_layer: Type[nn.Module],
                 norm_layer: Type[nn.Module],
                 return_features: bool = False):
        super().__init__()
        self.encoder_rgb = encoder_rgb
        self.encoder_ir = encoder_ir
        self.return_features = return_features
        self.ssmas = nn.ModuleList()
        for rgb_chs, ir_chs in zip(self.encoder_rgb.feature_info.channels(), self.encoder_ir.feature_info.channels()):
            self.ssmas.append(SSMA(rgb_channels=rgb_chs, ir_channels=ir_chs, act_layer=act_layer,
                                   norm_layer=norm_layer))

    @classmethod
    def create(cls, name: str, pretrained: bool, features_only: bool, out_indices: List[int], **kwargs) -> Encoder:
        act_layer = kwargs.pop("act_layer", partial(nn.ReLU, inplace=True))
        norm_layer = kwargs.pop("norm_layer", nn.BatchNorm2d)
        m_rgb = timm.create_model(name,
                                  pretrained=pretrained,
                                  features_only=features_only,
                                  out_indices=out_indices,
                                  **kwargs)
        m_ir = timm.create_model(name,
                                 pretrained=pretrained,
                                 features_only=features_only,
                                 out_indices=out_indices,
                                 **kwargs)
        return cls(encoder_rgb=m_rgb, encoder_ir=m_ir, act_layer=act_layer, norm_layer=norm_layer)

    @property
    def feature_info(self) -> FeatureInfo:
        return self.encoder_rgb.feature_info

    def forward(self, inputs: Tensor) -> Tensor:
        # expecting x to be [batch, 4, h, w]
        # we pass the first 3 to the RGB enc., the last one to the IR enc.
        rgb, ir = inputs[:, :-1], inputs[:, -1].unsqueeze(1)
        rgb_features = self.encoder_rgb(rgb)
        ir_features = self.encoder_ir(ir)
        out_features = []
        for module, rgb, ir in zip(self.ssmas, rgb_features, ir_features):
            out_features.append(module(rgb, ir))
        return out_features


# just a simple wrapper to include custom encoders into the list, if required
available_encoders = {name: timm.create_model for name in timm.list_models()}

from typing import Type

import torch
from torch import Tensor, nn

from saticl.models.base import Encoder
from saticl.models.modules import RotationHead


class PretextClassifier(nn.Module):

    def __init__(self,
                 encoder_rgb: Encoder,
                 encoder_ir: Encoder,
                 act_layer: Type[nn.Module],
                 norm_layer: Type[nn.Module],
                 hidden_dim: int = 100,
                 num_classes: int = 4) -> None:
        super().__init__()
        self.encoder_rgb = encoder_rgb
        self.encoder_ir = encoder_ir
        # first, check whether the final H and W are equal for the concatenation
        red_rgb = self.encoder_rgb.feature_info.reduction()[-1]
        red_ir = self.encoder_ir.feature_info.reduction()[-1]
        assert red_rgb == red_ir, f"Reductions not matching - RGB: {red_rgb}, IR: {red_ir}"
        # compute the final channel output
        channels_rgb = self.encoder_rgb.feature_info.channels()[-1]
        channels_ir = self.encoder_ir.feature_info.channels()[-1]
        self.head = RotationHead(channels_rgb + channels_ir,
                                 act_layer,
                                 norm_layer,
                                 hidden_dim=hidden_dim,
                                 num_classes=num_classes)

    def forward(self, rgb: Tensor, ir: Tensor) -> Tensor:
        # we only extract the bottom out layer
        x_a = self.encoder_rgb(rgb)[-1]
        x_b = self.encoder_ir(ir)[-1]
        # concat and forward on classifier
        x = torch.cat((x_a, x_b), dim=1)
        return self.head(x)

from typing import Any, Callable, List

import torch
from torch import Tensor, nn
from torch.linalg import norm

reductions = {"mean": torch.mean, "sum": torch.sum}


class Regularizer(nn.Module):
    """Wrapper class, just for naming.
    """


class MultiModalScaling(Regularizer):

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reductions.get(reduction, torch.mean)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, rgb_features: List[Tensor], ir_features: List[Tensor], **kwargs) -> Any:
        # they should have shape [batch, channels, h, w].
        # we are interested in keeping the channels coherent, so we reduce H and W
        result = torch.tensor(0.0, device=rgb_features[0].device)
        # construct a list of weights, one per feature map, weigh more at the bottom
        num_layers = len(rgb_features)
        weights = [(i / num_layers)**self.gamma for i in range(1, num_layers + 1)]
        for rgb, ir, w in zip(rgb_features, ir_features, weights):
            reduced_rgb = self.flatten(self.pooling(rgb))
            reduced_ir = self.flatten(self.pooling(ir))
            diff = norm(reduced_rgb) - norm(reduced_ir)
            result += w * diff
        return result


class AugmentationInvariance(Regularizer):

    def __init__(self, transform: Callable, reduction: str = "mean"):
        super().__init__()
        self.transform = transform
        self.criterion = nn.MSELoss(reduction=reduction)

    def apply_transform(self, *tensors: Tensor, label: Tensor = None) -> Tensor:
        return self.transform(*tensors, label=label)

    def forward(self, features: Tensor, rotated: Tensor) -> Tensor:
        return self.criterion(features, rotated)

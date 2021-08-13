from typing import Any, List

import torch
from torch import Tensor, nn
from torch.linalg import norm

reductions = {"mean": torch.mean, "sum": torch.sum}


class Regularizer(nn.Module):
    """Wrapper class, just for naming.
    """


class MultiModalScaling(Regularizer):

    def __init__(self, reduction: str = "mean", axes: tuple = (-2, -1)) -> None:
        super().__init__()
        self.reduction = reductions.get(reduction, torch.mean)
        self.axes = axes

    def forward(self, rgb_features: List[Tensor], ir_features: List[Tensor], **kwargs) -> Any:
        # they should have shape [batch, channels, h, w].
        # we are interested in keeping the channels coherent, so we reduce H and W
        regularizations = []
        for rgb, ir in zip(rgb_features, ir_features):
            reduced_rgb = self.reduction(rgb, dim=self.axes)
            reduced_ir = self.reduction(ir, dim=self.axes)
            regularizations.append(norm(reduced_rgb) - norm(reduced_ir))
        return sum(regularizations)


class AugmentationInvariance(Regularizer):

    def __init__(self):
        super().__init__()
        self.loss = nn.CosineEmbeddingLoss()
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, features: Tensor, augmented: Tensor) -> torch.Tensor:
        # build a simple target of ones as big as the batch size, which is expected
        # to be the same for every output layer
        targets = features.new_ones(features.size(0))
        f1 = self.flatten(features)
        f2 = self.flatten(augmented)
        return self.loss(f1, f2, targets)

from typing import Any, List

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

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.criterion = nn.CosineEmbeddingLoss()
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, features: List[Tensor], augmented: List[Tensor]) -> torch.Tensor:
        # build a simple target of ones as big as the batch size, which is expected
        # to be the same for every output layer
        f = features[0]
        targets = f.new_ones(f.size(0), dtype=f.dtype)
        # construct a list of weights, one per feature map, weigh more at the bottom
        num_layers = len(features)
        weights = [(i / num_layers)**self.gamma for i in range(1, num_layers + 1)]
        # sum up the losses for each layer
        total_loss = torch.tensor(0.0, device=f.device)
        for w, f, a in zip(weights, features, augmented):
            f1 = self.flatten(f)
            f2 = self.flatten(a)
            total_loss += w * self.criterion(f1, f2, targets)
        return total_loss

from functools import partial
from typing import Union

import torch
from torch import nn
from torch.nn import functional as func

from saticl.cli import Initializer
from saticl.losses.functional import one_hot_batch, smooth_weights, tanimoto_loss, unbiased_softmax


class CombinedLoss(nn.Module):

    def __init__(self, criterion_a: Initializer, criterion_b: Initializer, alpha: float = 0.5):
        super().__init__()
        self.criterion_a = criterion_a()
        self.criterion_b = criterion_b()
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_a = self.criterion_a(preds, targets)
        loss_b = self.criterion_b(preds, targets)
        return self.alpha * loss_a + (1 - self.alpha) * loss_b


class UnbiasedCrossEntropy(nn.Module):

    def __init__(self,
                 old_class_count: int,
                 reduction: str = "mean",
                 ignore_index: int = 255,
                 weight: torch.Tensor = None):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_class_count = old_class_count
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        outputs = unbiased_softmax(inputs, old_index=self.old_class_count)
        # just make sure we are not considering any of the old classes
        labels = targets.clone()    # B, H, W
        labels[targets < self.old_class_count] = 0
        loss = func.nll_loss(outputs,
                             labels,
                             ignore_index=self.ignore_index,
                             reduction=self.reduction,
                             weight=self.weight)
        return loss


class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255, weight: torch.Tensor = None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = func.cross_entropy(inputs,
                                     targets,
                                     reduction='none',
                                     ignore_index=self.ignore_index,
                                     weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean() if self.size_average else focal_loss.sum()


class UnbiasedFocalLoss(nn.Module):

    def __init__(self,
                 old_class_count: int,
                 reduction: str = "mean",
                 ignore_index: int = 255,
                 alpha: float = 1.0,
                 gamma: float = 2.0,
                 weight: torch.Tensor = None):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_class_count = old_class_count
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        outputs = unbiased_softmax(inputs, old_index=self.old_class_count)
        labels = targets.clone()    # B, H, W
        labels[targets < self.old_class_count] = 0    # just make sure we are not considering any of the old classes
        ce = func.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction="none", weight=self.weight)
        loss = self.alpha * (1 - torch.exp(-ce))**self.gamma * ce
        return loss


class FocalTverskyLoss(nn.Module):
    """Custom implementation
    """

    def __init__(self,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 gamma: float = 2.0,
                 ignore_index: int = 255,
                 weight: Union[float, torch.Tensor] = None,
                 **kwargs: dict):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weight = weight if weight is not None else 1.0
        self.ignore_index = ignore_index

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = preds.size(1)
        onehot = one_hot_batch(targets, num_classes=num_classes, ignore_index=self.ignore_index)
        onehot = onehot.float().to(preds.device)
        probs = func.softmax(preds, dim=1)

        # sum over batch, height width, leave classes (dim 1)
        dims = (0, 2, 3)
        tp = (onehot * probs).sum(dim=dims)
        fp = (probs * (1 - onehot)).sum(dim=dims)
        fn = ((1 - probs) * onehot).sum(dim=dims)

        index = self.weight * (tp / (tp + self.alpha * fp + self.beta * fn))
        return (1 - index.mean())**self.gamma


class UnbiasedFTLoss(nn.Module):
    """Custom implementation
    """

    def __init__(self,
                 old_class_count: int,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 gamma: float = 2.0,
                 ignore_index: int = 255,
                 weight: Union[float, torch.Tensor] = None,
                 **kwargs: dict):
        super().__init__()
        self.old_class_count = old_class_count
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weight = weight if weight is not None else 1.0
        self.ignore_index = ignore_index

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # just make sure we are not considering any of the old classes, labels is [batch, h, w]
        # compute the one-hot encoded ground truth, excluding old labels
        labels = targets.clone()
        labels[targets < self.old_class_count] = 0
        onehot = one_hot_batch(labels, num_classes=preds.size(1), ignore_index=self.ignore_index)
        onehot = onehot.float().to(preds.device)

        # compute the softmax (using log for numerical stability)
        # denominator becomes log(sum_i^N(exp(x_i)))
        # numerator becomes x_j - denominator
        # unbiased setting: p(0) = sum(p(old classes)) -> p of having an old class or background
        probs = unbiased_softmax(preds, old_index=self.old_class_count)
        probs = torch.exp(probs)
        # sum over batch, height width, leave classes (dim 1) to compute TP, FP, FN
        dims = (0, 2, 3)
        tp = (onehot * probs).sum(dim=dims)
        fp = (probs * (1 - onehot)).sum(dim=dims)
        fn = ((1 - probs) * onehot).sum(dim=dims)

        index = self.weight * (tp / (tp + self.alpha * fp + self.beta * fn))
        return (1 - index.mean())**self.gamma


class KDLoss(nn.Module):

    def __init__(self, alpha: float = 1.0, reduce: bool = True, **kwargs: dict):
        super().__init__()
        self.reduce = reduce
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        preds = preds.narrow(1, 0, targets.shape[1])
        outputs = torch.log_softmax(preds, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)
        loss = (outputs * labels).mean(dim=1)
        if mask is not None:
            loss = loss * mask.float()
        outputs = -torch.mean(loss) if self.reduce else -loss
        return outputs


class UnbiasedKDLoss(nn.Module):

    def __init__(self, alpha: float = 1.0, reduce: bool = True, **kwargs: dict):
        super().__init__()
        self.reduce = reduce
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]
        targets = targets * self.alpha
        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)
        den = torch.logsumexp(inputs, dim=1)    # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)    # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den    # B, H, W
        labels = torch.softmax(targets, dim=1)    # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        outputs = -torch.mean(loss) if self.reduce else -loss
        return outputs


class TanimotoLoss(nn.Module):

    def __init__(self, ignore_index: int = 255, gamma: float = 2.0, eps: float = 1e-6, **kwargs: dict):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.eps = eps
        if "old_class_count" in kwargs:
            self.old_class_count = kwargs.get("old_class_count")
            self.softmax = partial(unbiased_softmax, old_index=self.old_class_count)
        else:
            self.softmax = partial(func.softmax, dim=1)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the Dice loss using the Tanimoto formulation of the Jaccard index
        (https://en.wikipedia.org/wiki/Jaccard_index)

        Args:
            preds (torch.Tensor): prediction tensor, in the form [batch, classes, h, w]
            targets (torch.Tensor): target tensor with class indices, with shape [batch, h, w]

        Returns:
            torch.Tensor: Tanimoto loss, as described in https://arxiv.org/abs/1904.00592
        """
        num_classes = preds.size(1)
        targets_onehot = one_hot_batch(targets, num_classes=num_classes, ignore_index=self.ignore_index)
        # mean class volume per batch: sum over H and W, average over batch (dim 1 = one hot labels)
        # final tensor shape: (classes,)
        class_volume = targets_onehot.sum(dim=(2, 3)).mean(dim=0)
        vol_weights = smooth_weights(class_volume, normalize=True)
        # compute softmax probabilities
        dims = (0, 2, 3)
        probs = self.softmax(preds)
        tp = (targets_onehot * probs).sum(dim=dims)
        l2 = (targets_onehot * targets_onehot).sum(dim=dims)
        p2 = (probs * probs).sum(dim=dims)
        denominator = l2 + p2 - tp
        # compute weighted dot(p,t) / dot(t,t) + dot(p,p) - dot(p,t)
        index = ((vol_weights * tp) + self.eps) / ((vol_weights * denominator) + self.eps)
        return ((1 - index).mean())**self.gamma


class DualTanimotoLoss(nn.Module):

    def __init__(self, ignore_index: int = 255, alpha: float = 0.5, gamma: float = 1.0, **kwargs: dict):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        if "old_class_count" in kwargs:
            self.old_class_count = kwargs.get("old_class_count")
            self.softmax = partial(unbiased_softmax, old_index=self.old_class_count)
        else:
            self.softmax = partial(func.softmax, dim=1)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the Dice loss using the Tanimoto formulation of the Jaccard index
        (https://en.wikipedia.org/wiki/Jaccard_index). Also computes the dual formulation,
        then averages intersection and non-intersections.

        Args:
            preds (torch.Tensor): prediction tensor, in the form [batch, classes, h, w]
            targets (torch.Tensor): target tensor with class indices, with shape [batch, h, w]

        Returns:
            torch.Tensor: Tanimoto loss, as described in https://arxiv.org/abs/1904.00592
        """
        num_classes = preds.size(1)
        onehot_pos = one_hot_batch(targets, num_classes=num_classes, ignore_index=self.ignore_index)
        onehot_neg = 1 - onehot_pos
        probs_pos = self.softmax(preds)
        probs_neg = 1 - probs_pos
        # mean class volume per batch: sum over H and W, average over batch (dim 1 = one hot labels)
        # final tensor shape: (classes,)
        weights_pos = smooth_weights(onehot_pos.sum(dim=(2, 3)).mean(dim=0), normalize=True)
        weights_neg = smooth_weights(onehot_neg.sum(dim=(2, 3)).mean(dim=0), normalize=True)
        # compute die/tanimoto and dual dice
        dims = (0, 2, 3)
        loss = tanimoto_loss(probs_pos, onehot_pos, weights_pos, dims=dims, gamma=self.gamma)
        dual = tanimoto_loss(probs_neg, onehot_neg, weights_neg, dims=dims, gamma=self.gamma)
        return self.alpha * loss + (1 - self.alpha) * dual

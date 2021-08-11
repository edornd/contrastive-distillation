from typing import Union

import torch
from torch import nn
from torch.nn import functional as func

from saticl.cli import Initializer
from saticl.utils.ml import one_hot_batch


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


class FocalTverskyLoss(nn.Module):
    """Custom implementation
    """

    def __init__(self,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 gamma: float = 2.0,
                 ignore_index: int = 255,
                 weight: Union[float, torch.Tensor] = None):
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
                 weight: Union[float, torch.Tensor] = None):
        super().__init__()
        self.old_class_count = old_class_count
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weight = weight if weight is not None else 1.0
        self.ignore_index = ignore_index

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        oc = self.old_class_count
        num_classes = preds.size(1)
        # just make sure we are not considering any of the old classes, labels is [batch, h, w]
        # compute the one-hot encoded ground truth, excluding old labels
        labels = targets.clone()
        labels[targets < oc] = 0
        onehot = one_hot_batch(labels, num_classes=num_classes, ignore_index=self.ignore_index)
        onehot = onehot.float().to(preds.device)

        # compute the softmax (using log for numerical stability)
        # denominator becomes log(sum_i^N(exp(x_i)))
        # numerator becomes x_j - denominator
        # unbiased setting: p(0) = sum(p(old classes)) -> p of having an old class or background
        probs = torch.zeros_like(preds)
        denominator = torch.logsumexp(preds, dim=1)
        probs[:, 0] = torch.logsumexp(preds[:, :oc], dim=1) - denominator    # [batch, h, w] p(O)
        probs[:, oc:] = preds[:, oc:] - denominator.unsqueeze(dim=1)    # [batch, new, h, w] p(new_i)
        probs = torch.exp(probs)
        # sum over batch, height width, leave classes (dim 1) to compute TP, FP, FN
        dims = (0, 2, 3)
        tp = (onehot * probs).sum(dim=dims)
        fp = (probs * (1 - onehot)).sum(dim=dims)
        fn = ((1 - probs) * onehot).sum(dim=dims)

        index = self.weight * (tp / (tp + self.alpha * fp + self.beta * fn))
        return (1 - index.mean())**self.gamma


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

        oc = self.old_class_count
        # contruct a zero-initialized tensor with same dims as input [batch, (bkgr + old + new), height, width]
        outputs = torch.zeros_like(inputs)
        # build log sum(exp(inputs))  [batch, height, width], denominator for the softmax
        denominator = torch.logsumexp(inputs, dim=1)
        # compute the softmax for background (based on old classes) and new classes (minus operator because of logs)
        outputs[:, 0] = torch.logsumexp(inputs[:, :oc], dim=1) - denominator    # [batch, h, w] p(O)
        outputs[:, oc:] = inputs[:, oc:] - denominator.unsqueeze(dim=1)    # [batch, new, h, w] p(new_i)

        labels = targets.clone()    # B, H, W
        labels[targets < oc] = 0    # just make sure we are not considering any of the old classes
        loss = func.nll_loss(outputs,
                             labels,
                             ignore_index=self.ignore_index,
                             reduction=self.reduction,
                             weight=self.weight)
        return loss


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
        oc = self.old_class_count
        # same as before, build the container tensor [batch, (bkgr, old, new), h, w]
        outputs = torch.zeros_like(inputs)
        # compute the denominator, a.k.a sum(exp(predictions)) [batch, h, w]
        denominator = torch.logsumexp(inputs, dim=1)
        # compute sum(p(0) + p(old)) for p(background), standard softmax for p(new)
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:oc], dim=1) - denominator
        outputs[:, oc:] = inputs[:, oc:] - denominator.unsqueeze(dim=1)

        labels = targets.clone()    # B, H, W
        labels[targets < oc] = 0    # just make sure we are not considering any of the old classes
        ce = func.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction="none", weight=self.weight)
        loss = self.alpha * (1 - torch.exp(-ce))**self.gamma * ce
        return loss


class KDLoss(nn.Module):

    def __init__(self, alpha: float = 1.0, reduce: bool = True):
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

    def __init__(self, alpha: float = 1.0, reduce: bool = True):
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

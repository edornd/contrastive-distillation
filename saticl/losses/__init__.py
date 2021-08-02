import torch
from torch import nn
from torch.nn import functional as func

from saticl.cli import Initializer
from saticl.utils.ml import one_hot_batch


class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = func.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
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
                 weights: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weights = weights
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

        index = self.weights * (tp / (tp + self.alpha * fp + self.beta * fn))
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

    def __init__(self, old_classes=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_classes = old_classes

    def forward(self, inputs, targets):

        oc = self.old_classes
        outputs = torch.zeros_like(inputs)    # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)    # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:oc], dim=1) - den    # B, H, W       p(O)
        outputs[:, oc:] = inputs[:, oc:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets.clone()    # B, H, W
        labels[targets < oc] = 0    # just to be sure that all labels old belongs to zero

        loss = func.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)
        return loss


class UnbiasedFocalLoss(nn.Module):

    def __init__(self, old_classes=None, reduction="mean", ignore_index=255, alpha=1, gamma=2):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_classes = old_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        oc = self.old_classes
        outputs = torch.zeros_like(inputs)    # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)    # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:oc], dim=1) - den    # B, H, W       p(O)
        outputs[:, oc:] = inputs[:, oc:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets.clone()    # B, H, W
        labels[targets < oc] = 0    # just to be sure that all labels old belongs to zero
        ce = func.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction="none")
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

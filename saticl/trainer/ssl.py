from enum import Enum
from typing import Any, Dict

import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer

from saticl.logging import BaseLogger
from saticl.metrics import Metric
from saticl.tasks import Task
from saticl.trainer.base import Trainer, TrainerStage


class SSLStage(str, Enum):
    ssl = "ssl"


class SSLTrainer(Trainer):

    def __init__(self,
                 accelerator: Accelerator,
                 task: Task,
                 new_model: nn.Module,
                 old_model: nn.Module,
                 ssl_model: nn.Module,
                 optimizer: Optimizer,
                 scheduler: Any,
                 old_classes: Dict[int, str],
                 new_classes: Dict[int, str],
                 seg_criterion: nn.Module,
                 ssl_criterion: nn.Module,
                 kdd_criterion: nn.Module,
                 kde_criterion: nn.Module = None,
                 kdd_lambda: float = 0.0,
                 kde_lambda: float = 0.0,
                 ssl_lambda: float = 0.25,
                 train_metrics: Dict[str, Metric] = None,
                 val_metrics: Dict[str, Metric] = None,
                 logger: BaseLogger = None,
                 samples: int = None,
                 stage: str = "train",
                 debug: bool = False) -> None:
        super().__init__(accelerator,
                         task,
                         new_model,
                         old_model,
                         optimizer,
                         scheduler,
                         old_classes,
                         new_classes,
                         seg_criterion,
                         kdd_criterion,
                         kde_criterion=kde_criterion,
                         kdd_lambda=kdd_lambda,
                         kde_lambda=kde_lambda,
                         train_metrics=train_metrics,
                         val_metrics=val_metrics,
                         logger=logger,
                         samples=samples,
                         stage=stage,
                         debug=debug)
        self.ssl_model = ssl_model
        self.ssl_criterion = ssl_criterion
        self.ssl_lambda = ssl_lambda

    def train_batch(self, batch: Any) -> torch.Tensor:
        # init losses and retrieve x, y
        (rgb, ir, rgb_rot, ir_rot, y_rot), y = batch
        # forward and loss on segmentation task
        with self.accelerator.autocast():
            # standard segmentation task
            new_out, new_features = self.model((rgb, ir))
            seg_loss = self.criterion(new_out, y)
            # pretext task (relative rotation prediction)
            ssl_out = self.ssl_model(rgb_rot, ir_rot)
            ssl_loss = self.ssl_criterion(ssl_out, y_rot)
            # forward and loss on knowledge distillation task
            kdd_loss = torch.tensor(0, device=seg_loss.device, dtype=seg_loss.dtype)
            if self.task.step > 0:
                old_out, old_features = self.old_model((rgb, ir))
                kdd_loss = self.criterion_kdd(new_out, old_out)
            # sum everything
            total = seg_loss + self.ssl_lambda * ssl_loss + self.kdd_lambda * kdd_loss
        # gather and update metrics
        y_true_seg = self.accelerator.gather(y)
        y_pred_seg = self.accelerator.gather(new_out)
        self._update_metrics(y_true=y_true_seg, y_pred=y_pred_seg, stage=TrainerStage.train)
        y_true_ssl = self.accelerator.gather(y_rot)
        y_pred_ssl = self.accelerator.gather(ssl_out)
        self._update_metrics(y_true=y_true_ssl, y_pred=y_pred_ssl, stage=SSLStage.ssl)
        return {"tot_loss": total, "seg_loss": seg_loss, "ssl_loss": ssl_loss, "kdd_loss": kdd_loss}

    def validation_batch(self, batch: Any, batch_index: int):
        # init losses and retrieve x, y
        x, y = batch
        rgb, ir = x[:, :-1], x[:, -1].unsqueeze(1)
        seg_loss, kdd_loss = torch.tensor(0.0), torch.tensor(0.0)
        # forward and loss on main task, using AMP
        with self.accelerator.autocast():
            new_out, new_features = self.model((rgb, ir))
            seg_loss = self.criterion(new_out, y)
            # forward and loss for KD
            if self.task.step > 0:
                old_out, old_features = self.old_model((rgb, ir))
                kdd_loss = self.criterion_kdd(new_out, old_out)
            total = seg_loss + self.kdd_lambda * kdd_loss
        y_true = self.accelerator.gather(y)
        y_pred = self.accelerator.gather(new_out)
        # store samples for visualization, if present. Requires a plot callback
        # better to unpack now, so that we don't have to deal with the batch size later
        if self.sample_batches is not None and batch_index in self.sample_batches:
            images = self.accelerator.gather(x)
            self._store_samples(images, y_pred, y_true)
        # update metrics and return losses
        self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.val)
        return {"tot_loss": total, "seg_loss": seg_loss, "kdd_loss": kdd_loss}

    def train_epoch_end(self, train_losses: dict, train_times: list):
        super().train_epoch_end(train_losses, train_times)
        with torch.no_grad():
            self._compute_metrics(stage=SSLStage.ssl)
        self._log_metrics(stage=SSLStage.ssl)

    def test_batch(self, batch: Any, batch_index: int):
        x, y = batch
        x = x.to(self.accelerator.device)
        y = y.to(self.accelerator.device)
        rgb, ir = x[:, :-1], x[:, -1].unsqueeze(1)
        # forward and loss on main task, using AMP
        with self.accelerator.autocast():
            preds, _ = self.model((rgb, ir))
            loss = self.criterion(preds, y)
        # gather info
        images = self.accelerator.gather(x)
        y_true = self.accelerator.gather(y)
        y_pred = self.accelerator.gather(preds)
        # store samples for visualization, if present. Requires a plot callback
        # better to unpack now, so that we don't have to deal with the batch size later
        if self.sample_batches is not None and batch_index in self.sample_batches:
            self._store_samples(images, y_pred, y_true)
        # update metrics and return losses
        self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.test)
        return loss, (images.cpu(), y_true.cpu(), torch.argmax(y_pred, dim=1).cpu())

import logging
from typing import Any, Dict

import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer

from saticl.logging import BaseLogger
from saticl.logging.console import DistributedLogger
from saticl.metrics import Metric
from saticl.tasks import Task
from saticl.trainer.base import Trainer, TrainerStage

LOG = DistributedLogger(logging.getLogger(__name__))


class AugInvarianceTrainer(Trainer):

    def __init__(self,
                 accelerator: Accelerator,
                 task: Task,
                 new_model: nn.Module,
                 old_model: nn.Module,
                 optimizer: Optimizer,
                 scheduler: Any,
                 old_classes: Dict[int, str],
                 new_classes: Dict[int, str],
                 seg_criterion: nn.Module,
                 kdd_criterion: nn.Module,
                 kde_criterion: nn.Module = None,
                 aug_criterion: nn.Module = None,
                 kdd_lambda: float = 0.0,
                 kde_lambda: float = 0.0,
                 aug_lambda: float = 0.0,
                 aug_layers: int = 1,
                 temperature: float = 4.0,
                 temp_epochs: int = 20,
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
        layers = max(len(self.model.encoder.feature_info.channels()), aug_layers)
        if layers < aug_layers:
            LOG.warn("The number of proposed layers (%d) exceeds the available ones (%d)", aug_layers, layers)
        self.criterion_aug = aug_criterion
        self.aug_lambda = aug_lambda
        self.aug_layers = layers
        self.temperature = temperature
        self.temp_epochs = temp_epochs

    def step(self) -> None:
        super().step()
        if not self.temperature > 0:
            return
        if self.current_epoch >= self.temp_epochs:
            exp = 0.0
        else:
            exp = ((self.temp_epochs - self.current_epoch) / self.temp_epochs)**2
        self.temperature = self.temperature**exp

    def train_epoch_end(self, train_losses: dict, train_times: list):
        super().train_epoch_end(train_losses, train_times)
        self.logger.log_scalar("train/temp", self.temperature)

    def train_batch(self, batch: Any) -> torch.Tensor:
        # init losses and retrieve x, y
        x1, x2, y1, y2 = batch
        full_x = torch.cat((x1, x2), dim=0)
        full_y = torch.cat((y1, y2), dim=0).long()
        # forward and loss on segmentation task
        with self.accelerator.autocast():
            split = x1.size(0)
            new_out, new_features = self.model(full_x)
            if self.temperature:
                new_out /= self.temperature
            seg_loss = self.criterion(new_out, full_y)
            # rotation invariance loss
            # since we are feeding the same images augmented twice, the output features contain both
            # and we need to split in the middle, at 'split' size
            enc1 = [f[:split] for f in new_features[-self.aug_layers:]]
            enc2 = [f[split:] for f in new_features[-self.aug_layers:]]
            rot_loss = self.aug_lambda * self.criterion_aug(enc1, enc2)
            # knowledge distillation from the old model
            # this only has effect from step 1 onwards
            kdd_loss = torch.tensor(0, device=seg_loss.device, dtype=seg_loss.dtype)
            if self.task.step > 0:
                old_out, _ = self.old_model(full_x)
                kdd_loss = self.kdd_lambda * self.criterion_kdd(new_out, old_out)
            # sum up losses
            total = seg_loss + kdd_loss + rot_loss
        # gather and update metrics
        # we group only the 'standard' images, not the rotated ones
        y_true = self.accelerator.gather(full_y)
        y_pred = self.accelerator.gather(new_out)
        self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.train)
        # debug if active
        if self.debug:
            self._debug_training(x=x1.dtype, y=y1.dtype, pred=new_out.dtype, seg_loss=seg_loss, kdd_loss=kdd_loss)
        return {"tot_loss": total, "seg_loss": seg_loss, "kdd_loss": kdd_loss, "rot_loss": rot_loss}

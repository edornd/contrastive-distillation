import logging
from typing import Any, Dict

import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer

from saticl.logging import BaseLogger
from saticl.logging.console import DistributedLogger
from saticl.losses.regularization import AugmentationInvariance
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
                 aug_criterion: AugmentationInvariance = None,
                 kdd_lambda: float = 0.0,
                 kde_lambda: float = 0.0,
                 aug_lambda: float = 0.0,
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
        self.regularizer = aug_criterion
        self.aug_lambda = aug_lambda
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
        x, y = batch
        # forward and loss on segmentation task
        with self.accelerator.autocast():
            rot_loss = torch.tensor(0.0, device=self.accelerator.device)
            kdd_loss = torch.tensor(0.0, device=self.accelerator.device)
            logits, features = self.model(x)
            tensors = [x, features]
            # handle multimodal output, if any
            if self.multimodal:
                raise NotImplementedError("Not working on it")
            # knowledge distillation from the old model
            # this only has effect from step 1 onwards
            if self.task.step > 0:
                old_logits, old_features = self.old_model(x)
                kdd_loss = self.criterion_kdd(logits, old_logits)
                kdd_loss = self.kdd_lambda * torch.nan_to_num(kdd_loss, nan=0.0)
                tensors.append(old_features)
            # rotation/augmentation invariance:
            # z1 = f1(rot(x))
            # z2 = rot(f2(x))
            # f1 = new model, f2 = new model for step 0, old model in incremental setups
            if self.aug_lambda > 0:
                tensors_rot, y_rot = self.regularizer.apply_transform(*tensors, label=y)
                rot_x, rot_f = tensors_rot[0], tensors_rot[1]
                # compute forward on rotated image, merge rotated and non-rotated for segmentation
                rot_logits, f_rot = self.model(rot_x)
                logits = torch.cat((logits, rot_logits), axis=0)
                y = torch.cat((y, y_rot), axis=0)
                # compute rotation/augmentation invariance loss
                # incremental: rot(old_model(x)) <-> new_model(rot(x))
                rot_loss = self.aug_lambda * torch.nan_to_num(self.regularizer(rot_f, f_rot), nan=0.0)
                if len(tensors_rot) > 2:
                    old_rot_loss = torch.nan_to_num(self.regularizer(f_rot, tensors_rot[-1]), nan=0.0)
                    rot_loss = self.aug_lambda * 0.5 * old_rot_loss
            # sum up losses
            seg_loss = self.criterion(logits, y)
            total = seg_loss + kdd_loss + rot_loss
        # gather and update metrics
        # we group only the 'standard' images, not the rotated ones
        y_true = self.accelerator.gather(y)
        y_pred = self.accelerator.gather(logits)
        self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.train)
        # debug if active
        if self.debug:
            self._debug_training(x=x.dtype,
                                 y=y.dtype,
                                 pred=logits.dtype,
                                 seg_loss=seg_loss,
                                 kdd_loss=kdd_loss,
                                 rot_loss=rot_loss)
        return {"tot_loss": total, "seg_loss": seg_loss, "kdd_loss": kdd_loss, "rot_loss": rot_loss}

from typing import Any, Dict

import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer

from saticl.logging import BaseLogger
from saticl.metrics import Metric
from saticl.tasks import Task
from saticl.trainer.base import Trainer, TrainerStage


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
                 rot_criterion: nn.Module = None,
                 kdd_lambda: float = 0.0,
                 kde_lambda: float = 0.0,
                 rot_lambda: float = 0.0,
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
        self.criterion_rot = rot_criterion
        self.rot_lambda = rot_lambda

    def train_batch(self, batch: Any) -> torch.Tensor:
        # init losses and retrieve x, y
        x, xr, y, yr = batch
        full_x = torch.cat((x, xr), dim=0)
        full_y = torch.cat((y, yr), dim=0)
        # forward and loss on segmentation task
        with self.accelerator.autocast():
            split = x.size(0)
            new_out, new_features = self.model(full_x)
            seg_loss = self.criterion(new_out, full_y)
            # rotation invariance loss
            # since we are feeding rotated and non-rotated, the output features contain both
            enc_out = new_features[-1]
            rot_loss = self.rot_lambda * self.criterion_rot(enc_out[:split], enc_out[split:])
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
            self._debug_training(x=x.dtype, y=y.dtype, pred=new_out.dtype, seg_loss=seg_loss, kdd_loss=kdd_loss)
        return {"tot_loss": total, "seg_loss": seg_loss, "kdd_loss": kdd_loss, "rot_loss": rot_loss}

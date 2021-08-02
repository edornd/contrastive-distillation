from __future__ import annotations

import time
from collections import defaultdict
from enum import Enum
from posix import listdir
from typing import TYPE_CHECKING, Any, Dict, Iterable

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from saticl.logging import BaseLogger
from saticl.logging.empty import EmptyLogger
from saticl.metrics import Metric
from saticl.tasks import Task
from saticl.utils.common import get_logger, progressbar

if TYPE_CHECKING:
    from saticl.trainer.callbacks import BaseCallback

LOG = get_logger(__name__)


class TrainerStage(str, Enum):
    train = "train"
    val = "val"
    test = "test"


class Trainer:

    def __init__(self,
                 accelerator: Accelerator,
                 task: Task,
                 new_model: nn.Module,
                 old_model: nn.Module,
                 optimizer: Optimizer,
                 scheduler: Any,
                 train_metrics: Dict[str, Metric],
                 val_metrics: Dict[str, Metric],
                 old_classes: Dict[int, str],
                 new_classes: Dict[int, str],
                 seg_criterion: nn.Module,
                 kdd_criterion: nn.Module,
                 kde_criterion: nn.Module = None,
                 kdd_lambda: float = 0.0,
                 kde_lambda: float = 0.0,
                 logger: BaseLogger = None) -> None:
        assert task.step == 0 or old_model is not None, "ICL steps require the old model for KD"
        self.accelerator = accelerator
        self.model = new_model
        self.old_model = old_model
        self.criterion = seg_criterion
        # knowledge distillation: KDD = KD on decoder, KDE = KD on encoder
        self.criterion_kdd = kdd_criterion
        self.criterion_kde = kde_criterion
        self.kdd_lambda = kdd_lambda
        self.kde_lambda = kde_lambda
        # optimizer, scheduler, metrics and logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = {TrainerStage.train.value: train_metrics, TrainerStage.val.value: val_metrics}
        self.logger = logger or EmptyLogger()
        # ICL information for loggers
        self.task = task
        self.old_classes = old_classes
        self.new_classes = new_classes
        # internal state
        self.current_epoch = -1
        self.global_step = -1
        # internal monitoring
        self.current_scores = {TrainerStage.train.value: dict(), TrainerStage.val.value: dict()}
        self.best_epoch = None
        self.best_score = None
        self.best_state_dict = None
        self.callbacks: listdir[BaseCallback] = list()

    def _prepare(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None) -> None:
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        train_dataloader = self.accelerator.prepare(train_dataloader)
        if val_dataloader is not None:
            val_dataloader = self.accelerator.prepare(val_dataloader)
        return train_dataloader, val_dataloader

    def _update_metrics(self,
                        y_true: torch.Tensor,
                        y_pred: torch.Tensor,
                        stage: TrainerStage = TrainerStage.train) -> None:
        with torch.no_grad():
            for metric in self.metrics[stage.value].values():
                metric(y_true, y_pred)

    def _compute_metrics(self, stage: TrainerStage = TrainerStage.train) -> None:
        result = dict()
        with torch.no_grad():
            for name, metric in self.metrics[stage.value].items():
                result[name] = metric.compute()
        self.current_scores[stage.value] = result

    def _reset_metrics(self, stage: TrainerStage = TrainerStage.train) -> None:
        for metric in self.metrics[stage.value].values():
            metric.reset()

    def _log_metrics(self, stage: TrainerStage = TrainerStage.train, exclude: Iterable[str] = None) -> None:
        log_strings = []
        exclude = exclude or []
        scores = self.current_scores[stage.value]
        classwise = dict()
        # first log scalars
        for metric_name, score in scores.items():
            if metric_name in exclude:
                continue
            if score.ndim > 0:
                # store for later
                classwise[metric_name] = score
                continue
            self.logger.log_scalar(f"{stage.value}/{metric_name}", score)
            log_strings.append(f"{stage.value}/{metric_name}: {score:.4f}")
            LOG.info(", ".join(log_strings))
        # then log class-wise results in a single table
        if classwise:
            LOG.debug("Classwise: %s", str(classwise))
            header = list(self.new_classes.values())
            self.logger.log_results(f"{stage.value}/results", headers=header, results=classwise)

    def add_callback(self, callback: BaseCallback) -> Trainer:
        self.callbacks.append(callback)
        return self

    def setup_callbacks(self) -> None:
        for callback in self.callbacks:
            callback.setup(self)

    def dispose_callbacks(self) -> None:
        for callback in self.callbacks:
            callback.dispose(self)

    def step(self) -> None:
        self.global_step += 1
        self.logger.step()

    def train_epoch_start(self):
        self._reset_metrics(stage=TrainerStage.train)

    def train_batch(self, batch: Any) -> torch.Tensor:
        # init losses and retrieve x, y
        x, y = batch
        # forward and loss on segmentation task
        new_out, new_features = self.model(x)
        seg_loss = self.criterion(new_out, y)
        # forward and loss on knowledge distillation task
        if self.task.step > 0:
            old_out, old_features = self.old_model(x)
            kdd_loss = self.criterion_kdd(new_out, old_out)
        else:
            kdd_loss = torch.tensor(0, device=seg_loss.device, dtype=seg_loss.dtype)
        # total loss, gather stuff and update metrics
        total_loss = seg_loss + kdd_loss
        y_true = self.accelerator.gather(y)
        y_pred = self.accelerator.gather(new_out)
        self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.train)
        return total_loss, seg_loss, kdd_loss

    def train_epoch_end(self, train_losses: list, train_times: list):
        with torch.no_grad():
            self._compute_metrics(stage=TrainerStage.train)
        self.logger.log_scalar("train/tot_loss", np.mean(train_losses["tot"]))
        self.logger.log_scalar("train/seg_loss", np.mean(train_losses["seg"]))
        self.logger.log_scalar("train/kdd_loss", np.mean(train_losses["kdd"]))
        self.logger.log_scalar("train/time", np.mean(train_times))
        self._log_metrics(stage=TrainerStage.train)

    def validation_epoch_start(self):
        self._reset_metrics(stage=TrainerStage.val)

    def validation_batch(self, batch: Any):
        # init losses and retrieve x, y
        x, y = batch
        seg_loss, kdd_loss = torch.tensor(0.0), torch.tensor(0.0)
        # forward and loss on main task
        new_out, new_features = self.model(x)
        seg_loss = self.criterion(new_out, y)
        # forward and loss for KD
        if self.task.step > 0:
            old_out, old_features = self.old_model(x)
            kdd_loss = self.criterion_kdd(new_out, old_out)
        # total loss, gather and update metrics
        total_loss = seg_loss + kdd_loss
        y_true = self.accelerator.gather(y)
        y_pred = self.accelerator.gather(new_out)
        self._update_metrics(y_true=y_true, y_pred=y_pred, stage=TrainerStage.val)
        return total_loss, seg_loss, kdd_loss

    def validation_epoch_end(self, val_losses: list, val_times: list):
        with torch.no_grad():
            self._compute_metrics(stage=TrainerStage.val)
        self.logger.log_scalar("val/tot_loss", np.mean(val_losses["tot"]))
        self.logger.log_scalar("val/seg_loss", np.mean(val_losses["seg"]))
        self.logger.log_scalar("val/kdd_loss", np.mean(val_losses["kdd"]))

        self.logger.log_scalar("val/time", np.mean(val_times))
        self._log_metrics(stage=TrainerStage.val)

    def test_batch(self, batch: Any):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        preds = self.model(x)
        loss = self.criterion(preds, y)
        self._update_metrics(y_true=y, y_pred=preds, stage=TrainerStage.test)
        return loss, (x, y, torch.argmax(preds, dim=1))

    def train_epoch(self, epoch: int, train_dataloader: DataLoader) -> Any:
        timings = []
        losses = defaultdict(list)
        train_tqdm = progressbar(train_dataloader, epoch=epoch, stage=TrainerStage.train.value)

        self.model.train()
        for batch in train_tqdm:
            start = time.time()
            self.optimizer.zero_grad()
            loss, seg_loss, kdd_loss = self.train_batch(batch=batch)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()
            # measure elapsed time
            elapsed = (time.time() - start)
            # store training info
            loss_val = loss.mean().item()
            train_tqdm.set_postfix({"loss": loss_val})
            self.logger.log_scalar("train/loss_iter", loss_val)
            self.logger.log_scalar("train/lr", self.optimizer.param_groups[0]["lr"])
            self.logger.log_scalar("train/time_iter", elapsed)
            losses["tot"].append(loss_val)
            losses["seg"].append(seg_loss.mean().item())
            losses["kdd"].append(kdd_loss.mean().item())
            timings.append(elapsed)
            # step the logger
            self.step()

        return losses, timings

    def validation_epoch(self, epoch: int, val_dataloader: DataLoader) -> Any:
        val_tqdm = progressbar(val_dataloader, epoch=epoch, stage=TrainerStage.val.value)
        timings = []
        losses = defaultdict(list)

        with torch.no_grad():
            self.model.eval()
            for batch in val_tqdm:
                start = time.time()
                loss, sg_loss, kd_loss = self.validation_batch(batch=batch)
                elapsed = (time.time() - start)
                # gather info
                sg_loss = self.accelerator.gather(sg_loss)
                kd_loss = self.accelerator.gather(kd_loss)
                loss = self.accelerator.gather(loss)
                loss_val = loss.mean().item()
                val_tqdm.set_postfix({"loss": loss_val})
                # we do not log 'iter' versions for loss and timings, since we do notadvance the logger step
                # during validation (also, it's kind of useless)
                losses["tot"].append(loss_val)
                losses["seg"].append(sg_loss.mean().item())
                losses["kdd"].append(kd_loss.mean().item())
                timings.append(elapsed)

        return losses, timings

    def fit(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None, max_epochs: int = 100):
        train_dataloader, val_dataloader = self._prepare(train_dataloader, val_dataloader)
        self.best_state_dict = self.model.state_dict()
        self.setup_callbacks()
        self.global_step = 0

        for curr_epoch in range(max_epochs):
            self.current_epoch = curr_epoch
            LOG.info(f"[Epoch {self.current_epoch:>2d}]")
            try:
                self.train_epoch_start()
                t_losses, t_times = self.train_epoch(epoch=self.current_epoch, train_dataloader=train_dataloader)
                self.train_epoch_end(t_losses, t_times)

                if val_dataloader is not None:
                    self.validation_epoch_start()
                    v_losses, v_times = self.validation_epoch(epoch=self.current_epoch, val_dataloader=val_dataloader)
                    self.validation_epoch_end(v_losses, v_times)

                for callback in self.callbacks:
                    callback(self)

            except KeyboardInterrupt:
                LOG.info("[Epoch %2d] Interrupting training", curr_epoch)
                break

        self.dispose_callbacks()
        return self

    def predict(self, test_dataloader: DataLoader, metrics: Dict[str, Metric], logger_exclude: Iterable[str]):
        self.metrics[TrainerStage.test.value] = metrics
        self._reset_metrics(stage=TrainerStage.test)
        test_tqdm = progressbar(test_dataloader, stage=TrainerStage.test.value)
        losses, timings, results = [], [], []

        with torch.no_grad():
            self.model.eval()
            for batch in test_tqdm:
                start = time.time()
                loss, data = self.test_batch(batch=batch)
                elapsed = (time.time() - start)
                loss_value = loss.item()
                test_tqdm.set_postfix({"loss": loss_value})
                # we do not log 'iter' versions, as for validation
                losses.append(loss_value)
                timings.append(elapsed)
                results.append(data)

            self.logger.log_scalar("test/loss", np.mean(losses))
            self.logger.log_scalar("test/time", np.mean(timings))
            self._compute_metrics(stage=TrainerStage.test)
            self._log_metrics(stage=TrainerStage.test, exclude=logger_exclude)

        return losses, results

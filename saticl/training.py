from itertools import chain
from pathlib import Path
from typing import Tuple

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from saticl.config import Configuration, SSLConfiguration
from saticl.datasets.icl import ICLDataset
from saticl.datasets.transforms import geom_transforms, inverse_transform, ssl_transforms
from saticl.datasets.wrappers import RotationDataset, SSLDataset
from saticl.logging.tensorboard import TensorBoardLogger
from saticl.losses.regularization import AugmentationInvariance
from saticl.models.icl import ICLSegmenter
from saticl.prepare import prepare_dataset, prepare_metrics, prepare_metrics_ssl, prepare_model, prepare_model_ssl
from saticl.tasks import Task
from saticl.trainer.base import Trainer
from saticl.trainer.callbacks import Checkpoint, DisplaySamples, EarlyStopping, EarlyStoppingCriterion
from saticl.trainer.invariance import AugInvarianceTrainer
from saticl.trainer.ssl import SSLStage, SSLTrainer
from saticl.utils.common import flatten_config, get_logger, store_config
from saticl.utils.ml import checkpoint_path, init_experiment, seed_everything, seed_worker


LOG = get_logger(__name__)


def init_from_previous_step(config: Configuration, new_model: ICLSegmenter, old_model: ICLSegmenter,
                            model_folder: Path, task: Task) -> Tuple[ICLSegmenter, ICLSegmenter]:
    if task.step == 0:
        LOG.info("Step 0: training from scratch without old model")
        return new_model, old_model

    LOG.info("Loading checkpoint from step: %d", task.step - 1)
    if config.task.step_checkpoint is not None:
        ckpt_path = Path(config.task.step_checkpoint)
    else:
        ckpt_path = checkpoint_path(model_folder, task_name=task.name, step=task.step - 1)
    assert ckpt_path.exists() and ckpt_path.is_file(), f"Checkpoint for step {task.step-1} not found at {str(ckpt_path)}"

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    # load checkpoint into the new model, without strict matching because of ICL heads
    new_model.load_state_dict(checkpoint, strict=False)
    if config.model.init_balanced:
        new_model.init_classifier()
    # load the same checkpoint into the old model, this time strict since it's the very same
    old_model.load_state_dict(checkpoint, strict=True)
    old_model.freeze()
    old_model.eval()
    del checkpoint
    return new_model, old_model


def train(config: Configuration):
    # assertions before starting
    assert config.name is not None or config.task.step == 0, "Specify the experiment name with ICL steps >= 1!"
    assert torch.backends.cudnn.enabled, "AMP requires CUDNN backend to be enabled."

    # prepare accelerator ASAP
    accelerator = Accelerator(fp16=config.trainer.amp, cpu=config.trainer.cpu)

    # Create the directory tree:
    # outputs
    # |-- dataset
    #     |--task_name
    #        |-- exp_name
    #            |-- models
    #            |-- logs
    accelerator.wait_for_everyone()
    log_name = f"output-{config.task.step}.log"
    exp_id, out_folder, model_folder, logs_folder = init_experiment(config=config, log_name=log_name)
    config_path = out_folder / f"segmenter-config-s{config.task.step}.yaml"
    store_config(config, path=config_path)
    LOG.info("Run started")
    LOG.info("Experiment ID: %s", exp_id)
    LOG.info("Output folder: %s", out_folder)
    LOG.info("Models folder: %s", model_folder)
    LOG.info("Logs folder:   %s", logs_folder)
    LOG.info("Configuration: %s", config_path)

    # seeding everything
    LOG.info("Using seed: %d", config.seed)
    seed_everything(config.seed)
    # prepare datasets
    LOG.info("Loading datasets...")
    train_set, valid_set = prepare_dataset(config=config)
    LOG.info("Full sets - train set: %d samples, validation set: %d samples", len(train_set), len(valid_set))

    add_background = not train_set.has_background()
    task = Task(dataset=config.dataset,
                name=config.task.name,
                step=config.task.step,
                add_background=add_background)
    train_mask, valid_mask = 0, 255
    train_set = ICLDataset(dataset=train_set, task=task, mask_value=train_mask, overlap=config.task.overlap)
    valid_set = ICLDataset(dataset=valid_set, task=task, mask_value=valid_mask, overlap=config.task.overlap)
    if config.ce.aug_factor > 0:
        LOG.info("Wrapping into rotation-augmented dataset")
        train_set = RotationDataset(train_set, transform=geom_transforms())
    train_loader = DataLoader(dataset=train_set,
                              batch_size=config.trainer.batch_size,
                              shuffle=True,
                              num_workers=config.trainer.num_workers,
                              worker_init_fn=seed_worker,
                              drop_last=True)
    valid_loader = DataLoader(dataset=valid_set,
                              batch_size=config.trainer.batch_size,
                              shuffle=False,
                              num_workers=config.trainer.num_workers,
                              worker_init_fn=seed_worker)
    LOG.info("ICL sets  - Train set: %d samples, validation set: %d samples", len(train_set), len(valid_set))

    # prepare models
    LOG.info("Preparing model...")
    new_model = prepare_model(config=config, task=task)
    new_model = new_model.to(accelerator.device)
    if task.step > 0:
        old_task = Task(dataset=config.dataset,
                        name=config.task.name,
                        step=task.step - 1,
                        add_background=add_background)
        old_model = prepare_model(config=config, task=old_task)
        old_model = old_model.to(accelerator.device)
    else:
        old_model = None
    new_model, old_model = init_from_previous_step(config, new_model, old_model, model_folder, task)
    LOG.info("Done preparing models")

    # prepare optimizer and scheduler
    optimizer = config.optimizer.instantiate(new_model.parameters())
    scheduler = config.scheduler.instantiate(optimizer)
    # prepare losses
    weights = None
    if config.class_weights:
        weights = train_set.load_class_weights(Path(config.class_weights),
                                               device=accelerator.device,
                                               normalize=config.ce.tversky)
        LOG.info("Using class weights: %s", str(weights))
    segment_loss = config.ce.instantiate(ignore_index=255, old_class_count=task.old_class_count(), weight=weights)
    distill_loss = config.kd.instantiate()
    if task.step > 0 and config.ce.unbiased:
        seg_loss_name = str(type(segment_loss))
        kdd_loss_name = str(type(distill_loss))
        assert "Unbiased" in seg_loss_name, f"Wrong loss '{seg_loss_name}' for step {task.step}"
        assert "Unbiased" in kdd_loss_name, f"Wrong loss '{kdd_loss_name}' for step {task.step}"
    # prepare metrics and logger
    monitored = config.trainer.monitor.name
    train_metrics, valid_metrics = prepare_metrics(task=task, device=accelerator.device)
    logger = TensorBoardLogger(log_folder=logs_folder,
                               filename_suffix=f"step-{task.step}",
                               icl_step=task.step,
                               comment=config.comment)
    # logging configuration to tensorboard
    LOG.debug("Logging flattened config. to TensorBoard")
    logger.log_table("config", flatten_config(config.dict()))

    # prepare trainer
    LOG.info("Visualize: %s, num. batches for visualization: %s", str(config.visualize), str(config.num_samples))
    num_samples = int(config.visualize) * config.num_samples
    # choose trainer class depending on task or regularization
    trainer_class = Trainer
    kwargs = dict()
    if config.ce.aug_factor > 0:
        kwargs.update(rot_criterion=AugmentationInvariance(), rot_lambda=config.ce.aug_factor)
        trainer_class = AugInvarianceTrainer
    trainer = trainer_class(accelerator=accelerator,
                            task=task,
                            new_model=new_model,
                            old_model=old_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            train_metrics=train_metrics,
                            val_metrics=valid_metrics,
                            old_classes=train_set.old_categories(),
                            new_classes=train_set.new_categories(),
                            seg_criterion=segment_loss,
                            kdd_criterion=distill_loss,
                            kde_criterion=None,
                            kdd_lambda=config.kd.decoder_factor,
                            kde_lambda=config.kd.encoder_factor,
                            logger=logger,
                            samples=num_samples,
                            debug=config.debug,
                            **kwargs)
    trainer.add_callback(EarlyStopping(call_every=1, metric=monitored,
                                       criterion=EarlyStoppingCriterion.maximum,
                                       patience=config.trainer.patience)) \
           .add_callback(Checkpoint(call_every=1,
                                    model_folder=model_folder,
                                    name_format=f"task{task.name}_step-{task.step}",
                                    save_best=True)) \
           .add_callback(DisplaySamples(inverse_transform=inverse_transform(),
                                        color_palette=train_set.palette()))
    trainer.fit(train_dataloader=train_loader, val_dataloader=valid_loader, max_epochs=config.trainer.max_epochs)
    LOG.info(f"Training completed at epoch {trainer.current_epoch:<2d} "
             f"(best {monitored}: {trainer.best_score:.4f})")
    LOG.info("Experiment %s completed!", exp_id)


def train_ssl(config: SSLConfiguration):
    # assertions before starting
    assert config.name is not None or config.task.step == 0, "Specify the experiment name with ICL steps >= 1!"
    assert torch.backends.cudnn.enabled, "AMP requires CUDNN backend to be enabled."
    if config.in_channels != 4:
        LOG.warn("Forcing input channels to 4 (previous value: %d)", config.in_channels)
        config.in_channels = 4
    # prepare accelerator ASAP
    accelerator = Accelerator(fp16=config.trainer.amp, cpu=config.trainer.cpu)

    # Create the directory tree:
    # outputs
    # |-- dataset
    #     |--task_name
    #        |-- exp_name
    #            |-- models
    #            |-- logs
    accelerator.wait_for_everyone()
    log_name = f"output-{config.task.step}.log"
    exp_id, out_folder, model_folder, logs_folder = init_experiment(config=config, log_name=log_name)
    config_path = out_folder / f"segmenter-config-s{config.task.step}.yaml"
    store_config(config, path=config_path)
    LOG.info("Run started")
    LOG.info("Experiment ID: %s", exp_id)
    LOG.info("Output folder: %s", out_folder)
    LOG.info("Models folder: %s", model_folder)
    LOG.info("Logs folder:   %s", logs_folder)
    LOG.info("Configuration: %s", config_path)

    # seeding everything
    LOG.info("Using seed: %d", config.seed)
    seed_everything(config.seed)
    # prepare datasets
    LOG.info("Loading datasets...")
    train_set, valid_set = prepare_dataset(config=config)
    train_set = SSLDataset(train_set, transform=ssl_transforms())
    LOG.info("Full sets - train set: %d samples, validation set: %d samples", len(train_set), len(valid_set))

    add_background = not train_set.has_background()
    task = Task(dataset=config.dataset,
                name=config.task.name,
                step=config.task.step,
                add_background=add_background)
    train_mask, valid_mask = 0, 255
    train_set = ICLDataset(dataset=train_set, task=task, mask_value=train_mask, overlap=config.task.overlap)
    valid_set = ICLDataset(dataset=valid_set, task=task, mask_value=valid_mask, overlap=config.task.overlap)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=config.trainer.batch_size,
                              shuffle=True,
                              num_workers=config.trainer.num_workers,
                              worker_init_fn=seed_worker,
                              drop_last=True)
    valid_loader = DataLoader(dataset=valid_set,
                              batch_size=config.trainer.batch_size,
                              shuffle=False,
                              num_workers=config.trainer.num_workers,
                              worker_init_fn=seed_worker)
    LOG.info("ICL sets  - Train set: %d samples, validation set: %d samples", len(train_set), len(valid_set))

    # prepare models
    LOG.info("Preparing model...")
    new_model, ssl_model = prepare_model_ssl(config=config, task=task)
    new_model = new_model.to(accelerator.device)
    ssl_model = ssl_model.to(accelerator.device)
    if task.step > 0:
        old_task = Task(dataset=config.dataset,
                        name=config.task.name,
                        step=task.step - 1,
                        add_background=add_background)
        old_model = prepare_model(config=config, task=old_task)
        old_model = old_model.to(accelerator.device)
    else:
        old_model = None
    new_model, old_model = init_from_previous_step(config, new_model, old_model, model_folder, task)
    LOG.info("Done preparing models")

    # prepare optimizer and scheduler
    parameters = chain(new_model.parameters(), ssl_model.head.parameters())
    optimizer = config.optimizer.instantiate(parameters)
    scheduler = config.scheduler.instantiate(optimizer)
    # prepare losses, including SSL
    segment_loss = config.ce.instantiate(ignore_index=255, old_class_count=task.old_class_count())
    distill_loss = config.kd.instantiate()
    pretext_loss = config.ssl_loss()
    # asserts to verify their validity
    if task.step > 0 and config.ce.unbiased:
        seg_loss_name = str(type(segment_loss))
        kdd_loss_name = str(type(distill_loss))
        assert "Unbiased" in seg_loss_name, f"Wrong loss '{seg_loss_name}' for step {task.step}"
        assert "Unbiased" in kdd_loss_name, f"Wrong loss '{kdd_loss_name}' for step {task.step}"
    # prepare metrics and logger
    monitored = config.trainer.monitor.name
    train_metrics, valid_metrics = prepare_metrics(task=task, device=accelerator.device)
    ssl_metrics = prepare_metrics_ssl(num_classes=config.model.pretext_classes, device=accelerator.device)
    logger = TensorBoardLogger(log_folder=logs_folder,
                               filename_suffix=f"step-{task.step}",
                               icl_step=task.step,
                               comment=config.comment)
    # logging configuration to tensorboard
    LOG.debug("Logging flattened config. to TensorBoard")
    logger.log_table("config", flatten_config(config.dict()))

    # prepare trainer
    LOG.info("Visualize: %s, num. batches for visualization: %s", str(config.visualize), str(config.num_samples))
    num_samples = int(config.visualize) * config.num_samples
    trainer = SSLTrainer(accelerator=accelerator,
                         task=task,
                         new_model=new_model,
                         old_model=old_model,
                         ssl_model=ssl_model,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         train_metrics=train_metrics,
                         val_metrics=valid_metrics,
                         old_classes=train_set.old_categories(),
                         new_classes=train_set.new_categories(),
                         seg_criterion=segment_loss,
                         ssl_criterion=pretext_loss,
                         kdd_criterion=distill_loss,
                         kde_criterion=None,
                         kdd_lambda=config.kd.decoder_factor,
                         kde_lambda=config.kd.encoder_factor,
                         logger=logger,
                         samples=num_samples,
                         debug=config.debug)
    trainer.add_metrics(SSLStage.ssl, metrics=ssl_metrics)
    trainer.add_callback(EarlyStopping(call_every=1, metric=monitored,
                                       criterion=EarlyStoppingCriterion.maximum,
                                       patience=config.trainer.patience)) \
           .add_callback(Checkpoint(call_every=1,
                                    model_folder=model_folder,
                                    name_format=f"task{task.name}_step-{task.step}",
                                    save_best=True)) \
           .add_callback(DisplaySamples(inverse_transform=inverse_transform(),
                                        color_palette=train_set.palette()))
    trainer.fit(train_dataloader=train_loader, val_dataloader=valid_loader, max_epochs=config.trainer.max_epochs)
    LOG.info(f"Training completed at epoch {trainer.current_epoch:<2d} "
             f"(best {monitored}: {trainer.best_score:.4f})")
    LOG.info("Experiment %s completed!", exp_id)

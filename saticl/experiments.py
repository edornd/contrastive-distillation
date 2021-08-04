from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from saticl.config import Configuration, Metrics, TestConfiguration
from saticl.datasets import create_dataset
from saticl.datasets.icl import ICLDataset
from saticl.datasets.transforms import inverse_transform, test_transforms, train_transforms
from saticl.logging.tensorboard import TensorBoardLogger
from saticl.metrics import ConfusionMatrix, F1Score, IoU
from saticl.models import create_decoder, create_encoder
from saticl.models.icl import ICLSegmenter
from saticl.tasks import Task
from saticl.trainer.base import Trainer
from saticl.trainer.callbacks import Checkpoint, DisplaySamples, EarlyStopping, EarlyStoppingCriterion
from saticl.utils.common import flatten_config, get_logger, load_config, prepare_folder, store_config
from saticl.utils.ml import (
    checkpoint_path,
    init_experiment,
    mask_set,
    plot_confusion_matrix,
    save_grid,
    seed_everything,
    seed_worker,
)
from tqdm import tqdm


LOG = get_logger(__name__)


def prepare_dataset(config: Configuration) -> ICLDataset:
    # instantiate transforms for training and evaluation
    data_root = Path(config.data_root)
    train_transform = train_transforms(image_size=config.image_size, in_channels=config.in_channels)
    eval_transform = test_transforms(in_channels=config.in_channels)
    LOG.debug("Train transforms: %s", str(train_transform))
    LOG.debug("Eval. transforms: %s", str(eval_transform))
    # create the train dataset, then split or create the ad hoc validation set
    train_dataset = create_dataset(config.dataset,
                                   path=data_root,
                                   subset="train",
                                   transform=train_transform,
                                   channels=config.in_channels)
    if not config.has_val:
        # Many datasets do not have a validation set or a test set with annotations.
        # In the first case (like this one), a portion of training is reserved for validation
        # In the second case, the validation becomes testing and the training is again split randomly
        train_mask, val_mask, _ = mask_set(len(train_dataset), val_size=config.val_size, test_size=0.0)
        LOG.debug("Creating val. set from training, split: %d - %d", len(train_mask), len(val_mask))
        val_dataset = create_dataset(config.dataset,
                                     path=data_root,
                                     subset="train",
                                     transform=eval_transform,
                                     channels=config.in_channels)
        train_dataset.add_mask(train_mask)
        val_dataset.add_mask(val_mask, stage="valid")
    else:
        # When the validation set is present, by all means use it
        val_dataset = create_dataset(config.dataset,
                                     path=data_root,
                                     subset="valid",
                                     transform=eval_transform,
                                     channels=config.in_channels)
    return train_dataset, val_dataset


def prepare_model(config: Configuration, task: Task) -> nn.Module:
    cfg = config.model
    encoder = create_encoder(name=cfg.encoder,
                             decoder=cfg.decoder,
                             pretrained=cfg.pretrained,
                             freeze=cfg.freeze,
                             output_stride=cfg.output_stride,
                             act_layer=cfg.act,
                             norm_layer=cfg.norm)
    decoder = create_decoder(name=cfg.decoder, feature_info=encoder.feature_info, act_layer=cfg.act, norm_layer=cfg.norm)
    # extract intermediate features when encoder KD is required
    extract_features = config.kd.encoder_factor > 0
    icl_model = ICLSegmenter(encoder, decoder, classes=task.num_classes_per_task(), return_features=extract_features)
    return icl_model


def prepare_metrics(task: Task, device: torch.device) -> Tuple[dict, dict]:
    # prepare metrics
    num_classes = task.old_class_count() + task.current_class_count()
    t_metrics = (Metrics.f1, Metrics.iou)
    v_metrics = (m for m in Metrics)
    train_metrics = {e.name: e.value(num_classes=num_classes, device=device) for e in t_metrics}
    valid_metrics = {e.name: e.value(num_classes=num_classes, device=device) for e in v_metrics}
    valid_metrics.update(dict(class_iou=IoU(num_classes=num_classes, reduction=None, device=device),
                              class_f1=F1Score(num_classes=num_classes, reduction=None, device=device)))
    LOG.debug("Train metrics: %s", str(list(train_metrics.keys())))
    LOG.debug("Eval. metrics: %s", str(list(valid_metrics.keys())))
    return train_metrics, valid_metrics


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
    accelerator = Accelerator(fp16=config.trainer.amp)

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
    segment_loss = config.ce.instantiate(ignore_index=255, old_class_count=task.old_class_count())
    distill_loss = config.kd.instantiate()
    if task.step > 0 and config.ce.unbiased:
        seg_loss_name = str(type(segment_loss))
        kdd_loss_name = str(type(distill_loss))
        assert seg_loss_name.startswith("Unbiased"), f"Wrong loss '{seg_loss_name}' for step {task.step}"
        assert kdd_loss_name.startswith("Unbiased"), f"Wrong loss '{kdd_loss_name}' for step {task.step}"
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
    sample_ids = list()
    if config.num_samples and config.visualize:
        sample_ids = np.random.choice(len(valid_loader), config.num_samples, replace=False)
    trainer = Trainer(accelerator=accelerator,
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
                      samples=sample_ids,
                      debug=config.debug)
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


def test(test_config: TestConfiguration):
    # assertions before starting
    assert test_config.name is not None, "Specify the experiment name to test!"
    assert torch.backends.cudnn.enabled, "AMP requires CUDNN backend to be enabled."

    # prepare the test log
    log_name = f"output-{test_config.task.step}-test.log"
    exp_id, out_folder, model_folder, logs_folder = init_experiment(config=test_config, log_name=log_name)
    config_path = out_folder / f"segmenter-config-s{test_config.task.step}.yaml"
    config = load_config(path=config_path, config_class=Configuration)

    # prepare accelerator
    accelerator = Accelerator(fp16=config.trainer.amp)
    accelerator.wait_for_everyone()

    # seeding everything
    LOG.info("Using seed: %d", config.seed)
    seed_everything(config.seed)
    # prepare datasets
    LOG.info("Loading test dataset...")
    eval_transform = test_transforms(in_channels=config.in_channels)
    LOG.debug("Eval. transforms: %s", str(eval_transform))
    # create the train dataset, then split or create the ad hoc validation set
    test_dataset = create_dataset(config.dataset,
                                  path=Path(config.data_root),
                                  subset="test",
                                  transform=eval_transform,
                                  channels=config.in_channels)
    add_background = not test_dataset.has_background()
    task = Task(dataset=config.dataset,
                name=config.task.name,
                step=config.task.step,
                add_background=add_background)
    test_set = ICLDataset(dataset=test_dataset, task=task, mask_value=0, overlap=config.task.overlap)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=test_config.trainer.batch_size,
                             shuffle=False,
                             num_workers=test_config.trainer.num_workers,
                             worker_init_fn=seed_worker)
    LOG.info("ICL sets  - Test set: %d samples", len(test_set))

    # prepare model for inference
    LOG.info("Preparing model...")
    model = prepare_model(config=config, task=task)
    ckpt_path = checkpoint_path(model_folder, task_name=task.name, step=task.step)
    assert ckpt_path.exists(), f"Checkpoint '{str(ckpt_path)}' not found"
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"), strict=True)
    LOG.info("Model restored from: %s", str(ckpt_path))

    # prepare losses
    segment_loss = config.ce.instantiate(ignore_index=255, old_class_count=task.old_class_count())
    distill_loss = config.kd.instantiate()
    # prepare metrics and logger
    logger = TensorBoardLogger(log_folder=logs_folder,
                               filename_suffix=f"step-{task.step}-test",
                               icl_step=task.step,
                               comment=config.comment)

    # prepare trainer
    sample_ids = list()
    if config.num_samples and config.visualize:
        sample_ids = np.random.choice(len(test_loader), config.num_samples, replace=False)
    trainer = Trainer(accelerator=accelerator,
                      task=task,
                      new_model=model,
                      old_model=None,
                      optimizer=None,
                      scheduler=None,
                      old_classes=test_set.old_categories(),
                      new_classes=test_set.new_categories(),
                      seg_criterion=segment_loss,
                      kdd_criterion=distill_loss,
                      kde_criterion=None,
                      kdd_lambda=config.kd.decoder_factor,
                      kde_lambda=config.kd.encoder_factor,
                      logger=logger,
                      samples=sample_ids,
                      stage="test",
                      debug=config.debug)
    trainer.add_callback(DisplaySamples(inverse_transform=inverse_transform(),
                                        color_palette=test_set.palette(),
                                        stage="test"))
    # prepare testing metrics, same as validation with the addition of a confusion matrix
    num_classes = task.old_class_count() + task.current_class_count()
    _, eval_metrics = prepare_metrics(task=task, device=accelerator.device)
    eval_metrics.update(dict(conf_mat=ConfusionMatrix(num_classes=num_classes, device=accelerator.device)))
    # execute predict
    losses, data = trainer.predict(test_dataloader=test_loader, metrics=eval_metrics, logger_exclude=["conf_mat"])
    scores = trainer.current_scores["test"]
    # logging stuff to file and storing images if required
    LOG.info("Testing completed, average loss: %.4f", np.mean(losses))
    LOG.info("Storing predicted images: %s", str(test_config.store_predictions).lower())

    if test_config.store_predictions:
        for i, (images, y_trues, y_preds) in tqdm(enumerate(data)):
            results_folder = prepare_folder(out_folder / "images" / f"step-{task.step}")
            save_grid(images, y_trues, y_preds,
                      filepath=results_folder,
                      filename=f"{i:06d}",
                      palette=test_set.palette())

    LOG.info("Average results:")
    classwise = dict()
    for i, (name, score) in enumerate(scores.items()):
        # only printing reduced metrics
        if score.ndim == 0:
            LOG.info(f"{name:<20s}: {score.item():.4f}")
        elif name != "conf_mat":
            classwise[name] = score

    LOG.info("Class-wise results:")
    header = f"{'score':<20s}  " + "|".join([f"{l:<15s}" for l in trainer.all_classes.values()])
    LOG.info(header)
    for i, (name, score) in enumerate(classwise.items()):
        scores_str = [f"{v:.4f}" for v in score]
        scores_str = "|".join(f"{s:<15s}" for s in scores_str)
        LOG.info(f"{name:<20s}: {scores_str}")

    LOG.info("Plotting confusion matrix...")
    cm_name = f"cm_{Path(ckpt_path).stem}"
    plot_folder = prepare_folder(out_folder / "plots")
    plot_confusion_matrix(scores["conf_mat"].cpu().numpy(),
                          destination=plot_folder / f"{cm_name}.png",
                          labels=trainer.all_classes.values(),
                          title=cm_name,
                          normalize=False)
    LOG.info("Testing done!")

from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

import tqdm
from saticl.config import Configuration, SSLConfiguration, TestConfiguration
from saticl.datasets import create_dataset
from saticl.datasets.icl import ICLDataset
from saticl.datasets.transforms import inverse_transform, test_transforms
from saticl.logging.tensorboard import TensorBoardLogger
from saticl.metrics import ConfusionMatrix
from saticl.prepare import prepare_metrics, prepare_model, prepare_model_ssl
from saticl.tasks import Task
from saticl.trainer.base import Trainer
from saticl.trainer.callbacks import DisplaySamples
from saticl.trainer.ssl import SSLTrainer
from saticl.utils.common import get_logger, load_config, prepare_folder
from saticl.utils.ml import (
    checkpoint_path,
    init_experiment,
    plot_confusion_matrix,
    save_grid,
    seed_everything,
    seed_worker,
)


LOG = get_logger(__name__)


def test(test_config: TestConfiguration):
    # assertions before starting
    assert test_config.name is not None, "Specify the experiment name to test!"
    assert torch.backends.cudnn.enabled, "AMP requires CUDNN backend to be enabled."

    # prepare the test log
    log_name = f"output-{test_config.task.step}-test.log"
    exp_id, out_folder, model_folder, logs_folder = init_experiment(config=test_config, log_name=log_name)
    config_path = out_folder / f"segmenter-config-s{test_config.task.step}.yaml"
    config: Configuration = load_config(path=config_path, config_class=Configuration)

    # prepare accelerator
    accelerator = Accelerator(fp16=config.trainer.amp, cpu=config.trainer.cpu)
    accelerator.wait_for_everyone()

    # seeding everything
    LOG.info("Using seed: %d", config.seed)
    seed_everything(config.seed)
    # prepare datasets
    subset = "valid" if test_config.test_on_val else "test"
    LOG.info("Loading %s dataset...", subset)
    eval_transform = test_transforms(in_channels=config.in_channels)
    LOG.debug("Eval. transforms: %s", str(eval_transform))
    # create the train dataset, then split or create the ad hoc validation set
    test_dataset = create_dataset(config.dataset,
                                  path=Path(config.data_root),
                                  subset=subset,
                                  transform=eval_transform,
                                  channels=config.in_channels)
    add_background = not test_dataset.has_background()
    task = Task(dataset=config.dataset,
                name=config.task.name,
                step=config.task.step,
                add_background=add_background)
    test_set = ICLDataset(dataset=test_dataset, task=task, mask_value=0, overlap=config.task.overlap, mask_old=False)
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
    weights = None
    if config.class_weights:
        weights = test_set.load_class_weights(Path(config.class_weights),
                                              device=accelerator.device,
                                              normalize=config.ce.tversky)
        LOG.info("Using class weights: %s", str(weights))
    segment_loss = config.ce.instantiate(ignore_index=255, old_class_count=task.old_class_count(), weight=weights)
    distill_loss = config.kd.instantiate()
    # prepare metrics and logger
    logger = TensorBoardLogger(log_folder=logs_folder,
                               filename_suffix=f"step-{task.step}-test",
                               icl_step=task.step,
                               comment=config.comment)

    # prepare trainer
    LOG.info("Visualize: %s, num. batches for visualization: %s", str(config.visualize), str(config.num_samples))
    num_samples = int(config.visualize) * config.num_samples
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
                      samples=num_samples,
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
    losses, data = trainer.predict(test_dataloader=test_loader,
                                   metrics=eval_metrics,
                                   logger_exclude=["conf_mat"],
                                   return_preds=test_config.store_predictions)
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


def test_ssl(test_config: TestConfiguration):
    # assertions before starting
    assert test_config.name is not None, "Specify the experiment name to test!"
    assert torch.backends.cudnn.enabled, "AMP requires CUDNN backend to be enabled."

    # prepare the test log
    log_name = f"output-{test_config.task.step}-test.log"
    exp_id, out_folder, model_folder, logs_folder = init_experiment(config=test_config, log_name=log_name)
    config_path = out_folder / f"segmenter-config-s{test_config.task.step}.yaml"
    config: Configuration = load_config(path=config_path, config_class=SSLConfiguration)

    # prepare accelerator
    accelerator = Accelerator(fp16=config.trainer.amp, cpu=config.trainer.cpu)
    accelerator.wait_for_everyone()

    # seeding everything
    LOG.info("Using seed: %d", config.seed)
    seed_everything(config.seed)
    # prepare datasets
    subset = "valid" if test_config.test_on_val else "test"
    LOG.info("Loading %s dataset...", subset)
    eval_transform = test_transforms(in_channels=config.in_channels)
    LOG.debug("Eval. transforms: %s", str(eval_transform))
    # create the train dataset, then split or create the ad hoc validation set
    test_dataset = create_dataset(config.dataset,
                                  path=Path(config.data_root),
                                  subset=subset,
                                  transform=eval_transform,
                                  channels=config.in_channels)
    add_background = not test_dataset.has_background()
    task = Task(dataset=config.dataset,
                name=config.task.name,
                step=config.task.step,
                add_background=add_background)
    test_set = ICLDataset(dataset=test_dataset, task=task, mask_value=0, overlap=config.task.overlap, mask_old=False)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=test_config.trainer.batch_size,
                             shuffle=False,
                             num_workers=test_config.trainer.num_workers,
                             worker_init_fn=seed_worker)
    LOG.info("ICL sets  - Test set: %d samples", len(test_set))

    # prepare model for inference, we only require the main ICL model, not the SSL pretext classifier
    LOG.info("Preparing model...")
    model, _ = prepare_model_ssl(config=config, task=task)
    ckpt_path = checkpoint_path(model_folder, task_name=task.name, step=task.step)
    assert ckpt_path.exists(), f"Checkpoint '{str(ckpt_path)}' not found"
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"), strict=True)
    LOG.info("Model restored from: %s", str(ckpt_path))

    # prepare losses
    weights = None
    if config.class_weights:
        weights = test_set.load_class_weights(Path(config.class_weights),
                                              device=accelerator.device,
                                              normalize=config.ce.tversky)
        LOG.info("Using class weights: %s", str(weights))
    segment_loss = config.ce.instantiate(ignore_index=255, old_class_count=task.old_class_count(), weight=weights)
    distill_loss = config.kd.instantiate()
    # prepare metrics and logger
    logger = TensorBoardLogger(log_folder=logs_folder,
                               filename_suffix=f"step-{task.step}-test",
                               icl_step=task.step,
                               comment=config.comment)

    # prepare trainer
    LOG.info("Visualize: %s, num. batches for visualization: %s", str(config.visualize), str(config.num_samples))
    num_samples = int(config.visualize) * config.num_samples
    trainer = SSLTrainer(accelerator=accelerator,
                         task=task,
                         new_model=model,
                         old_model=None,
                         ssl_model=None,
                         optimizer=None,
                         scheduler=None,
                         old_classes=test_set.old_categories(),
                         new_classes=test_set.new_categories(),
                         seg_criterion=segment_loss,
                         kdd_criterion=distill_loss,
                         kde_criterion=None,
                         ssl_criterion=None,
                         kdd_lambda=config.kd.decoder_factor,
                         kde_lambda=config.kd.encoder_factor,
                         logger=logger,
                         samples=num_samples,
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

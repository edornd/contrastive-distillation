import io
import os
import random
from contextlib import redirect_stdout
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

import seaborn as sns
from matplotlib import pyplot as plt
from pydantic.env_settings import BaseSettings
from saticl.datasets.transforms import Denormalize
from saticl.utils import common as utils
from timm.models.tresnet import SpaceToDepthModule
from torchsummary import summary

F32_EPS = np.finfo(np.float32).eps
F16_EPS = np.finfo(np.float16).eps


def identity(*args: Any) -> Any:
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def string_summary(model: torch.nn.Module, input_size: Tuple[int, int, int], batch_size: int = -1, device: str = "cpu"):
    output: str = None
    with io.StringIO() as buf, redirect_stdout(buf):
        summary(model, input_size=input_size, batch_size=batch_size, device=device)
        output = buf.getvalue()
    return output


def checkpoint_path(model_folder: Path, task_name: str, step: int) -> Path:
    return model_folder / f"task{task_name}_step-{step}.pth"


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight)
    elif isinstance(module, (nn.SyncBatchNorm, nn.BatchNorm2d)):
        module.weight.data.fill_(1)
        module.bias.data.zero_()


def compute_class_weights(data: Dict[Any, int], smoothing: float = 0.15, clip: float = 10.0):
    assert smoothing >= 0 and smoothing <= 1, "Smoothing factor out of range"
    if smoothing > 0:
        # the larger the smooth factor, the bigger the quantities to sum to the remaining counts (additive smoothing)
        smoothed_maxval = max(list(data.values())) * smoothing
        for k in data.keys():
            data[k] += smoothed_maxval
    # retrieve the (new) max value, divide by counts, round to 2 digits and clip to the given value
    # max / value allows to keep the majority class' weights to 1, while the others will be >= 1 and <= clip
    majority = max(data.values())
    return {k: np.clip(round(float(majority / v), ndigits=2), 0, clip) for k, v in data.items()}


def load_class_weights(weights_path: Path) -> torch.Tensor:
    # load class weights, if any
    if weights_path is None or not weights_path.exists() or not weights_path.is_file():
        raise ValueError(f"Path '{str(weights_path)}' does not exist or it's not a numpy array")

    weights = np.load(weights_path).astype(np.float32)
    return torch.from_numpy(weights)


def one_hot(target: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
    """source: https://github.com/PhoenixDL/rising. Computes one-hot encoding of input tensor.
    Args:
        target (torch.Tensor): tensor to be converted
        num_classes (Optional[int], optional): number of classes. If None, the maximum value of target is used.
    Returns:
        torch.Tensor: one-hot encoded tensor of the target
    """
    if num_classes is None:
        num_classes = int(target.max().detach().item() + 1)
    dtype, device = target.dtype, target.device
    target_onehot = torch.zeros(*target.shape, num_classes, dtype=dtype, device=device)
    return target_onehot.scatter_(1, target.unsqueeze_(1), 1.0)


def _copy_channel(layer: nn.Module, channel: int = 0, num_copies: int = 1) -> torch.Tensor:
    input_weights = layer.weight
    extra_weights = input_weights[:, channel].unsqueeze(dim=1).repeat(1, num_copies, 1, 1)    # make it  [64, n, 7, 7]
    return torch.cat((input_weights, extra_weights), dim=1)    # obtain  [64, 3+n, 7, 7]


def _copy_channel_depthwise(layer: nn.Module, copy_block: int = 0, num_copies: int = 1) -> torch.Tensor:
    # layer weight should be a 3x3 conv [64, 48, 3, 3]
    input_weights = layer.weight
    maps_per_channel = input_weights.shape[1] // 3    # divide 48 by RGB input
    idx_a = copy_block * maps_per_channel
    idx_b = (copy_block + 1) * maps_per_channel
    extra_weights = input_weights[:, idx_a:idx_b].repeat(1, num_copies, 1, 1)    # obtain [64, 16*n, 3, 3] weights
    return torch.cat((input_weights, extra_weights), dim=1)    # concat to make them [64, 48+16*n, 3, 3]


def expand_input(model: nn.Module, input_layer: str = None, copy_channel: int = 0, num_copies: int = 1) -> nn.Module:
    # when we know the layer name
    if input_layer is not None:
        model[input_layer].weight = nn.Parameter(
            _copy_channel(model[input_layer], channel=copy_channel, num_copies=num_copies))
    else:
        children = list(model.children())
        input_layer = children[0]
        while children and len(children) > 0:
            input_layer = children[0]
            children = list(children[0].children())

        assert not list(input_layer.children()), f"layer '{input_layer}' still has children!"
        if isinstance(input_layer, SpaceToDepthModule):
            input_layer = model["body_conv1"][0]
            input_layer.weight = nn.Parameter(
                _copy_channel_depthwise(input_layer, copy_block=copy_channel, num_copies=num_copies))
        else:
            input_layer.weight = nn.Parameter(_copy_channel(input_layer, channel=copy_channel, num_copies=num_copies))

    return model


def init_experiment(config: BaseSettings, log_name: str = "output.log"):
    # initialize experiment
    experiment_id = config.name or utils.current_timestamp()
    out_folder = Path(config.output_folder) / config.dataset / config.task.name
    # prepare folders and log outputs
    output_folder = utils.prepare_folder(out_folder, experiment_id=experiment_id)
    utils.prepare_file_logging(output_folder, filename=log_name)

    # prepare experiment directories
    model_folder = utils.prepare_folder(output_folder / "models")
    logs_folder = utils.prepare_folder(output_folder / "logs")
    if config.name is not None:
        assert model_folder.exists() and logs_folder.exists()
    return experiment_id, output_folder, model_folder, logs_folder


def find_best_checkpoint(folder: Path, model_name: str = "*.pth", divider: str = "_") -> Path:
    wildcard_path = folder / model_name
    models = list(glob(str(wildcard_path)))
    assert len(models) > 0, f"No models found for pattern '{wildcard_path}'"
    current_best = None
    current_best_metric = None

    for model_path in models:
        model_name = os.path.basename(model_path).replace(".pth", "")
        mtype, _, metric_str = model_name.split(divider)
        assert mtype == "classifier" or mtype == "segmenter", f"Unknown model type '{mtype}'"
        model_metric = float(metric_str.split("-")[-1])
        if not current_best_metric or current_best_metric < model_metric:
            current_best_metric = model_metric
            current_best = model_path

    return current_best


def plot_confusion_matrix(cm: np.ndarray,
                          destination: Path,
                          labels: List[str],
                          title: str = "confusion matrix",
                          normalize: bool = True) -> None:
    # annot=True to annotate cells, ftm='g' to disable scientific notation
    fig = plt.figure(figsize=(6, 6))
    if normalize:
        cm /= cm.max()
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(title)
    # set labels and ticks
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    # save figure
    fig.savefig(str(destination))


def mask_to_rgb(mask: np.ndarray, palette: Dict[int, tuple]) -> np.ndarray:
    """Given an input batch, or single picture with dimensions [B, H, W] or [H, W], the utility generates
    an equivalent [B, H, W, 3] or [H, W, 3] array corresponding to an RGB version.
    The conversion uses the given palette, which should be provided as simple dictionary of indices and tuples, lists
    or arrays indicating a single RGB color. (e.g. {0: (255, 255, 255)})
    Args:
        mask (np.ndarray): input mask of indices. Each index should be present in the palette
        palette (Dict[int, tuple]): dictionary of pairs <index - color>, where colors can be provided in RGB tuple fmt
    Returns:
        np.ndarray: tensor containing the RGB version of the input index tensor
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for index, color in palette.items():
        lut[index] = np.array(color, dtype=np.uint8)
    return lut[mask]


def make_grid(inputs: np.ndarray, rgb_true: np.ndarray, rgb_pred: np.ndarray) -> np.ndarray:
    assert inputs.ndim == 3, "Input must be a single RGB image (channels last)"
    assert inputs.shape == rgb_true.shape == rgb_pred.shape, \
        f"Shapes not matching: {inputs.shape}, {rgb_true.shape}, {rgb_pred.shape}"
    # image = Denormalize()(input_batch[0]).cpu().numpy()[:3].transpose(1, 2, 0)
    # image = (image * 255).astype(np.uint8)
    return np.concatenate((inputs, rgb_true, rgb_pred), axis=1).astype(np.uint8)


def save_grid(inputs: torch.Tensor,
              targets: torch.Tensor,
              preds: torch.Tensor,
              filepath: Path,
              filename: str,
              palette: Dict[int, tuple],
              offset: int = 0) -> None:
    assert targets.shape == preds.shape, f"Shapes not matching: {targets.shape}, {preds.shape}"
    assert inputs.ndim >= 3, "Image must be at least a 3-channel tensor (channels first)"
    if inputs.ndim == 4:
        for i in range(inputs.shape[0]):
            save_grid(inputs[i], targets[i], preds[i], filepath=filepath, filename=filename, palette=palette, offset=i)
    else:
        image = (Denormalize()(inputs) * 255).astype(np.uint8)
        # targets and predictions still have a channel dim
        if targets.ndim > 2:
            targets = targets.squeeze(0)
            preds = preds.squeeze(0)
        rgb_true = mask_to_rgb(targets.numpy(), palette=palette)
        rgb_pred = mask_to_rgb(preds.numpy(), palette=palette)
        grid = make_grid(image, rgb_true, rgb_pred)
        plt.imsave(filepath / f"{filename}-{offset}.png", grid)


def mask_set(dataset_length: int,
             val_size: float = 0.1,
             test_size: float = 0.1) -> Tuple[List[bool], List[bool], List[bool]]:
    """Returns three boolean arrays of length `dataset_length`,
    representing the train set, validation set and test set. These
    arrays can be passed to `Dataset.add_mask` to yield the appropriate
    datasets.
    """
    mask = np.random.rand(dataset_length)
    train_mask = mask < (1 - (val_size + test_size))
    val_mask = (mask >= (1 - (val_size + test_size))) & (mask < 1 - test_size)
    test_mask = mask >= (1 - test_size)
    # just checking indices
    train_indices = np.argwhere(np.array(train_mask))
    valid_indices = np.argwhere(np.array(val_mask))
    test_indices = np.argwhere(np.array(test_mask))
    assert len(np.intersect1d(train_indices, valid_indices)) == 0
    assert len(np.intersect1d(train_indices, test_indices)) == 0
    # we checked they are not overlapping, good to go
    return train_mask, val_mask, test_mask

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch

from ordered_set import OrderedSet
from saticl.datasets.base import DatasetBase
from saticl.tasks import Task
from saticl.utils.common import get_logger
from saticl.utils.ml import load_class_weights

LOG = get_logger(__name__)


class ICLDataset(DatasetBase):

    def __init__(self,
                 dataset: DatasetBase,
                 task: Task,
                 mask_value: int = 0,
                 filter_mode: str = "overlap",
                 mask_old: bool = True) -> None:
        super().__init__()
        self.dataset = dataset
        self.task = task
        self._categories = dataset.categories().copy()
        self._palette = dataset.palette().copy()
        # if the dataset doesn't include the background on its own, add it manually
        if not dataset.has_background():
            self._categories = {min(k + 1, 255): v for k, v in self._categories.items()}
            self._palette = {min(k + 1, 255): v for k, v in self._palette.items()}
            self._categories.update({0: "background"})
            self._palette.update({0: (0, 0, 0)})
        # check the (original, non-shifted) task labels are in the dataset
        assert all([index in dataset.categories() for index in task.seen_labels]), \
            f"Label index out of bounds for dataset {dataset.name()}"
        # safe to proceed
        self.mask_value = mask_value
        self._has_background = dataset.has_background()
        self.dataset.add_mask(task.filter_images(dataset, mode=str(filter_mode)))
        # prepare lookup tables to transform from normal -> ICL indices
        # first shift labels if the background is not already included and set 0 as first
        new_labels = self._process_labels(task.new_labels)
        old_labels = self._process_labels(task.old_labels)
        # prepare a dictionary label -> index for old + new labels (old + new <= full)
        self.new_labels = new_labels
        self.old_labels = old_labels
        self.label2index = old_labels.union(new_labels).map
        self.label2index[dataset.ignore_index()] = mask_value
        self.index2label = {v: k for k, v in self.label2index.items()}
        # prepare a similar lookup, including all available classes
        self.label_transform = dict()
        for key in self._categories:
            available = new_labels if mask_old else old_labels.union(new_labels)
            substitute = self.label2index.get(key, mask_value) if key in available else mask_value
            self.label_transform[key] = substitute

    def _process_labels(self, labels: OrderedSet) -> OrderedSet:
        shift = 0 if self._has_background else 1
        tmp = OrderedSet([x + shift for x in labels])
        # move any background in the wrong place to index 0
        tmp.discard(0)
        return OrderedSet([0]).union(tmp)

    def _remap(self, label: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """(relatively) fast mapping exploiting the `unique` functions, in numpy and torch.
        The main idea is: first track unique values and their indices (where they are located),
        then swap each unique value with its match and use the inverse mapping to remap the full tensor.
        The shape must be rebuilt after the lookup.

        Args:
            label (Union[np.ndarray, torch.Tensor]): input label, as np array or torch tensor, depends on dataset
                                                     and whether we applied a transform or not before this.

        Returns:
            Union[np.ndarray, torch.Tensor]: returns a matrix with same shape and type, but with remapped labels,
                                             following the current incremental step.
        """
        if isinstance(label, torch.Tensor):
            unique, inverse = torch.unique(label, return_inverse=True)
            result = torch.tensor([self.label_transform[x.item()] for x in unique])[inverse].reshape(label.shape)
        else:
            # for numpy is actually the same, just without the item()
            unique, inverse = np.unique(label, return_inverse=True)
            result = np.array([self.label_transform[x] for x in unique])[inverse].reshape(label.shape)
        return result

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, label = self.dataset[index]
        # increment (valid) indices by 1 to account for background at 0
        if not self._has_background:
            label[label != self.ignore_index()] += 1
        # use the lookup dictionary for indexing
        if self.label_transform is not None:
            # alternative: label.apply_(lambda x: self.label_transform[x])
            label = self._remap(label)
        return image, label

    def name(self) -> str:
        return self.dataset.name()

    def stage(self) -> str:
        return self.dataset.stage()

    def categories(self) -> Dict[int, str]:
        return self._categories

    def palette(self) -> Dict[int, tuple]:
        return {k: self._palette[v] for k, v in self.index2label.items()}

    def old_categories(self) -> Dict[int, str]:
        return OrderedDict((i, self._categories[i]) for i in self.old_labels)

    def new_categories(self) -> Dict[int, str]:
        return OrderedDict((i, self._categories[i]) for i in self.new_labels)

    def add_mask(self, mask: List[bool], stage: str = None) -> None:
        return super().add_mask(mask, stage)

    def ignore_index(self) -> int:
        return self.dataset.ignore_index()

    def has_background(self) -> bool:
        return True

    def load_class_weights(self, weights_path: Path, device: torch.device, normalize: bool = False) -> torch.Tensor:
        if not weights_path:
            return None
        # load the weights for the standard classes, add background if not included (min. weight)
        weights = load_class_weights(weights_path=weights_path)
        if not self._has_background:
            weights = torch.cat((weights.min().view(1), weights))
        # this are the standard class weights, we need to remap them
        labels = self.old_labels.union(self.new_labels)
        remapped = torch.tensor([weights[l] for l in labels])
        if normalize:
            remapped /= remapped.max()
        return remapped.to(device)

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

import numpy as np

from ordered_set import OrderedSet
from saticl.datasets.base import DatasetBase
from saticl.logging.console import DistributedLogger
from saticl.tasks.agrivision import ICL_AGRIVISION
from saticl.tasks.isaid import ICL_ISAID
from saticl.tasks.isprs import ICL_ISPRS
from saticl.utils.common import prepare_folder
from tqdm import tqdm

LOG = DistributedLogger(logging.getLogger(__name__))

AVAILABLE_TASKS = {"potsdam": ICL_ISPRS, "vaihingen": ICL_ISPRS, "agrivision": ICL_AGRIVISION, "isaid": ICL_ISAID}


def filter_with_overlap(image_labels: Iterable[int], new_labels: Iterable[int], *args, **kwargs) -> bool:
    """Returns whether the current image must be maintained or discarded for the given step,
    based on the labels on the current image and the labels required at the step.

    Args:
        image_labels (List[int]): indices of the labels present in the current image
        new_labels (List[int]): indices of the labels needed at the current step

    Returns:
        bool: true if any of the current labels are present, false otherwise
    """
    return any(x in new_labels for x in image_labels)


def filter_without_overlap(image_labels: Iterable[int], new_labels: Iterable[int], curr_labels: Iterable[int]) -> bool:
    """Filters out any image that contains data with no labels belonging to the current step, including
    those images that contain future labels (potentially dangerous if an image contains more or less every label).

    Args:
        image_labels (List[int]): indices of unique labels for the current image
        new_labels (List[int]): indices of labels for the step T
        curr_labels (List[int]): indices of labels from steps 1 .. T - 1 + labels from step T + [0, 255]

    Returns:
        bool: true whether the image must be kept, false otherwise
    """
    # check whether at least one of the labels of the current image is present in the current step labels
    contains_new_labels = any(x in new_labels for x in image_labels)
    # also check that the image ONLY contains labels from the current step or previous steps
    no_future_labels = all(x in curr_labels for x in image_labels)
    return contains_new_labels and no_future_labels


def filter_with_split(dataset: DatasetBase, new_labels: set):
    """Subdivides the given dataset into equal partitions, one per category (excluding background, if present).
    If the dataset has N classes, this function divides into N splits, where each split i contributes with only
    the label i.

    Args:
        dataset (DatasetBase): dataset to be filtered
        new_labels (set): set of labels for current step

    Returns:
        List[bool]: list of values to be kept for the current step
    """
    # count samples and classes, we do not care about background for splits
    num_samples = len(dataset)
    shift = int(dataset.has_background())
    num_classes = len(dataset.categories()) - shift
    # create a dict of <index label: list of tiles> and a supporting count array
    label2tile = defaultdict(list)
    shuffled = random.sample(list(range(num_samples)), k=num_samples)
    tile_counts = np.zeros(num_classes)

    for i in tqdm(shuffled):
        _, mask = dataset[i]
        # extract unique labels, remove background if it's included
        available_labels = np.unique(mask)
        if dataset.has_background():
            available_labels = available_labels[available_labels != 0]
        # retrieve the currently less populated category (excluding background if present)
        # e.g. tile counts[ 34, 45, 12] -> index = 2
        # then use this index to retrieve the corresponding label
        # last, store the tile for that label and increment the count
        index = np.argmin(tile_counts[available_labels - shift])
        label = available_labels[index]
        label2tile[label].append(i)
        tile_counts[label - shift] += 1
    # create a list of booleans, one per sample, true when included, false otherwise
    filtered = [False] * num_samples
    for label, tiles in label2tile.items():
        for index in tiles:
            filtered[index] = label in new_labels
    return filtered


class Task:

    def __init__(self,
                 dataset: str,
                 name: str,
                 step: int = 0,
                 add_background: bool = False,
                 data_folder: Path = Path("data/tasks")) -> None:
        # sanity checks:
        # - the data folder can be used to cache indices
        # - the dataset exists in the task list
        # - the task name appears in the entries associated with the dataset
        # - the step exists in the given task
        assert data_folder.exists() and data_folder.is_dir(), f"Wrong path: {str(data_folder)}"
        assert dataset in AVAILABLE_TASKS, f"No tasks for dataset: {dataset}"
        tasks = AVAILABLE_TASKS.get(dataset, {})
        assert name in tasks, f"Unknown task: {name}"
        task_dict = tasks[name]
        assert step in task_dict, f"Step {step} out of range for: {task_dict}"
        # we made sure dataset and task exist, and the step is within range
        self.task_dict = task_dict
        new_labels = OrderedSet([label for label in task_dict[step]])
        old_labels = OrderedSet([label for s in range(step) for label in task_dict[s]])
        # step 0 - sanity check: only new labels by definitions
        if step == 0:
            assert len(old_labels) == 0 and len(new_labels) > 0, "step 0: expected only new labels"
        # step N - sanity check: old and new are non-empty and disjoint sets
        else:
            assert len(new_labels) > 0 and len(old_labels) > 0, f"step {step}: Old and new must be non-empty sets"
            assert not new_labels.intersection(old_labels), "Old and new labels are not disjoint sets"
        self.seen_labels = new_labels.union(old_labels)
        self.new_labels = new_labels
        self.old_labels = old_labels
        self.data_root = data_folder
        # save information, save shift for sets without background class
        # useful for actual class counts
        self.shift = int(add_background)
        self.dataset_name = dataset
        self.name = name
        self.step = step

    def task_name(self) -> str:
        return f"{self.name}_step-{self.step}"

    def num_classes_per_task(self) -> List[int]:
        """Counts the number of classes for each step, including the current one.
        Future steps are not involved yet. In case of dataset with missing background,
        a shift is applied (it does nothing when `add_background` is false, since shift=0)

        Returns:
            List[int]: list containing the class count at each step from 0 to t
        """
        counts = [len(self.task_dict[s]) for s in range(self.step + 1)]
        counts[0] += self.shift
        return counts

    def old_class_count(self) -> int:
        """Counts the total amount of classes seen until now, excluding the current step.
        Formally, returns sum(classes_0 ... classes_t-1).
        Shift accounts for datasets missing background.

        Returns:
            int: [description]
        """
        if self.step == 0:
            return 0
        return sum([len(self.task_dict[s]) for s in range(self.step)]) + self.shift

    def current_class_count(self) -> int:
        """Returns the number of classes at the current step.
        If we are at step 0 and the dataset doesn't have its own background class, add 1.

        Returns:
            int: class count at step t
        """
        shift = int(self.step == 0) * self.shift
        return len(self.task_dict[self.step]) + shift

    def filter_images(self, dataset: DatasetBase, mode: str = "overlap") -> List[bool]:
        """Iterates the given dataset, storing in a numpy array which image indices are fit
        for the current task. The fit criterion is defined with or without overlap.
        WITH OVERLAP (default): the image is kept when the mask contains one of the task labels,
                                regardless of the other labels in the mask.
        WITHOUT OVERLAP: the image is kept when the mask contains one of the task labels AND
                         the other labels are ∈ old labels (i.e. does not include labels from future tasks)
        WITH SPLIT: first, the dataset is divided into evenly sized chunks of N/num_classes samples each.
                    Then, each chunk_i is assigned to class i, removing any tile that does not contain any pixel
                    with label=i.
                    The image is kept at step T if (and only if) it belongs to the chunk i and contains
                    at least one pixel labeled as i, for each i in C_T, the set of current new classes.

        Args:
            dataset (DatasetBase): original dataset to be filtered
            overlap (str, optional): filter with overlap, without overlap, or splitting. Defaults to overlap.

        Returns:
            List[bool]: a list of bool values, one per image, where true means to keep it.
        """
        assert mode in ("overlap", "noov", "split")
        postfix = "" if mode == "overlap" else f"_{mode}"
        cached_name = f"{self.task_name()}_{dataset.stage()}{postfix}.npy"
        cached_path = self.data_root / self.dataset_name / cached_name
        if cached_path.exists() and cached_path.is_file():
            filtered = np.load(cached_path)
        else:
            filtered = list()
            if mode != "split":
                # first select the right function for the filtering
                # then iterate dataset to decide which image to keep and which not
                filter_fn = filter_with_overlap if mode == "overlap" else filter_without_overlap
                for _, mask in tqdm(dataset, desc=f"Creating {cached_name}"):
                    mask_indices = np.unique(mask)
                    filtered.append(filter_fn(mask_indices, self.new_labels, self.seen_labels))
            else:
                filtered = filter_with_split(dataset, new_labels=self.new_labels)

            # create a numpy array of indices, then store it to file
            filtered = np.array(filtered)
            cached_path = prepare_folder(cached_path.parent)
            np.save(str(cached_path / cached_name), filtered)
        assert any(filtered), "Current filter does not include any images"
        return filtered

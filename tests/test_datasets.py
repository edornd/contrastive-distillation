import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import albumentations as alb
from albumentations.pytorch import ToTensorV2
from saticl.datasets import create_dataset
from saticl.datasets.icl import ICLDataset
from saticl.datasets.isprs import PotsdamDataset
from saticl.datasets.transforms import test_transforms, train_transforms
from saticl.tasks import Task
from saticl.utils.ml import mask_set, seed_everything
from tqdm import tqdm

LOG = logging.getLogger(__name__)


def test_dataset_potsdam(potsdam_path: Path):
    dataset = PotsdamDataset(potsdam_path, subset="train", include_dsm=False, transform=None, channels=3)
    assert len(dataset.categories()) == 6
    image, mask = dataset.__getitem__(0)
    assert image.shape == (512, 512, 3)
    assert mask.shape == (512, 512)
    assert image.min() >= 0 and image.max() <= 255
    # zero is not included unless we're in incremental learning
    assert mask.min() >= 0 and mask.max() <= 5


def test_dataset_potsdam_ir(potsdam_path: Path):
    dataset = PotsdamDataset(potsdam_path, subset="train", include_dsm=False, transform=None, channels=4)
    assert len(dataset.categories()) == 6
    image, mask = dataset.__getitem__(0)
    assert image.shape == (512, 512, 4)
    assert mask.shape == (512, 512)
    assert image.min() >= 0 and image.max() <= 255
    # zero is not included unless we're in incremental learning
    assert mask.min() >= 0 and mask.max() <= 5


def test_dataset_potsdam_ir_transform(potsdam_path: Path):
    transforms = alb.Compose(
        [alb.Normalize(mean=(0.485, 0.456, 0.406, 0.485), std=(0.229, 0.224, 0.225, 0.229)),
         ToTensorV2()])
    dataset = PotsdamDataset(potsdam_path, subset="train", include_dsm=False, transform=transforms, channels=4)
    assert len(dataset.categories()) == 6
    image, mask = dataset.__getitem__(0)
    assert image.shape == (4, 512, 512)
    assert mask.shape == (512, 512)
    assert image.min() >= -100 and image.max() <= +100
    assert mask.min() >= 0 and mask.max() <= 5


def test_dataset_potsdam_icl_step0(potsdam_path: Path):
    # instantiate transforms for training
    seed_everything(1337)
    train_transform = test_transforms(in_channels=4)
    eval_transform = test_transforms(in_channels=4)
    # create the train dataset, then split or create the ad hoc validation set
    train_dataset = create_dataset("potsdam", path=potsdam_path, subset="train", transform=train_transform, channels=4)
    train_mask, val_mask, _ = mask_set(len(train_dataset), val_size=0.1, test_size=0.0)
    val_dataset = create_dataset("potsdam", path=potsdam_path, subset="train", transform=eval_transform, channels=4)
    train_dataset.add_mask(train_mask)
    val_dataset.add_mask(val_mask, stage="valid")
    original_length = len(train_dataset)

    task = Task(dataset="potsdam", name="222a")
    icl_set = ICLDataset(train_dataset, task, overlap=True)
    # check it includes the previously missing background + handpicked check
    assert len(icl_set.categories()) == 7
    assert icl_set.categories()[1] == "impervious_surfaces"
    assert len(icl_set) < original_length
    # some logs
    LOG.info("ICL map    : %s", str(icl_set.label2index))
    LOG.info("ICL inverse: %s", str(icl_set.index2label))
    LOG.info("ICL transf : %s", str(icl_set.label_transform))
    assert icl_set.label2index == {0: 0, 1: 1, 3: 2, 255: 0}
    assert icl_set.index2label == {0: 255, 1: 1, 2: 3}
    assert icl_set.label_transform == {1: 1, 2: 0, 3: 2, 4: 0, 5: 0, 6: 0, 0: 0}
    # check every image
    loader = DataLoader(icl_set, batch_size=4, num_workers=8)
    for image, mask in tqdm(loader):
        assert image.shape == (4, 4, 512, 512)
        assert mask.shape == (4, 512, 512)
        # labels = np.unique(mask.numpy())
        # LOG.info(str(labels))
        assert torch.all(sum(mask == i for i in (0, 1, 2)).bool())


def test_dataset_potsdam_icl_step1(potsdam_path: Path):
    # instantiate transforms for training
    seed_everything(1337)
    train_transform = train_transforms(image_size=512, in_channels=4)
    eval_transform = test_transforms(in_channels=4)
    # create the train dataset, then split or create the ad hoc validation set
    train_dataset = create_dataset("potsdam", path=potsdam_path, subset="train", transform=train_transform, channels=4)
    train_mask, val_mask, _ = mask_set(len(train_dataset), val_size=0.1, test_size=0.0)
    val_dataset = create_dataset("potsdam", path=potsdam_path, subset="train", transform=eval_transform, channels=4)
    train_dataset.add_mask(train_mask)
    val_dataset.add_mask(val_mask, stage="valid")
    original_length = len(train_dataset)

    task = Task(dataset="potsdam", name="222a", step=1)
    icl_set = ICLDataset(train_dataset, task, overlap=True)
    # check it includes the previously missing background + handpicked check
    assert len(icl_set.categories()) == 7
    assert icl_set.categories()[1] == "impervious_surfaces"
    assert len(icl_set) < original_length
    # some logs
    LOG.info("ICL map    : %s", str(icl_set.label2index))
    LOG.info("ICL inverse: %s", str(icl_set.index2label))
    LOG.info("ICL transf : %s", str(icl_set.label_transform))
    assert icl_set.label2index == {0: 0, 1: 1, 3: 2, 2: 3, 4: 4, 255: 0}
    assert icl_set.index2label == {0: 255, 1: 1, 2: 3, 3: 2, 4: 4}
    assert icl_set.label_transform == {1: 1, 2: 3, 3: 2, 4: 4, 5: 0, 6: 0, 0: 0}
    # check every image
    loader = DataLoader(icl_set, batch_size=4, num_workers=8)
    for image, mask in tqdm(loader):
        assert image.shape == (4, 4, 512, 512)
        assert mask.shape == (4, 512, 512)
        # labels = np.unique(mask.numpy())
        # LOG.info(str(labels))
        assert torch.all(sum(mask == i for i in (0, 1, 2, 3, 4)).bool())


def test_dataset_potsdam_icl_step2(potsdam_path: Path):
    # instantiate transforms for training
    seed_everything(1337)
    train_transform = train_transforms(image_size=512, in_channels=4)
    eval_transform = test_transforms(in_channels=4)
    # create the train dataset, then split or create the ad hoc validation set
    train_dataset = create_dataset("potsdam", path=potsdam_path, subset="train", transform=train_transform, channels=4)
    train_mask, val_mask, _ = mask_set(len(train_dataset), val_size=0.1, test_size=0.0)
    val_dataset = create_dataset("potsdam", path=potsdam_path, subset="train", transform=eval_transform, channels=4)
    train_dataset.add_mask(train_mask)
    val_dataset.add_mask(val_mask, stage="valid")
    original_length = len(train_dataset)

    task = Task(dataset="potsdam", name="222a", step=2)
    icl_set = ICLDataset(train_dataset, task, overlap=True)
    # check it includes the previously missing background + handpicked check
    assert len(icl_set.categories()) == 7
    assert icl_set.categories()[1] == "impervious_surfaces"
    assert len(icl_set) < original_length
    # some logs
    LOG.info("ICL map    : %s", str(icl_set.label2index))
    LOG.info("ICL inverse: %s", str(icl_set.index2label))
    LOG.info("ICL transf : %s", str(icl_set.label_transform))
    assert icl_set.label2index == {0: 0, 1: 1, 3: 2, 2: 3, 4: 4, 5: 5, 6: 6, 255: 0}
    assert icl_set.index2label == {0: 255, 1: 1, 2: 3, 3: 2, 4: 4, 5: 5, 6: 6}
    assert icl_set.label_transform == {1: 1, 2: 3, 3: 2, 4: 4, 5: 5, 6: 6, 0: 0}
    # check every image
    loader = DataLoader(icl_set, batch_size=4, num_workers=8)
    for image, mask in tqdm(loader):
        assert image.shape == (4, 4, 512, 512)
        assert mask.shape == (4, 512, 512)
        # labels = np.unique(mask.numpy())
        # LOG.info(str(labels))
        assert torch.all(sum(mask == i for i in (0, 1, 2, 3, 4, 5, 6)).bool())

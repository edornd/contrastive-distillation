import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import albumentations as alb
from albumentations.pytorch import ToTensorV2
from saticl.datasets import create_dataset
from saticl.datasets.icl import ICLDataset
from saticl.datasets.isaid import ISAIDDataset
from saticl.datasets.isprs import PotsdamDataset
from saticl.datasets.transforms import geom_transforms, test_transforms, train_transforms
from saticl.datasets.wrappers import ContrastiveDataset
from saticl.tasks import Task
from saticl.transforms import ContrastiveTransform
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


def test_dataset_isaid(isaid_path: Path):
    dataset = ISAIDDataset(isaid_path, subset="train", transform=None, channels=3)
    assert len(dataset.categories()) == 16
    image, mask = dataset.__getitem__(0)
    assert image.shape == (512, 512, 3)
    assert mask.shape == (512, 512)
    assert image.min() >= 0 and image.max() <= 255
    # zero is not included unless we're in incremental learning
    assert mask.min() >= 0 and mask.max() <= 15


def test_dataset_isaid_transform(isaid_path: Path):
    transforms = alb.Compose([alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
    dataset = ISAIDDataset(isaid_path, subset="train", transform=transforms)
    assert len(dataset.categories()) == 16
    assert dataset.has_background()
    for image, mask in tqdm(dataset):
        assert image.shape == (3, 512, 512)
        assert mask.shape == (512, 512)
        assert image.min() >= -100 and image.max() <= +100
        assert mask.min() >= 0 and mask.max() <= 15


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
    icl_set = ICLDataset(train_dataset, task, filter_mode="overlap")
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

    task = Task(dataset="potsdam", name="222a", step=1, add_background=not train_dataset.has_background())
    icl_set = ICLDataset(train_dataset, task, filter_mode="overlap")
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
    assert icl_set.label_transform == {1: 0, 2: 3, 3: 0, 4: 4, 5: 0, 6: 0, 0: 0}
    # check every image
    loader = DataLoader(icl_set, batch_size=4, num_workers=8)
    for image, mask in tqdm(loader):
        assert image.shape == (4, 4, 512, 512)
        assert mask.shape == (4, 512, 512)
        # labels = np.unique(mask.numpy())
        # LOG.info(str(labels))
        assert torch.all(sum(mask == i for i in (0, 3, 4)).bool())


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
    icl_set = ICLDataset(train_dataset, task, filter_mode="overlap")
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
    assert icl_set.label_transform == {1: 0, 2: 0, 3: 0, 4: 0, 5: 5, 6: 6, 0: 0}
    # check every image
    loader = DataLoader(icl_set, batch_size=4, num_workers=8)
    for image, mask in tqdm(loader):
        assert image.shape == (4, 4, 512, 512)
        assert mask.shape == (4, 512, 512)
        # labels = np.unique(mask.numpy())
        # LOG.info(str(labels))
        assert torch.all(sum(mask == i for i in (0, 5, 6)).bool())


def test_dataset_potsdam_icl_step0_weights(potsdam_path: Path, potsdam_weights: Path):
    # instantiate transforms for training
    seed_everything(1337)
    train_transform = test_transforms(in_channels=4)
    # create the train dataset, then split or create the ad hoc validation set
    train_dataset = create_dataset("potsdam", path=potsdam_path, subset="train", transform=train_transform, channels=4)
    original_length = len(train_dataset)

    task = Task(dataset="potsdam", name="321", step=0)
    icl_set = ICLDataset(train_dataset, task, filter_mode="overlap")
    # check it includes the previously missing background + handpicked check
    assert len(icl_set.categories()) == 7
    assert icl_set.categories()[1] == "impervious_surfaces"
    assert len(icl_set) < original_length
    # load weights
    weights = icl_set.load_class_weights(potsdam_weights, device=torch.device("cpu"))
    assert len(weights) == 4
    LOG.info("Weights: %s", weights)
    LOG.info("Normalized: %s", icl_set.load_class_weights(potsdam_weights, device=torch.device("cpu"), normalize=True))


def test_dataset_potsdam_icl_step1_weights(potsdam_path: Path, potsdam_weights: Path):
    # instantiate transforms for training
    seed_everything(1337)
    train_transform = test_transforms(in_channels=4)
    # create the train dataset, then split or create the ad hoc validation set
    train_dataset = create_dataset("potsdam", path=potsdam_path, subset="train", transform=train_transform, channels=4)
    original_length = len(train_dataset)

    task = Task(dataset="potsdam", name="321", step=1)
    icl_set = ICLDataset(train_dataset, task, filter_mode="overlap")
    # check it includes the previously missing background + handpicked check
    assert len(icl_set.categories()) == 7
    assert icl_set.categories()[1] == "impervious_surfaces"
    assert len(icl_set) < original_length
    # load weights
    weights = icl_set.load_class_weights(potsdam_weights, device=torch.device("cpu"))
    assert len(weights) == 6
    LOG.info("Weights: %s", weights)
    LOG.info("Normalized: %s", icl_set.load_class_weights(potsdam_weights, device=torch.device("cpu"), normalize=True))


def test_dataset_potsdam_icl_step2_weights(potsdam_path: Path, potsdam_weights: Path):
    # instantiate transforms for training
    seed_everything(1337)
    train_transform = test_transforms(in_channels=4)
    # create the train dataset, then split or create the ad hoc validation set
    train_dataset = create_dataset("potsdam", path=potsdam_path, subset="train", transform=train_transform, channels=4)
    original_length = len(train_dataset)

    task = Task(dataset="potsdam", name="321", step=2)
    icl_set = ICLDataset(train_dataset, task, filter_mode="overlap")
    # check it includes the previously missing background + handpicked check
    assert len(icl_set.categories()) == 7
    assert icl_set.categories()[1] == "impervious_surfaces"
    assert len(icl_set) < original_length
    # load weights
    weights = icl_set.load_class_weights(potsdam_weights, device=torch.device("cpu"))
    assert len(weights) == 7
    LOG.info("Weights: %s", weights)
    LOG.info("Normalized: %s", icl_set.load_class_weights(potsdam_weights, device=torch.device("cpu"), normalize=True))


def test_dataset_potsdam_augmentations(potsdam_path: Path):
    # instantiate transforms for training
    seed_everything(1337)
    # create the train dataset, then split or create the ad hoc validation set
    train_transform = train_transforms(image_size=512,
                                       in_channels=4,
                                       channel_dropout=0.5,
                                       normalize=False,
                                       tensorize=False)
    train_dataset = create_dataset("potsdam", path=potsdam_path, subset="train", transform=train_transform, channels=4)
    extra_trf = geom_transforms(in_channels=4, normalize=True, tensorize=True)
    rotation_set = ContrastiveDataset(train_dataset, transform=ContrastiveTransform(extra_trf, extra_trf))
    loader = DataLoader(rotation_set, batch_size=4, num_workers=1)
    img1, img2, mask1, mask2 = next(iter(loader))
    LOG.info("%s - %s", str(img1.shape), str(mask1.shape))
    assert img1.shape == (4, 4, 512, 512)
    assert mask1.shape == (4, 512, 512)
    assert img1.shape == img2.shape
    assert mask1.shape == mask2.shape
    # zero is not included unless we're in incremental learning
    assert mask1.min() >= 0 and mask1.max() <= 5
    # we expect different transforms
    LOG.info(torch.nn.MSELoss(reduce="mean")(img1, img2))
    assert torch.any(img1 != img2)

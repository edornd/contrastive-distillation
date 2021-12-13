import logging
import os
from glob import glob
from pathlib import Path

import numpy as np

import cv2
from natsort import natsorted
from saticl.preproc import DatasetInfo
from saticl.preproc.config import ISAIDPreprocConfig
from saticl.preproc.utils import convert_mask
from tqdm import tqdm

LOG = logging.getLogger(__name__)


class ISAIDDatasetInfo(DatasetInfo):

    def __init__(self) -> None:
        super().__init__()
        self.index2label: dict = {
            0: "background",
            1: "ship",
            2: "storage_tank",
            3: "baseball_diamond",
            4: "tennis_court",
            5: "basketball_court",
            6: "ground_track_field",
            7: "bridge",
            8: "large_vehicle",
            9: "small_vehicle",
            10: "helicopter",
            11: "swimming_pool",
            12: "roundabout",
            13: "soccer_ball_field",
            14: "plane",
            15: "harbor"
        }
        self.index2color = {
            0: (0, 0, 0),
            1: (0, 0, 63),
            2: (0, 191, 127),
            3: (0, 63, 0),
            4: (0, 63, 127),
            5: (0, 63, 191),
            6: (0, 63, 255),
            7: (0, 127, 63),
            8: (0, 127, 127),
            9: (0, 0, 127),
            10: (0, 0, 191),
            11: (0, 0, 255),
            12: (0, 63, 63),
            13: (0, 127, 191),
            14: (0, 127, 255),
            15: (0, 100, 155)
        }
        self.label2index: dict = {v: k for k, v in self.index2label.items()}
        self.color2index: dict = {v: k for k, v in self.index2color.items()}
        self.num_classes = len(self.index2label)


def swap_channels(image: np.ndarray) -> np.ndarray:
    return image[:, :, ::-1]


def main(config: ISAIDPreprocConfig):
    LOG.info(str(config))
    info = ISAIDDatasetInfo()
    assert len(config.subsets) <= 2, "way too many sets my friend"
    assert all([s in ("train", "valid") for s in config.subsets]), "Only train and validation supported"
    label_suffix = "instance_color_RGB"

    for subset in config.subsets:
        src_path = Path(config.src) / subset / config.subdir
        dst_path = Path(config.dst) / subset
        os.makedirs(str(dst_path), exist_ok=True)

        files = glob(str(src_path / "*.png"))
        files = [os.path.split(i)[-1].split('.')[0] for i in files if '_' not in os.path.split(i)[-1]]
        files = natsorted(files)

        for file_id in tqdm(files):
            if file_id == 'P1527' or file_id == 'P1530':
                continue
            image_id = f"{file_id}.png"
            label_id = f"{file_id}_{label_suffix}.png"
            image_filename = src_path / image_id
            label_filename = src_path / label_id
            # read files
            image = cv2.imread(str(image_filename))
            label = cv2.imread(str(label_filename))
            assert image.shape[:-1] == label.shape[:-1], f"Shape mismatch for image: {file_id}"
            h, w, _ = image.shape
            tile_size = config.patch_size
            overlap = config.overlap

            if h > config.patch_size and w > tile_size:
                for x in range(0, w, tile_size - overlap):
                    for y in range(0, h, tile_size - overlap):
                        x1 = x
                        x2 = x + tile_size
                        if x2 > w:
                            diff_x = x2 - w
                            x1 -= diff_x
                            x2 = w
                        y1 = y
                        y2 = y + tile_size
                        if y2 > h:
                            diff_y = y2 - h
                            y1 -= diff_y
                            y2 = h
                        image_tile = image[y1:y2, x1:x2, :]
                        label_tile = label[y1:y2, x1:x2, :]
                        label_tile = convert_mask(swap_channels(label_tile), lut=info.color2index)
                        img_name = f"{file_id}_{y1}_{y2}_{x1}_{x2}.png"
                        lab_name = f"{file_id}_{y1}_{y2}_{x1}_{x2}_mask.png"

                        save_path = dst_path / img_name
                        if not os.path.isfile(save_path):
                            cv2.imwrite(str(dst_path / img_name), image_tile)
                            cv2.imwrite(str(dst_path / lab_name), label_tile)
            else:
                image = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
                label = convert_mask(swap_channels(label), lut=info.color2index)
                cv2.imwrite(str(dst_path / f"{file_id}.png"), image)
                cv2.imwrite(str(dst_path / f"{file_id}_mask.png"), label)

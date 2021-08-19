import os
from typing import Union

import numpy as np


def lenient_makedirs(path: str) -> None:
    """Simple wrapper around makedirs that first checks for existence.

    Args:
        path (str): path to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)


def tile_overlapped(image: np.ndarray,
                    tile_size: Union[tuple, int] = 256,
                    channels_first: bool = False,
                    tile_rows: int = None,
                    tile_cols: int = None) -> np.ndarray:
    if len(image.shape) == 2:
        axis = 0 if channels_first else -1
        image = np.expand_dims(image, axis=axis)
    if channels_first:
        image = np.moveaxis(image, 0, -1)
    # assume height, width, channels from now on
    height, width, channels = image.shape
    tile_h, tile_w = tile_size if isinstance(tile_size, tuple) else (tile_size, tile_size)
    if height <= tile_h and width <= tile_w:
        raise ValueError("Image is smaller than the required tile size")
    # number of expected tiles, manually defined or inferred
    exact = [height / float(tile_h), width / float(tile_w)]
    outer = [int(np.ceil(v)) for v in exact]
    # the required number of tiles is given by the ceiling
    tile_count_h = tile_rows or outer[0]
    tile_count_w = tile_cols or outer[1]
    # compute total remainder for the expanded window
    remainder_h = (tile_count_h * tile_h) - height
    remainder_w = (tile_count_w * tile_w) - width
    # divide remainders among tiles as overlap
    overlap_h = int(np.floor(remainder_h / float(tile_count_h))) if tile_count_h > 1 else 0
    overlap_w = int(np.floor(remainder_w / float(tile_count_w))) if tile_count_w > 1 else 0
    # create the empty tensor to contain tiles
    tiles = np.empty((tile_count_h, tile_count_w, tile_h, tile_w, channels), dtype=image.dtype)
    stride_h = tile_h - overlap_h
    stride_w = tile_w - overlap_w
    # iterate over tiles and copy content from image windows
    for row in range(tile_count_h):
        for col in range(tile_count_w):
            # get the starting indices, accounting for initial positions
            # overlap is halved to distribute in left/right and top/bottom
            x = max(row * stride_h - overlap_h // 2, 0)
            y = max(col * stride_w - overlap_w // 2, 0)
            # if it exceeds horizontally or vertically in the last rows or cols, increase overlap to fit
            if (x + tile_h) >= height:
                x -= abs(x + tile_h - height)
            if (y + tile_w) >= width:
                y -= abs(y + tile_w - width)
            # assign tile to final tensor
            tiles[row, col] = image[x:x + tile_h, y:y + tile_w, :]
    return tiles

# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from torch.cuda import amp
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

import albumentations as alb
from albumentations.pytorch import ToTensorV2
from inplace_abn import InPlaceABNSync
from timm import create_model
from tqdm import tqdm

# ugly fix to import from package
sys.path.append("../")

from saticl.datasets.isprs import PotsdamDataset
from saticl.models import create_decoder

# from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor

########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a ResNet50 on the Oxford-IIT Pet Dataset
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


class Segmenter(torch.nn.Module):

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.out = torch.nn.Conv2d(decoder.output(), num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.out(x)


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
    return train_mask, val_mask, test_mask


def training_function(config, args):
    # Initialize accelerator
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu)

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    image_size = config["image_size"]
    if not isinstance(image_size, (list, tuple)):
        image_size = (image_size, image_size)
    data_dir = Path(args.data_dir)

    # Set the seed before splitting the data.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For training we use a simple RandomResizedCrop
    train_tfm = alb.Compose([alb.Flip(), alb.RandomRotate90(), alb.Normalize(), ToTensorV2()])
    # train_tfm = Compose([RandomResizedCrop(image_size, scale=(0.5, 1.0)), ToTensor()])
    train_dataset = PotsdamDataset(path=data_dir, subset="train", include_dsm=False, transform=train_tfm)
    train_split, val_split, _ = mask_set(len(train_dataset), test_size=0)
    train_dataset.add_mask(train_split)

    # Build the label correspondences
    id_to_label = train_dataset.categories()
    label_to_id = {lbl: i for i, lbl in id_to_label.items()}

    # For evaluation, we use a deterministic Resize
    eval_tfm = alb.Compose([alb.Normalize(), ToTensorV2()])
    # eval_tfm = Compose([Resize(image_size), ToTensor()])
    eval_dataset = PotsdamDataset(path=data_dir, subset="train", include_dsm=False, transform=eval_tfm)
    eval_dataset.add_mask(val_split)

    # Instantiate dataloaders.
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, num_workers=8)

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    act_layer = torch.nn.Identity    # partial(torch.nn.ReLU, inplace=True)
    norm_layer = partial(InPlaceABNSync, activation="leaky_relu", activation_param=0.01)
    encoder = create_model("tresnet_l", pretrained=True, features_only=True, num_classes=len(label_to_id))
    decoder = create_decoder("unet", encoder.feature_info, act_layer=act_layer, norm_layer=norm_layer)

    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = Segmenter(encoder, decoder, num_classes=len(label_to_id))
    model = model.to(accelerator.device)

    # Instantiate optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader,
                                                                              eval_dataloader)
    # create loss and scaler
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction=args.reduction)
    # scaler = amp.grad_scaler.GradScaler() if args.fp16 else None
    # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
    # may change its length.
    lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(train_dataloader))
    # Now we train the model
    try:
        for epoch in range(num_epochs):
            accelerator.print(f"[Epoch {epoch:2d}]")
            model.train()
            pbar = tqdm(train_dataloader)
            for x, y in pbar:
                # with amp.autocast():
                #     outputs = model(x)
                #     loss = criterion(outputs, y.long())
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()

                with accelerator.autocast():
                    outputs = model(x)
                    loss = criterion(outputs, y.long())
                accelerator.backward(loss)
                lr_scheduler.step()
                optimizer.zero_grad()
                pbar.set_postfix(dict(loss=f"{loss.item():.4f}", type=f"{loss.dtype}"))

            model.eval()
            accurate = 0
            num_elems = 0
            accelerator.print("Evaluating...")
            for x, y in tqdm(eval_dataloader):
                with torch.no_grad():
                    outputs = model(x)
                predictions = outputs.argmax(dim=1)
                accurate_preds = accelerator.gather(predictions) == accelerator.gather(y)
                num_elems += accurate_preds.numel()
                accurate += accurate_preds.long().sum()

            eval_metric = accurate.item() / num_elems
            # Use accelerator.print to print only on the main process.
            accelerator.print(f"epoch {epoch}: {100 * eval_metric:.2f}")
    except KeyboardInterrupt:
        exit(0)


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--data_dir", required=True, help="The data folder on disk.")
    parser.add_argument("--reduction", default="mean", choices=["mean", "sum"], help="Reduction type for the loss")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    args = parser.parse_args()
    config = {"lr": 1e-3, "num_epochs": 3, "seed": 42, "batch_size": 8, "image_size": 512}
    training_function(config, args)


if __name__ == "__main__":
    main()

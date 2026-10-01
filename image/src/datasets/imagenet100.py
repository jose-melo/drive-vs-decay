# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger

import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()

# ImageNet statistics
IMAGENET100_MEAN = (0.485, 0.456, 0.406)
IMAGENET100_STD = (0.229, 0.224, 0.225)

_IMG_EXTS = ('.jpg', '.jpeg', '.png')


def _is_real_image(path):
    name = os.path.basename(path)
    if name.startswith('._') or name == '.DS_Store':
        return False
    return name.lower().endswith(_IMG_EXTS)


def _split_dir(root_path, training):
    split = 'train' if training else 'val'
    path = os.path.join(root_path, split)
    if not os.path.isdir(path):
        raise FileNotFoundError(f'{path} not found (expected train/ and val/ class folders)')
    return path


def make_imagenet100(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path='./data',
    training=True,
    drop_last=True,
):
    path = _split_dir(root_path, training)
    dataset = torchvision.datasets.ImageFolder(
        root=path, transform=transform, is_valid_file=_is_real_image)

    if len(dataset) == 0:
        raise RuntimeError(f'ImageNet-100 split at {path!r} contains 0 images.')
    if len(dataset.classes) != 100:
        logger.warning(
            'ImageNet-100 split at %s has %d classes, expected 100.',
            path, len(dataset.classes))

    logger.info('ImageNet-100 dataset created (%d images, %d classes)',
                len(dataset), len(dataset.classes))

    use_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    if use_ddp:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
        )
    else:
        dist_sampler = torch.utils.data.RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    logger.info('ImageNet-100 data loader created')

    return dataset, data_loader, dist_sampler


def make_imagenet100_linear_probe(
    batch_size=256,
    num_workers=4,
    root_path='./data',
    crop_size=128,
):
    normalize = torchvision.transforms.Normalize(
        mean=IMAGENET100_MEAN,
        std=IMAGENET100_STD,
    )

    resize_short = int(round(crop_size * 1.14))

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(crop_size, scale=(0.6, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize,
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize_short),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=_split_dir(root_path, True), transform=train_transform,
        is_valid_file=_is_real_image)
    test_dataset = torchvision.datasets.ImageFolder(
        root=_split_dir(root_path, False), transform=test_transform,
        is_valid_file=_is_real_image)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, test_loader

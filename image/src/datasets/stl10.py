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

STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)

def make_stl10(
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
    split = 'train+unlabeled' if training else 'test'
    dataset = torchvision.datasets.STL10(
        root=root_path,
        split=split,
        download=True,
        transform=transform,
    )
    logger.info(f'STL-10 dataset created (split={split}, n={len(dataset)})')

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
    logger.info('STL-10 data loader created')

    return dataset, data_loader, dist_sampler

def make_stl10_linear_probe(
    batch_size=256,
    num_workers=4,
    root_path='./data',
    crop_size=96,
):
    normalize = torchvision.transforms.Normalize(
        mean=STL10_MEAN,
        std=STL10_STD,
    )

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize,
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.STL10(
        root=root_path, split='train', download=True, transform=train_transform)
    test_dataset = torchvision.datasets.STL10(
        root=root_path, split='test', download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, test_loader

if __name__ == "__main__":

    import logging
    logging.basicConfig(level=logging.INFO)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
    ])

    dataset, loader, _ = make_stl10(
        transform=transform, batch_size=4, num_workers=0, drop_last=False,
        training=True,
    )
    imgs, labels = next(iter(loader))
    assert imgs.shape == (4, 3, 96, 96), f'unexpected pretraining shape {imgs.shape}'

    train_loader, test_loader = make_stl10_linear_probe(batch_size=4, num_workers=0)
    imgs_p, labels_p = next(iter(train_loader))
    assert imgs_p.shape[1:] == (3, 96, 96), f'unexpected probe shape {imgs_p.shape}'
    assert labels_p.min() >= 0 and labels_p.max() <= 9, f'probe labels out of range: {labels_p}'

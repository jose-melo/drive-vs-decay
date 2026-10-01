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

def make_cifar100(
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
    dataset = torchvision.datasets.CIFAR100(
        root=root_path,
        train=training,
        download=True,
        transform=transform,
    )
    logger.info('CIFAR-100 dataset created')

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
    logger.info('CIFAR-100 data loader created')

    return dataset, data_loader, dist_sampler

def make_cifar100_linear_probe(
    batch_size=256,
    num_workers=4,
    root_path='./data',
    crop_size=32,
):
    normalize = torchvision.transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
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

    train_dataset = torchvision.datasets.CIFAR100(
        root=root_path, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR100(
        root=root_path, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, test_loader

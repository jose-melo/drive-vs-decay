# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch

import src.models.vision_transformer as vit
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        if target_encoder is not None:
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch

def init_model(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384,
    predictor_type='standard',
    beta_init=10.0,
    no_zero_init=False,
    mimetic_alpha=0.5,
):
    known = ('standard', 'residualpred', 'frozen_rp', 'skiponly', 'mimetic')
    if predictor_type not in known:
        raise ValueError(f'unknown predictor_type {predictor_type!r}; expected one of {known}')

    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)

    if predictor_type in ('residualpred', 'frozen_rp'):
        predictor = vit.__dict__['vit_residualpred'](
            num_patches=encoder.patch_embed.num_patches,
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=pred_emb_dim,
            depth=pred_depth,
            num_heads=encoder.num_heads,
            beta_init=beta_init,
            no_zero_init=no_zero_init)
    elif predictor_type == 'skiponly':
        predictor = vit.__dict__['vit_skiponly'](
            num_patches=encoder.patch_embed.num_patches,
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=pred_emb_dim,
            num_heads=encoder.num_heads)
    else:
        predictor = vit.__dict__['vit_predictor'](
            num_patches=encoder.patch_embed.num_patches,
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=pred_emb_dim,
            depth=pred_depth,
            num_heads=encoder.num_heads)

    if predictor_type == 'mimetic':
        from src.baseline_predictors import mimetic_init_predictor
        info = mimetic_init_predictor(predictor, alpha_m=mimetic_alpha, beta_v=0.5)
        logger.info(f'mimetic init: alpha_m={mimetic_alpha} blocks={info["blocks"]}')

    if predictor_type == 'frozen_rp':
        for p in predictor.parameters():
            p.requires_grad_(False)
        logger.info('frozen_rp: every predictor parameter is frozen')

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(f'Predictor type: {predictor_type}')

    if predictor_type == 'skiponly':
        assert len(predictor.predictor_blocks) == 0
        logger.info('skiponly: no predictor blocks, so the prediction does not '
                    'depend on the context tokens')

    n_train = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in predictor.parameters())
    logger.info(f'Predictor params: {n_train} trainable / {n_total} total')
    return encoder, predictor

def _is_zero_init_param(name):

    return ('predictor_blocks' in name and
            (name.endswith('mlp.fc2.weight') or name.endswith('mlp.fc2.bias')))

def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25,
    predictor_type='standard',
    no_zero_init=False,
):

    is_residualpred = (predictor_type == 'residualpred') and (not no_zero_init)

    def _pred_params(pred):
        return [(n, p) for n, p in pred.named_parameters() if p.requires_grad]

    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in _pred_params(predictor)
                       if ('bias' not in n) and (len(p.shape) != 1)
                       and not (is_residualpred and _is_zero_init_param(n)))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in _pred_params(predictor)
                       if (('bias' in n) or (len(p.shape) == 1))
                       and not (is_residualpred and _is_zero_init_param(n))),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    for g in param_groups:
        g['params'] = list(g['params'])
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    if is_residualpred:
        zero_init_params = [p for n, p in _pred_params(predictor)
                            if _is_zero_init_param(n)]
        if zero_init_params:
            param_groups.append({
                'params': zero_init_params,
                'WD_exclude': True,
                'weight_decay': 0,
            })
            logger.info(f'ResidualPred: {len(zero_init_params)} zero-init params excluded from WD')

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))

    scaler = None
    return optimizer, scaler, scheduler, wd_scheduler

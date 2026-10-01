# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import argparse
import copy
import json
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.cifar100 import make_cifar100, make_cifar100_linear_probe
from src.datasets.cifar10 import (
    make_cifar10, make_cifar10_linear_probe, CIFAR10_MEAN, CIFAR10_STD,
)
from src.datasets.imagenet100 import (
    make_imagenet100, make_imagenet100_linear_probe,
    IMAGENET100_MEAN, IMAGENET100_STD,
)
from src.datasets.stl10 import (
    make_stl10, make_stl10_linear_probe, STL10_MEAN, STL10_STD,
)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

def _resolve_dataset(name):
    if name == 'cifar10':
        return (make_cifar10, make_cifar10_linear_probe,
                CIFAR10_MEAN, CIFAR10_STD, 10)
    if name == 'cifar100':
        return (make_cifar100, make_cifar100_linear_probe,
                CIFAR100_MEAN, CIFAR100_STD, 100)
    if name == 'stl10':
        return (make_stl10, make_stl10_linear_probe,
                STL10_MEAN, STL10_STD, 10)
    if name == 'imagenet100':
        return (make_imagenet100, make_imagenet100_linear_probe,
                IMAGENET100_MEAN, IMAGENET100_STD, 100)
    if name == 'imagenet1k':
        # same loader as ImageNet-100, 1000 classes
        return (make_imagenet100, make_imagenet100_linear_probe,
                IMAGENET100_MEAN, IMAGENET100_STD, 1000)
    raise ValueError(
        f'Unknown dataset {name!r}; '
        f'expected cifar10/cifar100/stl10/imagenet100/imagenet1k')

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms
from src.aux_losses import sigreg_loss, vicreg_var_cov
from src.mu_utils import (
    compute_erank,
    compute_mu_at_init,
    compute_predictor_sensitivity,
    linear_probe,
)

log_timings = True
log_freq = 10
checkpoint_freq = 25  # module default; overridden per-run inside main() from config

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args, resume_preempt=False):

    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    predictor_type = args['meta'].get('predictor_type', 'standard')
    beta_init = args['meta'].get('beta_init', 10.0)
    no_zero_init = args['meta'].get('no_zero_init', False)
    mimetic_alpha = float(args['meta'].get('mimetic_alpha', 0.5))
    sigreg_lambda = float(args['meta'].get('sigreg_lambda', 0.0))
    sigreg_slices = int(args['meta'].get('sigreg_slices', 48))
    sigreg_samples = int(args['meta'].get('sigreg_samples', 384))
    vicreg_var_weight = float(args['meta'].get('vicreg_var_weight', 0.0))
    vicreg_cov_weight = float(args['meta'].get('vicreg_cov_weight', 0.0))
    use_sigreg = sigreg_lambda > 0.0
    use_vicreg = (vicreg_var_weight > 0.0) or (vicreg_cov_weight > 0.0)

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    dataset_name = args['data'].get('dataset', 'cifar100')
    make_pretrain, make_probe, ds_mean, ds_std, ds_default_num_classes = (
        _resolve_dataset(dataset_name)
    )

    allow_overlap = args['mask']['allow_overlap']
    patch_size = args['mask']['patch_size']
    num_enc_masks = args['mask']['num_enc_masks']
    min_keep = args['mask']['min_keep']
    enc_mask_scale = args['mask']['enc_mask_scale']
    num_pred_masks = args['mask']['num_pred_masks']
    pred_mask_scale = args['mask']['pred_mask_scale']
    aspect_ratio = args['mask']['aspect_ratio']

    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    checkpoint_freq = args['meta'].get('checkpoint_freq', 25)
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    eval_cfg = args.get('evaluation', {})
    probe_every = eval_cfg.get('probe_every', 10)
    erank_every = eval_cfg.get('erank_every', 5)
    probe_epochs = eval_cfg.get('linear_probe_epochs', 50)
    probe_lr = eval_cfg.get('linear_probe_lr', 0.1)
    num_classes = eval_cfg.get('num_classes', ds_default_num_classes)

    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    os.makedirs(folder, exist_ok=True)

    dump = os.path.join(folder, f'{tag}_params.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    metrics_path = os.path.join(folder, f'{tag}_metrics.json')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        predictor_type=predictor_type,
        beta_init=beta_init,
        no_zero_init=no_zero_init,
        mimetic_alpha=mimetic_alpha)
    target_encoder = copy.deepcopy(encoder)

    cifar100_mean = ds_mean
    cifar100_std = ds_std

    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter,
        normalization=(cifar100_mean, cifar100_std))

    _, unsupervised_loader, unsupervised_sampler = make_pretrain(
        transform=transform,
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        training=True,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        drop_last=True)
    ipe = len(unsupervised_loader)

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
        predictor_type=predictor_type,
        no_zero_init=no_zero_init)

    use_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    if use_ddp:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        if any(p.requires_grad for p in predictor.parameters()):
            predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # VICReg on the context tokens before the final LayerNorm
    _preln_tap = {}
    if use_vicreg:
        _enc_module = encoder.module if hasattr(encoder, 'module') else encoder
        def _pre_norm_hook(mod, inputs):
            _preln_tap['z'] = inputs[0]
        _enc_module.norm.register_forward_pre_hook(_pre_norm_hook)

    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0

    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    _pred_mod = getattr(predictor, 'module', predictor)
    _effective_depth = len(getattr(_pred_mod, 'predictor_blocks', []))

    metrics_log = {
        'config': {
            'predictor_type': predictor_type,
            'pred_depth': pred_depth,
            'effective_pred_depth': _effective_depth,
            'dataset': dataset_name,
            'predictor_trainable_params': sum(
                p.numel() for p in _pred_mod.parameters() if p.requires_grad),
            'model_name': model_name,
            'patch_size': patch_size,
            'crop_size': crop_size,
            'enc_mask_scale': enc_mask_scale,
            'pred_mask_scale': pred_mask_scale,
            'beta_init': beta_init,
            'sigreg_lambda': sigreg_lambda,
            'sigreg_slices': sigreg_slices,
            'sigreg_samples': sigreg_samples,
            'vicreg_var_weight': vicreg_var_weight,
            'vicreg_cov_weight': vicreg_cov_weight,
            'no_zero_init': no_zero_init,
        },
        'mu_init': {},
        'erank': {},
        'probe_acc': {},
    }

    if rank == 0 and start_epoch == 0:
        logger.info('Computing mu at initialization...')
        try:

            context_ratio = (enc_mask_scale[0] + enc_mask_scale[1]) / 2.0
            mu_stats = compute_mu_at_init(
                encoder=encoder,
                predictor=predictor,
                data_loader=unsupervised_loader,
                device=device,
                context_ratio=context_ratio,
                patch_size=patch_size,
                crop_size=crop_size,
                num_batches=30)
            metrics_log['mu_init'] = {
                k: v for k, v in mu_stats.items()
                if k != 'mu_values'
            }

            mu_vals_path = os.path.join(folder, f'{tag}_mu_init_values.npy')
            np.save(mu_vals_path, mu_stats['mu_values'])

            logger.info(f'mu at init: mu_25: {mu_stats["mu_25"]:.4f}, '
                        f'mu_50: {mu_stats["mu_50"]:.4f}, '
                        f'mu_75: {mu_stats["mu_75"]:.4f}, '
                        f'frac_learning: {mu_stats["frac_learning"]:.4f}')
        except Exception as e:
            logger.warning(f'Failed to compute mu at init: {e}')

        try:
            sens_stats = compute_predictor_sensitivity(
                encoder=encoder,
                predictor=predictor,
                data_loader=unsupervised_loader,
                device=device,
                n_perturbations=20,
                num_batches=10)
            metrics_log['predictor_sensitivity'] = sens_stats
            logger.info(f'Predictor sensitivity: alpha_eff: {sens_stats["alpha_eff"]:.6f}, '
                        f'mu_approx: {sens_stats["mu_approx"]:.2f}, '
                        f'context_sens: {sens_stats["context_sensitivity"]:.6f}')
        except Exception as e:
            logger.warning(f'Failed to compute predictor sensitivity: {e}')

    if rank == 0 and start_epoch == 0:
        _write_health_status(folder, tag, metrics_log, 0, num_epochs)

    if rank == 0 and start_epoch == 0:
        try:
            erank_val = _compute_erank_from_loader(
                target_encoder, unsupervised_loader, device, num_batches=20)
            metrics_log['erank']['epoch_0'] = erank_val
            logger.info(f'ERank at init: {erank_val:.2f}')
        except Exception as e:
            logger.warning(f'Failed to compute ERank at init: {e}')

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if checkpoint_freq > 0 and (epoch) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch}'))

    def save_metrics():
        if rank == 0:
            with open(metrics_path, 'w') as f:
                json.dump(metrics_log, f, indent=2, default=str)

    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        if hasattr(unsupervised_sampler, 'set_epoch'):
            unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()
        sigreg_meter = AverageMeter()
        vicreg_var_meter = AverageMeter()
        vicreg_cov_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))
                        B = len(h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    z_ctx = encoder(imgs, masks_enc)
                    z = predictor(z_ctx, masks_enc, masks_pred)
                    return z_ctx, z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z_ctx, z = forward_context()
                    loss_pred = loss_fn(z, h)

                # logged loss = prediction term only
                loss = loss_pred
                aux_sg = aux_var = aux_cov = None
                if use_sigreg:
                    with torch.cuda.amp.autocast(enabled=False):
                        sg = sigreg_loss(z_ctx.float(),
                                         n_slices=sigreg_slices,
                                         n_samples=sigreg_samples)
                    loss = loss + sigreg_lambda * sg
                    aux_sg = float(sg)
                if use_vicreg:
                    with torch.cuda.amp.autocast(enabled=False):
                        zt = _preln_tap.pop('z').float()
                        v_term, c_term = vicreg_var_cov(zt)
                    loss = loss + vicreg_var_weight * v_term \
                                + vicreg_cov_weight * c_term
                    aux_var, aux_cov = float(v_term), float(c_term)

                assert bool(torch.isfinite(loss)), 'total loss non-finite'

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss_pred), (aux_sg, aux_var, aux_cov),
                        _new_lr, _new_wd, grad_stats)
            (loss, aux_vals, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step, log_timings=(itr % log_freq == 0))
            loss_meter.update(loss)
            if aux_vals[0] is not None:
                sigreg_meter.update(aux_vals[0])
            if aux_vals[1] is not None:
                vicreg_var_meter.update(aux_vals[1])
                vicreg_cov_meter.update(aux_vals[2])
            time_meter.update(etime)

            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (use_sigreg or use_vicreg) and (itr % log_freq == 0):
                    logger.info('[%d, %5d] aux: sigreg %.4f vicreg_var %.4f '
                                'vicreg_cov %.4f'
                                % (epoch + 1, itr, sigreg_meter.avg,
                                   vicreg_var_meter.avg, vicreg_cov_meter.avg))
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        logger.info('avg. loss %.3f' % loss_meter.avg)
        if (use_sigreg or use_vicreg) and rank == 0:
            metrics_log.setdefault('aux_by_epoch', {})[f'epoch_{epoch+1}'] = {
                'pred_loss': loss_meter.avg,
                'sigreg': sigreg_meter.avg if use_sigreg else None,
                'vicreg_var': vicreg_var_meter.avg if use_vicreg else None,
                'vicreg_cov': vicreg_cov_meter.avg if use_vicreg else None,
            }
            save_metrics()
        save_checkpoint(epoch + 1)

        if rank == 0 and ((epoch + 1) % erank_every == 0 or epoch == num_epochs - 1):
            try:
                erank_val = _compute_erank_from_loader(
                    target_encoder, unsupervised_loader, device, num_batches=20)
                metrics_log['erank'][f'epoch_{epoch+1}'] = erank_val
                logger.info(f'ERank at epoch {epoch+1}: {erank_val:.2f}')
                save_metrics()
            except Exception as e:
                logger.warning(f'Failed to compute ERank at epoch {epoch+1}: {e}')

        if rank == 0 and ((epoch + 1) % probe_every == 0 or epoch == num_epochs - 1):
            try:
                probe_train_loader, probe_test_loader = make_probe(
                    batch_size=256, num_workers=2, root_path=root_path,
                    crop_size=crop_size)
                test_acc, train_acc = linear_probe(
                    encoder=target_encoder,
                    train_loader=probe_train_loader,
                    test_loader=probe_test_loader,
                    device=device,
                    num_classes=num_classes,
                    num_epochs=probe_epochs,
                    lr=probe_lr)
                metrics_log['probe_acc'][f'epoch_{epoch+1}'] = {
                    'test_acc': test_acc,
                    'train_acc': train_acc,
                }
                logger.info(f'Probe acc at epoch {epoch+1}: '
                            f'test={test_acc:.4f}, train={train_acc:.4f}')
                save_metrics()
            except Exception as e:
                logger.warning(f'Failed linear probe at epoch {epoch+1}: {e}')

        if rank == 0 and ((epoch + 1) in (5, 10, 20) or
                          (epoch + 1) % erank_every == 0 or
                          (epoch + 1) % probe_every == 0 or
                          epoch == num_epochs - 1):
            _write_health_status(folder, tag, metrics_log, epoch + 1, num_epochs)

    save_metrics()
    _write_health_status(folder, tag, metrics_log, num_epochs, num_epochs)
    logger.info('Training complete.')

def _write_health_status(folder, tag, metrics_log, epoch, total_epochs):
    health_path = os.path.join(folder, f'{tag}_health.json')

    erank_keys = sorted(metrics_log.get('erank', {}).keys(),
                        key=lambda k: int(k.split('_')[1])) if metrics_log.get('erank') else []
    probe_keys = sorted(metrics_log.get('probe_acc', {}).keys(),
                        key=lambda k: int(k.split('_')[1])) if metrics_log.get('probe_acc') else []

    latest_erank = metrics_log['erank'].get(erank_keys[-1]) if erank_keys else None
    latest_probe = (metrics_log['probe_acc'][probe_keys[-1]].get('test_acc')
                    if probe_keys else None)
    mu_50 = metrics_log.get('mu_init', {}).get('mu_50')
    frac_learning = metrics_log.get('mu_init', {}).get('frac_learning')
    alpha_eff = metrics_log.get('predictor_sensitivity', {}).get('alpha_eff')
    context_sens = metrics_log.get('predictor_sensitivity', {}).get('context_sensitivity')

    alerts = []
    status = 'RUNNING'

    if latest_erank is not None and latest_erank < 2.0:
        alerts.append(f'COLLAPSE: ERank={latest_erank:.2f} < 2.0')
    if latest_erank is not None and latest_erank < 5.0 and epoch >= 20:
        alerts.append(f'LOW_ERANK: ERank={latest_erank:.2f} < 5.0 at epoch {epoch}')
    if latest_probe is not None and latest_probe < 0.05 and epoch >= 20:
        alerts.append(f'LOW_PROBE: test_acc={latest_probe:.4f} < 5% at epoch {epoch}')
    if mu_50 is not None and mu_50 < 0.5:
        alerts.append(f'LOW_MU: mu_50={mu_50:.4f} < 0.5 (likely collapse)')

    if any('COLLAPSE' in a for a in alerts):
        status = 'COLLAPSE_ALERT'
    elif any('LOW_PROBE' in a for a in alerts):
        status = 'LOW_PROBE'
    elif epoch >= total_epochs:
        status = 'COMPLETE'
    elif latest_erank is not None and latest_erank >= 5.0:
        status = 'HEALTHY'

    health = {
        'tag': tag,
        'status': status,
        'epoch': epoch,
        'total_epochs': total_epochs,
        'alerts': alerts,
        'predictor_type': metrics_log.get('config', {}).get('predictor_type', '?'),
        'pred_depth': metrics_log.get('config', {}).get('pred_depth', '?'),
        'enc_mask_scale': metrics_log.get('config', {}).get('enc_mask_scale', '?'),
        'mu_50': mu_50,
        'frac_learning': frac_learning,
        'alpha_eff': alpha_eff,
        'context_sensitivity': context_sens,
        'beta_init': metrics_log.get('config', {}).get('beta_init', '?'),
        'latest_erank': latest_erank,
        'latest_probe_acc': latest_probe,
    }

    with open(health_path, 'w') as f:
        json.dump(health, f, indent=2, default=str)

@torch.no_grad()
def _compute_erank_from_loader(encoder, data_loader, device, num_batches=20):
    from src.mu_utils import compute_erank
    module = encoder.module if hasattr(encoder, 'module') else encoder
    module.eval()

    all_embeddings = []
    for batch_idx, (udata, masks_enc, masks_pred) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        imgs = udata[0] if isinstance(udata, (list, tuple)) else udata
        imgs = imgs.to(device, non_blocking=True)

        h = module(imgs)

        h = h.mean(dim=1).float()
        all_embeddings.append(h)

    Z = torch.cat(all_embeddings, dim=0)
    erank = compute_erank(Z)

    module.train()
    return erank

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--predictor_type', type=str, default=None,
                        help='Override predictor type: standard, residualpred, frozen_rp, skiponly or mimetic')
    parser.add_argument('--pred_depth', type=int, default=None,
                        help='Override predictor depth')
    parser.add_argument('--enc_mask_scale_low', type=float, default=None,
                        help='Override encoder mask scale lower bound (context ratio)')
    parser.add_argument('--enc_mask_scale_high', type=float, default=None,
                        help='Override encoder mask scale upper bound')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--tag', type=str, default=None,
                        help='Override experiment tag')
    parser.add_argument('--folder', type=str, default=None,
                        help='Override log folder')
    parser.add_argument('--beta_init', type=float, default=None,
                        help='Override diagonal attention bias (ResidualPred)')
    parser.add_argument('--no_zero_init', action='store_true', default=False,
                        help='ResidualPred without the zero-initialised mlp.fc2 '
                             '(the bias-only arm).')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['cifar10', 'cifar100', 'stl10', 'imagenet100', 'imagenet1k'],
                        help='Override dataset choice.')
    parser.add_argument('--root_path', type=str, default=None,
                        help='Override data root path (data.root_path).')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of pretraining epochs '
                             '(optimization.epochs).')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Save a permanent -ep{N}.pth.tar every N epochs '
                             '(0 = only the every-epoch -latest.pth.tar). '
                             'Use 0 for large backbones to bound disk.')
    parser.add_argument('--sigreg_lambda', type=float, default=None,
                        help='SIGReg (Epps-Pulley) weight on the context-encoder '
                             'tokens. 0 = off (default).')
    parser.add_argument('--sigreg_slices', type=int, default=None,
                        help='SIGReg: random slices, resampled every step (default 48).')
    parser.add_argument('--sigreg_samples', type=int, default=None,
                        help='SIGReg: token rows subsampled per step (default 384).')
    parser.add_argument('--vicreg_var_weight', type=float, default=None,
                        help='VICReg variance-hinge weight. 0 = off.')
    parser.add_argument('--vicreg_cov_weight', type=float, default=None,
                        help='VICReg covariance weight. 0 = off.')
    parser.add_argument('--mimetic_alpha', type=float, default=None,
                        help='Mimetic initialisation alpha_m (predictor_type=mimetic).')

    cli_args = parser.parse_args()

    with open(cli_args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    if cli_args.predictor_type is not None:
        params['meta']['predictor_type'] = cli_args.predictor_type
    if cli_args.pred_depth is not None:
        params['meta']['pred_depth'] = cli_args.pred_depth
    if cli_args.enc_mask_scale_low is not None:
        params['mask']['enc_mask_scale'][0] = cli_args.enc_mask_scale_low
    if cli_args.enc_mask_scale_high is not None:
        params['mask']['enc_mask_scale'][1] = cli_args.enc_mask_scale_high
    if cli_args.tag is not None:
        params['logging']['write_tag'] = cli_args.tag
    if cli_args.folder is not None:
        params['logging']['folder'] = cli_args.folder
    if cli_args.beta_init is not None:
        params['meta']['beta_init'] = cli_args.beta_init
    if cli_args.no_zero_init:

        params['meta']['no_zero_init'] = True
    if cli_args.dataset is not None:
        params['data']['dataset'] = cli_args.dataset
    if cli_args.root_path is not None:
        params['data']['root_path'] = cli_args.root_path
    if cli_args.epochs is not None:
        params['optimization']['epochs'] = cli_args.epochs
    if cli_args.checkpoint_freq is not None:
        params['meta']['checkpoint_freq'] = cli_args.checkpoint_freq
    if cli_args.sigreg_lambda is not None:
        params['meta']['sigreg_lambda'] = cli_args.sigreg_lambda
    if cli_args.sigreg_slices is not None:
        params['meta']['sigreg_slices'] = cli_args.sigreg_slices
    if cli_args.sigreg_samples is not None:
        params['meta']['sigreg_samples'] = cli_args.sigreg_samples
    if cli_args.vicreg_var_weight is not None:
        params['meta']['vicreg_var_weight'] = cli_args.vicreg_var_weight
    if cli_args.vicreg_cov_weight is not None:
        params['meta']['vicreg_cov_weight'] = cli_args.vicreg_cov_weight
    if cli_args.mimetic_alpha is not None:
        params['meta']['mimetic_alpha'] = cli_args.mimetic_alpha
    if cli_args.seed is not None:
        _GLOBAL_SEED = cli_args.seed
        np.random.seed(_GLOBAL_SEED)
        torch.manual_seed(_GLOBAL_SEED)

    main(args=params)

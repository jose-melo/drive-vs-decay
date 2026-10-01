import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger()

def estimate_target_mask_coverage(collator, num_patches, num_steps=64,
                                   batch_size=8):
    H = W = int(round(math.sqrt(num_patches)))
    assert H * W == num_patches, f"num_patches={num_patches} not square"

    dummy_image = torch.zeros(3, H * collator.patch_size, W * collator.patch_size)
    dummy_batch = [(dummy_image, 0) for _ in range(batch_size)]

    coverage = torch.zeros(num_patches, dtype=torch.float64)
    n_samples = 0
    for _ in range(num_steps):
        _, _, masks_pred = collator(dummy_batch)
        for b_idx in range(batch_size):
            indicator = torch.zeros(num_patches, dtype=torch.bool)
            for m in masks_pred:

                indicator[m[b_idx].long()] = True
            coverage += indicator.to(torch.float64)
            n_samples += 1

    per_patch = coverage / max(1, n_samples)
    return float(per_patch.mean()), per_patch

def compute_erank(Z):

    S = torch.linalg.svdvals(Z.float())

    S = S[S > 1e-10]
    if len(S) == 0:
        return 1.0

    p = S / S.sum()

    entropy = -(p * torch.log(p)).sum()
    erank = torch.exp(entropy).item()
    return erank

@torch.no_grad()
def compute_predictor_sensitivity(
    encoder,
    predictor,
    data_loader,
    device,
    n_perturbations=30,
    eps=1e-3,
    num_batches=10,
):
    encoder.eval()
    predictor.eval()

    sensitivities = []
    context_diffs = []

    for batch_idx, (udata, masks_enc, masks_pred) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        imgs = udata[0] if isinstance(udata, (list, tuple)) else udata
        imgs = imgs.to(device, non_blocking=True)
        masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
        masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]

        h = encoder(imgs, masks_1)
        out_base = predictor(h, masks_1, masks_2)

        for _ in range(n_perturbations):
            delta = torch.randn_like(h)
            delta = delta / delta.norm() * eps
            out_pert = predictor(h + delta, masks_1, masks_2)
            sensitivities.append((out_pert - out_base).norm() / delta.norm())

    for batch_idx, (udata, masks_enc, masks_pred) in enumerate(data_loader):
        if batch_idx >= min(5, num_batches):
            break
        imgs = udata[0] if isinstance(udata, (list, tuple)) else udata
        imgs = imgs.to(device, non_blocking=True)
        masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
        masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]

        B = imgs.shape[0]
        if B < 2:
            continue

        h = encoder(imgs, masks_1)
        out = predictor(h, masks_1, masks_2)

        diff = (out[0] - out[1]).abs().mean()
        mag = out.abs().mean()
        context_diffs.append(diff / mag.clamp(min=1e-10))

    if sensitivities:
        sens_tensor = torch.stack(sensitivities)
        alpha_eff = sens_tensor.mean().item()
        alpha_max = sens_tensor.max().item()
    else:
        alpha_eff, alpha_max = 0.0, 0.0
    if context_diffs:
        ctx_sens = torch.stack(context_diffs).mean().item()
    else:
        ctx_sens = 0.0

    stats = {
        'alpha_eff': alpha_eff,
        'alpha_max': alpha_max,
        'mu_approx': 1.0 / alpha_eff if alpha_eff > 1e-10 else float('inf'),
        'context_sensitivity': ctx_sens,
        'n_samples': len(sensitivities),
    }

    encoder.train()
    predictor.train()
    return stats

@torch.no_grad()
def compute_mu_at_init(
    encoder,
    predictor,
    data_loader,
    device,
    context_ratio=0.7,
    patch_size=4,
    crop_size=32,
    num_batches=50,
):
    encoder.eval()
    predictor.eval()

    num_patches_per_side = crop_size // patch_size
    num_patches = num_patches_per_side ** 2

    all_embeddings = []
    for batch_idx, (udata, masks_enc, masks_pred) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        imgs = udata[0] if isinstance(udata, (list, tuple)) else udata
        imgs = imgs.to(device, non_blocking=True)

        with torch.no_grad():

            module = encoder.module if hasattr(encoder, 'module') else encoder
            x = module.patch_embed(imgs)
            pos = module.pos_embed
            if pos.shape[1] == x.shape[1]:
                x = x + pos
            elif pos.shape[1] == x.shape[1] + 1:
                x = x + pos[:, 1:]
            else:
                x = x + module.interpolate_pos_encoding(x, pos)

        all_embeddings.append(x.float().cpu())

    Z = torch.cat(all_embeddings, dim=0)
    B_total, N, D = Z.shape

    Z_flat = Z.reshape(-1, D)

    Sigma_XX = (Z_flat.T @ Z_flat) / Z_flat.shape[0]

    p = context_ratio
    Sigma_11 = p * torch.diag(torch.diag(Sigma_XX)) + p**2 * (Sigma_XX - torch.diag(torch.diag(Sigma_XX)))
    Sigma_X1 = p * Sigma_XX

    pred_module = predictor.module if hasattr(predictor, 'module') else predictor

    W_embed = pred_module.predictor_embed.weight.data.float().cpu()
    W_proj = pred_module.predictor_proj.weight.data.float().cpu()

    W_p = W_proj @ W_embed
    WpTWp = W_p.T @ W_p

    eig_Sigma_11 = torch.linalg.eigvalsh(Sigma_11.double()).flip(0)
    eig_Sigma_X1_sq = torch.linalg.eigvalsh((Sigma_X1.T @ Sigma_X1).double()).flip(0)
    eig_Sigma_X1 = torch.sqrt(torch.clamp(eig_Sigma_X1_sq, min=0))

    eig_WpTWp = torch.linalg.eigvalsh(WpTWp.double()).flip(0)

    target_ratio = 1.0 - context_ratio

    sigma_all = torch.outer(
        eig_Sigma_11.clamp(min=0),
        eig_WpTWp.clamp(min=0)
    ).flatten()

    eig_WpT_M2 = target_ratio * torch.sqrt(eig_WpTWp.clamp(min=0))

    gamma_all = torch.outer(
        eig_Sigma_X1.clamp(min=0),
        eig_WpT_M2
    ).flatten()

    sigma_sorted, _ = sigma_all.sort(descending=True)
    gamma_sorted, _ = gamma_all.sort(descending=True)

    n_modes = len(sigma_sorted)

    valid = sigma_sorted > 1e-12
    mu_values = torch.zeros(n_modes, dtype=torch.float64)
    mu_values[valid] = gamma_sorted[valid] / sigma_sorted[valid]

    mu_values[~valid] = float('inf')

    finite_mu = mu_values[mu_values.isfinite()]
    if len(finite_mu) == 0:
        finite_mu = torch.tensor([1.0])

    mu_stats = {
        'mu_25': torch.quantile(finite_mu.float(), 0.25).item(),
        'mu_50': torch.quantile(finite_mu.float(), 0.50).item(),
        'mu_75': torch.quantile(finite_mu.float(), 0.75).item(),
        'mu_mean': finite_mu.float().mean().item(),
        'mu_values': mu_values.cpu().numpy(),
        'n_modes': n_modes,
        'n_learning': (finite_mu > 1.0).sum().item(),
        'n_total': len(finite_mu),
        'frac_learning': (finite_mu > 1.0).float().mean().item(),
    }

    encoder.train()
    predictor.train()

    return mu_stats

@torch.no_grad()
@torch.no_grad()
def extract_features(encoder, data_loader, device):
    encoder.eval()
    module = encoder.module if hasattr(encoder, 'module') else encoder

    all_features = []
    all_labels = []

    for imgs, targets in data_loader:
        imgs = imgs.to(device, non_blocking=True)

        x = module.patch_embed(imgs)
        pos = module.pos_embed
        if pos.shape[1] == x.shape[1]:
            x = x + pos
        elif pos.shape[1] == x.shape[1] + 1:
            x = x + pos[:, 1:]
        else:
            x = x + module.interpolate_pos_encoding(x, pos)

        for blk in module.blocks:
            x = blk(x)
        if module.norm is not None:
            x = module.norm(x)

        features = x.mean(dim=1)
        all_features.append(features.cpu())
        all_labels.append(targets)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels

def linear_probe(
    encoder,
    train_loader,
    test_loader,
    device,
    num_classes=100,
    num_epochs=50,
    lr=0.01,
    batch_size=256,
):
    logger.info('Extracting features for linear probe...')
    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)

    D = train_features.shape[1]

    train_features = train_features.to(device)
    test_features = test_features.to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)

    train_mean = train_features.mean(dim=0, keepdim=True)
    train_std = train_features.std(dim=0, keepdim=True) + 1e-6
    train_features = (train_features - train_mean) / train_std
    test_features = (test_features - train_mean) / train_std

    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)

    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dl = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    classifier = nn.Linear(D, num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        classifier.train()
        for feats, targets in train_dl:
            logits = classifier(feats)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    classifier.eval()
    correct_train, total_train = 0, 0
    correct_test, total_test = 0, 0

    with torch.no_grad():
        for feats, targets in train_dl:
            preds = classifier(feats).argmax(dim=1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        for feats, targets in test_dl:
            preds = classifier(feats).argmax(dim=1)
            correct_test += (preds == targets).sum().item()
            total_test += targets.size(0)

    train_acc = correct_train / max(total_train, 1)
    test_acc = correct_test / max(total_test, 1)

    logger.info(f'Linear probe: train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}')
    return test_acc, train_acc

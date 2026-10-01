
import math

import torch
import torch.nn.functional as F


def sigreg_loss(Z, n_slices=48, n_samples=384, eps=1e-6):
    """SIGReg: Epps-Pulley statistic of random 1-D projections of Z against N(0, 1)."""
    Z = Z.reshape(-1, Z.shape[-1]).float()
    N, D = Z.shape
    if N > n_samples:
        idx = torch.randperm(N, device=Z.device)[:n_samples]
        Z = Z[idx]
    Z = Z - Z.mean(dim=0, keepdim=True)
    sigma = Z.pow(2).mean().sqrt()
    Z = Z / (sigma + eps)

    V = torch.randn(D, n_slices, device=Z.device, dtype=Z.dtype)
    V = V / (V.norm(dim=0, keepdim=True) + eps)
    P = Z @ V                                        # [n, S]

    d = P.unsqueeze(0) - P.unsqueeze(1)              # [n, n, S]
    t1 = torch.exp(-0.5 * d.pow(2)).mean(dim=(0, 1))
    t2 = (2.0 / math.sqrt(2.0)) * torch.exp(-0.25 * P.pow(2)).mean(dim=0)
    T = t1 - t2 + 1.0 / math.sqrt(3.0)
    return T.mean()


def vicreg_var_cov(Z, gamma=1.0, eps=1e-4, max_rows=4096):
    """VICReg variance hinge and covariance term."""
    Z = Z.reshape(-1, Z.shape[-1]).float()
    N, D = Z.shape
    if N > max_rows:
        idx = torch.randperm(N, device=Z.device)[:max_rows]
        Z = Z[idx]
        N = max_rows
    Zc = Z - Z.mean(dim=0, keepdim=True)
    std = torch.sqrt(Zc.var(dim=0, unbiased=False) + eps)
    var_term = F.relu(gamma - std).mean()

    C = (Zc.transpose(0, 1) @ Zc) / max(N - 1, 1)
    off_sq = C.pow(2).sum() - C.diagonal().pow(2).sum()
    cov_term = off_sq / D
    return var_term, cov_term

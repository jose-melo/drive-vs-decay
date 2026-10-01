
import torch

from src.utils.tensors import trunc_normal_


def _factor_target(T):
    U, S, Vh = torch.linalg.svd(T)
    sq = S.clamp_min(0).sqrt()
    A = U * sq.unsqueeze(0)          # d x d
    B = sq.unsqueeze(1) * Vh         # d x d
    return A, B


@torch.no_grad()
def mimetic_init_predictor(predictor, alpha_m=0.5, beta_v=0.5, init_std=0.02):
    """Mimetic initialisation of the predictor attention (Trockman and Kolter, 2023)."""
    info = {"alpha_m": alpha_m, "beta_v": beta_v, "blocks": 0,
            "qk_diag_mean": [], "vp_diag_mean": []}
    for blk in predictor.predictor_blocks:
        attn = blk.attn
        W = attn.qkv.weight
        d = W.shape[1]
        dev, dt = W.device, W.dtype
        I = torch.eye(d, device=dev, dtype=dt)

        Z1 = torch.empty(d, d, device=dev, dtype=dt)
        Z2 = torch.empty(d, d, device=dev, dtype=dt)
        trunc_normal_(Z1, std=init_std)
        trunc_normal_(Z2, std=init_std)
        T_qk = alpha_m * I + Z1 @ Z2
        A, B = _factor_target(T_qk)
        W[0:d].copy_(A.transpose(0, 1))       # W_q
        W[d:2 * d].copy_(B)                   # W_k

        Z3 = torch.empty(d, d, device=dev, dtype=dt)
        Z4 = torch.empty(d, d, device=dev, dtype=dt)
        trunc_normal_(Z3, std=init_std)
        trunc_normal_(Z4, std=init_std)
        T_vp = -beta_v * I + Z3 @ Z4
        A2, B2 = _factor_target(T_vp)
        attn.proj.weight.copy_(A2)            # W_proj
        W[2 * d:3 * d].copy_(B2)              # W_v

        if attn.qkv.bias is not None:
            attn.qkv.bias.zero_()
        if attn.proj.bias is not None:
            attn.proj.bias.zero_()

        info["blocks"] += 1
        info["qk_diag_mean"].append(
            float((W[0:d].transpose(0, 1) @ W[d:2 * d]).diagonal().mean()))
        info["vp_diag_mean"].append(
            float((attn.proj.weight @ W[2 * d:3 * d]).diagonal().mean()))
    return info

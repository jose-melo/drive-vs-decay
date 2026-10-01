#!/usr/bin/env python3
"""kNN (k=20) and attentive probes on a frozen checkpoint."""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import src.models.vision_transformer as vit
from src.utils.tensors import trunc_normal_
from src.datasets.imagenet100 import (
    IMAGENET100_MEAN, IMAGENET100_STD, _is_real_image, _split_dir,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _read_arch_from_metrics(ckpt_path):
    d = os.path.dirname(ckpt_path)
    m = glob.glob(os.path.join(d, "*_metrics.json"))
    if not m:
        return {}
    cfg = json.load(open(m[0])).get("config", {})
    return {k: cfg[k] for k in ("model_name", "patch_size", "crop_size") if k in cfg}


def load_encoder(ckpt_path, model_name, patch_size, crop_size):
    enc = vit.__dict__[model_name](img_size=[crop_size], patch_size=patch_size)
    ck = torch.load(ckpt_path, map_location="cpu")
    if "target_encoder" not in ck:
        raise KeyError(f"'target_encoder' not in checkpoint keys: {list(ck)[:8]}")
    sd = {k.replace("module.", "", 1): v for k, v in ck["target_encoder"].items()}
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    # pos_embed is fixed (sincos); anything else missing is an error
    real_missing = [k for k in missing if "pos_embed" not in k]
    if real_missing:
        raise RuntimeError(f"missing encoder params: {real_missing[:8]}")
    enc.to(DEVICE).eval()
    return enc, int(ck.get("epoch", -1))


def _eval_transform(crop_size):
    resize = int(round(crop_size * 1.14))
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET100_MEAN, IMAGENET100_STD),
    ])


@torch.no_grad()
def _balanced_indices(targets, max_per_class):
    """First max_per_class indices of each class."""
    per, keep = {}, []
    for i, y in enumerate(targets):
        y = int(y)
        c = per.get(y, 0)
        if c < max_per_class:
            keep.append(i)
            per[y] = c + 1
    return keep


@torch.no_grad()
def extract_tokens(enc, data_root, split, crop_size, batch_size=256, num_workers=8,
                   max_per_class=0):
    enc.eval()
    ds = torchvision.datasets.ImageFolder(
        root=_split_dir(data_root, split == "train"),
        transform=_eval_transform(crop_size), is_valid_file=_is_real_image)
    # subsample the train bank only
    if max_per_class and max_per_class > 0:
        idx = _balanced_indices(ds.targets, max_per_class)
        n_cls = len(ds.classes)
        print(f"[eval] subsampling {split}: {len(idx)} / {len(ds)} imgs "
              f"(<= {max_per_class}/class over {n_cls} classes)", flush=True)
        ds = torch.utils.data.Subset(ds, idx)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    toks, labs = [], []
    for imgs, y in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        m = enc.module if hasattr(enc, "module") else enc
        x = m.patch_embed(imgs)
        pos = m.pos_embed
        if pos.shape[1] == x.shape[1]:
            x = x + pos
        elif pos.shape[1] == x.shape[1] + 1:
            x = x + pos[:, 1:]
        else:
            x = x + m.interpolate_pos_encoding(x, pos)
        for blk in m.blocks:
            x = blk(x)
        if m.norm is not None:
            x = m.norm(x)
        toks.append(x.half().cpu())          # (B, N, D) fp16 to save RAM
        labs.append(y.clone())
    return torch.cat(toks), torch.cat(labs)


@torch.no_grad()
def knn_eval(train_tok, train_lab, val_tok, val_lab, k=20, tau=0.07, num_classes=100):
    # reduce in fp32 without copying the token bank to fp32
    tr = F.normalize(train_tok.mean(1, dtype=torch.float32), dim=1).to(DEVICE)   # (Ntr, D)
    va = F.normalize(val_tok.mean(1, dtype=torch.float32), dim=1).to(DEVICE)
    trl = train_lab.to(DEVICE)
    correct = 0
    for i in range(0, va.size(0), 512):
        sims = va[i:i + 512] @ tr.T                                 # (b, Ntr)
        sim, idx = sims.topk(k, dim=1)
        lab = trl[idx]                                              # (b, k)
        w = (sim / tau).exp()
        oh = F.one_hot(lab, num_classes).float() * w.unsqueeze(-1)
        pred = oh.sum(1).argmax(1)
        correct += (pred == val_lab[i:i + 512].to(DEVICE)).sum().item()
    return correct / va.size(0)


class AttentiveClassifier(nn.Module):
    def __init__(self, dim, num_classes, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        trunc_normal_(self.query, std=0.02)
        self.ln_x = nn.LayerNorm(dim)
        self.ln_q = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln_z = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):                      # x: (B, N, D)
        b = x.size(0)
        q = self.ln_q(self.query).expand(b, -1, -1)
        z, _ = self.cross(q, self.ln_x(x), self.ln_x(x))
        z = z + self.mlp(self.ln_z(z))
        return self.head(z.squeeze(1))


def _carve_val(n, frac=0.10, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(round(n * frac))
    return perm[n_val:], perm[:n_val]          # train_idx, holdout_idx


def train_attentive(train_tok, train_lab, val_tok, val_lab, dim,
                    lr, epochs=20, batch=512, num_classes=100, wd=0.05):
    tr_idx, ho_idx = _carve_val(train_tok.size(0), seed=0)
    Xtr, Ytr = train_tok[tr_idx].to(DEVICE), train_lab[tr_idx].to(DEVICE)
    Xho, Yho = train_tok[ho_idx].to(DEVICE), train_lab[ho_idx].to(DEVICE)
    clf = AttentiveClassifier(dim, num_classes).to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = -(-Xtr.size(0) // batch)          # ceil division
    steps = max(1, steps_per_epoch * epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(1)
    for _ in range(epochs):
        clf.train()
        perm = torch.randperm(Xtr.size(0), generator=g)
        for i in range(0, Xtr.size(0), batch):
            b = perm[i:i + batch]
            xb = Xtr[b].float()
            opt.zero_grad()
            loss = F.cross_entropy(clf(xb), Ytr[b])
            loss.backward()
            opt.step()
            sched.step()
    # holdout accuracy for model selection
    clf.eval()
    with torch.no_grad():
        ho_acc = 0
        for i in range(0, Xho.size(0), batch):
            ho_acc += (clf(Xho[i:i + batch].float()).argmax(1) == Yho[i:i + batch]).sum().item()
        ho_acc /= Xho.size(0)
        Xva = val_tok.to(DEVICE)
        va_acc = 0
        for i in range(0, Xva.size(0), batch):
            va_acc += (clf(Xva[i:i + batch].float()).argmax(1)
                       == val_lab[i:i + batch].to(DEVICE)).sum().item()
        va_acc /= Xva.size(0)
    return ho_acc, va_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--patch_size", type=int, default=None)
    ap.add_argument("--crop_size", type=int, default=None)
    ap.add_argument("--num_classes", type=int, default=100)
    ap.add_argument("--attn_epochs", type=int, default=20)
    ap.add_argument("--lrs", default="5e-4,1e-3,2e-3")
    ap.add_argument("--max_per_class", type=int, default=0,
                    help="Cap the TRAIN kNN/attentive bank at N imgs/class "
                         "(0 = use all; use ~200 on ImageNet-1k to bound RAM).")
    a = ap.parse_args()

    arch = _read_arch_from_metrics(a.ckpt)
    model_name = a.model_name or arch.get("model_name", "vit_small")
    patch_size = a.patch_size or arch.get("patch_size", 16)
    crop_size = a.crop_size or arch.get("crop_size", 128)
    print(f"[eval] {model_name} patch{patch_size} crop{crop_size} :: {a.ckpt}", flush=True)

    enc, epoch = load_encoder(a.ckpt, model_name, patch_size, crop_size)
    dim = enc.embed_dim

    tr_tok, tr_lab = extract_tokens(enc, a.data_root, "train", crop_size,
                                    max_per_class=a.max_per_class)
    va_tok, va_lab = extract_tokens(enc, a.data_root, "val", crop_size)
    print(f"[eval] tokens train {tuple(tr_tok.shape)} val {tuple(va_tok.shape)}", flush=True)

    knn_acc = knn_eval(tr_tok, tr_lab, va_tok, va_lab, num_classes=a.num_classes)
    print(f"[eval] kNN(20) test acc = {knn_acc:.4f}", flush=True)

    lrs = [float(x) for x in a.lrs.split(",")]
    sweep = {}
    best = (-1.0, None, None)                    # (holdout, lr, val_acc)
    for lr in lrs:
        ho, va = train_attentive(tr_tok, tr_lab, va_tok, va_lab, dim, lr,
                                 epochs=a.attn_epochs, num_classes=a.num_classes)
        sweep[f"{lr:.0e}"] = {"holdout_acc": ho, "val_acc": va}
        print(f"[eval] attentive lr={lr:.0e} holdout={ho:.4f} val={va:.4f}", flush=True)
        if ho > best[0]:
            best = (ho, lr, va)

    result = {
        "ckpt": os.path.abspath(a.ckpt),
        "model_name": model_name, "patch_size": patch_size, "crop_size": crop_size,
        "epoch": epoch,
        "knn20_test_acc": knn_acc,
        "attentive_sweep": sweep,
        "attentive_best_lr": f"{best[1]:.0e}",
        "attentive_test_acc": best[2],            # val_acc at holdout-selected lr
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)
    print(f"[eval] wrote {a.out} :: attentive={best[2]:.4f} (lr {best[1]:.0e}) kNN={knn_acc:.4f}",
          flush=True)


if __name__ == "__main__":
    main()

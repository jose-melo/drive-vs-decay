import os
import copy
import json
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from .model import LinearTJEPA, LinearTJEPAConfig
from .mask import SimpleBinaryMaskCollator, compute_mask_correlation
from src.collapse_metrics import create_collapse_monitor, JacobianAnalyzer

class LinearTJEPATrainer:

    def __init__(
        self,
        model: LinearTJEPA,
        config: LinearTJEPAConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        probe_callback: Optional[callable] = None,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.probe_callback = probe_callback
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = optim.AdamW(
            [
                {"params": model.context_encoder.parameters()},
                {"params": model.predictor.parameters()},
            ],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=config.learning_rate * 0.01,
        )

        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.training_history = []
        self.collapse_monitor = create_collapse_monitor(device=torch.device(device))
        self.jacobian_analyzer = JacobianAnalyzer(device=torch.device(device))

    def _compute_momentum(self, epoch: int, total_epochs: int) -> float:
        progress = epoch / max(1, total_epochs - 1)
        return self.config.ema_momentum_start + progress * (
            self.config.ema_momentum_end - self.config.ema_momentum_start
        )

    def train(
        self,
        num_epochs: int,
        log_every: int = 10,
        probe_every: int = 0,
        save_every: int = 0,
    ) -> Dict[str, Any]:
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs * len(self.train_loader),
            eta_min=self.config.learning_rate * 0.01,
        )

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_metrics = self._train_epoch(log_every)

            log_dict = {
                "epoch": epoch,
                "train/epoch_loss": epoch_metrics["avg_loss"],
                "train/mse_loss": epoch_metrics["avg_mse_loss"],
                "train/lr": self.scheduler.get_last_lr()[0],
                **{f"stability/{k}": v for k, v in epoch_metrics["stability"].items()},
            }
            if "collapse" in epoch_metrics:
                log_dict.update(
                    {f"collapse/{k}": v for k, v in epoch_metrics["collapse"].items()}
                )

            if self.val_loader is not None:
                val_loss = self._validate()
                log_dict["val/loss"] = val_loss

            if probe_every > 0 and (epoch + 1) % probe_every == 0:
                if self.probe_callback is not None:
                    probe_results = self.probe_callback(self.model)
                    log_dict.update({f"probe/{k}": v for k, v in probe_results.items()})

            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(log_dict)

            if save_every > 0 and (epoch + 1) % save_every == 0:
                self._save_checkpoint(f"epoch_{epoch+1}.pt")

            if epoch_metrics["avg_loss"] < self.best_loss:
                self.best_loss = epoch_metrics["avg_loss"]
                self._save_checkpoint("best_model.pt")

            self.training_history.append(epoch_metrics)

        self._save_checkpoint("final_model.pt")

        return {
            "best_loss": self.best_loss,
            "final_loss": epoch_metrics["avg_loss"],
            "history": self.training_history,
        }

    def _train_epoch(self, log_every: int = 10) -> Dict[str, Any]:
        self.model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_var = 0.0
        num_batches = 0
        collapse_metrics = []

        momentum = self._compute_momentum(self.epoch, len(self.train_loader))

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")
        for batch_idx, (batch, context_mask, target_mask) in enumerate(pbar):
            batch = batch.float().to(self.device)
            context_mask = context_mask.float().to(self.device)
            target_mask = target_mask.float().to(self.device)

            self.optimizer.zero_grad()
            losses = self.model.compute_loss(batch, context_mask, target_mask)

            losses["total_loss"].backward()

            torch.nn.utils.clip_grad_norm_(
                list(self.model.context_encoder.parameters())
                + list(self.model.predictor.parameters()),
                max_norm=1.0,
            )

            self.optimizer.step()
            self.scheduler.step()

            self.model.update_target_encoder(momentum=momentum)

            total_loss += losses["total_loss"].item()
            total_mse += losses["mse_loss"].item()
            total_var += losses["var_loss"].item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix(
                {
                    "loss": f"{losses['total_loss'].item():.4f}",
                    "mse": f"{losses['mse_loss'].item():.4f}",
                }
            )

            if log_every > 0 and batch_idx % log_every == 0:
                step_metrics = {
                    "train/step_loss": losses["total_loss"].item(),
                    "train/step_mse": losses["mse_loss"].item(),
                    "train/step_var": losses["var_loss"].item(),
                    "train/momentum": momentum,
                    "global_step": self.global_step,
                }

                mask_stats = compute_mask_correlation(context_mask, target_mask)
                step_metrics.update({f"masks/{k}": v for k, v in mask_stats.items()})

                with torch.no_grad():
                    context_emb = self.model.context_encoder(
                        batch, mask=context_mask
                    ).unsqueeze(1)
                    target_emb = self.model.target_encoder(
                        batch, mask=target_mask
                    ).unsqueeze(1)
                    collapse = self.collapse_monitor.compute_all_metrics(
                        context_encoder=self.model.context_encoder,
                        target_encoder=self.model.target_encoder,
                        predictor=self.model.predictor,
                        embeddings=context_emb,
                        target_embeddings=target_emb,
                        loss=losses["total_loss"].item(),
                    )
                    collapse_metrics.append(collapse)
                    step_metrics.update(
                        {
                            f"collapse/{k}": v
                            for k, v in collapse.items()
                            if isinstance(v, (int, float)) and not np.isnan(v)
                        }
                    )

                    try:
                        context_flat = context_emb.squeeze(1)
                        target_flat = target_emb.squeeze(1)
                        context_centered = context_flat - context_flat.mean(
                            dim=0, keepdim=True
                        )
                        target_centered = target_flat - target_flat.mean(
                            dim=0, keepdim=True
                        )
                        if context_centered.shape[0] > 1:
                            data_cov = (context_centered.T @ context_centered) / (
                                context_centered.shape[0] - 1
                            )
                            cross_cov = (target_centered.T @ context_centered) / (
                                context_centered.shape[0] - 1
                            )
                        else:
                            data_cov = torch.eye(
                                context_centered.shape[1],
                                device=context_centered.device,
                            )
                            cross_cov = torch.zeros_like(data_cov)
                        beta = 1.0 - momentum
                        mask_density = target_mask.mean().item()
                        jacobian = (
                            self.jacobian_analyzer.estimate_stability_eigenvalues(
                                predictor=self.model.predictor,
                                data_cov=data_cov,
                                cross_cov=cross_cov,
                                beta=beta,
                                mask_density=mask_density,
                            )
                        )
                        if "error" not in jacobian:
                            for key, value in jacobian.items():
                                if isinstance(value, (int, float)) and not np.isnan(
                                    value
                                ):
                                    step_metrics[f"jacobian/{key}"] = value
                    except Exception:
                        pass

                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log(step_metrics)

        stability_metrics = self.model.compute_stability_metrics()
        collapse_summary = {}
        if collapse_metrics:
            for key in collapse_metrics[0].keys():
                values = [
                    m[key]
                    for m in collapse_metrics
                    if isinstance(m.get(key), (int, float)) and not np.isnan(m.get(key))
                ]
                if values:
                    collapse_summary[key] = float(np.mean(values))

        return {
            "avg_loss": total_loss / num_batches,
            "avg_mse_loss": total_mse / num_batches,
            "avg_var_loss": total_var / num_batches,
            "stability": stability_metrics,
            "collapse": collapse_summary,
        }

    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch, context_mask, target_mask in self.val_loader:
                batch = batch.float().to(self.device)
                context_mask = context_mask.float().to(self.device)
                target_mask = target_mask.float().to(self.device)

                losses = self.model.compute_loss(batch, context_mask, target_mask)
                total_loss += losses["total_loss"].item()
                num_batches += 1

        self.model.train()
        return total_loss / num_batches

    def _save_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_loss": self.best_loss,
                "config": self.config.to_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]

class MLPProber:

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [128, 64],
        task_type: str = "classification",
        dropout: float = 0.1,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

        if task_type == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
        elif task_type == "binary_class":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def to(self, device):
        self.mlp = self.mlp.to(device)
        return self

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        num_epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> Dict[str, float]:
        self.mlp = self.mlp.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        if X_val is not None:
            X_val = X_val.to(device)
            y_val = y_val.to(device)

        optimizer = optim.Adam(self.mlp.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_val_score = -float("inf")

        for epoch in range(num_epochs):
            self.mlp.train()

            perm = torch.randperm(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]

            total_loss = 0.0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i : i + batch_size]
                batch_y = y_train[i : i + batch_size]

                optimizer.zero_grad()
                pred = self.mlp(batch_X)

                if self.task_type == "binary_class":
                    loss = self.loss_fn(pred.squeeze(), batch_y.float())
                elif self.task_type == "regression":
                    loss = self.loss_fn(pred.squeeze(), batch_y.float())
                else:
                    loss = self.loss_fn(pred, batch_y)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            if X_val is not None:
                val_score = self.evaluate(X_val, y_val)
                best_val_score = max(best_val_score, val_score)

        return {
            "best_val_score": best_val_score,
            "final_train_loss": total_loss / (len(X_train) // batch_size + 1),
        }

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        self.mlp.eval()
        with torch.no_grad():
            pred = self.mlp(X)

            if self.task_type == "classification":
                pred_labels = pred.argmax(dim=1)
                accuracy = (pred_labels == y).float().mean().item()
                return accuracy
            elif self.task_type == "binary_class":
                pred_labels = (pred.squeeze() > 0).long()
                accuracy = (pred_labels == y).float().mean().item()
                return accuracy
            else:
                mse = nn.functional.mse_loss(pred.squeeze(), y.float())
                return -mse.item()

def create_probe_callback(
    dataset,
    embed_dim: int,
    task_type: str = "classification",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):

    def probe_callback(model: LinearTJEPA) -> Dict[str, float]:
        model.eval()

        with torch.no_grad():
            X_train = dataset.X_train.float().to(device)
            X_val = dataset.X_val.float().to(device)

            emb_train = model.get_embeddings(X_train, use_target=True)
            emb_val = model.get_embeddings(X_val, use_target=True)

        y_train = dataset.y_train.to(device)
        y_val = dataset.y_val.to(device)

        if task_type == "classification":
            output_dim = len(torch.unique(y_train))
        elif task_type == "binary_class":
            output_dim = 1
        else:
            output_dim = 1

        prober = MLPProber(
            input_dim=embed_dim,
            output_dim=output_dim,
            hidden_dims=[128, 64],
            task_type=task_type,
        ).to(device)

        results = prober.fit(
            emb_train,
            y_train,
            emb_val,
            y_val,
            num_epochs=50,
            device=device,
        )

        model.train()
        return results

    return probe_callback

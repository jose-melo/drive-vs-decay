import datetime
import os
import sys
import json
import argparse
import random
import copy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.configs import build_parser
from src.encoder import Encoder
from src.predictors import Predictors
from src.feedback_predictor import FeedbackPredictors
from src.collapse_metrics import create_collapse_monitor, JacobianAnalyzer
from src.torch_dataset import TorchDataset, DataModule
from src.mask import MaskCollator
from src.utils.encode_utils import encode_data
from src.utils.train_utils import init_weights, apply_masks_from_idx, AllReduce
from src.utils.optim_utils import init_optim
from src.datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP
from src.datasets.online_dataset import OnlineDataset
from src.benchmark.utils import (
    MODEL_CONFIG_BASE_PATH,
    MODEL_NAME_TO_MODEL_MAP,
    get_loss_from_task,
)
from src.utils.models_utils import BaseModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

MASTER_SCALES = [
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    0.8,
    1.0,
    2.0,
    5.0,
    10.0,
]


def set_callbacks_loggers(args: dict):
    callbacks = [
        ModelCheckpoint(
            monitor=f"{args["data_set"]}_val_loss",
            mode="min",
            save_top_k=1,
            dirpath="checkpoints/",
            filename="model-{epoch:02d}-{val_loss:.2f}",
        ),
        EarlyStopping(
            monitor=f"{args["data_set"]}_val_loss",
            patience=args["exp_patience"],
            mode="min",
        ),
    ]
    loggers = [
        WandbLogger(
            name="drive-decay",
            log_model=False,
            log_graph=False,
            save_code=False,
        ),
        TensorBoardLogger(
            "lightning_logs",
            version=f"{args['model_name']}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}",
        ),
    ]

    return callbacks, loggers


def get_primary_mu(result: Dict, fallback_key: str = "final_mu_v3") -> float:
    jacobian_mu = result.get("jacobian_worst_case_mu")
    if jacobian_mu is not None and not np.isnan(jacobian_mu):
        return jacobian_mu

    jacobian_mu = result.get("jacobian_mu")
    if jacobian_mu is not None and not np.isnan(jacobian_mu):
        return jacobian_mu

    mu_v3 = result.get("final_mu_v3")
    if mu_v3 is not None and not np.isnan(mu_v3):
        return mu_v3

    return result.get("final_mu", 0.0)


class StabilityExperiment:
    def __init__(
        self,
        base_args,
        output_dir: str = "./experiments/results",
        device: str = None,
    ):
        self.base_args = base_args
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)
        self.results = []

        self.jacobian_analyzer = JacobianAnalyzer(device=self.device)

        self.tjepa_step = 0
        self.probe_step = 0
        self.config_id = 0

        if wandb.run is not None:
            wandb.define_metric("tjepa_step")
            wandb.define_metric("tjepa/*", step_metric="tjepa_step")
            wandb.define_metric("probe_step")
            wandb.define_metric("probe/*", step_metric="probe_step")

        torch.manual_seed(base_args.torch_seed)
        np.random.seed(base_args.np_seed)
        random.seed(base_args.np_seed)

        self.loss_fn = nn.MSELoss()

    def setup_data(self, args):
        dataset = DATASET_NAME_TO_DATASET_MAP[args.data_set](args)
        dataset.load()

        torch_dataset = TorchDataset(
            dataset=dataset,
            mode="train",
            kwargs=args,
            device=self.device,
            preprocessing=encode_data,
        )

        mask_collator = MaskCollator(
            args.mask_allow_overlap,
            args.mask_min_ctx_share,
            args.mask_max_ctx_share,
            args.mask_min_trgt_share,
            args.mask_max_trgt_share,
            args.mask_num_preds,
            args.mask_num_encs,
            dataset.D,
            dataset.cardinalities,
        )

        dataloader = DataLoader(
            dataset=torch_dataset,
            batch_size=args.batch_size,
            num_workers=args.data_loader_nprocs,
            collate_fn=mask_collator,
            pin_memory=args.pin_memory,
            drop_last=False,
        )

        return dataset, torch_dataset, dataloader

    def create_models(self, args, dataset, use_feedback: bool = False):
        context_encoder = Encoder(
            idx_num_features=dataset.num_features,
            cardinalities=dataset.cardinalities,
            hidden_dim=args.model_dim_hidden,
            num_layers=args.model_num_layers,
            num_heads=args.model_num_heads,
            p_dropout=args.model_dropout_prob,
            layer_norm_eps=args.model_layer_norm_eps,
            gradient_clipping=args.exp_gradient_clipping,
            feature_type_embedding=args.model_feature_type_embedding,
            feature_index_embedding=args.model_feature_index_embedding,
            dim_feedforward=args.model_dim_feedforward,
            device=self.device,
            args=args,
        )

        if use_feedback:
            predictors = FeedbackPredictors(
                pred_type=args.pred_type,
                hidden_dim=args.model_dim_hidden,
                pred_embed_dim=args.pred_embed_dim,
                num_features=dataset.D,
                num_layers=args.pred_num_layers,
                num_heads=args.pred_num_heads,
                p_dropout=args.pred_p_dropout,
                layer_norm_eps=args.pred_layer_norm_eps,
                activation=args.pred_activation,
                device=self.device,
                cardinalities=dataset.cardinalities,
                pred_dim_feedforward=args.pred_dim_feedforward,
                use_residual=True,
                residual_scale=getattr(args, "pred_residual_scale", 0.0),
                identity_attn_bias=getattr(args, "pred_identity_attn_bias", 10.0),
                use_zero_init=True,
            )
        else:
            predictors = Predictors(
                pred_type=args.pred_type,
                hidden_dim=args.model_dim_hidden,
                pred_embed_dim=args.pred_embed_dim,
                num_features=dataset.D,
                num_layers=args.pred_num_layers,
                num_heads=args.pred_num_heads,
                p_dropout=args.pred_p_dropout,
                layer_norm_eps=args.pred_layer_norm_eps,
                activation=args.pred_activation,
                device=self.device,
                cardinalities=dataset.cardinalities,
                pred_dim_feedforward=args.pred_dim_feedforward,
            )

        for m in context_encoder.modules():
            init_weights(m, init_type=args.init_type)

        if not use_feedback and args.pred_type == "mlp":
            for pred in predictors.predictors:
                for m in pred.modules():
                    init_weights(m, init_type=args.init_type)

        pred_init_scale = getattr(args, "pred_init_scale", None)
        if pred_init_scale is not None:
            default_std = 0.02
            scale_factor = pred_init_scale / default_std
            with torch.no_grad():
                for param in predictors.parameters():
                    if param.dim() >= 2:
                        param.mul_(scale_factor)

        target_encoder = copy.deepcopy(context_encoder)

        context_encoder.to(self.device)
        target_encoder.to(self.device)
        predictors.to(self.device)

        for p in target_encoder.parameters():
            p.requires_grad = False

        return context_encoder, target_encoder, predictors

    def train_epoch(
        self,
        context_encoder: nn.Module,
        target_encoder: nn.Module,
        predictors: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        wd_scheduler,
        scaler: GradScaler,
        momentum_scheduler,
        dataloader: DataLoader,
        args,
        collapse_monitor,
    ) -> Dict:
        context_encoder.train()
        predictors.train()
        target_encoder.eval()

        loss_fn = self.loss_fn
        total_loss = 0
        num_batches = 0
        epoch_metrics = []
        current_momentum = 0.0

        pbar = tqdm(dataloader, desc="  Batches", leave=False, ncols=100)
        for batch, masks_enc, masks_pred in pbar:
            batch = batch.to(self.device, non_blocking=True)
            masks_enc = [m.to(self.device, non_blocking=True) for m in masks_enc]
            masks_pred = [m.to(self.device, non_blocking=True) for m in masks_pred]

            with torch.cuda.amp.autocast(enabled=args.model_amp):
                with torch.no_grad():
                    h = target_encoder(batch)
                    h_masked = apply_masks_from_idx(h, masks_pred)

                z = context_encoder(batch, masks_enc)
                z_full = context_encoder(batch, mask=None)

                if args.pred_type == "mlp":
                    z_flat = z.view(z.size(0), -1)
                    z_pred = predictors(z_flat, masks_enc, masks_pred)
                    loss = sum(
                        loss_fn(zp, hp) for zp, hp in zip(z_pred, h_masked)
                    ) / len(z_pred)
                else:
                    z_pred = predictors(z, masks_enc, masks_pred)
                    loss = loss_fn(z_pred, h_masked)

                loss = AllReduce.apply(loss)
                assert not np.isnan(loss.item()), "loss is NaN"

                optimizer.zero_grad()
                if args.model_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                current_momentum = next(momentum_scheduler)
                for param_q, param_k in zip(
                    context_encoder.parameters(),
                    target_encoder.parameters(),
                ):
                    param_k.data.mul_(current_momentum).add_(
                        (1 - current_momentum) * param_q.detach().data
                    )

            if scheduler is not None:
                scheduler.step()
            if wd_scheduler is not None:
                wd_scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if num_batches % 10 == 0:
                with torch.no_grad():
                    z_full_views = []
                    z_ctx_views = []
                    for mask in masks_enc:
                        mask_keep = torch.zeros(
                            z_full.size(0), z_full.size(1), 1, device=z_full.device
                        )
                        mask_keep.scatter_(1, mask.unsqueeze(-1), 1.0)
                        z_ctx = z_full * mask_keep
                        z_full_views.append(z_full)
                        z_ctx_views.append(z_ctx)

                    z_full_cov = torch.cat(z_full_views, dim=0)
                    z_ctx_cov = torch.cat(z_ctx_views, dim=0)

                    context_mask = torch.zeros_like(batch)
                    if masks_enc:
                        context_mask.scatter_(1, masks_enc[0], 1.0)

                    mask_density = (
                        sum(m.shape[1] for m in masks_enc)
                        / (len(masks_enc) * z_full.shape[1])
                        if masks_enc
                        else 0.5
                    )

                    metrics = collapse_monitor.compute_all_metrics(
                        context_encoder=context_encoder,
                        target_encoder=target_encoder,
                        predictor=predictors,
                        embeddings=z_full_cov,
                        target_embeddings=z_ctx_cov,
                        loss=loss.item(),
                        batch=batch,
                        context_mask=context_mask,
                        beta=1.0 - current_momentum,
                        mask_density=mask_density,
                    )

                    epoch_metrics.append(metrics)

                    tjepa_log = {
                        f"tjepa/{k}": v
                        for k, v in metrics.items()
                        if isinstance(v, (int, float)) and not np.isnan(v)
                    }
                    tjepa_log["tjepa/batch"] = num_batches
                    tjepa_log["tjepa/loss"] = loss.item()
                    tjepa_log["tjepa/config_id"] = self.config_id
                    tjepa_log["tjepa_step"] = self.tjepa_step
                    wandb.log(tjepa_log)
                    self.tjepa_step += 1

        avg_loss = total_loss / max(num_batches, 1)

        final_metrics = {"loss": avg_loss, "ema_momentum": current_momentum}
        if epoch_metrics:
            for key in epoch_metrics[0].keys():
                values = []
                for m in epoch_metrics:
                    val = m.get(key)
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        values.append(val)
                if values:
                    final_metrics[key] = np.mean(values)
                elif key in epoch_metrics[-1]:
                    final_metrics[key] = epoch_metrics[-1][key]

        if scheduler is not None:
            final_metrics["lr"] = scheduler.get_last_lr()[0]
        if wd_scheduler is not None:
            final_metrics["weight_decay"] = wd_scheduler.get_last_wd()[0]

        return final_metrics

    def run_probe(
        self,
        target_encoder: nn.Module,
        args,
        probe_model: str = "mlp",
    ) -> Dict[str, float]:
        from argparse import Namespace

        probe_metrics = {}

        online_dataset_args = {
            "data_set": args.data_set,
            "data_path": args.data_path,
            "batch_size": 512,
            "data_loader_nprocs": getattr(args, "data_loader_nprocs", 0),
            "pin_memory": getattr(args, "pin_memory", False),
            "mock": getattr(args, "mock", False),
            "test_size_ratio": 0,
            "random_state": getattr(args, "np_seed", 42),
            "val_size_ratio": 0,
            "full_dataset_cuda": getattr(args, "full_dataset_cuda", False),
            "val_batch_size": getattr(args, "val_batch_size", 512),
            "input_embed_dim": args.model_dim_hidden,
        }
        online_dataset_args = Namespace(**online_dataset_args)

        online_dataset = OnlineDataset(
            online_dataset_args,
            target_encoder,
        )
        online_dataset.load()

        X = online_dataset.X

        probe_metrics["embedding_mean"] = float(np.mean(X))
        probe_metrics["embedding_std"] = float(np.std(X))

        model_class: BaseModel = MODEL_NAME_TO_MODEL_MAP[probe_model]

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dataset_args = vars(online_dataset_args).copy()
        dataset_args.update(
            {
                "test_size_ratio": 0.1,
                "val_size_ratio": 0.1,
                "batch_size": 128,
                "task_type": online_dataset.task_type,
                "using_embedding": True,
                "exp_train_total_epochs": (
                    50 if not getattr(args, "mock", False) else 1
                ),
                "model_name": probe_model,
                "dataset_name": args.data_set,
                "exp_patience": 20,
                "n_cls_tokens": getattr(args, "n_cls_tokens", 1),
            }
        )
        dataset_args = Namespace(**dataset_args)

        datamodule = DataModule(
            dataset=online_dataset,
            test_size_ratio=dataset_args.test_size_ratio,
            val_size_ratio=dataset_args.val_size_ratio,
            random_state=dataset_args.random_state,
            device=device,
            batch_size=dataset_args.batch_size,
            workers=dataset_args.data_loader_nprocs,
            pin_memory=dataset_args.pin_memory,
            full_dataset_cuda=dataset_args.full_dataset_cuda,
            preprocessing=model_class.preprocessing,
            mock=dataset_args.mock,
            using_embedding=True,
        )

        base_config = {
            "dataset_name": args.data_set,
            "encoder_type": "linear_flatten",
        }

        config_path = MODEL_CONFIG_BASE_PATH.format(
            dataset_name=args.data_set,
            model_name=probe_model,
        )

        if os.path.exists(config_path):
            model_args = json.load(open(config_path))
        else:
            model_args = {
                "model_dim_hidden": 256,
                "model_num_layers": 2,
                "exp_lr": 1e-3,
            }

        model_args.update(base_config)
        model_args = Namespace(**model_args)

        model_args = model_class.get_model_args(
            datamodule,
            dataset_args,
            model_args,
        )

        loss_fn = get_loss_from_task(dataset_args.task_type)
        final_dataset_args = {**vars(dataset_args), **vars(model_args)}
        model = model_class(loss=loss_fn, **final_dataset_args)
        model = model.float()

        callbacks, loggers = set_callbacks_loggers(final_dataset_args)

        trainer = pl.Trainer(
            max_epochs=final_dataset_args["exp_train_total_epochs"],
            logger=loggers,
            callbacks=callbacks,
            log_every_n_steps=10,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(model, datamodule=datamodule)
        val_metrics = trainer.validate(model, datamodule=datamodule, verbose=False)
        test_metrics = trainer.test(model, datamodule=datamodule, verbose=False)

        if val_metrics:
            val_score = val_metrics[0].get(f"{args.data_set}_val_score", 0)
            probe_metrics["probe_val_score"] = float(val_score)

            for key, value in val_metrics[0].items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    probe_metrics[f"probe_val_{key}"] = float(value)

        if test_metrics:
            test_score = test_metrics[0].get(f"{args.data_set}_test_score", 0)
            probe_metrics["probe_test_score"] = float(test_score)

            for key, value in test_metrics[0].items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    probe_metrics[f"probe_test_{key}"] = float(value)

        return probe_metrics

    def run_single_config(
        self,
        config: Dict,
        num_epochs: int = 20,
        use_feedback: bool = False,
        probe_cadence: int = 0,
        probe_model: str = "mlp",
    ) -> Dict:
        self.config_id += 1

        args = copy.deepcopy(self.base_args)
        for key, value in config.items():
            setattr(args, key, value)

        dataset, torch_dataset, dataloader = self.setup_data(args)
        context_encoder, target_encoder, predictors = self.create_models(
            args, dataset, use_feedback=use_feedback
        )

        ipe = len(dataloader)
        ipe_scale = getattr(args, "exp_ipe_scale", 1.25)

        optimizer, scheduler, wd_scheduler = init_optim(
            context_encoder,
            predictors,
            ipe,
            args.exp_start_lr,
            args.exp_lr,
            args.exp_warmup,
            num_epochs,
            args.exp_weight_decay,
            args.exp_final_weight_decay,
            args.exp_final_lr,
            ipe_scale,
            getattr(args, "exp_scheduler", True),
            getattr(args, "exp_weight_decay_scheduler", True),
        )

        scaler = GradScaler(enabled=args.model_amp)

        ema_start = args.model_ema_start
        ema_end = args.model_ema_end
        momentum_scheduler = (
            ema_start + i * (ema_end - ema_start) / (ipe * num_epochs * ipe_scale)
            for i in range(int(ipe * num_epochs * ipe_scale) + 1)
        )

        collapse_monitor = create_collapse_monitor(self.device)

        results = {
            "config": config,
            "use_feedback": use_feedback,
            "epochs": [],
            "probe_results": [],
        }

        last_probe_metrics = {}

        epoch_pbar = tqdm(range(num_epochs), desc="  Epochs", ncols=100)
        for epoch in epoch_pbar:
            metrics = self.train_epoch(
                context_encoder,
                target_encoder,
                predictors,
                optimizer,
                scheduler,
                wd_scheduler,
                scaler,
                momentum_scheduler,
                dataloader,
                args,
                collapse_monitor,
            )

            metrics["epoch"] = epoch

            epoch_log = {
                "tjepa/train_loss": metrics.get("loss", 0),
                "tjepa/epoch": epoch,
                "tjepa/lr": metrics.get("lr", args.exp_lr),
                "tjepa/momentum": metrics.get("ema_momentum", ema_start),
                "tjepa/weight_decay": metrics.get(
                    "weight_decay", args.exp_weight_decay
                ),
                "tjepa/config_id": self.config_id,
            }
            for k, v in metrics.items():
                if (
                    isinstance(v, (int, float))
                    and not np.isnan(v)
                    and k not in ["epoch", "ema_momentum", "lr", "weight_decay", "loss"]
                ):
                    epoch_log[f"tjepa/{k}"] = v
            epoch_log["tjepa_step"] = self.tjepa_step
            wandb.log(epoch_log)
            self.tjepa_step += 1

            if probe_cadence > 0 and epoch % probe_cadence == 0:
                probe_metrics = self.run_probe(target_encoder, args, probe_model)
                metrics.update(probe_metrics)
                last_probe_metrics = probe_metrics
                results["probe_results"].append({"epoch": epoch, **probe_metrics})

                probe_log = {
                    f"probe/{k}": v
                    for k, v in probe_metrics.items()
                    if isinstance(v, (int, float)) and not np.isnan(v)
                }
                probe_log["probe/tjepa_epoch"] = epoch
                probe_log["probe/config_id"] = self.config_id
                probe_log["probe_step"] = self.probe_step
                wandb.log(probe_log)
                self.probe_step += 1

            results["epochs"].append(metrics)

            postfix = {
                "loss": f"{metrics.get('loss', 0):.4f}",
                "var": f"{metrics.get('total_variance', 0):.2e}",
            }
            jacobian_mu = metrics.get("jacobian_worst_case_mu", np.nan)
            if not np.isnan(jacobian_mu):
                postfix["μ_J"] = f"{jacobian_mu:.3f}"
            if last_probe_metrics.get("probe_val_score"):
                postfix["probe"] = f"{last_probe_metrics['probe_val_score']:.3f}"
            epoch_pbar.set_postfix(postfix)

            if metrics.get("total_variance", 1) < 1e-6:
                break

        if probe_cadence == -1:
            probe_metrics = self.run_probe(target_encoder, args, probe_model)
            last_probe_metrics = probe_metrics
            results["probe_results"].append({"epoch": epoch, **probe_metrics})

            probe_log = {
                f"probe/{k}": v
                for k, v in probe_metrics.items()
                if isinstance(v, (int, float)) and not np.isnan(v)
            }
            probe_log["probe/epoch"] = epoch
            probe_log["probe_step"] = self.probe_step
            wandb.log(probe_log)
            self.probe_step += 1

        results["collapsed"] = metrics.get("total_variance", 1) < 1e-4
        results["final_mu"] = metrics.get("mu_estimate", 0)
        results["final_mu_v3"] = metrics.get("mu_estimate_v3", 0)
        results["final_sigma_v3"] = metrics.get("sigma_v3", 0)
        results["final_gamma_v3"] = metrics.get("gamma_v3", 0)
        results["final_variance"] = metrics.get("total_variance", 0)
        results["regime"] = collapse_monitor.detect_collapse_regime()

        results["jacobian_mu"] = metrics.get("jacobian_mu", np.nan)
        results["jacobian_worst_case_mu"] = metrics.get(
            "jacobian_worst_case_mu", np.nan
        )
        results["jacobian_lambda1"] = metrics.get("jacobian_lambda1", np.nan)
        results["jacobian_lambda2"] = metrics.get("jacobian_lambda2", np.nan)
        results["jacobian_max_real_eigenvalue"] = metrics.get(
            "jacobian_max_real_eigenvalue", np.nan
        )
        results["jacobian_regime"] = metrics.get("jacobian_regime", "unknown")
        results["jacobian_stable"] = metrics.get("jacobian_stable", np.nan)
        results["jacobian_oscillatory"] = metrics.get("jacobian_oscillatory", np.nan)
        results["jacobian_discriminant"] = metrics.get("jacobian_discriminant", np.nan)

        if last_probe_metrics:
            results["final_probe_val_score"] = last_probe_metrics.get(
                "probe_val_score", None
            )
            results["final_probe_test_score"] = last_probe_metrics.get(
                "probe_test_score", None
            )

        return results


def experiment_critical_regime_2d(
    exp: StabilityExperiment, args, probe_cadence: int = 0, probe_model: str = "mlp"
):
    scales = MASTER_SCALES
    ctx_shares = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    results = []
    total_configs = len(scales) * len(ctx_shares)
    current = 0

    for scale in tqdm(scales, desc="Scales", ncols=100):
        for ctx in tqdm(ctx_shares, desc="  Context shares", leave=False, ncols=100):
            current += 1

            config = {
                "pred_init_scale": scale,
                "mask_min_ctx_share": ctx * 0.8,
                "mask_max_ctx_share": ctx,
                "data_set": args.data_set,
            }

            result = exp.run_single_config(
                config,
                num_epochs=40,
                probe_cadence=probe_cadence,
                probe_model=probe_model,
            )
            result["scale"] = scale
            result["ctx_share"] = ctx
            results.append(result)

    save_path = exp.output_dir / "critical_regime_2d_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    plot_critical_regime_2d(results, exp.output_dir)

    return results


def experiment_single_point(
    exp: StabilityExperiment, args, probe_cadence: int = 0, probe_model: str = "mlp"
):
    pred_init_scale = getattr(args, "pred_init_scale", 0.1)
    ctx_share = getattr(args, "mask_max_ctx_share", 0.5)

    config = {
        "pred_init_scale": pred_init_scale,
        "mask_min_ctx_share": ctx_share * 0.8,
        "mask_max_ctx_share": ctx_share,
        "data_set": args.data_set,
    }

    use_feedback = bool(getattr(args, "use_residual_predictor", False))

    result = exp.run_single_config(
        config,
        num_epochs=getattr(args, "exp_train_total_epochs", 40),
        use_feedback=use_feedback,
        probe_cadence=probe_cadence,
        probe_model=probe_model,
    )
    result["scale"] = pred_init_scale
    result["ctx_share"] = ctx_share
    result["use_residual_predictor"] = use_feedback

    if wandb.run is not None:
        wandb.log(
            {
                "final_probe_val_score": result.get("final_probe_val_score", 0.0),
                "final_mu": result.get("final_mu", 0.0),
                "final_mu_v3": result.get("final_mu_v3", 0.0),
                "collapsed": int(result.get("collapsed", False)),
            }
        )

    save_path = (
        exp.output_dir / f"single_point_scale_{pred_init_scale}_ctx_{ctx_share}.json"
    )
    with open(save_path, "w") as f:
        json.dump([result], f, indent=2, default=str)

    return [result]


def plot_critical_regime_2d(results, output_dir):
    scales = sorted(set(r["scale"] for r in results))
    ctx_shares = sorted(set(r["ctx_share"] for r in results))

    n_scales, n_ctx = len(scales), len(ctx_shares)
    acc_matrix = np.full((n_ctx, n_scales), np.nan)
    mu_matrix = np.full((n_ctx, n_scales), np.nan)

    for r in results:
        i = ctx_shares.index(r["ctx_share"])
        j = scales.index(r["scale"])
        acc_matrix[i, j] = r.get("final_probe_val_score", np.nan)
        mu_matrix[i, j] = get_primary_mu(r)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    im = ax.imshow(
        acc_matrix,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[min(scales), max(scales), min(ctx_shares), max(ctx_shares)],
    )
    ax.set_xscale("log")
    ax.set_xlabel("Predictor Scale ||W_p||")
    ax.set_ylabel("Context Share")
    ax.set_title("Probe Accuracy")
    plt.colorbar(im, ax=ax, label="Accuracy")

    if not np.all(np.isnan(acc_matrix)):
        best_idx = np.unravel_index(np.nanargmax(acc_matrix), acc_matrix.shape)
        best_ctx, best_scale = ctx_shares[best_idx[0]], scales[best_idx[1]]
        best_acc = acc_matrix[best_idx]
        ax.scatter(
            [best_scale],
            [best_ctx],
            c="red",
            s=200,
            marker="*",
            zorder=5,
            label=f"Best: {best_acc:.3f}",
        )
        ax.legend(loc="upper right")

    ax = axes[0, 1]
    im = ax.imshow(
        mu_matrix,
        aspect="auto",
        cmap="RdYlGn",
        origin="lower",
        extent=[min(scales), max(scales), min(ctx_shares), max(ctx_shares)],
        vmin=0,
        vmax=2,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Predictor Scale ||W_p||")
    ax.set_ylabel("Context Share")
    ax.set_title("Stability Ratio μ")
    plt.colorbar(im, ax=ax, label="μ")

    try:
        X, Y = np.meshgrid(scales, ctx_shares)
        contour = ax.contour(
            X, Y, mu_matrix, levels=[1.0], colors="black", linewidths=2
        )
        ax.clabel(contour, inline=True, fontsize=10, fmt="μ=1")
    except Exception:
        pass

    if not np.all(np.isnan(acc_matrix)):
        ax.scatter(
            [best_scale],
            [best_ctx],
            c="blue",
            s=200,
            marker="*",
            zorder=5,
            label=f"Best acc @ μ={mu_matrix[best_idx]:.2f}",
        )
        ax.legend(loc="upper right")

    ax = axes[1, 0]
    flat_acc = acc_matrix.flatten()
    flat_mu = mu_matrix.flatten()
    flat_ctx = np.array([[ctx] * n_scales for ctx in ctx_shares]).flatten()

    valid_mask = ~np.isnan(flat_acc) & ~np.isnan(flat_mu)
    valid_acc = flat_acc[valid_mask]
    valid_mu = flat_mu[valid_mask]
    valid_ctx = flat_ctx[valid_mask]

    if len(valid_acc) > 0:
        scatter = ax.scatter(
            valid_mu, valid_acc, c=valid_ctx, cmap="plasma", s=80, alpha=0.8
        )
        plt.colorbar(scatter, ax=ax, label="Context Share")

        try:
            log_mu = np.log10(valid_mu + 1e-10)
            coeffs = np.polyfit(log_mu, valid_acc, 2)
            mu_fit = np.logspace(
                np.log10(valid_mu.min()), np.log10(valid_mu.max()), 100
            )
            acc_fit = np.polyval(coeffs, np.log10(mu_fit))
            ax.plot(
                mu_fit, acc_fit, "k--", linewidth=2, alpha=0.5, label="Quadratic fit"
            )
        except Exception:
            pass

    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=2, label="μ=1")
    ax.set_xscale("log")
    ax.set_xlabel("Stability Ratio μ")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Accuracy vs μ")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    regime_matrix = np.zeros((n_ctx, n_scales))
    for r in results:
        i = ctx_shares.index(r["ctx_share"])
        j = scales.index(r["scale"])
        mu = get_primary_mu(r)
        if r.get("collapsed", False):
            regime_matrix[i, j] = 0
        elif mu < 0.8:
            regime_matrix[i, j] = 1
        elif mu > 1.2:
            regime_matrix[i, j] = 3
        else:
            regime_matrix[i, j] = 2

    cmap = plt.cm.colors.ListedColormap(["gray", "blue", "green", "red"])
    im = ax.imshow(
        regime_matrix,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=[min(scales), max(scales), min(ctx_shares), max(ctx_shares)],
        vmin=0,
        vmax=3,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Predictor Scale ||W_p||")
    ax.set_ylabel("Context Share")
    ax.set_title("Regime Classification")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="gray", label="Collapsed"),
        Patch(facecolor="blue", label="Subcritical (μ<0.8)"),
        Patch(facecolor="green", label="Critical (0.8≤μ≤1.2)"),
        Patch(facecolor="red", label="Supercritical (μ>1.2)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "critical_regime_2d_validation.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="critical_2d",
        choices=["critical_2d", "single_point"],
    )
    parser.add_argument("--data_set", type=str, default="jannis")
    parser.add_argument("--output_dir", type=str, default="./experiments/results")
    parser.add_argument("--hyperparams", type=str, default=None)
    parser.add_argument("--probe_cadence", type=int, default=10)
    parser.add_argument(
        "--probe_model",
        type=str,
        default="mlp",
        choices=list(MODEL_NAME_TO_MODEL_MAP.keys()),
    )
    parser.add_argument("--pred_init_scale", type=float, default=0.1)
    parser.add_argument("--mask_max_ctx_share", type=float, default=0.5)
    # Convenience aliases, mapped onto base_args fields below.
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Pretraining epochs (alias for exp_train_total_epochs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master seed (sets np_seed and torch_seed).",
    )
    parser.add_argument(
        "--context_ratio",
        type=float,
        default=None,
        help="Context-mask ratio (sets mask_min_ctx_share = 0.8*v, "
        "mask_max_ctx_share = v).",
    )
    parser.add_argument(
        "--use_residual_predictor",
        action="store_true",
        default=False,
        help="Use the ResidualPred predictor instead of the standard one.",
    )
    exp_args, _unknown = parser.parse_known_args()

    base_parser = build_parser()
    base_args = base_parser.parse_args([])

    hyperparams_file = exp_args.hyperparams
    hyperparams = {}
    if hyperparams_file is None:
        dataset_hyperparams = (
            Path(__file__).parent / f"{exp_args.data_set}_hyperparameters.json"
        )
        if dataset_hyperparams.exists():
            hyperparams_file = str(dataset_hyperparams)

    if hyperparams_file:
        with open(hyperparams_file, "r") as f:
            hyperparams = json.load(f)

        for key, value in hyperparams.items():
            setattr(base_args, key, value)

    base_args.data_set = exp_args.data_set
    if "batch_size" not in hyperparams:
        base_args.batch_size = 64
    base_args.mock = False

    base_args.pred_init_scale = exp_args.pred_init_scale
    base_args.mask_max_ctx_share = exp_args.mask_max_ctx_share

    if exp_args.num_epochs is not None:
        base_args.exp_train_total_epochs = exp_args.num_epochs
    if exp_args.seed is not None:
        base_args.np_seed = exp_args.seed
        base_args.torch_seed = exp_args.seed
    if exp_args.context_ratio is not None:
        base_args.mask_max_ctx_share = exp_args.context_ratio
        base_args.mask_min_ctx_share = 0.8 * exp_args.context_ratio
    base_args.use_residual_predictor = bool(exp_args.use_residual_predictor)

    try:
        wandb.init(
            project="drive-vs-decay",
            config=vars(base_args),
            mode=os.environ.get("WANDB_MODE", "online"),
        )
    except Exception:
        os.environ.setdefault("WANDB_MODE", "disabled")
        wandb.init(project="drive-vs-decay", config=vars(base_args), mode="disabled")

    exp = StabilityExperiment(base_args, output_dir=exp_args.output_dir)

    probe_cadence = exp_args.probe_cadence
    probe_model = exp_args.probe_model

    use_feedback = base_args.use_residual_predictor

    def _critical_2d():
        return experiment_critical_regime_2d(exp, base_args, probe_cadence, probe_model)

    def _single_point():
            return experiment_single_point(exp, base_args, probe_cadence, probe_model)

    experiments = {
        "critical_2d": _critical_2d,
        "single_point": _single_point,
    }

    experiments[exp_args.experiment]()

    wandb.finish()


if __name__ == "__main__":
    main()

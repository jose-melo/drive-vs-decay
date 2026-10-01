import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from src.linear_tjepa import (
    LinearTJEPA,
    LinearTJEPADataModule,
    SimpleBinaryMaskCollator,
)
from src.linear_tjepa.model import LinearTJEPAConfig
from src.linear_tjepa.trainer import MLPProber
from src.collapse_metrics import create_collapse_monitor, JacobianAnalyzer

MASTER_SCALES = [
    # Initialization scales of the Table 1 phase grid.
    0.0001,
    0.001,
    0.01,
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
]

PROBE_CADENCE_DISABLED = 0
PROBE_CADENCE_FINAL_ONLY = -1

def get_primary_mu(result: Dict[str, Any], fallback_key: str = "final_mu_v3") -> float:
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
        device: Optional[str] = None,
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

        torch.manual_seed(base_args.seed)
        np.random.seed(base_args.seed)
        random.seed(base_args.seed)

        self.loss_fn = nn.MSELoss()

    def _compute_momentum(self, epoch: int, total_epochs: int, args) -> float:
        progress = epoch / max(1, total_epochs - 1)
        return args.ema_momentum_start + progress * (
            args.ema_momentum_end - args.ema_momentum_start
        )

    def setup_data(self, args):
        data_module = LinearTJEPADataModule(
            dataset_name=args.data_set,
            data_path=args.data_path,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            normalization=args.normalization,
            seed=args.seed,
            mock=args.mock,
        )
        data_module.load()

        input_dim = data_module.info.input_dim
        mask_collator = SimpleBinaryMaskCollator(
            num_features=input_dim,
            context_ratio=args.context_ratio,
            target_ratio=args.target_ratio,
            allow_overlap=args.allow_mask_overlap,
            seed=args.seed,
        )

        train_loader, val_loader = data_module.create_data_loaders(
            mask_collator=mask_collator,
            num_workers=0,
            pin_memory=False,
        )

        return data_module, train_loader, val_loader

    def create_model(self, args, input_dim: int, use_feedback: bool = False):
        use_residual = use_feedback or args.use_residual_predictor
        config = LinearTJEPAConfig(
            input_dim=input_dim,
            embed_dim=args.embed_dim,
            encoder_type=args.encoder_type,
            predictor_type=args.predictor_type,
            use_residual_predictor=use_residual,
            predictor_init_scale=args.predictor_init_scale,
            encoder_init_scale=args.encoder_init_scale,
            encoder_init_type=args.encoder_init_type,
            regularize_residual=args.regularize_residual,
            use_variance_reg=args.use_variance_reg,
            variance_reg_weight=args.variance_reg_weight,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            ema_momentum_start=args.ema_momentum_start,
            ema_momentum_end=args.ema_momentum_end,
            context_ratio=args.context_ratio,
            target_ratio=args.target_ratio,
            allow_mask_overlap=args.allow_mask_overlap,
        )

        model = LinearTJEPA(
            input_dim=config.input_dim,
            embed_dim=config.embed_dim,
            encoder_type=config.encoder_type,
            predictor_type=config.predictor_type,
            use_residual_predictor=config.use_residual_predictor,
            predictor_init_scale=config.predictor_init_scale,
            encoder_init_scale=config.encoder_init_scale,
            encoder_init_type=config.encoder_init_type,
            regularize_residual=config.regularize_residual,
            use_variance_reg=config.use_variance_reg,
            variance_reg_weight=config.variance_reg_weight,
        ).to(self.device)

        return model, config

    def train_epoch(
        self,
        model: LinearTJEPA,
        optimizer: torch.optim.Optimizer,
        scheduler,
        dataloader: DataLoader,
        args,
        epoch: int,
        total_epochs: int,
        collapse_monitor,
    ) -> Dict[str, Any]:
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_var = 0.0
        num_batches = 0
        epoch_metrics: List[Dict[str, Any]] = []

        momentum = self._compute_momentum(epoch, total_epochs, args)
        pbar = tqdm(dataloader, desc="  Batches", leave=False, ncols=100)
        for batch, context_mask, target_mask in pbar:
            batch = batch.float().to(self.device)
            context_mask = context_mask.float().to(self.device)
            target_mask = target_mask.float().to(self.device)

            optimizer.zero_grad()
            losses = model.compute_loss(batch, context_mask, target_mask)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.context_encoder.parameters())
                + list(model.predictor.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()

            model.update_target_encoder(momentum=momentum)

            total_loss += losses["total_loss"].item()
            total_mse += losses["mse_loss"].item()
            total_var += losses["var_loss"].item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{losses['total_loss'].item():.4f}"})

            if num_batches % args.metric_every == 0:
                with torch.no_grad():
                    context_full = model.context_encoder(
                        batch, mask=context_mask
                    ).unsqueeze(1)
                    target_emb = model.target_encoder(
                        batch, mask=target_mask
                    ).unsqueeze(1)

                    mask_density = target_mask.mean().item()
                    beta = 1.0 - momentum

                    metrics = collapse_monitor.compute_all_metrics(
                        context_encoder=model.context_encoder,
                        target_encoder=model.target_encoder,
                        predictor=model.predictor,
                        embeddings=context_full,
                        target_embeddings=target_emb,
                        loss=losses["total_loss"].item(),
                        mask_density=mask_density,
                        batch=batch,
                        context_mask=context_mask,
                        beta=beta,
                    )

                    epoch_metrics.append(metrics)

                    step_log = {
                        f"tjepa/{k}": v
                        for k, v in metrics.items()
                        if isinstance(v, (int, float)) and not np.isnan(v)
                    }
                    step_log["tjepa/batch"] = num_batches
                    step_log["tjepa/loss"] = losses["total_loss"].item()
                    step_log["tjepa/config_id"] = self.config_id
                    step_log["tjepa_step"] = self.tjepa_step

                    if (
                        "jacobian_logmu_hist" in metrics
                        and "jacobian_logmu_bins" in metrics
                    ):
                        wandb_hist = wandb.Histogram(
                            np_histogram=(
                                np.array(metrics["jacobian_logmu_hist"]),
                                np.array(metrics["jacobian_logmu_bins"]),
                            )
                        )
                        step_log["tjepa/jacobian_logmu_hist"] = wandb_hist

                    wandb.log(step_log)

                    self.tjepa_step += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)
        avg_var = total_var / max(num_batches, 1)

        collapse_summary = {}
        if epoch_metrics:
            for key in epoch_metrics[0].keys():
                values = [
                    m[key]
                    for m in epoch_metrics
                    if isinstance(m.get(key), (int, float)) and not np.isnan(m.get(key))
                ]
                if values:
                    if "max_real_eigenvalue" in key:
                        collapse_summary[key] = float(np.max(values))
                    elif "mu" in key:
                        collapse_summary[key] = float(np.min(values))
                    else:
                        collapse_summary[key] = float(np.mean(values))

        return {
            "loss": avg_loss,
            "mse_loss": avg_mse,
            "var_loss": avg_var,
            "ema_momentum": momentum,
            "collapse": collapse_summary,
        }

    def run_probe(self, model: LinearTJEPA, data_module, args) -> Dict[str, float]:
        model.eval()
        device = self.device
        with torch.no_grad():
            X_train = data_module.X_train.to(device)
            X_val = data_module.X_val.to(device)
            X_test = data_module.X_test.to(device)
            emb_train = model.get_embeddings(X_train, use_target=True)
            emb_val = model.get_embeddings(X_val, use_target=True)
            emb_test = model.get_embeddings(X_test, use_target=True)

        y_train = data_module.y_train.to(device)
        y_val = data_module.y_val.to(device)
        y_test = data_module.y_test.to(device)

        task_type = data_module.info.task_type
        if task_type == "multi_class":
            probe_task = "classification"
            output_dim = data_module.info.n_classes
        elif task_type == "binary_class":
            probe_task = "binary_class"
            output_dim = 1
        else:
            probe_task = "regression"
            output_dim = 1

        prober = MLPProber(
            input_dim=args.embed_dim,
            output_dim=output_dim,
            hidden_dims=[128, 64],
            task_type=probe_task,
        ).to(device)

        results = prober.fit(
            emb_train,
            y_train,
            emb_val,
            y_val,
            num_epochs=args.probe_epochs,
            batch_size=args.probe_batch_size,
            lr=args.probe_lr,
            device=str(device),
        )

        probe_metrics = {
            "probe_val_score": results.get("best_val_score", 0.0),
            "probe_train_loss": results.get("final_train_loss", 0.0),
        }

        test_score = prober.evaluate(emb_test, y_test)
        probe_metrics["probe_test_score"] = test_score

        return probe_metrics

    def run_single_config(
        self,
        config: Dict[str, Any],
        num_epochs: int = 20,
        use_feedback: bool = False,
        probe_cadence: int = 0,
    ) -> Dict[str, Any]:
        self.config_id += 1

        current_seed = self.base_args.seed + self.config_id
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)

        args = argparse.Namespace(**vars(self.base_args))
        for key, value in config.items():
            setattr(args, key, value)

        data_module, train_loader, _ = self.setup_data(args)
        model, _ = self.create_model(
            args, data_module.info.input_dim, use_feedback=use_feedback
        )

        collapse_monitor = create_collapse_monitor(
            device=self.device, normalization_mode="raw"
        )

        optimizer = optim.AdamW(
            [
                {"params": model.context_encoder.parameters()},
                {"params": model.predictor.parameters()},
            ],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs * len(train_loader),
            eta_min=args.learning_rate * 0.01,
        )

        results = {
            "config": config,
            "use_feedback": use_feedback,
            "epochs": [],
            "probe_results": [],
        }

        last_probe_metrics = {}
        best_probe_val_score = None
        best_probe_epoch = None
        epoch_pbar = tqdm(range(num_epochs), desc="  Epochs", ncols=100)
        for epoch in epoch_pbar:
            metrics = self.train_epoch(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloader=train_loader,
                args=args,
                epoch=epoch,
                total_epochs=num_epochs,
                collapse_monitor=collapse_monitor,
            )
            metrics["epoch"] = epoch

            epoch_log = {
                "tjepa/train_loss": metrics.get("loss", 0.0),
                "tjepa/epoch": epoch,
                "tjepa/lr": scheduler.get_last_lr()[0],
                "tjepa/momentum": metrics.get("ema_momentum", args.ema_momentum_start),
                "tjepa/weight_decay": args.weight_decay,
                "tjepa/config_id": self.config_id,
            }
            for k, v in metrics.get("collapse", {}).items():
                if isinstance(v, (int, float)) and not np.isnan(v):
                    epoch_log[f"tjepa/{k}"] = v
            epoch_log["tjepa_step"] = self.tjepa_step
            wandb.log(epoch_log)
            self.tjepa_step += 1

            if probe_cadence > PROBE_CADENCE_DISABLED and epoch % probe_cadence == 0:
                probe_metrics = self.run_probe(model, data_module, args)
                last_probe_metrics = probe_metrics
                results["probe_results"].append({"epoch": epoch, **probe_metrics})

                current_val_score = probe_metrics.get("probe_val_score")
                if current_val_score is not None:
                    if (
                        best_probe_val_score is None
                        or current_val_score > best_probe_val_score
                    ):
                        best_probe_val_score = current_val_score
                        best_probe_epoch = epoch

                probe_log = {
                    f"probe/{k}": v
                    for k, v in probe_metrics.items()
                    if isinstance(v, (int, float)) and not np.isnan(v)
                }
                probe_log["probe/tjepa_epoch"] = epoch
                probe_log["probe/config_id"] = self.config_id
                probe_log["probe_step"] = self.probe_step
                if best_probe_val_score is not None:
                    probe_log["probe/best_val_score"] = best_probe_val_score
                wandb.log(probe_log)
                self.probe_step += 1

            results["epochs"].append(metrics)

            collapse_metrics = metrics.get("collapse", {})
            postfix = {
                "loss": f"{metrics.get('loss', 0):.4f}",
                "var": f"{collapse_metrics.get('total_variance', 0):.2e}",
            }
            jacobian_mu = collapse_metrics.get("jacobian_worst_case_mu", np.nan)
            if not np.isnan(jacobian_mu):
                postfix["μ_J"] = f"{jacobian_mu:.3f}"
            if last_probe_metrics.get("probe_val_score") is not None:
                postfix["probe"] = f"{last_probe_metrics['probe_val_score']:.3f}"
            epoch_pbar.set_postfix(postfix)

            if collapse_metrics.get("total_variance", 1.0) < 1e-6:
                break

        final_epoch_probed = (
            probe_cadence > PROBE_CADENCE_DISABLED and epoch % probe_cadence == 0
        )
        should_run_final_probe = probe_cadence == PROBE_CADENCE_FINAL_ONLY or (
            probe_cadence > PROBE_CADENCE_DISABLED and not final_epoch_probed
        )
        if should_run_final_probe:
            probe_metrics = self.run_probe(model, data_module, args)
            last_probe_metrics = probe_metrics
            results["probe_results"].append({"epoch": epoch, **probe_metrics})

            current_val_score = probe_metrics.get("probe_val_score")
            if current_val_score is not None:
                if (
                    best_probe_val_score is None
                    or current_val_score > best_probe_val_score
                ):
                    best_probe_val_score = current_val_score
                    best_probe_epoch = epoch

            probe_log = {
                f"probe/{k}": v
                for k, v in probe_metrics.items()
                if isinstance(v, (int, float)) and not np.isnan(v)
            }
            probe_log["probe/epoch"] = epoch
            probe_log["probe_step"] = self.probe_step
            if best_probe_val_score is not None:
                probe_log["probe/best_val_score"] = best_probe_val_score
            wandb.log(probe_log)
            self.probe_step += 1

        final_metrics = results["epochs"][-1] if results["epochs"] else {}
        collapse = final_metrics.get("collapse", {})
        results["collapsed"] = collapse.get("total_variance", 1.0) < 1e-4
        results["final_mu"] = collapse.get(
            "mu_estimate_v3", collapse.get("mu_estimate", 0.0)
        )
        results["final_mu_v3"] = collapse.get("mu_estimate_v3", 0.0)
        results["final_sigma_v3"] = collapse.get("sigma_v3", 0.0)
        results["final_gamma_v3"] = collapse.get("gamma_v3", 0.0)
        results["final_variance"] = collapse.get("total_variance", 0.0)
        results["regime"] = collapse_monitor.detect_collapse_regime()
        results["jacobian_mu"] = collapse.get("jacobian_mu", np.nan)
        results["jacobian_worst_case_mu"] = collapse.get(
            "jacobian_worst_case_mu", np.nan
        )
        results["jacobian_lambda1"] = collapse.get("jacobian_lambda1", np.nan)
        results["jacobian_lambda2"] = collapse.get("jacobian_lambda2", np.nan)
        results["jacobian_max_real_eigenvalue"] = collapse.get(
            "jacobian_max_real_eigenvalue", np.nan
        )
        results["jacobian_regime"] = collapse.get("jacobian_regime", "unknown")
        results["jacobian_stable"] = collapse.get("jacobian_stable", np.nan)
        results["jacobian_oscillatory"] = collapse.get("jacobian_oscillatory", np.nan)
        results["jacobian_discriminant"] = collapse.get("jacobian_discriminant", np.nan)

        if last_probe_metrics:
            results["final_probe_val_score"] = last_probe_metrics.get(
                "probe_val_score", None
            )
            results["final_probe_test_score"] = last_probe_metrics.get(
                "probe_test_score", None
            )

        results["best_probe_val_score"] = best_probe_val_score
        results["best_probe_epoch"] = best_probe_epoch

        return results

def experiment_critical_regime_2d(
    exp: StabilityExperiment, args, probe_cadence: int = 0
):
    scales = MASTER_SCALES
    # Context ratios of the Table 1 phase grid.
    ctx_ratios = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results = []
    total_configs = len(scales) * len(ctx_ratios)
    current = 0

    for scale in tqdm(scales, desc="Scales", ncols=100):
        for ctx in tqdm(ctx_ratios, desc="  Context ratios", leave=False, ncols=100):
            current += 1
            tgt = min(0.5, max(0.1, 1.0 - ctx))

            config = {
                "predictor_init_scale": scale,
                "context_ratio": ctx,
                "target_ratio": tgt,
                "data_set": args.data_set,
                "use_residual_predictor": False,
            }

            result = exp.run_single_config(
                config,
                num_epochs=args.num_epochs,
                probe_cadence=probe_cadence,
            )
            result["scale"] = scale
            result["ctx_ratio"] = ctx
            results.append(result)

    save_path = exp.output_dir / f"critical_regime_2d_results_seed_{args.seed}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results

def experiment_single_point(exp: StabilityExperiment, args, probe_cadence: int = 0):
    tgt = min(0.5, max(0.1, 1.0 - args.context_ratio))

    config = {
        "predictor_init_scale": args.predictor_init_scale,
        "context_ratio": args.context_ratio,
        "target_ratio": tgt,
        "use_variance_reg": args.use_variance_reg,
        "variance_reg_weight": args.variance_reg_weight,
        "use_residual_predictor": args.use_residual_predictor,
        "data_set": args.data_set,
    }

    result = exp.run_single_config(
        config,
        num_epochs=args.num_epochs,
        probe_cadence=probe_cadence,
    )

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
        exp.output_dir
        / f"single_point_scale_{args.predictor_init_scale}_ctx_{args.context_ratio}.json"
    )
    with open(save_path, "w") as f:
        json.dump([result], f, indent=2, default=str)

    return [result]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="critical_2d",
        choices=["critical_2d", "single_point"],
    )
    parser.add_argument("--data_set", type=str, default="jannis")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="./experiments/results")
    parser.add_argument("--normalization", type=str, default="minmax")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--encoder_type", type=str, default="linear")
    parser.add_argument("--predictor_type", type=str, default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe_cadence", type=int, default=10)
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--hyperparams", type=str, default=None)
    parser.add_argument("--predictor_init_scale", type=float, default=0.01)
    parser.add_argument("--context_ratio", type=float, default=0.7)
    parser.add_argument("--variance_reg_weight", type=float, default=0.1)
    parser.add_argument("--use_variance_reg", action="store_true", default=False)
    parser.add_argument("--use_residual_predictor", action="store_true", default=False)
    parser.add_argument("--fast", action="store_true")

    exp_args, unknown = parser.parse_known_args()

    base_args = argparse.Namespace(
        data_set=exp_args.data_set,
        data_path=exp_args.data_path,
        batch_size=256,
        val_ratio=0.1,
        test_ratio=0.1,
        normalization="minmax",
        mock=False,
        embed_dim=64,
        encoder_type="linear",
        predictor_type="linear",
        use_residual_predictor=exp_args.use_residual_predictor,
        predictor_init_scale=exp_args.predictor_init_scale,
        encoder_init_scale=1.0,
        encoder_init_type="kaiming",
        regularize_residual=0.0,
        use_variance_reg=exp_args.use_variance_reg,
        variance_reg_weight=exp_args.variance_reg_weight,
        context_ratio=exp_args.context_ratio,
        target_ratio=0.3,
        allow_mask_overlap=True,
        num_epochs=30,
        learning_rate=1e-3,
        weight_decay=1e-5,
        ema_momentum_start=0.996,
        ema_momentum_end=1.0,
        metric_every=10,
        seed=42,
        probe_epochs=50,
        probe_batch_size=256,
        probe_lr=1e-3,
    )

    hyperparams_path = exp_args.hyperparams
    if hyperparams_path is None:
        possible_path = Path("hyperparameters") / f"{exp_args.data_set}.json"
        if possible_path.exists():
            hyperparams_path = str(possible_path)

    hyperparams = {}
    if hyperparams_path:
        with open(hyperparams_path, "r") as f:
            hyperparams = json.load(f)
        for key, value in hyperparams.items():
            setattr(base_args, key, value)

    if exp_args.output_dir == "./experiments/results":
        if exp_args.data_set not in exp_args.output_dir:
            base_args.output_dir = str(Path(exp_args.output_dir) / exp_args.data_set)
            exp_args.output_dir = base_args.output_dir

    cli_overrides = [
        "seed",
        "num_epochs",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "embed_dim",
        "encoder_type",
        "predictor_type",
        "normalization",
        "probe_epochs",
        "probe_batch_size",
        "probe_lr",
        "probe_cadence",
        "predictor_init_scale",
        "context_ratio",
        "use_variance_reg",
        "variance_reg_weight",
        "use_residual_predictor",
    ]

    for key in cli_overrides:
        val = getattr(exp_args, key, None)
        if val is not None:
            setattr(base_args, key, val)

    if exp_args.fast:
        base_args.probe_cadence = 0
        base_args.num_epochs = min(base_args.num_epochs, 5)

    wandb.init(project="drive-vs-decay", config=vars(base_args))

    exp = StabilityExperiment(base_args, output_dir=exp_args.output_dir)

    probe_cadence = exp_args.probe_cadence

    experiments = {
        "critical_2d": lambda: experiment_critical_regime_2d(
            exp, base_args, probe_cadence
        ),
        "single_point": lambda: experiment_single_point(exp, base_args, probe_cadence),
    }

    experiments[exp_args.experiment]()

    wandb.finish()

if __name__ == "__main__":
    main()

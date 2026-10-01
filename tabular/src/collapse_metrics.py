import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from scipy.stats import entropy
import warnings

def compute_effective_rank(embeddings: torch.Tensor) -> float:
    B, N, D = embeddings.shape
    embeddings_flat = embeddings.reshape(B, -1)

    max_samples = min(B, 256)
    if B > max_samples:
        idx = torch.randperm(B)[:max_samples]
        embeddings_flat = embeddings_flat[idx]

    try:
        _, s, _ = torch.linalg.svd(embeddings_flat, full_matrices=False)
        s = s.cpu().numpy()
        s_norm = s / (s.sum() + 1e-10)
        eff_rank = np.exp(entropy(s_norm + 1e-10))
        return float(eff_rank)
    except Exception:
        return float("nan")

class JacobianAnalyzer:
    def __init__(
        self, device: torch.device = None, max_multimode_combinations: int = 10000
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_multimode_combinations = max_multimode_combinations

    @torch.no_grad()
    def estimate_stability_eigenvalues(
        self,
        predictor: nn.Module,
        data_cov: torch.Tensor,
        cross_cov: torch.Tensor,
        beta: float,
        mask_density: float = 0.5,
        ctx_mask_density: float = 0.5,
    ) -> Dict[str, any]:
        results = {}

        W_p = self._get_predictor_matrix(predictor)
        if W_p is None:
            return {"error": "Could not extract predictor matrix"}

        W_p_spectral_norm = self._compute_spectral_norm(W_p)
        W_p_spectral_norm_sq = W_p_spectral_norm**2
        results["predictor_spectral_norm"] = W_p_spectral_norm

        lambda_max_data = self._compute_spectral_radius(data_cov)
        results["lambda_max_data_cov"] = lambda_max_data

        sigma = lambda_max_data * W_p_spectral_norm_sq
        results["sigma_estimate"] = sigma

        lambda_max_cross = self._compute_spectral_radius(cross_cov)
        results["lambda_max_cross_cov"] = lambda_max_cross

        gamma = lambda_max_cross * W_p_spectral_norm * mask_density
        results["gamma_estimate"] = gamma

        mu = gamma / (sigma + 1e-10)
        results["mu"] = mu

        worst_case_mu, mode_mus = self._compute_multimode_mu(
            data_cov, cross_cov, W_p, mask_density, ctx_mask_density
        )
        results["worst_case_mu"] = worst_case_mu
        results["mode_mus"] = mode_mus

        mu_for_stability = worst_case_mu if worst_case_mu is not None else mu
        results["mu_for_stability"] = mu_for_stability

        a = 1
        b = beta + sigma
        c = beta * sigma * (1 - mu_for_stability)

        discriminant = b**2 - 4 * a * c
        results["discriminant"] = discriminant

        if discriminant >= 0:
            lambda1 = (-b + np.sqrt(discriminant)) / (2 * a)
            lambda2 = (-b - np.sqrt(discriminant)) / (2 * a)
            results["lambda1"] = lambda1
            results["lambda2"] = lambda2
            results["max_real_eigenvalue"] = max(lambda1, lambda2)
            results["regime"] = (
                "supercritical" if max(lambda1, lambda2) > 0 else "subcritical"
            )
            results["oscillatory"] = False
        else:
            real_part = -b / (2 * a)
            imag_part = np.sqrt(-discriminant) / (2 * a)
            results["lambda_real"] = real_part
            results["lambda_imag"] = imag_part
            results["max_real_eigenvalue"] = real_part
            results["regime"] = "oscillatory"
            results["oscillatory"] = True

        results["stable"] = results["max_real_eigenvalue"] < 0

        return results

    def _compute_spectral_norm(self, matrix: torch.Tensor) -> float:
        try:
            s = torch.linalg.svdvals(matrix)
            return s[0].item()
        except Exception:
            return matrix.norm().item()

    def _compute_spectral_radius(self, matrix: torch.Tensor) -> float:
        try:
            eigenvalues = torch.linalg.eigvalsh(matrix)
            idx = torch.argmax(eigenvalues.abs())
            return eigenvalues[idx].item()
        except Exception:
            try:
                eigenvalues = torch.linalg.eigvals(matrix)
                eig_real = eigenvalues.real
                idx = torch.argmax(eig_real.abs())
                return eig_real[idx].item()
            except Exception:
                return matrix.norm().item()

    def _compute_multimode_mu(
        self,
        data_cov: torch.Tensor,
        cross_cov: torch.Tensor,
        W_p: torch.Tensor,
        mask_density: float,
        ctx_mask_density: float,
    ) -> Tuple[Optional[float], List[float]]:
        mode_mus = []

        if data_cov is None or cross_cov is None or W_p is None:
            return None, []

        if data_cov.numel() == 0 or cross_cov.numel() == 0 or W_p.numel() == 0:
            return None, []

        try:
            data_cov_cpu = data_cov.detach().cpu()
            cross_cov_cpu = cross_cov.detach().cpu()
            W_p_cpu = W_p.detach().cpu()

            eig_data = torch.linalg.eigvalsh(data_cov_cpu).numpy()
            eig_cross = torch.linalg.eigvalsh(cross_cov_cpu).numpy()
            sv_pred = torch.linalg.svdvals(W_p_cpu).numpy()

            if len(eig_data) == 0 or len(eig_cross) == 0 or len(sv_pred) == 0:
                return None, []

            max_combinations = getattr(self, "max_multimode_combinations", None)
            total_combinations = int(len(eig_data) * len(eig_cross) * len(sv_pred))

            if max_combinations is not None and total_combinations > max_combinations:
                K = int(np.floor(max_combinations ** (1.0 / 3.0)))
                K = max(K, 1)

                data_idx = np.argsort(np.abs(eig_data))[::-1][:K]
                cross_idx = np.argsort(np.abs(eig_cross))[::-1][:K]
                sv_idx = np.argsort(np.abs(sv_pred))[::-1][:K]

                eig_data_use = eig_data[data_idx]
                eig_cross_use = eig_cross[cross_idx]
                sv_use = sv_pred[sv_idx]
            else:
                eig_data_use = eig_data
                eig_cross_use = eig_cross
                sv_use = sv_pred

            for lambda_data_i in eig_data_use:
                for lambda_cross_k in eig_cross_use:
                    for sv_j in sv_use:
                        sigma_ijk = lambda_data_i * (sv_j**2)
                        gamma_ijk = lambda_cross_k * sv_j * mask_density
                        mu_ijk = gamma_ijk / (sigma_ijk + 1e-10)

                        if np.isfinite(mu_ijk):
                            mode_mus.append(float(mu_ijk))

            # eigenvalues solve l^2 + (beta + sigma) l + beta sigma (1 - mu) = 0,

            # so the most unstable mode has the largest mu
            if mode_mus:
                worst_case = max(mode_mus)
            else:
                worst_case = None

            return worst_case, mode_mus

        except torch.linalg.LinAlgError:
            return None, []
        except Exception:
            return None, []

    def _get_predictor_matrix(self, predictor: nn.Module) -> Optional[torch.Tensor]:
        for name, module in predictor.named_modules():
            if isinstance(module, nn.Linear):
                if "proj" in name or "out" in name:
                    return module.weight.data.clone()

        largest_layer = None
        largest_size = 0
        for module in predictor.modules():
            if isinstance(module, nn.Linear):
                size = module.weight.numel()
                if size > largest_size:
                    largest_size = size
                    largest_layer = module.weight.data.clone()

        if largest_layer is not None:
            return largest_layer

        weights = []
        for module in predictor.modules():
            if isinstance(module, nn.Linear):
                weights.append(module.weight.data.reshape(-1))

        if weights:
            total = torch.cat(weights)
            n = int(np.sqrt(len(total)))
            if n * n <= len(total):
                return total[: n * n].reshape(n, n)

        return None

def create_collapse_monitor(
    device: torch.device = None,
    normalization_mode: str = "adaptive",
    baseline_warmup_steps: int = 10,
) -> "CollapseMetrics":
    return CollapseMetrics(device=device)

class CollapseMetrics:
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.history = {
            "effective_rank": [],
            "jacobian_mu": [],
            "jacobian_worst_case_mu": [],
            "jacobian_lambda1": [],
            "jacobian_lambda2": [],
            "jacobian_max_real_eigenvalue": [],
            "jacobian_discriminant": [],
            "jacobian_num_modes": [],
            "jacobian_mu_mean": [],
            "jacobian_mu_std": [],
            "total_variance": [],
            "mu_estimate": [],
            "mu_estimate_v3": [],
        }
        self._jacobian_analyzer = JacobianAnalyzer(device=self.device)

    @torch.no_grad()
    def compute_all_metrics(
        self,
        context_encoder: nn.Module,
        target_encoder: nn.Module,
        predictor: nn.Module,
        embeddings: torch.Tensor,
        target_embeddings: torch.Tensor = None,
        loss: float = None,
        mask_density: float = 0.5,
        batch: torch.Tensor = None,
        context_mask: torch.Tensor = None,
        beta: float = 0.01,
    ) -> Dict[str, float]:
        metrics = {}

        B, N, D = embeddings.shape
        embeddings_flat = embeddings.reshape(B, -1)

        metrics["total_variance"] = embeddings_flat.var().item()

        eff_rank = compute_effective_rank(embeddings)
        metrics["effective_rank"] = eff_rank

        ctx_mask_density = (
            context_mask.mean().item() if context_mask is not None else 0.5
        )
        jacobian_metrics = self._compute_jacobian_metrics(
            predictor=predictor,
            batch=batch,
            context_mask=context_mask,
            beta=beta,
            mask_density=mask_density,
            ctx_mask_density=ctx_mask_density,
        )
        metrics.update(jacobian_metrics)

        metrics["mu_estimate"] = metrics.get("jacobian_worst_case_mu", 0.0)
        metrics["mu_estimate_v3"] = metrics.get("jacobian_worst_case_mu", 0.0)

        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

        return metrics

    def _compute_jacobian_metrics(
        self,
        predictor: nn.Module,
        batch: torch.Tensor = None,
        context_mask: torch.Tensor = None,
        beta: float = 0.01,
        mask_density: float = 0.5,
        ctx_mask_density: float = 0.5,
    ) -> Dict[str, float]:
        metrics = {}

        if batch is None or context_mask is None:
            return metrics

        B = batch.shape[0]

        try:
            masked_context = batch * context_mask
            masked_centered = masked_context - masked_context.mean(dim=0, keepdim=True)
            if B >= 2:
                data_cov = torch.mm(masked_centered.T, masked_centered) / (B - 1)
            else:
                data_cov = torch.eye(batch.shape[1], device=batch.device)

            batch_centered = batch - batch.mean(dim=0, keepdim=True)
            if B >= 2:
                full_cov = torch.mm(batch_centered.T, batch_centered) / (B - 1)
                cross_cov = full_cov * ctx_mask_density
            else:
                cross_cov = torch.zeros_like(data_cov)

            jacobian_results = self._jacobian_analyzer.estimate_stability_eigenvalues(
                predictor=predictor,
                data_cov=data_cov,
                cross_cov=cross_cov,
                beta=beta,
                mask_density=mask_density,
                ctx_mask_density=ctx_mask_density,
            )

            if "error" not in jacobian_results:
                metrics["jacobian_mu"] = jacobian_results.get("mu", float("nan"))
                metrics["jacobian_worst_case_mu"] = jacobian_results.get(
                    "worst_case_mu", float("nan")
                )
                metrics["jacobian_lambda1"] = jacobian_results.get(
                    "lambda1", jacobian_results.get("lambda_real", float("nan"))
                )
                metrics["jacobian_lambda2"] = jacobian_results.get(
                    "lambda2", jacobian_results.get("lambda_imag", float("nan"))
                )
                metrics["jacobian_max_real_eigenvalue"] = jacobian_results.get(
                    "max_real_eigenvalue", float("nan")
                )
                metrics["jacobian_discriminant"] = jacobian_results.get(
                    "discriminant", float("nan")
                )
                metrics["jacobian_stable"] = jacobian_results.get("stable", False)
                metrics["jacobian_regime"] = jacobian_results.get("regime", "unknown")

                mode_mus = jacobian_results.get("mode_mus", [])
                metrics["jacobian_num_modes"] = len(mode_mus)

                if mode_mus:
                    mode_mus_arr = np.array(mode_mus, dtype=np.float64)
                    log_mus = np.log10(np.clip(mode_mus_arr, 1e-12, None))

                    bin_edges = np.linspace(-4.0, 4.0, 81)
                    hist, _ = np.histogram(log_mus, bins=bin_edges, density=True)

                    metrics["jacobian_logmu_hist"] = hist.tolist()
                    metrics["jacobian_logmu_bins"] = bin_edges.tolist()
                    metrics["jacobian_mu_mean"] = float(np.mean(mode_mus))
                    metrics["jacobian_mu_std"] = float(np.std(mode_mus))
                    metrics["jacobian_mu_min"] = float(np.min(mode_mus))
                    metrics["jacobian_mu_max"] = float(np.max(mode_mus))
                    metrics["jacobian_mu_median"] = float(np.median(mode_mus))
                    metrics["jacobian_mu_p25"] = float(np.percentile(mode_mus, 25))
                    metrics["jacobian_mu_p75"] = float(np.percentile(mode_mus, 75))
            else:
                for key in [
                    "jacobian_mu",
                    "jacobian_worst_case_mu",
                    "jacobian_lambda1",
                    "jacobian_lambda2",
                    "jacobian_max_real_eigenvalue",
                    "jacobian_discriminant",
                    "jacobian_mu_mean",
                    "jacobian_mu_std",
                    "jacobian_mu_min",
                    "jacobian_mu_max",
                    "jacobian_mu_median",
                    "jacobian_mu_p25",
                    "jacobian_mu_p75",
                ]:
                    metrics[key] = float("nan")

        except Exception:
            for key in [
                "jacobian_mu",
                "jacobian_worst_case_mu",
                "jacobian_lambda1",
                "jacobian_lambda2",
                "jacobian_max_real_eigenvalue",
                "jacobian_discriminant",
                "jacobian_mu_mean",
                "jacobian_mu_std",
                "jacobian_mu_min",
                "jacobian_mu_max",
                "jacobian_mu_median",
                "jacobian_mu_p25",
                "jacobian_mu_p75",
            ]:
                metrics[key] = float("nan")

        return metrics

    def detect_collapse_regime(self) -> str:
        if len(self.history["jacobian_worst_case_mu"]) >= 1:
            mu_j = self.history["jacobian_worst_case_mu"][-1]
            if not np.isnan(mu_j):
                if len(self.history["jacobian_discriminant"]) >= 1:
                    disc = self.history["jacobian_discriminant"][-1]
                    if not np.isnan(disc) and disc < 0:
                        return "oscillatory"

                if mu_j > 1.2:
                    return "supercritical"
                elif mu_j < 0.8:
                    return "subcritical"
                else:
                    return "critical"

        return "unknown"

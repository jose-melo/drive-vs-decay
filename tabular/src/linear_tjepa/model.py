import copy
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from .encoder import LinearEncoder, LinearEncoderWithPositional
from .predictor import LinearPredictor, MLPPredictor, IdentityPredictor

class LinearTJEPA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        encoder_type: str = "linear",
        predictor_type: str = "linear",
        use_residual_predictor: bool = True,
        predictor_init_scale: float = 0.01,
        encoder_init_scale: float = 1.0,
        encoder_init_type: str = "kaiming",
        regularize_residual: float = 0.0,
        use_variance_reg: bool = False,
        variance_reg_weight: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_variance_reg = use_variance_reg
        self.variance_reg_weight = variance_reg_weight

        if encoder_type == "linear":
            self.context_encoder = LinearEncoder(
                input_dim=input_dim,
                embed_dim=embed_dim,
                init_scale=encoder_init_scale,
                init_type=encoder_init_type,
            )
        elif encoder_type == "linear_positional":
            self.context_encoder = LinearEncoderWithPositional(
                input_dim=input_dim,
                embed_dim=embed_dim,
                init_scale=encoder_init_scale,
            )

        if predictor_type == "linear":
            self.predictor = LinearPredictor(
                embed_dim=embed_dim,
                use_residual=use_residual_predictor,
                init_scale=predictor_init_scale,
                regularize_residual=regularize_residual,
            )
        elif predictor_type == "mlp":
            self.predictor = MLPPredictor(
                embed_dim=embed_dim,
                hidden_dim=embed_dim * 2,
                num_layers=2,
                use_residual=use_residual_predictor,
                init_scale=predictor_init_scale,
            )
        elif predictor_type == "identity":
            self.predictor = IdentityPredictor(embed_dim=embed_dim)

        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        context_emb = self.context_encoder(x, mask=context_mask)
        pred = self.predictor(context_emb)

        with torch.no_grad():
            target = self.target_encoder(x, mask=target_mask)

        return pred, target

    def compute_loss(
        self,
        x: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        pred, target = self.forward(x, context_mask, target_mask)

        mse_loss = nn.functional.mse_loss(pred, target)

        reg_loss = torch.tensor(0.0, device=x.device)
        if hasattr(self.predictor, "get_regularization_loss"):
            reg_loss = self.predictor.get_regularization_loss()

        var_loss = torch.tensor(0.0, device=x.device)
        if self.use_variance_reg:
            embedding_var = pred.var(dim=0).mean()
            var_loss = -self.variance_reg_weight * embedding_var

        total_loss = mse_loss + reg_loss + var_loss

        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "reg_loss": reg_loss,
            "var_loss": var_loss,
        }

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.99):
        for param_q, param_k in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.mul_(momentum).add_((1.0 - momentum) * param_q.data)

    def get_embeddings(
        self,
        x: torch.Tensor,
        use_target: bool = True,
    ) -> torch.Tensor:
        encoder = self.target_encoder if use_target else self.context_encoder
        with torch.no_grad():
            return encoder(x, mask=None)

    def compute_stability_metrics(self) -> Dict[str, float]:
        metrics = {}

        if hasattr(self.predictor, "get_predictor_norm"):
            metrics["predictor_norm"] = self.predictor.get_predictor_norm()
        if hasattr(self.predictor, "get_spectral_norm"):
            metrics["predictor_spectral_norm"] = self.predictor.get_spectral_norm()
        if hasattr(self.predictor, "get_residual_norm"):
            residual_norm = self.predictor.get_residual_norm()
            if residual_norm is not None:
                metrics["residual_norm"] = residual_norm

        if hasattr(self.context_encoder, "get_weight_norm"):
            metrics["encoder_norm"] = self.context_encoder.get_weight_norm()
        if hasattr(self.context_encoder, "get_spectral_norm"):
            metrics["encoder_spectral_norm"] = self.context_encoder.get_spectral_norm()

        if hasattr(self.target_encoder, "get_weight_norm"):
            metrics["target_encoder_norm"] = self.target_encoder.get_weight_norm()

        with torch.no_grad():
            ctx_params = list(self.context_encoder.parameters())
            tgt_params = list(self.target_encoder.parameters())
            divergence = (
                sum(((p1 - p2) ** 2).sum() for p1, p2 in zip(ctx_params, tgt_params))
                .sqrt()
                .item()
            )
            metrics["encoder_divergence"] = divergence

        if "predictor_norm" in metrics:
            pred_norm = metrics["predictor_norm"]
            if pred_norm > 0:
                metrics["estimated_mu_inv"] = pred_norm**2

        return metrics

    def compute_jacobian_eigenvalues(
        self,
        sigma_11: Optional[torch.Tensor] = None,
        sigma_x1: Optional[torch.Tensor] = None,
        beta: float = 0.01,
    ) -> Dict[str, torch.Tensor]:

        device = next(self.parameters()).device

        if sigma_11 is None:
            sigma_11 = torch.eye(self.input_dim, device=device)
        if sigma_x1 is None:
            sigma_x1 = torch.eye(self.input_dim, device=device)

        if hasattr(self.predictor, "get_predictor_matrix"):
            W_p = self.predictor.get_predictor_matrix()
        else:
            W_p = torch.eye(self.embed_dim, device=device)

        WpTWp = W_p.T @ W_p

        sigma_eigenvalues = torch.linalg.eigvalsh(WpTWp)

        eigenvalues = []
        for sigma_i in sigma_eigenvalues:
            a = 1.0
            b = beta + sigma_i.item()
            c = 0.0

            discriminant = b**2 - 4 * a * c
            if discriminant >= 0:
                lambda1 = (-b + math.sqrt(discriminant)) / (2 * a)
                lambda2 = (-b - math.sqrt(discriminant)) / (2 * a)
                eigenvalues.extend([lambda1, lambda2])
            else:
                real_part = -b / (2 * a)
                imag_part = math.sqrt(-discriminant) / (2 * a)
                eigenvalues.append(complex(real_part, imag_part))
                eigenvalues.append(complex(real_part, -imag_part))

        return {
            "sigma_eigenvalues": sigma_eigenvalues,
            "jacobian_eigenvalues": eigenvalues,
            "max_real_eigenvalue": max(
                e.real if isinstance(e, complex) else e for e in eigenvalues
            ),
            "is_stable": all(
                (e.real if isinstance(e, complex) else e) <= 0 for e in eigenvalues
            ),
        }

class LinearTJEPAConfig:
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        encoder_type: str = "linear",
        predictor_type: str = "linear",
        use_residual_predictor: bool = True,
        predictor_init_scale: float = 0.01,
        encoder_init_scale: float = 1.0,
        encoder_init_type: str = "kaiming",
        regularize_residual: float = 0.0,
        use_variance_reg: bool = False,
        variance_reg_weight: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        ema_momentum_start: float = 0.996,
        ema_momentum_end: float = 1.0,
        context_ratio: float = 0.7,
        target_ratio: float = 0.3,
        allow_mask_overlap: bool = True,
    ):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.encoder_type = encoder_type
        self.predictor_type = predictor_type
        self.use_residual_predictor = use_residual_predictor
        self.predictor_init_scale = predictor_init_scale
        self.encoder_init_scale = encoder_init_scale
        self.encoder_init_type = encoder_init_type
        self.regularize_residual = regularize_residual
        self.use_variance_reg = use_variance_reg
        self.variance_reg_weight = variance_reg_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ema_momentum_start = ema_momentum_start
        self.ema_momentum_end = ema_momentum_end
        self.context_ratio = context_ratio
        self.target_ratio = target_ratio
        self.allow_mask_overlap = allow_mask_overlap

    def to_dict(self) -> dict:
        return vars(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LinearTJEPAConfig":
        return cls(**d)

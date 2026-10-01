import math
import torch
import torch.nn as nn
from typing import Optional

class LinearPredictor(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        use_residual: bool = True,
        init_scale: float = 0.01,
        regularize_residual: float = 0.0,
        use_bias: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_residual = use_residual
        self.init_scale = init_scale
        self.regularize_residual = regularize_residual

        if use_residual:
            self.delta_W = nn.Linear(embed_dim, embed_dim, bias=use_bias)
            self._init_residual_weights()
        else:
            self.W_p = nn.Linear(embed_dim, embed_dim, bias=use_bias)
            self._init_standard_weights()

    def _init_residual_weights(self):
        nn.init.normal_(self.delta_W.weight, std=self.init_scale)
        if self.delta_W.bias is not None:
            nn.init.zeros_(self.delta_W.bias)

    def _init_standard_weights(self):
        nn.init.normal_(self.W_p.weight, std=self.init_scale)
        if self.W_p.bias is not None:
            nn.init.zeros_(self.W_p.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return z + self.delta_W(z)
        else:
            return self.W_p(z)

    def get_predictor_matrix(self) -> torch.Tensor:
        if self.use_residual:
            identity = torch.eye(
                self.embed_dim,
                device=self.delta_W.weight.device,
                dtype=self.delta_W.weight.dtype,
            )
            return identity + self.delta_W.weight.data
        else:
            return self.W_p.weight.data

    def get_WpTWp(self) -> torch.Tensor:
        W_p = self.get_predictor_matrix()
        return W_p.T @ W_p

    def get_predictor_norm(self) -> float:
        W_p = self.get_predictor_matrix()
        return torch.norm(W_p, p="fro").item()

    def get_spectral_norm(self) -> float:
        W_p = self.get_predictor_matrix()
        return torch.linalg.svdvals(W_p)[0].item()

    def get_residual_norm(self) -> Optional[float]:
        if self.use_residual:
            return torch.norm(self.delta_W.weight, p="fro").item()
        return None

    def get_regularization_loss(self) -> torch.Tensor:
        if self.use_residual and self.regularize_residual > 0:
            return (
                self.regularize_residual * torch.norm(self.delta_W.weight, p="fro") ** 2
            )
        return torch.tensor(0.0, device=self.get_predictor_matrix().device)

class MLPPredictor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_residual: bool = True,
        init_scale: float = 0.01,
        activation: str = "gelu",
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        layers = []
        dims = [embed_dim] + [hidden_dim] * (num_layers - 1) + [embed_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "relu":
                    layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=init_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.mlp(z)
        if self.use_residual:
            return z + out
        return out

class IdentityPredictor(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def get_predictor_matrix(self) -> torch.Tensor:
        return torch.eye(self.embed_dim)

    def get_predictor_norm(self) -> float:
        return math.sqrt(self.embed_dim)

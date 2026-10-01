import math
import torch
import torch.nn as nn
from typing import Optional

class LinearEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        use_bias: bool = True,
        init_scale: float = 1.0,
        init_type: str = "kaiming",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.init_scale = init_scale
        self.init_type = init_type

        self.encoder = nn.Linear(input_dim, embed_dim, bias=use_bias)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        if self.init_type == "kaiming":
            nn.init.kaiming_normal_(self.encoder.weight, a=math.sqrt(5))
        elif self.init_type == "xavier":
            nn.init.xavier_normal_(self.encoder.weight)
        elif self.init_type == "normal":
            nn.init.normal_(self.encoder.weight, std=0.02)
        elif self.init_type == "orthogonal":
            nn.init.orthogonal_(self.encoder.weight)
        elif self.init_type == "identity_like":
            with torch.no_grad():
                min_dim = min(self.input_dim, self.embed_dim)
                self.encoder.weight.zero_()
                self.encoder.weight[:min_dim, :min_dim] = torch.eye(min_dim)

        with torch.no_grad():
            self.encoder.weight.mul_(self.init_scale)

        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            x = x * mask

        z = self.encoder(x)
        z = self.layer_norm(z)
        return z

    def get_weight_matrix(self) -> torch.Tensor:
        return self.encoder.weight.data

    def get_weight_norm(self) -> float:
        return torch.norm(self.encoder.weight, p="fro").item()

    def get_spectral_norm(self) -> float:
        return torch.linalg.svdvals(self.encoder.weight)[0].item()

class LinearEncoderWithPositional(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        use_positional: bool = True,
        init_scale: float = 1.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_positional = use_positional

        self.feature_proj = nn.Linear(1, embed_dim, bias=True)

        if use_positional:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_dim, embed_dim))
            nn.init.normal_(self.pos_embed, std=0.02)

        self.output_proj = nn.Linear(input_dim * embed_dim, embed_dim)

        self.layer_norm = nn.LayerNorm(embed_dim)

        nn.init.kaiming_normal_(self.feature_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.feature_proj.bias)

        with torch.no_grad():
            self.feature_proj.weight.mul_(init_scale)
            self.output_proj.weight.mul_(init_scale)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        if mask is not None:
            x = x * mask

        x = x.unsqueeze(-1)

        x = self.feature_proj(x)

        if self.use_positional:
            x = x + self.pos_embed

        x = x.view(batch_size, -1)
        x = self.output_proj(x)

        x = self.layer_norm(x)

        return x

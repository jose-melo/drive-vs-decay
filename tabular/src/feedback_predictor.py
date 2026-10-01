import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.encoder import TabularEncoder
from src.utils.train_utils import (
    get_1d_sincos_pos_embed,
    trunc_normal_,
    apply_masks_from_idx,
)

class ResidualMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        p_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        activation: str = "gelu",
        residual_scale: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_identity = input_dim == output_dim

        layers = []
        in_features = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim, eps=layer_norm_eps))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "elu":
                layers.append(nn.ELU())
            layers.append(nn.Dropout(p=p_dropout))
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, output_dim))

        self.residual_branch = nn.Sequential(*layers)

        self.residual_gate = nn.Parameter(torch.tensor(residual_scale))

        self._init_weights(residual_scale)

        if not self.use_identity:
            self.input_proj = nn.Linear(input_dim, output_dim)
            nn.init.eye_(
                self.input_proj.weight[
                    : min(input_dim, output_dim), : min(input_dim, output_dim)
                ]
            )
            nn.init.zeros_(self.input_proj.bias)

    def _init_weights(self, scale: float):
        for module in self.residual_branch.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=scale * 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_identity:
            identity = x
        else:
            identity = self.input_proj(x)

        residual = self.residual_branch(x)

        output = identity + self.residual_gate * residual

        return output

class FeedbackTransformerPredictor(nn.Module):
    def __init__(
        self,
        num_features: int,
        model_hidden_dim: int,
        pred_embed_dim: int,
        num_layers: int,
        num_heads: int,
        p_dropout: float,
        layer_norm_eps: float,
        activation: str,
        init_std: float = 0.02,
        dim_feedforward: int = None,
        identity_attn_bias: float = 10.0,
        use_zero_init: bool = True,
    ):
        super().__init__()

        self.model_hidden_dim = model_hidden_dim
        self.pred_embed_dim = pred_embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.p_dropout = p_dropout
        self.layer_norm_eps = layer_norm_eps
        self.activation = activation
        self.num_features = num_features
        self.dim_feedforward = dim_feedforward or 4 * pred_embed_dim
        self.identity_attn_bias = identity_attn_bias
        self.use_zero_init = use_zero_init
        self.init_std = init_std

        self.predictor_emb = nn.Linear(
            self.model_hidden_dim, self.pred_embed_dim, bias=True
        )

        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_features, self.pred_embed_dim), requires_grad=False
        )
        predictor_pos_embed = get_1d_sincos_pos_embed(
            self.pred_embed_dim,
            np.arange(self.num_features),
        )
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(predictor_pos_embed).float().unsqueeze(0)
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.pred_embed_dim))
        trunc_normal_(self.mask_token, std=self.init_std)

        self.transformer = FeedbackTransformerEncoder(
            d_model=self.pred_embed_dim,
            nhead=self.num_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.p_dropout,
            activation=self.activation,
            identity_attn_bias=self.identity_attn_bias,
            use_zero_init=self.use_zero_init,
        )

        self.predictor_norm = nn.LayerNorm(self.pred_embed_dim)

        self.predictor_proj = nn.Linear(
            self.pred_embed_dim, self.model_hidden_dim, bias=True
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.predictor_emb, self.predictor_norm]:
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=self.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        if self.use_zero_init:
            nn.init.zeros_(self.predictor_proj.weight)
            nn.init.zeros_(self.predictor_proj.bias)

    def forward(self, x, masks_enc, masks_pred):
        B = len(x)

        x = self.predictor_emb(x)

        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x_pos_embed = apply_masks_from_idx(x_pos_embed, masks_enc)
        x += x_pos_embed

        _, N_ctxt, D = x.shape

        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks_from_idx(pos_embs, masks_pred)
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs

        x = x.repeat(len(masks_pred), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        x = self.transformer(x, n_context=N_ctxt)
        x = self.predictor_norm(x)

        x = x[:, N_ctxt:]

        x = self.predictor_proj(x)

        return x

class FeedbackTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        identity_attn_bias: float = 10.0,
        use_zero_init: bool = True,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                FeedbackTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    identity_attn_bias=identity_attn_bias,
                    use_zero_init=use_zero_init,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, n_context: int = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, n_context=n_context)
        return x

class FeedbackTransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        identity_attn_bias: float,
        use_zero_init: bool,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.identity_attn_bias = identity_attn_bias
        self.use_zero_init = use_zero_init

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "elu":
            self.activation = F.elu
        else:
            self.activation = F.relu

        self._init_weights()

    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.linear1]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        if self.use_zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            nn.init.zeros_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)
        else:
            nn.init.xavier_uniform_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            nn.init.xavier_uniform_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor, n_context: int = None) -> torch.Tensor:
        x = x + self._sa_block(self.norm1(x), n_context)

        x = x + self._ff_block(self.norm2(x))

        return x

    def _sa_block(self, x: torch.Tensor, n_context: int = None) -> torch.Tensor:
        B, N, D = x.shape
        head_dim = D // self.nhead

        q = self.q_proj(x).view(B, N, self.nhead, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.nhead, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.nhead, head_dim).transpose(1, 2)

        scale = math.sqrt(head_dim)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / scale

        if self.identity_attn_bias > 0:
            diag_bias = (
                torch.eye(N, device=x.device, dtype=x.dtype) * self.identity_attn_bias
            )
            attn_logits = attn_logits + diag_bias.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout1(attn_weights)

        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        return self.dropout(out)

class FeedbackPredictors(nn.Module):

    def __init__(
        self,
        pred_type: str,
        hidden_dim: int,
        pred_embed_dim: int,
        num_features: int,
        num_layers: int,
        num_heads: int,
        p_dropout: float,
        layer_norm_eps: float,
        activation: str,
        device: torch.device,
        cardinalities: list,
        pred_dim_feedforward: int = None,
        use_residual: bool = True,
        residual_scale: float = 0.0,
        identity_attn_bias: float = 10.0,
        use_zero_init: bool = True,
    ):
        super().__init__()

        self.pred_type = pred_type
        self.hidden_dim = hidden_dim
        self.device = device
        self.num_features = num_features
        self.cardinalities = cardinalities

        if pred_type == "mlp":
            self.predictors = []
            for _ in range(num_features):
                self.predictors.append(
                    ResidualMLP(
                        input_dim=hidden_dim * num_features,
                        output_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        p_dropout=p_dropout,
                        layer_norm_eps=layer_norm_eps,
                        activation=activation,
                        residual_scale=residual_scale,
                    ).to(device)
                )
        else:
            self.predictors = FeedbackTransformerPredictor(
                num_features=num_features,
                model_hidden_dim=hidden_dim,
                pred_embed_dim=pred_embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                p_dropout=p_dropout,
                layer_norm_eps=layer_norm_eps,
                activation=activation,
                dim_feedforward=pred_dim_feedforward,
                identity_attn_bias=identity_attn_bias,
                use_zero_init=use_zero_init,
            ).to(device)

    def forward(self, x, masks_enc, masks_pred):
        if self.pred_type == "mlp":
            return self.forward_mlp(x, masks_pred)
        else:
            return self.predictors(x, masks_enc, masks_pred)

    def forward_mlp(self, x, mask_pred):
        out = []
        for mask in mask_pred:
            out_mask = []
            for col_idx in range(self.num_features):
                out_batch = self.predictors[col_idx](x)
                out_mask.append(out_batch)
            out_mask = torch.stack(out_mask, dim=1)

            columns_to_keep = torch.ones(mask.shape[-1])
            card_idx = 0
            orig_idx = 0
            idx = 0
            while idx < mask.shape[-1]:
                if orig_idx in [card[0] for card in self.cardinalities]:
                    for _ in range(self.cardinalities[card_idx][1] - 1):
                        idx += 1
                        columns_to_keep[idx] = 0
                    card_idx += 1
                idx += 1
                orig_idx += 1
            mask = mask[:, columns_to_keep == 1]
            out_mask = out_mask * mask.unsqueeze(-1)
            out.append(out_mask)
        return out

    def state_dict_(self):
        if self.pred_type == "mlp":
            return [pred.state_dict() for pred in self.predictors]
        else:
            return self.predictors.state_dict()

    def load_state_dict_(self, state_dict):
        if self.pred_type == "mlp":
            for idx, pred in enumerate(self.predictors):
                pred.load_state_dict(state_dict[idx])
        else:
            self.predictors.load_state_dict(state_dict)

    def parameters(self):
        if self.pred_type == "mlp":
            params = []
            for pred in self.predictors:
                params.extend(list(pred.parameters()))
            return iter(params)
        else:
            return self.predictors.parameters()

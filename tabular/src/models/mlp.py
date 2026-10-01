from typing import OrderedDict

from src.datasets.base import BaseDataset
from src.utils.models_utils import (
    TASK_TYPE,
    BaseModel,
    EncodeEmbeddingFeatures,
)
import torch
import torch.nn as nn
import pytorch_lightning as pl


class MLP(BaseModel):

    def __init__(
        self,
        loss: nn.Module = torch.nn.MSELoss(),
        input_dim: int = 128,
        out_dim: int = 1,
        model_dim_hidden: int = 256,
        model_num_layers: int = 2,
        model_dropout_prob: float = 0.1,
        exp_lr: float = 1e-3,
        exp_weight_decay: float = 0.01,
        exp_eta_min: float = 0.0,
        dataset_name: str = None,
        iterations_per_epoch: int = None,
        num_epochs: int = None,
        input_embed_dim: int = 64,
        encoder_type: str = "linear_flatten",
        using_embedding: bool = False,
        **kwargs,
    ) -> None:
        self.encoder_type = encoder_type
        self.emb_dim = model_dim_hidden
        self.using_embedding = using_embedding
        self.input_embed_dim = input_embed_dim
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.model_dim_hidden = model_dim_hidden
        self.model_num_layers = model_num_layers
        self.model_dropout_prob = model_dropout_prob

        super(MLP, self).__init__(
            head_dimension=model_dim_hidden,
            out_dim=out_dim,
            loss=loss,
            lr=exp_lr,
            weight_decay=exp_weight_decay,
            T_max=max(1, (num_epochs or 1) * (iterations_per_epoch or 1)),
            eta_min=exp_eta_min,
            dataset_name=dataset_name,
        )

    def build_encoder(self):

        if self.using_embedding:
            front = EncodeEmbeddingFeatures(
                input_dim=self.input_dim,
                emb_dim=self.model_dim_hidden,
                encoder_type=self.encoder_type,
                input_embed_dim=self.input_embed_dim,
            )
            in_features = self.model_dim_hidden
        else:
            class _Flatten(nn.Module):
                def forward(self, x):
                    return x.view(x.size(0), -1)

            front = _Flatten()
            in_features = self.input_dim

        modules: list = [front]
        prev = in_features
        for _ in range(max(1, self.model_num_layers - 1)):
            modules.append(nn.Linear(prev, self.model_dim_hidden))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(self.model_dropout_prob))
            prev = self.model_dim_hidden
        modules.append(nn.Linear(prev, self.model_dim_hidden))

        return nn.Sequential(*modules)

    @staticmethod
    def get_model_args(
        datamodule: pl.LightningDataModule,
        args: OrderedDict,
        model_args: OrderedDict,
        dataset: BaseDataset = None,
        **kwargs,
    ) -> dict:
        if dataset is None:
            dataset = datamodule.dataset

        if hasattr(dataset, "H"):
            model_args.input_embed_dim = dataset.H
        else:
            model_args.input_embed_dim = None

        extra_cls = args.n_cls_tokens if args.using_embedding else 0
        model_args.input_dim = extra_cls + datamodule.dataset.D

        if not args.using_embedding:
            model_args.input_dim += sum(
                [x[1] - 1 for x in datamodule.dataset.cardinalities]
            )

        if args.task_type == TASK_TYPE.MULTI_CLASS:
            model_args.out_dim = len(set(datamodule.dataset.y))
        else:
            model_args.out_dim = 1

        if args.using_embedding:
            model_args.summary_input = (
                args.batch_size,
                model_args.input_dim,
                model_args.input_embed_dim,
            )
        else:
            model_args.summary_input = (args.batch_size, model_args.input_dim)

        if not hasattr(model_args, "dataset_name"):
            model_args.dataset_name = datamodule.dataset.name

        datamodule.setup("train")
        model_args.iterations_per_epoch = len(datamodule.train_dataloader())
        model_args.num_epochs = args.exp_train_total_epochs
        model_args.using_embedding = args.using_embedding

        return model_args

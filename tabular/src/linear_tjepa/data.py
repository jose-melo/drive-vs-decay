import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from src.datasets.base import BaseDataset
from src.datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP

@dataclass
class DatasetInfo:
    name: str
    input_dim: int
    n_classes: int
    task_type: str
    num_features: List[int]
    cat_features: List[int]
    cardinalities: List[Tuple[int, int]]

class TabularPreprocessor:
    def __init__(
        self,
        normalization: str = "minmax",
        handle_categorical: str = "label",
        handle_missing: str = "mean",
    ):
        self.normalization = normalization
        self.handle_categorical = handle_categorical
        self.handle_missing = handle_missing

        self.num_scalers: Dict[int, Union[StandardScaler, MinMaxScaler]] = {}
        self.cat_encoders: Dict[int, LabelEncoder] = {}
        self.fill_values: Dict[int, float] = {}

        self.is_fitted = False
        self.num_features: List[int] = []
        self.cat_features: List[int] = []
        self.cardinalities: List[Tuple[int, int]] = []
        self.input_dim: int = 0
        self.output_dim: int = 0

    def fit(
        self,
        X: np.ndarray,
        num_features: List[int],
        cat_features: List[int],
        cardinalities: List[Tuple[int, int]],
    ) -> "TabularPreprocessor":
        self.num_features = num_features
        self.cat_features = cat_features
        self.cardinalities = cardinalities
        self.input_dim = X.shape[1]

        X_clean = self._handle_missing_fit(X)

        for idx in num_features:
            col = X_clean[:, idx].reshape(-1, 1)

            if self.normalization == "standard":
                scaler = StandardScaler()
            elif self.normalization == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = None

            if scaler is not None:
                scaler.fit(col)
                self.num_scalers[idx] = scaler

        for idx in cat_features:
            col = X_clean[:, idx]
            encoder = LabelEncoder()
            encoder.fit(col.astype(str))
            self.cat_encoders[idx] = encoder

        if self.handle_categorical == "onehot":
            cat_dim = sum(card[1] for card in cardinalities)
            self.output_dim = len(num_features) + cat_dim
        else:
            self.output_dim = self.input_dim

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        X_out = X.copy().astype(np.float32)

        X_out = self._handle_missing_transform(X_out)

        for idx in self.num_features:
            if idx in self.num_scalers:
                col = X_out[:, idx].reshape(-1, 1)
                X_out[:, idx] = self.num_scalers[idx].transform(col).flatten()

        if self.handle_categorical == "label":
            for idx in self.cat_features:
                if idx in self.cat_encoders:
                    col = X_out[:, idx].astype(str)
                    encoder = self.cat_encoders[idx]
                    known_classes = set(encoder.classes_)
                    col = np.array(
                        [c if c in known_classes else encoder.classes_[0] for c in col]
                    )
                    X_out[:, idx] = encoder.transform(col)

        elif self.handle_categorical == "onehot":
            num_cols = X_out[:, self.num_features]
            cat_cols = []

            for idx, (feat_idx, n_cat) in enumerate(self.cardinalities):
                col = X_out[:, feat_idx].astype(int)
                onehot = np.zeros((len(X_out), n_cat))
                onehot[np.arange(len(X_out)), col.clip(0, n_cat - 1)] = 1
                cat_cols.append(onehot)

            if cat_cols:
                cat_cols = np.hstack(cat_cols)
                X_out = np.hstack([num_cols, cat_cols])
            else:
                X_out = num_cols

        return X_out.astype(np.float32)

    def fit_transform(
        self,
        X: np.ndarray,
        num_features: List[int],
        cat_features: List[int],
        cardinalities: List[Tuple[int, int]],
    ) -> np.ndarray:
        self.fit(X, num_features, cat_features, cardinalities)
        return self.transform(X)

    def _handle_missing_fit(self, X: np.ndarray) -> np.ndarray:
        X_out = X.copy()

        for idx in range(X.shape[1]):
            col = X[:, idx]
            mask = np.isnan(col) | np.isinf(col)

            if mask.any():
                if self.handle_missing == "mean":
                    fill_value = np.nanmean(col[~mask])
                elif self.handle_missing == "median":
                    fill_value = np.nanmedian(col[~mask])
                elif self.handle_missing == "zero":
                    fill_value = 0.0
                else:
                    fill_value = 0.0

                self.fill_values[idx] = fill_value
                X_out[mask, idx] = fill_value

        return X_out

    def _handle_missing_transform(self, X: np.ndarray) -> np.ndarray:
        X_out = X.copy()

        for idx in range(X.shape[1]):
            col = X_out[:, idx]
            mask = np.isnan(col) | np.isinf(col)

            if mask.any():
                fill_value = self.fill_values.get(idx, 0.0)
                X_out[mask, idx] = fill_value

        return X_out

class LinearTJEPADataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_labels: bool = True,
    ):
        self.X = X
        self.y = y
        self.return_labels = return_labels

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.return_labels and self.y is not None:
            return self.X[idx], self.y[idx]
        return (self.X[idx],)

class LinearTJEPADataModule:
    def __init__(
        self,
        dataset_name: str,
        data_path: str = "data",
        batch_size: int = 256,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        normalization: str = "minmax",
        handle_categorical: str = "label",
        seed: int = 42,
        mock: bool = False,
    ):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.normalization = normalization
        self.handle_categorical = handle_categorical
        self.seed = seed
        self.mock = mock

        self.preprocessor = TabularPreprocessor(
            normalization=normalization,
            handle_categorical=handle_categorical,
        )

        self.is_loaded = False
        self.info: Optional[DatasetInfo] = None

        self.X_train: Optional[torch.Tensor] = None
        self.X_val: Optional[torch.Tensor] = None
        self.X_test: Optional[torch.Tensor] = None
        self.y_train: Optional[torch.Tensor] = None
        self.y_val: Optional[torch.Tensor] = None
        self.y_test: Optional[torch.Tensor] = None

    def load(self):
        if self.is_loaded:
            return

        dataset_class = DATASET_NAME_TO_DATASET_MAP[self.dataset_name]

        class DatasetArgs:
            def __init__(self, data_path):
                self.data_path = data_path
                self.mock = False

        args = DatasetArgs(self.data_path)
        dataset = dataset_class(args)
        dataset.load()

        X = dataset.X
        y = dataset.y

        if self.mock:
            X = X[:1000]
            y = y[:1000]

        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.test_ratio,
            random_state=self.seed,
            stratify=y if dataset.task_type != "regression" else None,
        )

        val_ratio_adjusted = self.val_ratio / (1 - self.test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_ratio_adjusted,
            random_state=self.seed,
            stratify=y_temp if dataset.task_type != "regression" else None,
        )

        X_train = self.preprocessor.fit_transform(
            X_train,
            num_features=dataset.num_features,
            cat_features=dataset.cat_features,
            cardinalities=dataset.cardinalities,
        )

        X_val = self.preprocessor.transform(X_val)
        X_test = self.preprocessor.transform(X_test)

        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)

        if dataset.task_type == "regression":
            y_scaler = StandardScaler()
            y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
            self.y_scaler = y_scaler

            self.y_train = torch.tensor(y_train, dtype=torch.float32)
            self.y_val = torch.tensor(y_val, dtype=torch.float32)
            self.y_test = torch.tensor(y_test, dtype=torch.float32)
        else:
            if y_train.dtype == np.object_ or not np.issubdtype(
                y_train.dtype, np.integer
            ):
                label_encoder = LabelEncoder()
                label_encoder.fit(y)
                y_train = label_encoder.transform(y_train)
                y_val = label_encoder.transform(y_val)
                y_test = label_encoder.transform(y_test)
                self.label_encoder = label_encoder
            self.y_train = torch.tensor(y_train, dtype=torch.long)
            self.y_val = torch.tensor(y_val, dtype=torch.long)
            self.y_test = torch.tensor(y_test, dtype=torch.long)

        self.info = DatasetInfo(
            name=self.dataset_name,
            input_dim=self.preprocessor.output_dim,
            n_classes=len(np.unique(y)) if dataset.task_type != "regression" else 1,
            task_type=dataset.task_type,
            num_features=dataset.num_features,
            cat_features=dataset.cat_features,
            cardinalities=dataset.cardinalities,
        )

        self.is_loaded = True

        if self.info.task_type != "regression":

            pass
    def get_train_dataset(self, return_labels: bool = False) -> LinearTJEPADataset:
        if not self.is_loaded:
            self.load()
        return LinearTJEPADataset(
            self.X_train, self.y_train, return_labels=return_labels
        )

    def get_val_dataset(self, return_labels: bool = True) -> LinearTJEPADataset:
        if not self.is_loaded:
            self.load()
        return LinearTJEPADataset(self.X_val, self.y_val, return_labels=return_labels)

    def get_test_dataset(self, return_labels: bool = True) -> LinearTJEPADataset:
        if not self.is_loaded:
            self.load()
        return LinearTJEPADataset(self.X_test, self.y_test, return_labels=return_labels)

    def create_data_loaders(
        self,
        mask_collator,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> Tuple[DataLoader, DataLoader]:
        if not self.is_loaded:
            self.load()

        train_dataset = self.get_train_dataset(return_labels=False)
        val_dataset = self.get_val_dataset(return_labels=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=mask_collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=mask_collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader

def load_all_datasets(
    data_path: str = "data",
    datasets: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, LinearTJEPADataModule]:
    if datasets is None:
        datasets = ["adult", "helena", "jannis", "higgs", "aloi", "california"]

    data_modules = {}
    for name in datasets:
        try:
            dm = LinearTJEPADataModule(
                dataset_name=name,
                data_path=data_path,
                **kwargs,
            )
            dm.load()
            data_modules[name] = dm
        except Exception as e:

            pass
    return data_modules

import torch
import numpy as np
from typing import Tuple, Optional
from multiprocessing import Value

class SimpleBinaryMaskCollator:
    def __init__(
        self,
        num_features: int,
        context_ratio: float = 0.7,
        target_ratio: float = 0.3,
        allow_overlap: bool = True,
        fixed_masks: bool = False,
        seed: Optional[int] = None,
    ):
        self.num_features = num_features
        self.context_ratio = context_ratio
        self.target_ratio = target_ratio
        self.allow_overlap = allow_overlap
        self.fixed_masks = fixed_masks

        self.num_context = max(1, int(num_features * context_ratio))
        self.num_target = max(1, int(num_features * target_ratio))

        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)
        else:
            self._generator.seed()

        self._itr_counter = Value("i", -1)

    def step(self) -> int:
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return self._itr_counter.value

    def _generate_mask(
        self,
        batch_size: int,
        num_ones: int,
        exclude_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        masks = torch.zeros(batch_size, self.num_features)

        for i in range(batch_size):
            available = torch.ones(self.num_features, dtype=torch.bool)
            if exclude_indices is not None:
                available[exclude_indices[i].bool()] = False

            available_idx = torch.where(available)[0]

            if len(available_idx) >= num_ones:
                perm = torch.randperm(len(available_idx), generator=self._generator)[
                    :num_ones
                ]
                selected = available_idx[perm]
            else:
                selected = available_idx

            masks[i, selected] = 1.0

        return masks

    def __call__(
        self,
        batch: list,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        self.step()

        if isinstance(batch[0], tuple):
            data = [b[0] for b in batch]
        else:
            data = batch

        collated_batch = torch.stack(
            [torch.tensor(d) if not isinstance(d, torch.Tensor) else d for d in data]
        )

        batch_size = len(collated_batch)

        if self.fixed_masks:
            context_mask = self._generate_mask(1, self.num_context)
            context_mask = context_mask.repeat(batch_size, 1)

            if self.allow_overlap:
                target_mask = self._generate_mask(1, self.num_target)
            else:
                target_mask = self._generate_mask(1, self.num_target, context_mask[:1])
            target_mask = target_mask.repeat(batch_size, 1)
        else:
            context_mask = self._generate_mask(batch_size, self.num_context)

            if self.allow_overlap:
                target_mask = self._generate_mask(batch_size, self.num_target)
            else:
                target_mask = self._generate_mask(
                    batch_size, self.num_target, context_mask
                )

        return collated_batch, context_mask, target_mask

class RangeBasedMaskCollator:

    def __init__(
        self,
        num_features: int,
        min_context_ratio: float = 0.5,
        max_context_ratio: float = 0.9,
        min_target_ratio: float = 0.1,
        max_target_ratio: float = 0.5,
        allow_overlap: bool = True,
        seed: Optional[int] = None,
    ):
        self.num_features = num_features
        self.min_context_ratio = min_context_ratio
        self.max_context_ratio = max_context_ratio
        self.min_target_ratio = min_target_ratio
        self.max_target_ratio = max_target_ratio
        self.allow_overlap = allow_overlap

        self._torch_generator = torch.Generator()
        if seed is not None:
            self._torch_generator.manual_seed(seed)
            self._np_rng = np.random.default_rng(seed)
        else:
            self._torch_generator.seed()
            self._np_rng = np.random.default_rng()

        self._itr_counter = Value("i", -1)

    def step(self) -> int:
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return self._itr_counter.value

    def _sample_ratio(self, min_r: float, max_r: float) -> float:
        return min_r + self._np_rng.random() * (max_r - min_r)

    def __call__(
        self,
        batch: list,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.step()

        if isinstance(batch[0], tuple):
            data = [b[0] for b in batch]
        else:
            data = batch

        collated_batch = torch.stack(
            [torch.tensor(d) if not isinstance(d, torch.Tensor) else d for d in data]
        )

        batch_size = len(collated_batch)

        context_ratio = self._sample_ratio(
            self.min_context_ratio, self.max_context_ratio
        )
        target_ratio = self._sample_ratio(self.min_target_ratio, self.max_target_ratio)

        num_context = max(1, int(self.num_features * context_ratio))
        num_target = max(1, int(self.num_features * target_ratio))

        if not self.allow_overlap:
            total = num_context + num_target
            if total > self.num_features:
                num_target = self.num_features - num_context

        context_mask = torch.zeros(batch_size, self.num_features)
        target_mask = torch.zeros(batch_size, self.num_features)

        for i in range(batch_size):
            perm = torch.randperm(self.num_features, generator=self._torch_generator)
            context_mask[i, perm[:num_context]] = 1.0

            if self.allow_overlap:
                perm2 = torch.randperm(
                    self.num_features, generator=self._torch_generator
                )
                target_mask[i, perm2[:num_target]] = 1.0
            else:
                remaining = perm[num_context:]
                target_mask[i, remaining[:num_target]] = 1.0

        return collated_batch, context_mask, target_mask

def compute_mask_correlation(
    context_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict:
    overlap = (context_mask * target_mask).sum(dim=1)

    context_size = context_mask.sum(dim=1)
    target_size = target_mask.sum(dim=1)

    union = ((context_mask + target_mask) > 0).float().sum(dim=1)
    jaccard = overlap / (union + 1e-8)

    total_features = context_mask.shape[1]
    coverage = union / total_features

    return {
        "overlap_mean": overlap.mean().item(),
        "overlap_std": overlap.std().item(),
        "context_size_mean": context_size.mean().item(),
        "target_size_mean": target_size.mean().item(),
        "jaccard_mean": jaccard.mean().item(),
        "coverage_mean": coverage.mean().item(),
    }

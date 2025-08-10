import math
import random
from typing import Iterator, List, Literal, Optional, Sequence

from torch.utils.data import Sampler

__all__ = ["BalancedBatchSampler"]


class BalancedBatchSampler(Sampler[List[int]]):
    """Generates batch indices ensuring each batch contains at least a minimum number of positive samples.
    Designed for highly imbalanced binary classification scenarios.

    Parameters
    ----------
    labels : Sequence[int]
        Hard labels (0/1) for each sample. Length = len(dataset).
    batch_size : int
        The batch size.
    min_pos_per_batch : int, default 1
        Minimum number of positive samples per batch.
    target_pos_fraction : float, optional
        Target positive sample ratio (0-1). If None, only min_pos_per_batch is used.
        Actual positive samples per batch = max(min_pos_per_batch, round(batch_size * target_pos_fraction)).
    epoch_size_mode : {"max_of_sets","len_dataset","fixed"}, default "len_dataset"
        - "len_dataset": Total samples per epoch ≈ len(dataset), truncating or padding as needed.
        - "max_of_sets": Uses strategies like max(len(pos), len(neg)) * 2 (less common).
        - "fixed": Uses fixed_num_batches to specify number of batches.
    fixed_num_batches : int, optional
        Only valid when epoch_size_mode="fixed".
    shuffle : bool, default True
    drop_last : bool, default False
    reseed_each_epoch : bool, default True
    seed : int, default 42
        Random seed for reproducibility.

    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        min_pos_per_batch: int = 1,
        target_pos_fraction: Optional[float] = None,
        epoch_size_mode: Literal["max_of_sets", "len_dataset", "fixed"] = "len_dataset",
        fixed_num_batches: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        reseed_each_epoch: bool = True,
        seed: int = 42,
    ):
        self.labels = [int(lb) for lb in labels]
        self.batch_size = batch_size
        self.min_pos_per_batch = min_pos_per_batch
        self.target_pos_fraction = target_pos_fraction
        self.epoch_size_mode = epoch_size_mode
        self.fixed_num_batches = fixed_num_batches
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.reseed_each_epoch = reseed_each_epoch
        self.base_seed = seed

        self.pos_indices = [i for i, l in enumerate(self.labels) if l == 1]
        self.neg_indices = [i for i, l in enumerate(self.labels) if l == 0]

        if len(self.pos_indices) == 0:
            raise ValueError("No positive samples found; cannot use BalancedBatchSampler.")
        if len(self.neg_indices) == 0:
            raise ValueError("No negative samples found; cannot use BalancedBatchSampler.")
        if self.min_pos_per_batch > self.batch_size:
            raise ValueError("min_pos_per_batch cannot exceed batch_size.")

        if self.epoch_size_mode not in ["len_dataset", "max_of_sets", "fixed"]:
            raise ValueError("Unsupported epoch_size_mode.")

        if self.epoch_size_mode == "fixed" and self.fixed_num_batches is None:
            raise ValueError("fixed_num_batches must be set when epoch_size_mode='fixed'.")

        self._epoch = 0

    def __len__(self) -> int:
        if self.epoch_size_mode == "len_dataset":
            if self.drop_last:
                return len(self.labels) // self.batch_size
            else:
                return math.ceil(len(self.labels) / self.batch_size)
        elif self.epoch_size_mode == "fixed":
            return self.fixed_num_batches
        elif self.epoch_size_mode == "max_of_sets":
            total = max(len(self.pos_indices), len(self.neg_indices))
            return math.ceil(total / self.batch_size)
        return 0

    def _maybe_shuffle(self, rng: random.Random):
        if self.shuffle:
            rng.shuffle(self.pos_indices)
            rng.shuffle(self.neg_indices)

    def __iter__(self) -> Iterator[List[int]]:
        if self.reseed_each_epoch:
            rng = random.Random(self.base_seed + self._epoch)
        else:
            rng = random.Random(self.base_seed)

        self._maybe_shuffle(rng)

        pos_ptr = 0
        neg_ptr = 0
        n_pos = len(self.pos_indices)
        n_neg = len(self.neg_indices)

        num_batches = len(self)

        for _ in range(num_batches):
            # Calculate required positive samples for this batch
            if self.target_pos_fraction is not None:
                need_pos = max(
                    self.min_pos_per_batch,
                    int(round(self.batch_size * self.target_pos_fraction)),
                )
                need_pos = min(need_pos, self.batch_size - 1)  # Reserve at least 1 slot for negatives
            else:
                need_pos = self.min_pos_per_batch
            need_pos = min(need_pos, self.batch_size)

            # Handle insufficient positives by recycling
            if pos_ptr + need_pos > n_pos:
                # Not enough remaining - reshuffle
                remain = self.pos_indices[pos_ptr:]
                self._maybe_shuffle(rng)
                pos_ptr = 0
                take = need_pos - len(remain)
                batch_pos = remain + self.pos_indices[pos_ptr : pos_ptr + take]
                pos_ptr += take
            else:
                batch_pos = self.pos_indices[pos_ptr : pos_ptr + need_pos]
                pos_ptr += need_pos

            need_neg = self.batch_size - len(batch_pos)
            if need_neg <= 0:
                batch_indices = batch_pos
            else:
                if neg_ptr + need_neg > n_neg:
                    remain = self.neg_indices[neg_ptr:]
                    self._maybe_shuffle(rng)
                    neg_ptr = 0
                    take = need_neg - len(remain)
                    batch_neg = remain + self.neg_indices[neg_ptr : neg_ptr + take]
                    neg_ptr += take
                else:
                    batch_neg = self.neg_indices[neg_ptr : neg_ptr + need_neg]
                    neg_ptr += need_neg
                batch_indices = batch_pos + batch_neg

            if self.shuffle:
                rng.shuffle(batch_indices)
            print(f"{len(batch_indices) = }")
            yield batch_indices

        self._epoch += 1

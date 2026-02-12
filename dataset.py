""" """

import gzip
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch_ecg._preprocessors import PreprocManager
from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.utils.misc import ReprMixin
from torch_ecg.utils.utils_data import one_hot_encode  # noqa: F401
from torch_ecg.utils.utils_nn import default_collate_fn
from tqdm.auto import tqdm

from cfg import TrainCfg
from const import LABEL_CACHE_DIR, PROJECT_DIR, SampleType
from data_reader import CINC2025

__all__ = [
    "CINC2025Dataset",
]

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


class CINC2025Dataset(Dataset, ReprMixin):
    """Dataset for the CinC2025 Challenge.

    Parameters
    ----------
    config : CFG
        configuration for the dataset
    training : bool, default True
        whether the dataset is for training or validation
    lazy : bool, default True
        whether to load all data into memory at initialization
    reader_kwargs : dict, optional
        keyword arguments for the data reader class.

    """

    __name__ = "CINC2025Dataset"

    def __init__(
        self,
        config: CFG,
        training: bool = True,
        lazy: bool = True,
        **reader_kwargs,
    ) -> None:
        super().__init__()
        self.config = CFG(deepcopy(TrainCfg))
        if config is not None:
            self.config.update(deepcopy(config))
        self.training = training
        self.lazy = lazy

        if self.config.get("db_dir", None) is None:
            self.config.db_dir = reader_kwargs.pop("db_dir", None)
            assert self.config.db_dir is not None, "db_dir must be specified"
        else:
            reader_kwargs.pop("db_dir", None)
        if self.config.get("use_dbs", None) is not None:
            reader_kwargs["use_dbs"] = self.config.use_dbs
        self.config.db_dir = Path(self.config.db_dir).expanduser().resolve()
        # subsampling is performed in the dataset (train-val split is fixed)
        self.config.subsample = reader_kwargs.pop("subsample", self.config.get("subsample", 1))
        assert 0 < self.config.subsample <= 1, "subsample must be in (0, 1]"
        self.config.extra_experiment = reader_kwargs.pop("extra_experiment", self.config.get("extra_experiment", False))

        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        self.__cache = None
        self.reader = CINC2025(db_dir=self.config.db_dir, **reader_kwargs)
        self.records = self._train_test_split()
        # add a column of sample_type to split the samples into 4 categories:
        # 0: negative samples
        # 1: self-reported positive samples
        # 2: self-reported uncertain samples
        # 3: doctor-confirmed positive samples
        self.reader._df_records["sample_type"] = np.full(
            (len(self.reader._df_records),), fill_value=SampleType.NEGATIVE_SAMPLE.value, dtype=int
        )
        self.reader._df_records.loc[
            self.reader._df_records["chagas"] & (self.reader._df_records["source"] == "CODE-15%"), "sample_type"
        ] = SampleType.SELF_REPORTED_POSITIVE_SAMPLE.value
        self.reader._df_records.loc[
            (~self.reader._df_records["chagas"]) & (self.reader._df_records["source"] == "CODE-15%"), "sample_type"
        ] = SampleType.SELF_REPORTED_UNCERTAIN_SAMPLE.value
        self.reader._df_records.loc[
            self.reader._df_records["chagas"] & (self.reader._df_records["source"] == "SaMi-Trop"), "sample_type"
        ] = SampleType.DOCTOR_CONFIRMED_POSITIVE_SAMPLE.value
        # add columns of hard labels and soft labels
        hard_label_dict = {
            SampleType.NEGATIVE_SAMPLE.value: np.array([1, 0]),
            SampleType.SELF_REPORTED_POSITIVE_SAMPLE.value: np.array([0, 1]),
            SampleType.SELF_REPORTED_UNCERTAIN_SAMPLE.value: np.array([1, 0]),
            SampleType.DOCTOR_CONFIRMED_POSITIVE_SAMPLE.value: np.array([0, 1]),
        }
        self.reader._df_records["hard_label"] = self.reader._df_records["sample_type"].map(hard_label_dict)
        soft_label_dict = {
            st: (1 - self.config.label_smooth.smoothing[str(st)]) * hard_label_vec
            + self.config.label_smooth.smoothing[str(st)] / len(self.config.chagas_classes)
            for st, hard_label_vec in hard_label_dict.items()
        }
        # print(f"{soft_label_dict = }")
        self.reader._df_records["soft_label"] = self.reader._df_records["sample_type"].map(soft_label_dict)

        # demographic features
        self.reader._df_records["age"] = self.reader._df_records["age"].map(float) / self.config.age_scale
        self.reader._df_records["sex"] = self.reader._df_records["sex"].map(self.config.sex_mapping)

        ppm_config = CFG(random=False)
        ppm_config.update(deepcopy(self.config))
        self.ppm = PreprocManager.from_config(ppm_config)

        self.fdr = FastDataReader(self.reader, self.records, self.config, self.ppm)

        if not self.lazy:
            self._load_all_data()

    def __len__(self) -> int:
        if self.cache is None:
            return len(self.fdr)
        return len(self.cache["signals"])

    def __getitem__(self, index: Union[int, slice]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        if self.cache is None:
            return self.fdr[index]
        return {k: v[index] for k, v in self.cache.items()}

    def _load_all_data(self, batch_size: int = 256, num_workers: Optional[int] = None) -> None:
        """Load all data into memory using DataLoader for multi-process acceleration.

        Parameters
        ----------
        batch_size : int, default 256
            Number of samples to load in each batch.
        num_workers : int, optional
            Number of worker processes for data loading.
            Set to 0 to disable multiprocessing (useful for debugging).

        .. warning::

            Caching all data into memory is not recommended, which would certainly cause OOM error.
            The RAM of the Challenge is only 64GB.

        """
        if num_workers is None:
            cpu_count = os.cpu_count() or 4
            num_workers = min(max(2, int(cpu_count * 0.8)), 8)
            print(f"Auto-detected num_workers: {num_workers} (CPU count: {cpu_count})")

        dataset_size = len(self)

        temp_loader = DataLoader(
            self.fdr,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=default_collate_fn,
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )

        self.__cache = {
            "record_idx": torch.empty((dataset_size,), dtype=torch.int64),
            "signals": torch.empty((dataset_size, self.config.n_leads, self.config.input_len), dtype=self.config.torch_dtype),
            "chagas": torch.empty((dataset_size, len(self.config.chagas_classes)), dtype=self.config.torch_dtype),
            "sample_type": torch.empty((dataset_size,), dtype=torch.int64),
            "demographics": torch.empty((dataset_size, len(self.config.demographic_features)), dtype=self.config.torch_dtype),
        }

        start_time = time.time()
        current_idx = 0

        for batch_data in tqdm(
            temp_loader,
            desc="Loading data into memory",
            unit="batch",
            mininterval=0.5,
            dynamic_ncols=True,
            total=len(temp_loader),
        ):
            batch_len = len(batch_data["record_idx"])
            end_idx = current_idx + batch_len

            self.__cache["record_idx"][current_idx:end_idx] = batch_data["record_idx"]
            self.__cache["signals"][current_idx:end_idx] = batch_data["signals"]
            self.__cache["chagas"][current_idx:end_idx] = batch_data["chagas"]
            self.__cache["sample_type"][current_idx:end_idx] = batch_data["sample_type"]
            self.__cache["demographics"][current_idx:end_idx] = batch_data["demographics"]

            current_idx = end_idx

        elapsed = time.time() - start_time
        samples_per_sec = dataset_size / elapsed if elapsed > 0 else 0
        print(f"\nLoaded {dataset_size} samples in {elapsed:.2f}s ({samples_per_sec:.1f} samples/s)")

        del temp_loader

    def _train_test_split(self, train_ratio: float = 0.8, part: Optional[Literal["train", "val", "test"]] = None) -> List[str]:
        """Split the dataset into training and validation sets
        in a stratified manner.

        Parameters
        ----------
        train_ratio : float, default 0.8
            The ratio of the training set.
        part : {"train", "val", "test"}, optional
            The part of the dataset to return.
            If None, it will be determined based on the training flag.

        Returns
        -------
        list
            List of record names.

        """
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0, "train_ratio and test_ratio must be positive"

        if self.config.extra_experiment:
            print("Using the extra experiment data split. It is fixed and train_ratio is ignored.")
            with gzip.open(Path(PROJECT_DIR) / "utils" / "code-15-data-split-64-16-20.json.gz", "rt") as f:
                code_15_data_split = json.load(f)
            with gzip.open(Path(PROJECT_DIR) / "utils" / "ptb-xl-data-split-64-16-20.json.gz", "rt") as f:
                ptb_xl_data_split = json.load(f)
            with gzip.open(Path(PROJECT_DIR) / "utils" / "sami-trop-data-split-64-16-20.json.gz", "rt") as f:
                sami_trop_data_split = json.load(f)
            if part is None:
                part = "train" if self.training else "val"
        else:
            if _train_ratio != 80:
                print("Call the _train_test_split method of the 3 database reader classes.")
                return []

            with gzip.open(Path(PROJECT_DIR) / "utils" / "code-15-data-split-82.json.gz", "rt") as f:
                code_15_data_split = json.load(f)
            with gzip.open(Path(PROJECT_DIR) / "utils" / "ptb-xl-data-split-82.json.gz", "rt") as f:
                ptb_xl_data_split = json.load(f)
            with gzip.open(Path(PROJECT_DIR) / "utils" / "sami-trop-data-split-82.json.gz", "rt") as f:
                sami_trop_data_split = json.load(f)

        if 100 in self.reader._df_records.fs.unique():
            suffix = "_lr"
        else:
            suffix = "_hr"
        for k, v in ptb_xl_data_split.items():
            ptb_xl_data_split[k] = [f"{x}{suffix}" for x in v]
            if part is None:
                part = "train" if self.training else "test"

        # do subsampling if needed
        if self.config.subsample < 1.0:
            code_15_data_split[part] = DEFAULTS.RNG.choice(
                code_15_data_split[part], size=int(len(code_15_data_split[part]) * self.config.subsample), replace=False
            ).tolist()
            ptb_xl_data_split[part] = DEFAULTS.RNG.choice(
                ptb_xl_data_split[part], size=int(len(ptb_xl_data_split[part]) * self.config.subsample), replace=False
            ).tolist()
            sami_trop_data_split[part] = DEFAULTS.RNG.choice(
                sami_trop_data_split[part], size=int(len(sami_trop_data_split[part]) * self.config.subsample), replace=False
            ).tolist()
        records = code_15_data_split[part] + ptb_xl_data_split[part] + sami_trop_data_split[part]

        # keep only the records that are in the database (self.reader.all_records)
        # and drop the records that have signal length less than `self.config.min_len`
        # to avoid data processing errors (e.g. bandpass filtering)
        records = list(
            set(records) & set(self.reader._df_records[self.reader._df_records.sig_len >= self.config.min_len].index)
        )

        # if the cached split has no common records with the current database due to unknown reasons,
        # re-split the dataset
        if len(records) == 0:
            df = self.reader._df_records[self.reader._df_records.sig_len >= self.config.min_len].copy()
            split_file = Path(LABEL_CACHE_DIR) / "data-split-82.csv"
            if not split_file.exists():
                df["split"] = np.where(np.random.rand(len(df)) < 0.8, "train", "test")
                df[["split"]].to_csv(split_file)
            else:
                df_split = pd.read_csv(split_file, dtype={"record": str, "split": str})
                df = pd.concat([df, df_split.set_index("record", drop=True)], axis=1)
            records = df[df["split"] == part].index.tolist()

        if self.config.upsample_positive_chagas and self.training:
            # upsample the positive class of the Chagas disease
            records = self._upsample_positive_samples(records)

        if self.training:
            DEFAULTS.RNG.shuffle(records)

        return records

    def _upsample_positive_samples(self, records: List[str]) -> List[str]:
        """Upsample the positive samples in the dataset.

        Parameters
        ----------
        records : list
            List of record names.

        Returns
        -------
        list
            List of record names after upsampling.

        """
        print(f"Upsampling positive samples: {self.config.upsample_positive_chagas}")
        df = self.reader._df_records[self.reader._df_records.index.isin(records)].copy()
        for source, rate in self.config.upsample_positive_chagas.items():
            records += df[(df["source"] == source) & (df["chagas"])].sample(frac=rate - 1, replace=True).index.tolist()
        return records

    def _adjust_upsample_rates(self, ratios: Union[float, Dict[str, float]]) -> None:
        """Adjust the upsample rates of the positive samples in the dataset.

        Parameters
        ----------
        ratios : float or dict
            Ratio of upsampling for the positive samples.
            If float, the same ratio is used for all sources.
            If dict, the keys are the sources and the values are the ratios.

        """
        if isinstance(ratios, (int, float)):
            ratios = {k: ratios for k in self.config.upsample_positive_chagas.keys()}

        self.config.upsample_positive_chagas = {
            k: v * ratios[k] if k in ratios else v for k, v in self.config.upsample_positive_chagas.items()
        }

    def _get_sample_weights(self) -> torch.Tensor:
        """Get the sample weights tensor.

        Returns
        -------
        torch.Tensor
            Sample weights tensor.

        """
        if (not self.training) or (not self.config.upsample_positive_chagas):
            return torch.ones(len(self), dtype=torch.float32)

        df = self.reader._df_records[self.reader._df_records.index.isin(self.records)].copy()
        df = df.loc[self.records]
        df["weight"] = 1.0
        for source, rate in self.config.upsample_positive_chagas.items():
            if source in self.records:
                df.loc[(df["source"] == source) & (df["chagas"]), "weight"] = rate

        return torch.tensor(df["weight"].values, dtype=torch.float32)

    @property
    def cache(self) -> Union[None, Dict[str, torch.Tensor]]:
        return self.__cache

    @property
    def data_fields(self) -> Set[str]:
        # return set(["signals", "chagas", "is_normal", "arr_diag"])
        return set(["record_idx", "signals", "chagas", "sample_type", "demographics"])

    @property
    def hard_chagas_labels(self):
        return (self.reader._df_records.loc[self.records, "chagas"]).astype(int).values

    def extra_repr_keys(self) -> List[str]:
        return ["reader", "training"]

    def reset_resample_fs(self, new_fs: int, reload: bool = False) -> None:
        """Reset the resampling frequency of the preprocessor.

        Parameters
        ----------
        new_fs : int
            New resampling frequency.
        reload : bool, default False
            Whether to reload all data.
            If data is cached and `reload` is False,
            the cached data will be cleared.

        """
        self.config.fs = new_fs
        candidates = [idx for idx, pp in enumerate(self.ppm.preprocessors) if pp.__name__.lower() == "resample"]
        if len(candidates) == 0:
            # no resample preprocessor
            self.ppm._add_resample(fs=new_fs)
        else:
            assert len(candidates) == 1, "Only one resample preprocessor is allowed."
            self.ppm.preprocessors[candidates[0]].fs = new_fs

        if reload:
            self._load_all_data()
        else:
            self.__cache = None

    def reset_input_len(self, new_len: int, reload: bool = False) -> None:
        """Reset the input length of the dataset.

        Parameters
        ----------
        new_len : int
            New input length.
        reload : bool, default False
            Whether to reload all data.
            If data is cached and `reload` is False,
            the cached data will be cleared.

        """
        self.config.input_len = new_len
        if reload:
            self._load_all_data()
        else:
            self.__cache = None


class FastDataReader(ReprMixin, Dataset):
    def __init__(
        self,
        reader: CINC2025,
        records: Sequence[str],
        config: CFG,
        ppm: Optional[PreprocManager] = None,
    ) -> None:
        self.reader = reader
        self.records = records
        self.config = config
        self.ppm = ppm
        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: Union[int, list, slice]) -> Dict[str, np.ndarray]:
        if isinstance(index, slice):
            return default_collate_fn([self[i] for i in range(*index.indices(len(self)))])
        elif isinstance(index, list):
            return default_collate_fn([self[i] for i in index])
        rec = self.records[index]
        signal, sig_fs = self.reader.load_data(
            rec,
            data_format="channel_first",
            # fs=self.config.fs,
            return_fs=True,
        )  # (n_leads, n_samples)
        if self.ppm is not None:
            signal, _ = self.ppm(signal, sig_fs)
        # ensure the length of the signal equals to the expected length `self.config.input_len`
        pad_len = self.config.input_len - signal.shape[1]
        if pad_len > 0:
            pad_shift = DEFAULTS.RNG.integers(0, pad_len + 1)
            signal = np.pad(signal, ((0, 0), (pad_shift, pad_len - pad_shift)), mode="constant")
        elif pad_len < 0:
            pad_shift = DEFAULTS.RNG.integers(0, -pad_len + 1)
            signal = signal[:, pad_shift : pad_shift + self.config.input_len]
        # chagas_label = self.reader.load_ann(rec)
        sample_type = self.reader._df_records.at[rec, "sample_type"]

        if self.config.label_smooth:
            if DEFAULTS.RNG.random() <= self.config.label_smooth.prob:
                chagas_label = self.reader._df_records.at[rec, "soft_label"]
            else:
                chagas_label = self.reader._df_records.at[rec, "hard_label"]
        else:
            chagas_label = self.reader.load_ann(rec)

        demographics = self.reader._df_records.loc[rec, self.config.demographic_features].values.astype(self.dtype)

        # if `index` is a slice, the output shapes are:
        # signal: (batch_size, n_leads, n_samples)
        # chagas_label: (batch_size,)
        # bin_label: (batch_size,)
        # arr_diag_label: (batch_size, n_classes)
        return {  # type: ignore
            "record_idx": index,
            "signals": signal.astype(self.dtype),  # (n_leads, n_samples)
            "chagas": chagas_label,  # scalar or (n_classes,)
            # "is_normal": bin_label,  # scalar
            # "arr_diag": arr_diag_label,  # (n_classes,)
            "sample_type": sample_type,  # scalar
            "demographics": demographics,  # (n_demographic_features,)
        }

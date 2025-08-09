"""
"""

import gzip
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd
import torch
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

        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        self.__cache = None
        self.reader = CINC2025(db_dir=self.config.db_dir, **reader_kwargs)
        self.records = self._train_test_split()
        # add a column of sample_type to split the samples into 3 categories:
        # 0: negative samples
        # 1: self-reported positive samples
        # 2: doctor-confirmed positive samples
        self.reader._df_records["sample_type"] = np.full(
            (len(self.reader._df_records),), fill_value=SampleType.NEGATIVE_SAMPLE.value, dtype=int
        )
        self.reader._df_records.loc[
            self.reader._df_records["chagas"] & (self.reader._df_records["source"] == "CODE-15%"), "sample_type"
        ] = SampleType.SELF_REPORTED_POSITIVE_SAMPLE.value
        self.reader._df_records.loc[
            self.reader._df_records["chagas"] & (self.reader._df_records["source"] == "SaMi-Trop"), "sample_type"
        ] = SampleType.DOCTOR_CONFIRMED_POSITIVE_SAMPLE.value
        # add columns of hard labels and soft labels
        soft_label_dict = {
            SampleType.NEGATIVE_SAMPLE.value: np.array([1, 0]),
            SampleType.SELF_REPORTED_POSITIVE_SAMPLE.value: np.array([0, 1]),
            SampleType.DOCTOR_CONFIRMED_POSITIVE_SAMPLE.value: np.array([0, 1]),
        }
        self.reader._df_records["hard_label"] = self.reader._df_records["sample_type"].map(soft_label_dict)
        soft_label_dict = {
            st: (1 - self.config.label_smooth.smoothing[str(st)]) * prob
            + self.config.label_smooth.smoothing[str(st)] / len(self.config.chagas_classes)
            for st, prob in soft_label_dict.items()
        }
        self.reader._df_records["soft_label"] = self.reader._df_records["sample_type"].map(soft_label_dict)

        ppm_config = CFG(random=False)
        ppm_config.update(deepcopy(self.config))
        self.ppm = PreprocManager.from_config(ppm_config)

        self.fdr = FastDataReader(self.reader, self.records, self.config, self.ppm)

        if not self.lazy:
            self._load_all_data()

    def __len__(self) -> int:
        if self.cache is None:
            return len(self.fdr)
        return len(self.cache["signal"])

    def __getitem__(self, index: Union[int, slice]) -> Dict[str, np.ndarray]:
        if self.cache is None:
            return self.fdr[index]
        return {k: v[index] for k, v in self.cache.items()}

    def _load_all_data(self) -> None:
        """Load all data into memory.

        .. warning::

            caching all data into memory is not recommended, which would certainly cause OOM error.
            The RAM of the Challenge is only 64GB.

        """
        self.__cache = {
            "signals": np.empty((len(self), self.config.n_leads, self.config.input_len), dtype=self.dtype),
            "chagas": np.empty((len(self),), dtype=np.int64),
            # "is_normal": np.empty((len(self),), dtype=np.int64),
            # "arr_diag": np.empty((len(self), len(self.config.arr_diag_class_map)), dtype=self.dtype),
        }
        for idx in tqdm(range(len(self)), desc="loading data", unit="record", mininterval=1, dynamic_ncols=True):
            data = self.fdr[idx]
            for k, v in data.items():
                self.__cache[k][idx] = v

    def _train_test_split(self, train_ratio: float = 0.8) -> List[str]:
        """Split the dataset into training and validation sets
        in a stratified manner.

        Parameters
        ----------
        train_ratio : float, default 0.8
            The ratio of the training set.

        Returns
        -------
        list
            List of record names.

        """
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0, "train_ratio and test_ratio must be positive"

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

        part = "train" if self.training else "test"
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
    def cache(self) -> Dict[str, np.ndarray]:
        return self.__cache

    @property
    def data_fields(self) -> Set[str]:
        # return set(["signals", "chagas", "is_normal", "arr_diag"])
        return set(["record_idx", "signals", "chagas"])

    def extra_repr_keys(self) -> List[str]:
        return ["reader", "training"]


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

    def __getitem__(self, index: Union[int, slice]) -> Dict[str, np.ndarray]:
        if isinstance(index, slice):
            return default_collate_fn([self[i] for i in range(*index.indices(len(self)))])
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
            if DEFAULTS.RNG.random() > self.config.label_smooth.prob:
                chagas_label = self.reader._df_records.at[rec, "soft_label"]
            else:
                chagas_label = self.reader._df_records.at[rec, "hard_label"]
        else:
            chagas_label = self.reader.load_ann(rec)

        # chagas_label = self.reader.load_chagas_ann(rec)  # categorical: 0 or 1
        # bin_label = self.reader.load_binary_ann(rec)  # categorical: 0 or 1
        # arr_diag_label = self.reader.load_ann(rec, class_map=self.config.arr_diag_class_map, augmented=True)
        # arr_diag_label = one_hot_encode([arr_diag_label], len(self.config.arr_diag_class_map))[0]  # (n_classes,)

        # if `index` is a slice, the output shapes are:
        # signal: (batch_size, n_leads, n_samples)
        # chagas_label: (batch_size,)
        # bin_label: (batch_size,)
        # arr_diag_label: (batch_size, n_classes)
        return {
            "record_idx": index,
            "signals": signal.astype(self.dtype),  # (n_leads, n_samples)
            "chagas": chagas_label,  # scalar or (n_classes,)
            # "is_normal": bin_label,  # scalar
            # "arr_diag": arr_diag_label,  # (n_classes,)
            "sample_type": sample_type,  # scalar
        }

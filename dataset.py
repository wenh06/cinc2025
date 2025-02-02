"""
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Union

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch_ecg._preprocessors import PreprocManager
from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.utils.misc import ReprMixin
from torch_ecg.utils.utils_data import one_hot_encode, stratified_train_test_split  # noqa: F401
from torch_ecg.utils.utils_nn import default_collate_fn
from tqdm.auto import tqdm  # noqa: F401

from cfg import BaseCfg, TrainCfg
from data_reader import CODE15

__all__ = [
    "CinC2025Dataset",
]


class CinC2025Dataset(Dataset, ReprMixin):
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

    __name__ = "CinC2025Dataset"

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
        self.config.db_dir = Path(self.config.db_dir).expanduser().resolve()

        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        self.__cache = None
        self.reader = CODE15(db_dir=self.config.db_dir, **reader_kwargs)
        self.subjects = self._train_test_split()
        self.records = self.reader._df_records[self.reader._df_records["patient_id"].isin(self.subjects)].index.tolist()

        ppm_config = CFG(random=False)
        ppm_config.update(deepcopy(self.config))
        self.ppm = PreprocManager.from_config(ppm_config)

        self.fdr = FastDataReader(self.reader, self.records, self.config, self.ppm)

        if not self.lazy:
            self._load_all_data()

    def __len__(self) -> int:
        if self.cache is None:
            return len(self.fdr)
        return len(self.cache["images"])

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
        raise NotImplementedError

    def _train_test_split(self, train_ratio: float = 0.8, force_recompute: bool = False) -> List[str]:
        """Split the dataset into training and validation sets
        in a stratified manner.

        Parameters
        ----------
        train_ratio : float, default 0.8
            The ratio of the training set.
        force_recompute : bool, default False
            Whether to recompute the split.

        Returns
        -------
        list
            List of record names.

        """
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0, "train_ratio and test_ratio must be positive"

        writable = True
        if os.access(self.reader.db_dir, os.W_OK):
            train_file = self.reader.db_dir / f"train_ratio_{_train_ratio}.json"
            test_file = self.reader.db_dir / f"test_ratio_{_test_ratio}.json"
        elif os.access(self.reader.working_dir, os.W_OK):
            train_file = self.reader.working_dir / f"train_ratio_{_train_ratio}.json"
            test_file = self.reader.working_dir / f"test_ratio_{_test_ratio}.json"
        else:
            train_file = None
            test_file = None
            writable = False

        # aux files are only used for recording the split, not for actual training
        (BaseCfg.project_dir / "utils").mkdir(exist_ok=True)
        aux_train_file = BaseCfg.project_dir / "utils" / f"train_ratio_{_train_ratio}.json"
        aux_test_file = BaseCfg.project_dir / "utils" / f"test_ratio_{_test_ratio}.json"

        if not force_recompute:
            if writable and train_file.exists() and test_file.exists():
                if self.training:
                    return json.loads(train_file.read_text())
                else:
                    return json.loads(test_file.read_text())
            elif aux_train_file.exists() and aux_test_file.exists():
                train_set = json.loads(aux_train_file.read_text())
                test_set = json.loads(aux_test_file.read_text())
                if writable:
                    train_file.write_text(json.dumps(train_set, ensure_ascii=False))
                    test_file.write_text(json.dumps(test_set, ensure_ascii=False))
                if self.training:
                    return train_set
                else:
                    return test_set

        df_subjects = self.reader._df_records[["age", "sex", "patient_id"]].copy()
        df_subjects["chagas"] = self.reader._df_chagas["chagas"].values
        # group by patient_id, and set `chagas` to `True` if any record of the patient is chagas
        df_subjects = df_subjects.groupby("patient_id").agg(
            {
                "age": "first",
                "sex": "first",
                "chagas": "max",
            }
        )
        # make `age` categorical
        df_subjects["age"] = df_subjects["age"].apply(lambda x: f"{int(x // 10)}0s")
        df_train, df_test = stratified_train_test_split(
            df_subjects,
            ["age", "sex", "chagas"],
            test_ratio=1 - train_ratio,
            reset_index=False,
        )
        train_set = df_train.index.tolist()
        test_set = df_test.index.tolist()
        print(f"train set: {len(df_train)}, with positive rate: {df_train['chagas'].mean()}")
        print(f"test set: {len(df_test)}, with positive rate: {df_test['chagas'].mean()}")

        if (writable and force_recompute) or not train_file.exists() or not test_file.exists():
            train_file.write_text(json.dumps(train_set, ensure_ascii=False))
            test_file.write_text(json.dumps(test_set, ensure_ascii=False))

        if force_recompute or not aux_train_file.exists() or not aux_test_file.exists():
            aux_train_file.write_text(json.dumps(train_set, ensure_ascii=False))
            aux_test_file.write_text(json.dumps(test_set, ensure_ascii=False))

        DEFAULTS.RNG.shuffle(train_set)
        DEFAULTS.RNG.shuffle(test_set)

        if self.training:
            return train_set
        else:
            return test_set

    @property
    def cache(self) -> Dict[str, np.ndarray]:
        return self.__cache

    @property
    def data_fields(self) -> Set[str]:
        # fmt: off
        return set(["signal", "chagas_label", "bin_label", "diag_label"])
        # fmt: on

    def extra_repr_keys(self) -> List[str]:
        return ["reader", "training"]


class FastDataReader(ReprMixin, Dataset):
    def __init__(
        self,
        reader: CODE15,
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
        signal = self.reader.load_data(
            rec,
            data_format="channel_first",
            return_fs=False,
        )  # (n_leads, n_samples)
        if self.ppm is not None:
            signal = self.ppm(signal)
        chagas_label = self.reader.load_chagas_ann(rec)
        bin_label = self.reader.load_bin_ann(rec)
        diag_label = self.reader.load_ann(rec, class_map=self.config.diag_class_map)
        return {
            "signal": signal.astype(self.dtype),
            "chagas_label": chagas_label,
            "bin_label": bin_label,
            "diag_label": diag_label,
        }

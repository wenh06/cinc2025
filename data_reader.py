"""
"""

import multiprocessing as mp
import os
import re
import shutil
import urllib.parse
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import wfdb
from torch_ecg.cfg import CFG
from torch_ecg.databases.base import DEFAULT_FIG_SIZE_PER_SEC, DataBaseInfo, _DataBase, wfdb_get_version
from torch_ecg.databases.physionet_databases import PTBXL as PTBXL_Reader
from torch_ecg.utils.download import http_get, url_is_reachable
from torch_ecg.utils.misc import add_docstring, str2bool, timeout
from torch_ecg.utils.utils_data import stratified_train_test_split
from tqdm.auto import tqdm

from cfg import BaseCfg
from helper_code import is_integer
from prepare_code15_data import convert_dat_to_mat as code15_convert_dat_to_mat
from prepare_code15_data import fix_checksums as code15_fix_checksums
from prepare_ptbxl_data import convert_dat_to_mat as ptbxl_convert_dat_to_mat
from prepare_ptbxl_data import fix_checksums as ptbxl_fix_checksums
from prepare_samitrop_data import convert_dat_to_mat as samitrop_convert_dat_to_mat
from prepare_samitrop_data import fix_checksums as samitrop_fix_checksums
from utils.misc import trim_zeros

__all__ = [
    "CODE15",
    "SamiTrop",
]


if np.__version__ >= "2.2":
    trim_zeros_func = np.trim_zeros
else:
    trim_zeros_func = trim_zeros


_CODE15_INFO = DataBaseInfo(
    title="CODE-15%: a large scale annotated dataset of 12-lead ECGs",
    about="""
    1. The database contains 345,779 exams from 233,770 patients, obtained through stratified sampling from the CODE dataset ( 15% of the patients). It can be downloaded from Zenodo [1]_. The paper describing the dataset is available in Nature Communications [2]_. The dataset is also used in the 2025 Moody Challenge [4]_.
    2. The "exams.csv" file contains the labels and demographic information of the patients with the following columns:

        - "exam_id": id used for identifying the exam;
        - "age": patient age in years at the moment of the exam;
        - "is_male": true if the patient is male;
        - "nn_predicted_age": age predicted by a neural network to the patient. As described in [3]_;
        - "1dAVb": Whether or not the patient has 1st degree AV block;
        - "RBBB": Whether or not the patient has right bundle branch block;
        - "LBBB": Whether or not the patient has left bundle branch block;
        - "SB": Whether or not the patient has sinus bradycardia;
        - "AF": Whether or not the patient has atrial fibrillation;
        - "ST": Whether or not the patient has sinus tachycardia;
        - "patient_id": id used for identifying the patient;
        - "normal_ecg": True if automatic annotation system say it is a normal ECG;
        - "death": true if the patient dies in the follow-up time. This data is available only in the first exam of the patient. Other exams will have this as an empty field;
        - "timey": if the patient dies it is the time to the death of the patient. If not, it is the follow-up time. This data is available only in the first exam of the patient. Other exams will have this as an empty field;
        - "trace_file": identify in which hdf5 file the file corresponding to this patient is located. This data is available only in the first exam of the patient. Other exams will have this as an empty field;
    3. The signal files are of the format "exams_part{i}.hdf5", containing two fields named `tracings` and `exam_id`. The `exam_id` is a tensor of dimension `(N,)` containing the exam id (the same as in the csv file) and the field `tracings` is a `(N, 4096, 12)` tensor containing the ECG tracings in the same order.
    4. The signals are sampled at 400 Hz. Some signals originally have a duration of 10 seconds (10 * 400 = 4000 samples) and others of 7 seconds (7 * 400 = 2800 samples). The latter were zero-padded (centered) to 10 seconds. (Actually, the length of the signals is 4096 samples).
    5. The binary Chagas labels are self-reported and therefore may or may not have been validated.
    6. The ratio of positive samples is 1.795%.
    """,
    usage=[
        "ECG arrhythmia detection",
        "Self-Supervised Learning",
    ],
    note="""
    """,
    issues="""
    1. A small part of the database has signals with all zeros.
    """,
    references=[
        "https://zenodo.org/records/4916206",
        "https://github.com/antonior92/automatic-ecg-diagnosis",
        "https://github.com/antonior92/ecg-age-prediction",
        "https://moody-challenge.physionet.org/2025/",
    ],
    doi=[
        "10.5281/zenodo.4916206",
        "10.1038/s41467-020-15432-4",
        "10.1038/s41467-021-25351-7",
    ],
)


@add_docstring(_CODE15_INFO.format_database_docstring(), mode="prepend")
class CODE15(_DataBase):
    """
    Parameters
    ----------
    db_dir : `path-like`, optional
        Storage path of the database.
        If not specified, data will be fetched from Physionet.
    working_dir : `path-like`, optional
        Working directory, to store intermediate files and log files.
    verbose : int, default 1
        Level of logging verbosity.
    kwargs : dict, optional
        Auxilliary key word arguments.

    """

    __name__ = "CODE15"
    __dl_base_url__ = "https://zenodo.org/records/4916206/files/"
    __data_files__ = {f"exams_part{i}": f"exams_part{i}.hdf5" for i in range(18)}
    __label_file__ = "exams.csv"
    __chagas_label_file__ = "code15_chagas_labels.csv"
    __chagas_label_file_url__ = "https://moody-challenge.physionet.org/2025/data/code15_chagas_labels.zip"
    __default_wfdb_data_dir__ = "wfdb_format_files"
    __label_cols__ = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
    __normal_ecg_name__ = "NORM"
    __abnormal_ecg_name__ = "OTHER"

    def __init__(
        self,
        db_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        working_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            db_name="CODE-15%",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.wfdb_data_dir = Path(kwargs.pop("wfdb_data_dir", self.__default_wfdb_data_dir__))
        self.wfdb_data_ext = kwargs.pop("wfdb_data_ext", "dat")
        self.__config = CFG(BaseCfg.copy())
        self.__config.update(kwargs)

        self.data_ext = "hdf5"
        self.ann_ext = self.__label_file__
        self.chagas_ann_ext = self.__chagas_label_file__
        self.fs = 400
        self.all_leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        self._h5_data_files = []
        self._df_records = pd.DataFrame()
        self._df_chagas = pd.DataFrame()
        self._all_records = []
        self._all_subjects = []
        self._subject_records = {}
        self._is_converted_to_wfdb_format = False
        self._label_file = self.__config.get("label_file", None)
        self._chagas_label_file = self.__config.get("chagas_label_file", None)
        self._ls_rec()

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in a dataframe.
        """
        # find all hdf5 files
        self._h5_data_files = list(self.db_dir.rglob("exams_part*.hdf5"))
        if len(self._h5_data_files) > 0:
            assert len(set([f.parent for f in self._h5_data_files])) == 1, "All hdf5 files should be in the same directory."
            self.db_dir = self._h5_data_files[0].parent

        if not self.wfdb_data_dir.is_absolute():
            self.wfdb_data_dir = self.db_dir / self.wfdb_data_dir
        self.wfdb_data_dir.mkdir(parents=True, exist_ok=True)

        # find all records in the wfdb data directory
        df_wfdb_records = pd.DataFrame(
            {
                "wfdb_signal_file": list(self.wfdb_data_dir.rglob(f"*.{self.wfdb_data_ext}")),
                "exam_id": None,
            }
        )
        df_wfdb_records.wfdb_signal_file = df_wfdb_records.wfdb_signal_file.apply(lambda x: x.with_suffix(""))
        # note that the ".mat" files are named {exam_id}m.mat in function `convert_dat_to_mat`
        df_wfdb_records.exam_id = df_wfdb_records.wfdb_signal_file.apply(lambda x: int(re.sub("\\D", "", x.stem)))

        if self._label_file is None:
            self._label_file = self.db_dir / self.__label_file__
            if not self._label_file.exists():
                # self.download(["labels"], refresh=False)
                self._label_file = None
        else:
            self._label_file = Path(self._label_file).expanduser().resolve()
        if self._chagas_label_file is None:
            self._chagas_label_file = self.db_dir / self.__chagas_label_file__
            if not self._chagas_label_file.exists():
                # self.download(["chagas-labels"], refresh=False)
                self._chagas_label_file = None
        else:
            self._chagas_label_file = Path(self._chagas_label_file).expanduser().resolve()

        early_exit = False

        if len(self._h5_data_files) == 0 and df_wfdb_records.empty:
            self.logger.warning("No data files found in the database directory. Call `download()` to download the database.")
            early_exit = True

        # else: some data files are found, proceed to load the metadata

        # assert (
        #     self._label_file is not None and self._label_file.exists()
        # ), f"Label file {self.__label_file__} not found in the given directory."
        # assert (
        #     self._chagas_label_file is not None and self._chagas_label_file.exists()
        # ), f"Chagas label file {self.__chagas_label_file__} not found in the given directory."
        if self._label_file is None or not self._label_file.exists():
            self.logger.warning(f"Label file {self.__label_file__} not found in the given directory.")
            self._label_file = None
            early_exit = True
        if self._chagas_label_file is None or not self._chagas_label_file.exists():
            self.logger.warning(f"Chagas label file {self.__chagas_label_file__} not found in the given directory.")
            self._chagas_label_file = None
            early_exit = True

        if early_exit:
            self._df_records = pd.DataFrame()
            self._df_chagas = pd.DataFrame()
            self._all_records = []
            self._all_subjects = []
            self._subject_records = {}
            return

        self._df_records = pd.read_csv(self._label_file)
        self._df_records["sex"] = self._df_records["is_male"].map({True: "Male", False: "Female"})
        self._df_chagas = pd.read_csv(self._chagas_label_file)
        self._all_records = list(
            set(self._df_records.exam_id.unique().tolist()).intersection(self._df_chagas.exam_id.unique().tolist())
        )
        self._df_records = self._df_records[self._df_records.exam_id.isin(self._all_records)]
        self._df_chagas = self._df_chagas[self._df_chagas.exam_id.isin(self._all_records)]
        df_wfdb_records = df_wfdb_records[df_wfdb_records.exam_id.isin(self._all_records)]

        if df_wfdb_records.empty:
            self._is_converted_to_wfdb_format = False

            # perhaps only a part of the dataset is downloaded
            # so we need to filter out the records that are not downloaded
            dl_rec_list = []
            for h5_file in self._h5_data_files:
                with h5py.File(h5_file, "r") as h5f:
                    dl_rec_list.extend(h5f["exam_id"][:].tolist())
            self._df_records = self._df_records[self._df_records.exam_id.isin(dl_rec_list)]
            self._df_chagas = self._df_chagas[self._df_chagas.exam_id.isin(dl_rec_list)]
            del dl_rec_list

            self._df_records["record"] = self._df_records["exam_id"].astype(str)
            self._subject_records = self._df_records.groupby("patient_id")["record"].apply(sorted).to_dict()
            self._df_records.set_index("record", inplace=True)
            self._all_records = self._df_records.index.tolist()
            self._all_subjects = self._df_records.patient_id.unique().tolist()
            self._df_chagas["record"] = self._df_chagas["exam_id"].astype(str)
            self._df_chagas.set_index("record", inplace=True)

            return

        self._is_converted_to_wfdb_format = True
        self._df_records = pd.merge(self._df_records, df_wfdb_records, on="exam_id", how="inner")
        self._df_records["record"] = self._df_records["exam_id"].astype(str)
        self._subject_records = self._df_records.groupby("patient_id")["record"].apply(sorted).to_dict()
        self._df_records.set_index("record", inplace=True)
        self._all_records = self._df_records.index.tolist()
        self._all_subjects = self._df_records.patient_id.unique().tolist()
        self._df_chagas = self._df_chagas[self._df_chagas.exam_id.isin(self._df_records.exam_id)]
        self._df_chagas["record"] = self._df_chagas["exam_id"].astype(str)
        self._df_chagas.set_index("record", inplace=True)

    def get_absolute_path(self, rec: Union[str, int], ext: Literal["hdf5", "dat", "mat"] = "dat") -> Path:
        """Get the absolute path of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
            NOTE: DO NOT confuse index (int) and record name (exam_id, str).
        ext : {"hdf5", "dat", "mat"}, default "dat"
            Extension of the file.

        Returns
        -------
        path : pathlib.Path
            Absolute path of the record.

        """
        if isinstance(rec, int):
            rec = self[rec]
        row = self._df_records.loc[rec]
        if ext == "hdf5":
            path = self.db_dir / row["trace_file"]
        else:
            path = row["wfdb_signal_file"].with_suffix(f".{ext}")
        if not path.exists():
            self.logger.warning(f"File {path} does not exist.")
        return path

    def load_data(
        self,
        rec: Union[str, int],
        data_format: str = "channel_first",
        units: Union[str, type(None)] = "mV",
        fs: Optional[Real] = None,
        return_fs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Real]]:
        """Load physical (converted from digital) ECG data,
        or load digital signal directly.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
            NOTE: DO NOT confuse index (int) and record name (exam_id, str).
        data_format : str, default "channel_first"
            Format of the ECG data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first"), or
            "flat" (alias "plain").
        units : str or None, default "mV"
            Units of the output signal, can also be "μV" (aliases "uV", "muV");
            None for digital data, without digital-to-physical conversion.
        fs : numbers.Real, optional
            Sampling frequency of the output signal.
            If not None, the loaded data will be resampled to this frequency,
            otherwise, the original sampling frequency will be used.
        return_fs : bool, default False
            Whether to return the sampling frequency of the output signal.

        Returns
        -------
        data : numpy.ndarray
            The loaded ECG data.
        data_fs : numbers.Real, optional
            Sampling frequency of the output signal.
            Returned if `return_fs` is True.

        .. note::
            Since the duration of the signals are short (<= 10 seconds),
            parameters `sampfrom` and `sampto` are not provided.

        """
        if isinstance(rec, int):
            rec = self[rec]
        if not self._is_converted_to_wfdb_format:
            # load data from hdf5 file
            h5_file = self.db_dir / self._df_records.loc[rec, "trace_file"]
            with h5py.File(h5_file, "r") as h5f:
                data = h5f["tracings"][h5f["exam_id"][:] == rec][0]  # shape (n_samples, n_leads)
        else:
            # load data from wfdb files
            record_path = self._df_records.loc[rec, "wfdb_signal_file"]
            data = wfdb.rdsamp(record_path)[0]  # shape (n_samples, n_leads)
        data = data.astype(np.float32)  # typically in most deep learning tasks, we use float32
        if units.lower() in ["uv", "μv", "muv"]:
            data = data * 1e3
        if fs is not None and fs != self.fs:
            data = wfdb.processing.resample_sig(data, self.fs, fs)
        else:
            fs = self.fs
        if data_format.lower() in ["channel_first", "lead_first"]:
            data = data.T
        if return_fs:
            return data, fs
        return data

    def load_ann(
        self, rec: Union[str, int], class_map: Optional[Dict[str, int]] = None, augmented: bool = False
    ) -> Union[List[str], List[int]]:
        """Load the arrhythmia annotations of the record.

        The arrhythmia annotations are:

            - 1dAVb: 1st degree AV block
            - RBBB: right bundle branch block
            - LBBB: left bundle branch block
            - SB: sinus bradycardia
            - AF: atrial fibrillation
            - ST: sinus tachycardia

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
            NOTE: DO NOT confuse index (int) and record name (exam_id, str).
        class_map : dict, optional
            Mapping of the arrhythmia classes to integers.
            If not provided, the conversion will not be performed.
        augmented : bool, default False
            Whether to augment the annotations with the binary label,
            i.e., adding two more classes "Normal" and "Other".

        Returns
        -------
        ann : list of str or int
            List of the arrhythmia annotations or
            their corresponding integer labels w.r.t. `class_map`.

        """
        if isinstance(rec, int):
            rec = self[rec]
        ann = self._df_records.loc[rec, self.__label_cols__].to_dict()
        ann = [k for k, v in ann.items() if v]
        if augmented and len(ann) == 0:
            if self.load_binary_ann(rec):
                ann.append(self.__normal_ecg_name__)
            else:
                ann.append(self.__abnormal_ecg_name__)
        if class_map is not None:
            ann = [class_map[an] for an in ann]
        return ann

    def load_binary_ann(self, rec: Union[str, int]) -> int:
        """Load the binary annotations of the record.

        This corresponds to the "normal_ecg" column in the label file.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
            NOTE: DO NOT confuse index (int) and record name (`exam_id`, str).

        Returns
        -------
        bin_ann : int
            Binary annotation of the record.
            0 for abnormal ECG, 1 for normal ECG.

        """
        if isinstance(rec, int):
            rec = self[rec]
        bin_ann = int(self._df_records.loc[rec, "normal_ecg"])
        return bin_ann

    def load_chagas_ann(self, rec: Union[str, int]) -> int:
        """Load the Chagas label of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
            NOTE: DO NOT confuse index (int) and record name (exam_id, str).

        Returns
        -------
        chagas_ann : int
            Chagas label of the record.
            0 for negative, 1 for positive.

        """
        if isinstance(rec, int):
            rec = self[rec]
        chagas_ann = int(self._df_chagas.loc[rec, "chagas"])
        return chagas_ann

    def load_demographics(self, rec: Union[str, int]) -> Dict[str, Any]:
        """Load the demographic information of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
            NOTE: DO NOT confuse index (int) and record name (exam_id, str).

        Returns
        -------
        demographics : dict
            Demographic information of the record,
            including "age", "sex".

        """
        if isinstance(rec, int):
            rec = self[rec]
        demographics = self._df_records.loc[rec, ["age", "sex"]].to_dict()
        return demographics

    def plot(
        self,
        rec: Union[str, int],
        data: Optional[np.ndarray] = None,
        ticks_granularity: int = 0,
        leads: Optional[Union[str, Sequence[str]]] = None,
        same_range: bool = False,
        **kwargs: Any,
    ) -> None:
        """Plot the signals of a record or external signals (units in μV),
        along with the annotations.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        data : numpy.ndarray, optional
            (12-lead) ECG signal to plot,
            should be of the format "channel_first",
            and compatible with `leads`.
            If is not None, data of `rec` will not be used.
            This is useful when plotting filtered data.
        ticks_granularity : int, default 0
            Granularity to plot axis ticks, the higher the more ticks.
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        leads : str or List[str], optional
            The leads of the ECG signal to plot.
        same_range : bool, default False
            If True, all leads are forced to have the same y range.
        kwargs : dict, optional
            Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot`.

        """
        if isinstance(rec, int):
            rec = self[rec]

        if "plt" not in dir():
            import matplotlib.pyplot as plt

            plt.MultipleLocator.MAXTICKS = 3000

        _leads = self._normalize_leads(leads, numeric=False)
        lead_indices = [self.all_leads.index(ld) for ld in _leads]

        if data is None:
            _data = self.load_data(rec, data_format="channel_first", units="μV")[lead_indices]
        else:
            units = self._auto_infer_units(data)
            self.logger.info(f"input data is auto detected to have units in {units}")
            if units.lower() == "mv":
                _data = 1000 * data
            else:
                _data = data
            assert _data.shape[0] == len(
                _leads
            ), f"number of leads from data of shape ({_data.shape[0]}) does not match the length ({len(_leads)}) of `leads`"

        if same_range:
            y_ranges = np.ones((_data.shape[0],)) * np.max(np.abs(_data)) + 100
        else:
            y_ranges = np.max(np.abs(_data), axis=1) + 100

        dem_row = self._df_records.loc[rec]
        chagas_ann = "Chagas - " + ("True" if self.load_chagas_ann(rec) else "False")
        arr_diag_ann = "Diagnosis - " + ("Normal" if self.load_binary_ann(rec) else "Abnormal")
        if dem_row[self.__label_cols__].any():
            arr_diag_ann += " - " + ", ".join(self.load_ann(rec))

        plot_alpha = 0.4
        nb_leads = len(_leads)

        t = np.arange(_data.shape[1]) / self.fs
        duration = len(t) / self.fs
        fig_sz_w = int(round(DEFAULT_FIG_SIZE_PER_SEC * duration))
        fig_sz_h = 6 * np.maximum(y_ranges, 750) / 1500
        fig, axes = plt.subplots(nb_leads, 1, sharex=False, figsize=(fig_sz_w, np.sum(fig_sz_h)))
        if nb_leads == 1:
            axes = [axes]
        for idx in range(nb_leads):
            axes[idx].plot(
                t,
                _data[idx],
                color="black",
                linewidth="2.0",
                label=f"lead - {_leads[idx]}",
            )
            axes[idx].axhline(y=0, linestyle="-", linewidth="1.0", color="red")
            # NOTE that `Locator` has default `MAXTICKS` equal to 1000
            if ticks_granularity >= 1:
                axes[idx].xaxis.set_major_locator(plt.MultipleLocator(0.2))
                axes[idx].yaxis.set_major_locator(plt.MultipleLocator(500))
                axes[idx].grid(which="major", linestyle="-", linewidth="0.4", color="red")
            if ticks_granularity >= 2:
                axes[idx].xaxis.set_minor_locator(plt.MultipleLocator(0.04))
                axes[idx].yaxis.set_minor_locator(plt.MultipleLocator(100))
                axes[idx].grid(which="minor", linestyle=":", linewidth="0.2", color="gray")
            # add extra info. to legend
            # https://stackoverflow.com/questions/16826711/is-it-possible-to-add-a-string-as-a-legend-item-in-matplotlib
            axes[idx].plot(
                [],
                [],
                " ",
                label=f"Exam ID - {rec}; Patient ID - {dem_row.patient_id}; Age - {dem_row.age}; Sex - {dem_row.sex}",
            )
            axes[idx].plot([], [], " ", label=f"{chagas_ann}; {arr_diag_ann}")
            axes[idx].legend(loc="upper left", fontsize=14)
            axes[idx].set_xlim(t[0], t[-1])
            axes[idx].set_ylim(min(-600, -y_ranges[idx]), max(600, y_ranges[idx]))
            axes[idx].set_xlabel("Time [s]", fontsize=16)
            axes[idx].set_ylabel("Voltage [μV]", fontsize=16)
        plt.subplots_adjust(hspace=0.05)
        fig.tight_layout()
        if kwargs.get("save_path", None):
            plt.savefig(kwargs["save_path"], dpi=200, bbox_inches="tight")
        else:
            plt.show()

    @property
    def all_subjects(self) -> List[str]:
        """List of all subject IDs."""
        return self._all_subjects

    @property
    def subject_records(self) -> Dict[str, List[str]]:
        """Dict of subject IDs and their corresponding records."""
        return self._subject_records

    @property
    def url(self) -> Dict[str, str]:
        files = {f"exams-part{i}": f"{self.__dl_base_url__}exams_part{i}.zip?download=1" for i in range(18)}
        files.update(
            {
                "labels": f"{self.__dl_base_url__}{self.__label_file__}?download=1",
                "chagas-labels": self.__chagas_label_file_url__,
            }
        )
        return files

    def download(self, files: Optional[Union[str, Sequence[str]]] = None, refresh: bool = True) -> None:
        """Download the database files.

        Parameters
        ----------
        files : str or list of str, optional
            The files to download.
            If not specified, download all files.
            The available files are:

                - "exams-part{i}" for i in range(18)
                - "labels"
                - "chagas-labels"
        refresh : bool, default True
            Whether to call `self._ls_rec()` after downloading the files.

        """
        if files is None:
            files = list(self.url.keys())
        elif isinstance(files, str):
            files = [files]

        for file in files:
            if file not in self.url:
                self.logger.warning(f"File {file} is not in the list of available files.")
                continue
            # skip downloading if the file already exists
            filename = Path(urllib.parse.urlparse(self.url[file]).path)
            if file.startswith("exams_part"):
                filename = filename.with_suffix(".hdf5").name
            else:
                filename = filename.with_suffix(".csv").name
            if (self.db_dir / filename).exists():
                self.logger.info(f"File {filename} already exists in the database directory.")
                continue
            http_get(self.url[file], self.db_dir)

        if refresh:
            self._ls_rec()

    def download_subset(self) -> None:
        """Download a subset of the database files."""
        self.download(["exams-part17", "labels", "chagas-labels"])

    @property
    def database_info(self) -> DataBaseInfo:
        return _CODE15_INFO

    def _train_test_split(self, train_ratio: float = 0.8) -> Dict[str, List[str]]:
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
        data_split : dict
            Dictionary containing the training and test (validation) sets
            of the record names.

        """
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0, "train_ratio and test_ratio must be positive"

        df_subjects = self._df_records[["age", "sex", "death", "patient_id"]].copy()
        df_subjects["chagas"] = self._df_chagas["chagas"]
        # group by patient_id, and set `chagas` to `True` if any record of the patient is chagas
        df_subjects = df_subjects.groupby("patient_id").agg(
            {
                "age": "first",
                "sex": "first",
                "death": "first",
                "chagas": "max",
            }
        )
        # make `age` categorical
        df_subjects["age"] = df_subjects["age"].apply(lambda x: f"{int(x // 10)}0s")
        df_train, df_test = stratified_train_test_split(
            df_subjects,
            ["age", "sex", "death", "chagas"],
            test_ratio=1 - train_ratio,
            reset_index=False,
        )
        data_split = {
            "train": self._df_records[self._df_records["patient_id"].isin(df_train.index)].index.tolist(),
            "test": self._df_records[self._df_records["patient_id"].isin(df_test.index)].index.tolist(),
        }
        return data_split

    def _convert_to_wfdb_format(
        self,
        signal_format: Literal["dat", "mat"] = "dat",
        trim_zeros: bool = True,
        overwrite: bool = False,
    ) -> List[Tuple[str, int]]:
        """Convert the CODE-15% dataset to WFDB format.

        Parameters
        ----------
        signal_format : {"dat", "mat"}, default "dat"
            The format of the signal files.
        trim_zeros : bool, default True
            Whether to trim the zero padding at the start and end of the signals.

            .. note::
                Signals corresponding some of the exam IDs have values all zeros,
                trimming the zeros will result in empty signals.
        overwrite : bool, default False
            Whether to overwrite the existing files.

        Returns
        -------
        excep_list : list of tuple
            List of tuples containing the h5 file name and the exam ID that failed to convert.
            This list is kept and returned for further inspection and debugging.

        """
        if len(self._h5_data_files) == 0:
            self.logger.warning("No hdf5 files found in the database directory. Call `download()` to download the database.")
            return

        excep_list = CODE15.convert_to_wfdb_format(
            signal_files=self._h5_data_files,
            df_demographics=self._df_records,
            df_chagas=self._df_chagas,
            output_path=self.wfdb_data_dir,
            signal_format=signal_format,
            trim_zeros=trim_zeros,
            overwrite=overwrite,
        )
        self._ls_rec()

        return excep_list

    @staticmethod
    def convert_to_wfdb_format(
        signal_files: Sequence[Union[str, bytes, os.PathLike]],
        df_demographics: pd.DataFrame,
        df_chagas: pd.DataFrame,
        output_path: Union[str, bytes, os.PathLike],
        signal_format: Literal["dat", "mat"] = "dat",
        trim_zeros: bool = True,
        overwrite: bool = False,
    ) -> List[Tuple[str, int]]:
        """Convert the CODE-15% dataset to WFDB format.

        Modified from the original script in `prepare_code15_data.py`,
        which is simplifed and enhanced with progress bar and error handling.

        TODO: use multi-processing for faster conversion.

        Parameters
        ----------
        signal_files : list of `path-like`
            List of paths to the signal files.
        df_demographics : pd.DataFrame
            DataFrame containing the demographic information.
        df_chagas : pd.DataFrame
            DataFrame containing the Chagas labels.
        output_path : `path-like`
            Output path to store the converted files.
        signal_format : {"dat", "mat"}, default "dat"
            The format of the signal files.
        trim_zeros : bool, default True
            Whether to trim the zero padding at the start and end of the signals.

            .. note::
                Signals corresponding some of the exam IDs have values all zeros,
                trimming the zeros will result in empty signals.
        overwrite : bool, default False
            Whether to overwrite the existing files.

        Returns
        -------
        excep_list : list of tuple
            List of tuples containing the h5 file name and the exam ID that failed to convert.
            This list is kept and returned for further inspection and debugging.

        """
        assert signal_format in ["dat", "mat"], f"Unsupported signal format: {signal_format}"
        df_demographics = df_demographics.copy()
        if "sex" not in df_demographics.columns:
            df_demographics["sex"] = df_demographics["is_male"].map({True: "Male", False: "Female"})
        exam_id_to_demographics = (
            df_demographics[["exam_id", "patient_id", "age", "sex"]].set_index("exam_id").to_dict(orient="index")
        )

        exam_id_to_chagas = df_chagas[["exam_id", "chagas"]].set_index("exam_id").to_dict()["chagas"]

        # Load and convert the signal data.
        # See https://zenodo.org/records/4916206 for more information about these values.
        lead_names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        sampling_frequency = 400
        units = "mV"

        # Define the paramters for the WFDB files.
        gain = 1000
        baseline = 0
        num_bits = 16
        fmt = str(num_bits)

        output_path = Path(output_path).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        # Iterate over the input signal files.
        excep_list = []
        with tqdm(signal_files, total=len(signal_files), desc="Converting signals", dynamic_ncols=True, mininterval=1) as pbar:
            for signal_file in pbar:
                signal_file = Path(signal_file).expanduser().resolve()
                pbar.set_postfix_str(f"Converting signals in {signal_file.stem}")
                signal_file = str(signal_file)
                with h5py.File(signal_file, "r") as h5_sig_file:
                    exam_ids = h5_sig_file["exam_id"][...]
                    num_exam_ids = len(exam_ids)

                    # Iterate over the exam IDs in each signal file.
                    for idx in tqdm(
                        range(num_exam_ids),
                        total=num_exam_ids,
                        desc=f"Converting signals in {Path(signal_file).stem}",
                        dynamic_ncols=True,
                        mininterval=1,
                        leave=False,
                    ):
                        exam_id = exam_ids[idx].item()

                        # Skip exam IDs without Chagas labels.
                        if exam_id not in exam_id_to_chagas:
                            continue

                        physical_signals = np.array(h5_sig_file["tracings"][idx], dtype=np.float32)

                        # Perform basic error checking on the signal.
                        num_samples, num_leads = np.shape(physical_signals)
                        assert num_leads == 12

                        if trim_zeros:
                            # Remove zero padding at the start and end of the signals.
                            physical_signals = trim_zeros_func(physical_signals, trim="fb", axis=0)
                            if physical_signals.shape[0] == 0:
                                excep_list.append((signal_file, exam_id))
                                continue

                        # Convert the signal to digital units;
                        # saturate the signal and represent NaNs as the lowest representable integer.
                        digital_signals = gain * physical_signals
                        digital_signals = np.round(digital_signals)
                        digital_signals = np.clip(digital_signals, -(2 ** (num_bits - 1)) + 1, 2 ** (num_bits - 1) - 1)
                        digital_signals[~np.isfinite(digital_signals)] = -(2 ** (num_bits - 1))
                        # We need to promote from 16-bit integers due to an error in the Python WFDB library.
                        digital_signals = np.asarray(digital_signals, dtype=np.int32)

                        # Add the exam ID, the patient ID, age, sex, the Chagas label, and data source.
                        patient_id = exam_id_to_demographics[exam_id]["patient_id"]
                        age = exam_id_to_demographics[exam_id]["age"]
                        sex = exam_id_to_demographics[exam_id]["sex"]
                        chagas = exam_id_to_chagas[exam_id]
                        source = "CODE-15%"
                        comments = [
                            # f"Exam ID: {exam_id}",
                            # f"Patient ID: {patient_id}",
                            f"Age: {age}",
                            f"Sex: {sex}",
                            f"Chagas label: {chagas}",
                            f"Source: {source}",
                        ]

                        # Save the signal.
                        record = str(exam_id)
                        if not overwrite and (output_path / record).with_suffix(f".{signal_format}").exists():
                            continue
                        wfdb.wrsamp(
                            record,
                            fs=sampling_frequency,
                            units=[units] * num_leads,
                            sig_name=lead_names,
                            d_signal=digital_signals,
                            fmt=[fmt] * num_leads,
                            adc_gain=[gain] * num_leads,
                            baseline=[baseline] * num_leads,
                            write_dir=str(output_path),
                            comments=comments,
                        )

                        if signal_format in ("mat", ".mat"):
                            code15_convert_dat_to_mat(record, write_dir=str(output_path))

                        # Recompute the checksums for the checksum due to an error in the Python WFDB library.
                        checksums = np.sum(digital_signals, axis=0, dtype=np.int16)
                        code15_fix_checksums(str(output_path / record), checksums)

        print("Conversion of the CODE-15% database is complete.")
        print(f"{num_exam_ids - len(excep_list)} signals converted successfully.")
        print(f"{len(excep_list)} signals failed to convert.")

        return excep_list

    @staticmethod
    def load_chagas_label_from_header(record_name: Union[str, bytes, os.PathLike]) -> int:
        """Load the Chagas label from the header file of the record.

        Parameters
        ----------
        record_name : str or `path-like`
            Record name or path to the record.

        Returns
        -------
        chagas_label : int
            Chagas label of the record.
            0 for negative, 1 for positive.

        """
        header_file = Path(record_name).expanduser().resolve().with_suffix(".hea")
        with open(header_file, "r") as f:
            for line in f:
                if "Chagas label" in line:
                    chagas_label = int(str2bool(line.split(":")[-1].strip()))
                    break
            else:
                raise ValueError(f"Chagas label not found in the header file {header_file}")
        return chagas_label


_SamiTrop_INFO = DataBaseInfo(
    title="Sami-Trop: 12-lead ECG traces with age and mortality annotations",
    about="""
    1. The whole Sami-Trop database consists of 12-lead ECG traces from 1959 patients **ALL** with chronic Chagas cardiomyopathy.
    2. The open-access database is a subset (1631 patients) of the Sami-Trop database, which contains annotations of age and mortality in addition to the 12-lead ECG traces. It can be downloaded from Zenodo [1]_, and also used in the 2025 Moody Challenge [3]_.
    3. The "exams.csv" file contains the following annotation columns:

        - "exam_id": the unique identifier of the exam
        - "age": patient age in years at the moment the of the exam
        - "is_male": boolean indicating whether the patient is male or not (female)
        - "normal_ecg": boolean indicating whether the ECG is normal or not
        - "death": boolean indicating whether the patient died in the follow-up time
        - "timey": time to death in years (if the patient died) or time to last follow-up (if the patient is alive)
        - "nn_predicted_age": age predicted by a neural network to the patient. As described in [1]_;
    4. The signal file "exams.hdf5" contains only one field named `tracings`, which is a `(N, 4096, 12)` tensor containing the ECG tracings (perhaps in the same order as in the "exams.csv" file).
    5. The signals are sampled at 400 Hz. Some signals originally have a duration of 10 seconds (10 * 400 = 4000 samples) and others of 7 seconds (7 * 400 = 2800 samples). The latter were zero-padded (centered) to 10 seconds. (Actually, the length of the signals is 4096 samples).
    """,
    usage=[
        "Electrocardiographic age prediction",
        "Mortality prediction",
    ],
    note="""
    """,
    issues="""
    """,
    references=[
        "https://zenodo.org/records/4905618",
        "https://github.com/antonior92/ecg-age-prediction",
        "https://moody-challenge.physionet.org/2025/",
    ],
    doi=[
        "10.5281/zenodo.4905618",
        "10.1038/s41467-021-25351-7",
    ],
)


@add_docstring(_SamiTrop_INFO.format_database_docstring(), mode="prepend")
class SamiTrop(_DataBase):
    """
    Parameters
    ----------
    db_dir : `path-like`, optional
        Storage path of the database.
        If not specified, data will be fetched from Physionet.
    working_dir : `path-like`, optional
        Working directory, to store intermediate files and log files.
    verbose : int, default 1
        Level of logging verbosity.
    kwargs : dict, optional
        Auxilliary key word arguments.

    """

    __name__ = "SamiTrop"
    __dl_base_url__ = "https://zenodo.org/record/4905618/files/"
    __data_file__ = "exams.zip"
    __label_file__ = "exams.csv"
    __chagas_label_file__ = "samitrop_chagas_labels.csv"
    __chagas_label_file_url__ = "https://moody-challenge.physionet.org/2025/data/samitrop_chagas_labels.zip"
    __default_wfdb_data_dir__ = "wfdb_format_files"
    __label_cols__ = ["age", "normal_ecg", "death", "timey"]

    def __init__(
        self,
        db_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        working_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            db_name="Sami-Trop",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.wfdb_data_dir = Path(kwargs.pop("wfdb_data_dir", self.__default_wfdb_data_dir__))
        self.wfdb_data_ext = kwargs.pop("wfdb_data_ext", "dat")
        self.__config = CFG(BaseCfg.copy())
        self.__config.update(kwargs)

        self.data_ext = "hdf5"
        self.ann_ext = self.__label_file__
        self.chagas_ann_ext = self.__chagas_label_file__
        self.fs = 400
        self.all_leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        self._h5_data_files = []
        self._df_records = pd.DataFrame()
        self._df_chagas = pd.DataFrame()
        self._all_records = []
        self._all_subjects = []
        self._is_converted_to_wfdb_format = False
        self._label_file = self.__config.get("label_file", None)
        self._chagas_label_file = self.__config.get("chagas_label_file", None)
        self._ls_rec()

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in a dataframe.
        """
        # find all hdf5 files
        self._h5_data_files = list(self.db_dir.rglob(str(Path(self.__data_file__).with_suffix(".hdf5"))))
        if len(self._h5_data_files) > 0:
            assert len(self._h5_data_files) == 1, "Multiple data files found in the database directory."
            self.db_dir = self._h5_data_files[0].parent

        if not self.wfdb_data_dir.is_absolute():
            self.wfdb_data_dir = self.db_dir / self.wfdb_data_dir
        self.wfdb_data_dir.mkdir(parents=True, exist_ok=True)

        # find all records in the wfdb data directory
        df_wfdb_records = pd.DataFrame(
            {
                "wfdb_signal_file": list(self.wfdb_data_dir.rglob(f"*.{self.wfdb_data_ext}")),
                "exam_id": None,
            }
        )
        df_wfdb_records.wfdb_signal_file = df_wfdb_records.wfdb_signal_file.apply(lambda x: x.with_suffix(""))
        # note that the ".mat" files are named {exam_id}m.mat in function `convert_dat_to_mat`
        df_wfdb_records.exam_id = df_wfdb_records.wfdb_signal_file.apply(lambda x: int(re.sub("\\D", "", x.stem)))

        if self._label_file is None:
            self._label_file = self.db_dir / self.__label_file__
            if not self._label_file.exists():
                # self.download(["labels"], refresh=False)
                self._label_file = None
        else:
            self._label_file = Path(self._label_file).expanduser().resolve()
        if self._chagas_label_file is None:
            self._chagas_label_file = self.db_dir / self.__chagas_label_file__
            if not self._chagas_label_file.exists():
                # self.download(["chagas-labels"], refresh=False)
                self._chagas_label_file = None
        else:
            self._chagas_label_file = Path(self._chagas_label_file).expanduser().resolve()

        early_exit = False

        if len(self._h5_data_files) == 0 and df_wfdb_records.empty:
            self.logger.warning("No data files found in the database directory. Call `download()` to download the database.")
            early_exit = True

        # else: some data files are found, proceed to load the metadata

        # assert (
        #     self._label_file is not None and self._label_file.exists()
        # ), f"Label file {self.__label_file__} not found in the given directory."
        # assert (
        #     self._chagas_label_file is not None and self._chagas_label_file.exists()
        # ), f"Chagas label file {self.__chagas_label_file__} not found in the given directory."
        if self._label_file is None or not self._label_file.exists():
            self.logger.warning(f"Label file {self.__label_file__} not found in the given directory.")
            self._label_file = None
            early_exit = True
        if self._chagas_label_file is None or not self._chagas_label_file.exists():
            self.logger.warning(f"Chagas label file {self.__chagas_label_file__} not found in the given directory.")
            self._chagas_label_file = None
            early_exit = True

        if early_exit:
            self._df_records = pd.DataFrame()
            self._df_chagas = pd.DataFrame()
            self._all_records = []
            return

        self._df_records = pd.read_csv(self._label_file)
        self._df_records["sex"] = self._df_records["is_male"].map({True: "Male", False: "Female"})
        self._df_chagas = pd.read_csv(self._chagas_label_file)
        self._all_records = list(
            set(self._df_records.exam_id.unique().tolist()).intersection(self._df_chagas.exam_id.unique().tolist())
        )
        self._df_records = self._df_records[self._df_records.exam_id.isin(self._all_records)]
        self._df_chagas = self._df_chagas[self._df_chagas.exam_id.isin(self._all_records)]
        df_wfdb_records = df_wfdb_records[df_wfdb_records.exam_id.isin(self._all_records)]

        if df_wfdb_records.empty:
            self._is_converted_to_wfdb_format = False

            self._df_records["record"] = self._df_records["exam_id"].astype(str)
            self._df_records.set_index("record", inplace=True)
            self._all_records = self._df_records.index.tolist()
            self._df_chagas["record"] = self._df_chagas["exam_id"].astype(str)
            self._df_chagas.set_index("record", inplace=True)

            return

        self._is_converted_to_wfdb_format = True
        self._df_records = pd.merge(self._df_records, df_wfdb_records, on="exam_id", how="inner")
        self._df_records["record"] = self._df_records["exam_id"].astype(str)
        self._df_records.set_index("record", inplace=True)
        self._all_records = self._df_records.index.tolist()
        self._df_chagas = self._df_chagas[self._df_chagas.exam_id.isin(self._df_records.exam_id)]
        self._df_chagas["record"] = self._df_chagas["exam_id"].astype(str)
        self._df_chagas.set_index("record", inplace=True)

    def get_absolute_path(self, rec: Union[str, int], ext: Literal["hdf5", "dat", "mat"] = "dat") -> Path:
        """Get the absolute path of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
            NOTE: DO NOT confuse index (int) and record name (exam_id, str).
        ext : {"hdf5", "dat", "mat"}, default "dat"
            Extension of the file.

        Returns
        -------
        path : pathlib.Path
            Absolute path of the record.

        """
        if isinstance(rec, int):
            rec = self[rec]
        row = self._df_records.loc[rec]
        if ext == "hdf5":
            path = self.db_dir / row["trace_file"]
        else:
            path = row["wfdb_signal_file"].with_suffix(f".{ext}")
        if not path.exists():
            self.logger.warning(f"File {path} does not exist.")
        return path

    def load_data(
        self,
        rec: Union[str, int],
        data_format: str = "channel_first",
        units: Union[str, type(None)] = "mV",
        fs: Optional[Real] = None,
        return_fs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Real]]:
        """Load physical (converted from digital) ECG data,
        or load digital signal directly.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
            NOTE: DO NOT confuse index (int) and record name (exam_id, str).
        data_format : str, default "channel_first"
            Format of the ECG data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first"), or
            "flat" (alias "plain").
        units : str or None, default "mV"
            Units of the output signal, can also be "μV" (aliases "uV", "muV");
            None for digital data, without digital-to-physical conversion.
        fs : numbers.Real, optional
            Sampling frequency of the output signal.
            If not None, the loaded data will be resampled to this frequency,
            otherwise, the original sampling frequency will be used.
        return_fs : bool, default False
            Whether to return the sampling frequency of the output signal.

        Returns
        -------
        data : numpy.ndarray
            The loaded ECG data.
        data_fs : numbers.Real, optional
            Sampling frequency of the output signal.
            Returned if `return_fs` is True.

        .. note::
            Since the duration of the signals are short (<= 10 seconds),
            parameters `sampfrom` and `sampto` are not provided.

        """
        if isinstance(rec, str):
            rec = self._df_records.index.get_loc(rec)
        if not self._is_converted_to_wfdb_format:
            # load data from hdf5 file
            h5_file = self._h5_data_files[0]
            with h5py.File(h5_file, "r") as h5f:
                data = h5f["tracings"][rec]
        else:
            # load data from wfdb files
            record_path = self._df_records.loc[rec, "wfdb_signal_file"]
            data = wfdb.rdsamp(record_path)[0]
        data = data.astype(np.float32)  # typically in most deep learning tasks, we use float32
        if units.lower() in ["uv", "μv", "muv"]:
            data = data * 1e3
        if fs is not None and fs != self.fs:
            data = wfdb.processing.resample_sig(data, self.fs, fs)
        else:
            fs = self.fs
        if data_format.lower() in ["channel_first", "lead_first"]:
            data = data.T
        if return_fs:
            return data, fs
        return data

    def load_ann(self, rec: Union[str, int]) -> Dict[str, Any]:
        """Load the annotations of the record.

        The annotations are:

            - age: age of the patient.
            - normal_ecg: binary label of the ECG.
            - death: binary label whether the patient died in the follow-up time.
            - timey: if the patient dies it is the time to the death of the patient.
              If not, it is the follow-up time.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
            NOTE: DO NOT confuse index (int) and record name (exam_id, str).

        Returns
        -------
        ann : dict
            Annotations of the record.

        """
        if isinstance(rec, int):
            rec = self[rec]
        ann = self._df_records.loc[rec, self.__label_cols__].to_dict()
        return ann

    def load_chagas_ann(self, rec: Union[str, int]) -> int:
        """Load the Chagas label of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
            NOTE: DO NOT confuse index (int) and record name (exam_id, str).

        Returns
        -------
        chagas_ann : int
            Chagas label of the record.
            0 for negative, 1 for positive.

        """
        if isinstance(rec, int):
            rec = self[rec]
        chagas_ann = int(self._df_chagas.loc[rec, "chagas"])
        return chagas_ann

    def plot(
        self,
        rec: Union[str, int],
        data: Optional[np.ndarray] = None,
        ticks_granularity: int = 0,
        leads: Optional[Union[str, Sequence[str]]] = None,
        same_range: bool = False,
        **kwargs: Any,
    ) -> None:
        """Plot the signals of a record or external signals (units in μV),
        along with the annotations.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        data : numpy.ndarray, optional
            (12-lead) ECG signal to plot,
            should be of the format "channel_first",
            and compatible with `leads`.
            If is not None, data of `rec` will not be used.
            This is useful when plotting filtered data.
        ticks_granularity : int, default 0
            Granularity to plot axis ticks, the higher the more ticks.
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        leads : str or List[str], optional
            The leads of the ECG signal to plot.
        same_range : bool, default False
            If True, all leads are forced to have the same y range.
        kwargs : dict, optional
            Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot`.

        """
        if isinstance(rec, int):
            rec = self[rec]

        if "plt" not in dir():
            import matplotlib.pyplot as plt

            plt.MultipleLocator.MAXTICKS = 3000

        _leads = self._normalize_leads(leads, numeric=False)
        lead_indices = [self.all_leads.index(ld) for ld in _leads]

        if data is None:
            _data = self.load_data(rec, data_format="channel_first", units="μV")[lead_indices]
        else:
            units = self._auto_infer_units(data)
            self.logger.info(f"input data is auto detected to have units in {units}")
            if units.lower() == "mv":
                _data = 1000 * data
            else:
                _data = data
            assert _data.shape[0] == len(
                _leads
            ), f"number of leads from data of shape ({_data.shape[0]}) does not match the length ({len(_leads)}) of `leads`"

        if same_range:
            y_ranges = np.ones((_data.shape[0],)) * np.max(np.abs(_data)) + 100
        else:
            y_ranges = np.max(np.abs(_data), axis=1) + 100

        row = self._df_records.loc[rec]
        ann = "; ".join([f"{k} - {v}" for k, v in self.load_ann(rec).items() if k not in ["age"]])

        plot_alpha = 0.4
        nb_leads = len(_leads)

        t = np.arange(_data.shape[1]) / self.fs
        duration = len(t) / self.fs
        fig_sz_w = int(round(DEFAULT_FIG_SIZE_PER_SEC * duration))
        fig_sz_h = 6 * np.maximum(y_ranges, 750) / 1500
        fig, axes = plt.subplots(nb_leads, 1, sharex=False, figsize=(fig_sz_w, np.sum(fig_sz_h)))
        if nb_leads == 1:
            axes = [axes]
        for idx in range(nb_leads):
            axes[idx].plot(
                t,
                _data[idx],
                color="black",
                linewidth="2.0",
                label=f"lead - {_leads[idx]}",
            )
            axes[idx].axhline(y=0, linestyle="-", linewidth="1.0", color="red")
            # NOTE that `Locator` has default `MAXTICKS` equal to 1000
            if ticks_granularity >= 1:
                axes[idx].xaxis.set_major_locator(plt.MultipleLocator(0.2))
                axes[idx].yaxis.set_major_locator(plt.MultipleLocator(500))
                axes[idx].grid(which="major", linestyle="-", linewidth="0.4", color="red")
            if ticks_granularity >= 2:
                axes[idx].xaxis.set_minor_locator(plt.MultipleLocator(0.04))
                axes[idx].yaxis.set_minor_locator(plt.MultipleLocator(100))
                axes[idx].grid(which="minor", linestyle=":", linewidth="0.2", color="gray")
            # add extra info. to legend
            # https://stackoverflow.com/questions/16826711/is-it-possible-to-add-a-string-as-a-legend-item-in-matplotlib
            axes[idx].plot(
                [],
                [],
                " ",
                label=f"Exam ID - {rec}; Age - {row.age}; Sex - {row.sex}",
            )
            axes[idx].plot([], [], " ", label=ann)
            axes[idx].legend(loc="upper left", fontsize=14)
            axes[idx].set_xlim(t[0], t[-1])
            axes[idx].set_ylim(min(-600, -y_ranges[idx]), max(600, y_ranges[idx]))
            axes[idx].set_xlabel("Time [s]", fontsize=16)
            axes[idx].set_ylabel("Voltage [μV]", fontsize=16)
        plt.subplots_adjust(hspace=0.05)
        fig.tight_layout()
        if kwargs.get("save_path", None):
            plt.savefig(kwargs["save_path"], dpi=200, bbox_inches="tight")
        else:
            plt.show()

    @property
    def url(self) -> Dict[str, str]:
        return {
            "exams": f"{self.__dl_base_url__}{self.__data_file__}?download=1",
            "labels": f"{self.__dl_base_url__}{self.__label_file__}?download=1",
            "chagas-labels": self.__chagas_label_file_url__,
        }

    def download(self, files: Optional[Union[str, Sequence[str]]] = None, refresh: bool = True) -> None:
        """Download the database files.

        Parameters
        ----------
        files : str or list of str, optional
            The files to download.
            If not specified, download all files.
            The available files are:

                - "exams"
                - "labels"
                - "chagas-labels"
        refresh : bool, default True
            Whether to call `self._ls_rec()` after downloading the files.

        """
        if files is None:
            files = list(self.url.keys())
        elif isinstance(files, str):
            files = [files]

        for file in files:
            if file not in self.url:
                self.logger.warning(f"File {file} is not in the list of available files.")
                continue
            # skip downloading if the file already exists
            filename = Path(urllib.parse.urlparse(self.url[file]).path)
            if file == "exams":
                filename = filename.with_suffix(".hdf5").name
            else:
                filename = filename.with_suffix(".csv").name
            if (self.db_dir / filename).exists():
                self.logger.info(f"File {filename} already exists in the database directory.")
                continue
            http_get(self.url[file], self.db_dir)

        if refresh:
            self._ls_rec()

    @property
    def database_info(self) -> DataBaseInfo:
        return _SamiTrop_INFO

    def _train_test_split(self, train_ratio: float = 0.8) -> Dict[str, List[str]]:
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
        data_split : dict
            Dictionary containing the training and test (validation) sets
            of the record names.

        """
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0, "train_ratio and test_ratio must be positive"

        df_records = self._df_records[["age", "sex", "death"]].copy()
        df_records["chagas"] = self._df_chagas["chagas"]
        # make `age` categorical
        df_records["age"] = df_records["age"].apply(lambda x: f"{int(x // 10)}0s")
        df_train, df_test = stratified_train_test_split(
            df_records,
            ["age", "sex", "death", "chagas"],
            test_ratio=1 - train_ratio,
            reset_index=False,
        )
        data_split = {
            "train": df_train.index.tolist(),
            "test": df_test.index.tolist(),
        }
        return data_split

    def _convert_to_wfdb_format(
        self,
        signal_format: Literal["dat", "mat"] = "dat",
        trim_zeros: bool = True,
        overwrite: bool = False,
    ) -> List[Tuple[str, int]]:
        """Convert the Sami-Trop dataset to WFDB format.

        Parameters
        ----------
        signal_format : {"dat", "mat"}, default "dat"
            The format of the signal files.
        trim_zeros : bool, default True
            Whether to trim the zero padding at the start and end of the signals.
        overwrite : bool, default False
            Whether to overwrite the existing files.

        Returns
        -------
        excep_list : list of tuple
            List of tuples containing the h5 file name and the exam ID that failed to convert.
            This list is kept and returned for further inspection and debugging.

        """
        if len(self._h5_data_files) == 0:
            self.logger.warning("No hdf5 files found in the database directory. Call `download()` to download the database.")
            return

        excep_list = SamiTrop.convert_to_wfdb_format(
            signal_file=self._h5_data_files[0],
            df_demographics=self._df_records,
            df_chagas=self._df_chagas,
            output_path=self.wfdb_data_dir,
            trim_zeros=trim_zeros,
            overwrite=overwrite,
        )
        self._ls_rec()

        return excep_list

    @staticmethod
    def convert_to_wfdb_format(
        signal_file: Union[str, bytes, os.PathLike],
        df_demographics: pd.DataFrame,
        df_chagas: pd.DataFrame,
        output_path: Union[str, bytes, os.PathLike],
        signal_format: Literal["dat", "mat"] = "dat",
        trim_zeros: bool = True,
        overwrite: bool = False,
    ) -> List[Tuple[str, int]]:
        """Convert the Sami-Trop dataset to WFDB format.

        Parameters
        ----------
        signal_file : `path-like`
            Path to the signal file.
        df_demographics : pd.DataFrame
            DataFrame containing the demographic information.
        df_chagas : pd.DataFrame
            DataFrame containing the Chagas labels.
        output_path : `path-like`
            Output path to store the converted files.
        signal_format : {"dat", "mat"}, default "dat"
            The format of the signal files.
        trim_zeros : bool, default True
            Whether to trim the zero padding at the start and end of the signals.
        overwrite : bool, default False
            Whether to overwrite the existing files.

        Returns
        -------
        excep_list : list of tuple
            List of tuples containing the h5 file name and the exam ID that failed to convert.
            This list is kept and returned for further inspection and debugging.

        """
        assert signal_format in ["dat", "mat"], f"Unsupported signal format: {signal_format}"
        df_demographics = df_demographics.copy()
        if "sex" not in df_demographics.columns:
            df_demographics["sex"] = df_demographics["is_male"].map({True: "Male", False: "Female"})
        exam_id_to_demographics = df_demographics[["exam_id", "age", "sex"]].set_index("exam_id").to_dict(orient="index")

        exam_id_to_chagas = df_chagas[["exam_id", "chagas"]].set_index("exam_id").to_dict()["chagas"]

        # Load and convert the signal data.

        # See https://zenodo.org/records/4905618 for more information about these values.
        lead_names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        sampling_frequency = 400
        units = "mV"

        # Define the paramters for the WFDB files.
        gain = 1000
        baseline = 0
        num_bits = 16
        fmt = str(num_bits)

        output_path = Path(output_path).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        # Iterate over the input signal files.
        exam_ids = df_demographics["exam_id"].values
        num_exam_ids = len(exam_ids)
        excep_list = []
        with h5py.File(signal_file, "r") as f:
            # Iterate over the exam IDs in each signal file.
            for idx in tqdm(
                range(num_exam_ids),
                total=num_exam_ids,
                desc=f"Converting signals in {Path(signal_file).stem}",
                dynamic_ncols=True,
                mininterval=1,
                leave=False,
            ):
                exam_id = exam_ids[idx]

                # Skip exam IDs without Chagas labels.
                if exam_id not in exam_id_to_chagas:
                    continue

                physical_signals = np.array(f["tracings"][idx], dtype=np.float32)

                # Perform basic error checking on the signal.
                num_samples, num_leads = np.shape(physical_signals)
                assert num_leads == 12

                if trim_zeros:
                    # Remove zero padding at the start and end of the signals.
                    physical_signals = trim_zeros_func(physical_signals, trim="fb", axis=0)
                    if physical_signals.shape[0] == 0:
                        excep_list.append((signal_file, exam_id))
                        continue

                # Convert the signal to digital units; saturate the signal and represent NaNs as the lowest representable integer.
                digital_signals = gain * physical_signals
                digital_signals = np.round(digital_signals)
                digital_signals = np.clip(digital_signals, -(2 ** (num_bits - 1)) + 1, 2 ** (num_bits - 1) - 1)
                digital_signals[~np.isfinite(digital_signals)] = -(2 ** (num_bits - 1))
                digital_signals = np.asarray(
                    digital_signals, dtype=np.int32
                )  # We need to promote from 16-bit integers due to an error in the Python WFDB library.

                # Add the exam ID, age, sex, the Chagas label, and data source.
                age = exam_id_to_demographics[exam_id]["age"]
                sex = exam_id_to_demographics[exam_id]["sex"]
                chagas = exam_id_to_chagas[exam_id]
                source = "SaMi-Trop"
                comments = [
                    # f"Exam ID: {exam_id}",
                    # f"Patient ID: {patient_id}",
                    f"Age: {age}",
                    f"Sex: {sex}",
                    f"Chagas label: {chagas}",
                    f"Source: {source}",
                ]

                # Save the signal.
                record = str(exam_id)
                wfdb.wrsamp(
                    record,
                    fs=sampling_frequency,
                    units=[units] * num_leads,
                    sig_name=lead_names,
                    d_signal=digital_signals,
                    fmt=[fmt] * num_leads,
                    adc_gain=[gain] * num_leads,
                    baseline=[baseline] * num_leads,
                    write_dir=str(output_path),
                    comments=comments,
                )

                if signal_format in ("mat", ".mat"):
                    samitrop_convert_dat_to_mat(record, write_dir=str(output_path))

                # Recompute the checksums as needed.
                checksums = np.sum(digital_signals, axis=0, dtype=np.int16)
                samitrop_fix_checksums(str(output_path / record), checksums)

        print("Conversion of the SaMi-Trop database is complete.")
        print(f"{num_exam_ids - len(excep_list)} signals converted successfully.")
        print(f"{len(excep_list)} signals failed to convert.")

        return excep_list


class PTBXL(PTBXL_Reader):

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in a dataframe.
        """
        super()._ls_rec()
        # fix errors when we have a subset of the PTB-XL dataset
        # the `_df_metadata` contains the metadata of the entire dataset
        self._df_metadata = self._df_metadata.loc[self._df_records.index]

    def _train_test_split(self, train_ratio: float = 0.8) -> Dict[str, List[str]]:
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
        data_split : dict
            Dictionary containing the training and test (validation) sets
            of the record names.

        """
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0, "train_ratio and test_ratio must be positive"

        df_subjects = self._df_records[["age", "sex", "patient_id"]].copy()
        df_subjects = df_subjects.groupby("patient_id").agg(
            {
                "age": "first",
                "sex": "first",
            }
        )
        # make `age` categorical
        df_subjects["age"] = df_subjects["age"].apply(lambda x: f"{int(x // 10)}0s")
        df_train, df_test = stratified_train_test_split(
            df_subjects,
            ["age", "sex"],
            test_ratio=1 - train_ratio,
            reset_index=False,
        )
        data_split = {
            "train": self._df_records[self._df_records["patient_id"].isin(df_train.index)].index.tolist(),
            "test": self._df_records[self._df_records["patient_id"].isin(df_test.index)].index.tolist(),
        }
        return data_split

    def _convert_to_wfdb_format(
        self,
        output_path: Union[str, bytes, os.PathLike] = "wfdb_format_files",
        fs: Literal[100, 500] = 500,
        signal_format: Literal["dat", "mat"] = "dat",
        overwrite: bool = False,
    ) -> None:
        """Convert the PTB-XL dataset to WFDB format.

        Typically, the chagas labels are added to the header files as comments.

        Parameters
        ----------
        output_path : `path-like`, default "wfdb_format_files"
            Output path to store the converted files.
            If not absolute, it is resolved relative to the database directory.
        fs : {100, 500}, default 500
            Sampling frequency of the signals to convert.
        signal_format : {"dat", "mat"}, default "dat"
            The format of the signal files.
        overwrite : bool, default False
            Whether to overwrite the existing files.

        Returns
        -------
        None

        """
        output_path = Path(output_path).expanduser()
        if not output_path.is_absolute():
            output_path = self.db_dir / output_path
        PTBXL.convert_to_wfdb_format(
            signal_dir=self.db_dir,
            df_demographics=self._df_metadata,
            output_path=output_path,
            fs=fs,
            signal_format=signal_format,
            overwrite=overwrite,
        )

    @staticmethod
    def convert_to_wfdb_format(
        signal_dir: Union[str, bytes, os.PathLike],
        df_demographics: pd.DataFrame,
        output_path: Union[str, bytes, os.PathLike],
        fs: Literal[100, 500] = 500,
        signal_format: Literal["dat", "mat"] = "dat",
        overwrite: bool = False,
    ) -> None:
        """Convert the PTB-XL dataset to WFDB format.

        Typically, the chagas labels are added to the header files as comments.

        TODO: use multi-processing for faster conversion.

        Parameters
        ----------
        signal_dir : `path-like`
            Path to the directory containing the signal files.
        df_demographics : pd.DataFrame
            DataFrame containing the demographic information.
        output_path : `path-like`
            Output path to store the converted files.
        fs : {100, 500}, default 500
            Sampling frequency of the signals to convert.
        signal_format : {"dat", "mat"}, default "dat"
            The format of the signal files.
        overwrite : bool, default False
            Whether to overwrite the existing files.

        Returns
        -------
        None

        """
        signal_dir = Path(signal_dir).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()
        assert fs in [100, 500], f"Unsupported sampling frequency: {fs}"
        file_col = {100: "filename_lr", 500: "filename_hr"}[fs]
        assert signal_format in ["dat", "mat"], f"Unsupported signal format: {signal_format}"
        df_demographics = df_demographics.copy()
        df_demographics["recording_date"] = pd.to_datetime(df_demographics["recording_date"])
        df_demographics["recording_date"] = df_demographics["recording_date"].dt.strftime("%X %d/%m/%Y")

        for row in tqdm(
            df_demographics.itertuples(),
            total=len(df_demographics),
            desc="Converting signals",
            dynamic_ncols=True,
            mininterval=1,
        ):
            age = row.age
            age = int(age) if is_integer(age) else float(age)

            sex = row.sex
            if sex == 0:
                sex = "Male"
            elif sex == 1:
                sex = "Female"
            else:
                sex = "Unknown"

            height = row.height
            height = int(height) if is_integer(height) else float(height)

            weight = row.weight
            weight = int(weight) if is_integer(weight) else float(weight)

            recording_date = row.recording_date

            # Assume that all of the patients are negative for Chagas, which is likely to be the case for every or almost every patient
            # in the PTB-XL dataset.
            label = False
            source = "PTB-XL"

            # Update the header file.
            input_header_file = (signal_dir / getattr(row, file_col)).with_suffix(".hea")
            output_header_file = (output_path / getattr(row, file_col)).with_suffix(".hea")
            output_header_file.parent.mkdir(parents=True, exist_ok=True)

            if not output_header_file.exists() or overwrite:
                input_header = input_header_file.read_text()

                lines = input_header.split("\n")
                record_line = " ".join(lines[0].strip().split(" ")[:4]) + "\n"
                signal_lines = "\n".join(line.strip() for line in lines[1:] if line.strip() and not line.startswith("#")) + "\n"
                comment_lines = (
                    "\n".join(
                        line.strip()
                        for line in lines[1:]
                        if line.startswith("#")
                        and not any(
                            (
                                line.startswith(x)
                                for x in ("# Age:", "# Sex:", "# Height:", "# Weight:", "# Chagas label:", "# Source:")
                            )
                        )
                    )
                    + "\n"
                )

                record_line = record_line.strip() + f" {recording_date}" + "\n"
                signal_lines = signal_lines.strip() + "\n"
                comment_lines = (
                    comment_lines.strip()
                    + f"# Age: {age}\n# Sex: {sex}\n# Height: {height}\n# Weight: {weight}\n"
                    + f"# Chagas label: {label}\n# Source: {source}\n"
                )

                output_header = record_line + signal_lines + comment_lines
                output_header_file.write_text(output_header)

            # Copy the signal files if the input and output folders are different.
            output_signal_file = (output_path / getattr(row, file_col)).with_suffix(f".{signal_format}")
            if not output_signal_file.exists():
                input_signal_file = (signal_dir / getattr(row, file_col)).with_suffix(".dat")
                shutil.copy2(input_signal_file, output_signal_file.with_suffix(".dat"))
                # Convert data from .dat files to .mat files as requested.
                if signal_format in ("mat", ".mat"):
                    ptbxl_convert_dat_to_mat(output_signal_file.stem, write_dir=str(output_signal_file.parent))

                # Recompute the checksums as needed.
                ptbxl_fix_checksums(str(output_signal_file.with_suffix("")))

        print("Conversion of the PTB-XL database is complete.")
        print(f"{len(df_demographics)} signals converted successfully.")


_CINC2025_INFO = DataBaseInfo(
    title="Detection of Chagas Disease from the ECG: The George B. Moody PhysioNet Challenge 2025",
    about="""
    1. The Challenge [1]_ uese the CODE-15% dataset [2]_, the SaMi-Trop dataset [3]_, and the PTB-XL dataset [4]_. It combines a large dataset with weak labels and two small datasets with strong labels.
    2. The CODE-15% dataset contains over 300,000 12-lead ECG records collected in Brazil between 2010 and 2016. Most recordings have a duration of either 7.3 s or 10.2 s and a sampling frequency of 400 Hz. The binary Chagas labels are self-reported and therefore may or may not have been validated.
    3. The SaMi-Trop dataset contains 1,631 12-lead ECG records collected from Chagas patients in Brazil between 2011 and 2012. Most recordings have a duration of either 7.3 s or 10.2 s and a sampling frequency of 400 Hz. The binary Chagas labels are validated by serological tests, and all are positive.
    4. The PTB-XL dataset contains 21,799 12-lead ECG records collected from presumably non-Chagas patients in Europe between 1989 and 1996. The recordings have a duration of 10 s and a sampling frequency of 500 Hz (or optionally 100 Hz). Based on geography, all or almost all of the patients are likely to be Chagas negative.
    """,
    usage=[
        "Chagas disease detection",
        "ECG arrhythmia detection",
        "Self-Supervised Learning",
    ],
    note="""
    """,
    issues="""
    """,
    references=[
        "https://moody-challenge.physionet.org/2025/",
        "https://zenodo.org/records/4916206",
        "https://zenodo.org/records/4905618",
        "https://physionet.org/content/ptb-xl/",
        "https://github.com/antonior92/automatic-ecg-diagnosis",
        "https://github.com/antonior92/ecg-age-prediction",
    ],
    doi=[
        "10.5281/zenodo.4916206",
        "10.5281/zenodo.4905618",
        "10.1038/s41467-020-15432-4",  # CODE15 paper
        "10.1038/s41467-021-25351-7",  # ECG-age prediction paper
        "10.1038/s41597-020-0495-6",  # PTB-XL paper
        "10.13026/6sec-a640",  # PTB-XL physionet
    ],
)


@add_docstring(_CINC2025_INFO.format_database_docstring(), mode="prepend")
class CINC2025(_DataBase):
    """
    Parameters
    ----------
    db_dir : `path-like`, optional
        Storage path of the database.
        If not specified, data will be fetched from Physionet.
    working_dir : `path-like`, optional
        Working directory, to store intermediate files and log files.
    verbose : int, default 1
        Level of logging verbosity.
    kwargs : dict, optional
        Auxilliary key word arguments.

    """

    __name__ = "CINC2025"

    def __init__(
        self,
        db_dir: Optional[Dict[str, Union[str, bytes, os.PathLike]]] = None,
        working_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            db_name="CINC2025",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.__config = CFG(BaseCfg.copy())
        self.__config.update(kwargs)
        self._use_dbs = self.__config.get("use_dbs", ["CODE-15%", "SaMi-Trop", "PTB-XL"])
        if self._use_dbs == ["all"]:
            self._use_dbs = ["CODE-15%", "SaMi-Trop", "PTB-XL"]
        if self._use_dbs == ["SaMi-Trop"] or self._use_dbs == ["PTB-XL"]:
            raise ValueError("Using database with only positive or negative labels is not allowed.")

        self.all_leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        self._df_records = pd.DataFrame()
        self._all_records = []
        self._ls_rec()

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in a dataframe.
        """
        columns = ["sig_len", "age", "sex", "fs", "chagas", "source"]
        self._df_records = pd.DataFrame(self.db_dir.rglob("*.hea"), columns=["path"])
        self._df_records["path"] = self._df_records["path"].apply(lambda x: x.with_suffix(""))
        # keep only those records that have a corresponding .dat file or .mat file
        self._df_records = self._df_records[
            self._df_records["path"].apply(lambda x: x.with_suffix(".dat").exists() or x.with_suffix(".mat").exists())
        ]
        if self._df_records.empty:
            self._df_records = pd.DataFrame(columns=["path"] + columns)
            self._df_records.index.name = "record"
            return

        self._df_records["record"] = self._df_records["path"].apply(lambda x: x.stem)

        # load the metadata (age, sex) and chagas labels from the header files
        self._df_records[columns] = None

        # comment_pattern = re.compile("(?P<key>[^:\\s]+): (?P<value>.+)", re.MULTILINE)
        # for row in tqdm(
        #     self._df_records.itertuples(),
        #     total=len(self._df_records),
        #     desc="Loading metadata",
        #     dynamic_ncols=True,
        #     mininterval=1,
        # ):
        #     comments = "\n".join(wfdb.rdheader(row.path).comments)
        #     for key, val in comment_pattern.findall(comments):
        #         key = key.lower().replace("label", "").strip()
        #         if key in columns:
        #             self._df_records.loc[row.Index, key] = val

        with mp.Pool(processes=max(1, mp.cpu_count() - 3)) as pool:
            metadata = pool.starmap(
                load_metadata_from_header,
                tqdm(
                    [(row.path,) for row in self._df_records.itertuples()],
                    total=len(self._df_records),
                    desc="Loading metadata",
                    dynamic_ncols=True,
                    mininterval=1,
                ),
            )
        metadata = pd.DataFrame(metadata, columns=columns + ["record"])
        metadata.set_index("record", inplace=True)
        self._df_records.set_index("record", inplace=True)
        self._df_records[columns] = metadata

        # drop records that are not from the selected databases
        if self._use_dbs is not None:
            self._df_records = self._df_records[self._df_records["source"].isin(self._use_dbs)]

        # drop records with missing chagas label
        self._df_records = self._df_records.dropna(subset=["chagas"])
        self._df_records["chagas"] = self._df_records["chagas"].apply(str2bool)
        self._all_records = self._df_records.index.tolist()

    def load_data(
        self,
        rec: Union[str, int],
        data_format: str = "channel_first",
        units: Union[str, type(None)] = "mV",
        fs: Optional[Real] = None,
        return_fs: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Real]]:
        """Load physical (converted from digital) ECG data,
        or load digital signal directly.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
            NOTE: DO NOT confuse index (int) and record name (exam_id, str).
        data_format : str, default "channel_first"
            Format of the ECG data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first"), or
            "flat" (alias "plain").
        units : str or None, default "mV"
            Units of the output signal, can also be "μV" (aliases "uV", "muV");
            None for digital data, without digital-to-physical conversion.
        fs : numbers.Real, optional
            Sampling frequency of the output signal.
            If not None, the loaded data will be resampled to this frequency,
            otherwise, the original sampling frequency will be used.
        return_fs : bool, default True
            Whether to return the sampling frequency of the output signal.

        Returns
        -------
        data : numpy.ndarray
            The loaded ECG data.
        data_fs : numbers.Real, optional
            Sampling frequency of the output signal.
            Returned if `return_fs` is True.

        .. note::
            Since the duration of the signals are short (<= 10 seconds),
            parameters `sampfrom` and `sampto` are not provided.

        """
        if isinstance(rec, int):
            rec = self[rec]
        record_path = self._df_records.loc[rec, "path"]
        data = wfdb.rdsamp(record_path)[0]  # shape (n_samples, n_leads)
        data = data.astype(np.float32)  # typically in most deep learning tasks, we use float32
        data_fs = wfdb.rdheader(record_path).fs
        if units.lower() in ["uv", "μv", "muv"]:
            data = data * 1e3
        if fs is not None and fs != data_fs:
            data = wfdb.processing.resample_sig(data, data_fs, fs)
        else:
            fs = data_fs
        if data_format.lower() in ["channel_first", "lead_first"]:
            data = data.T
        if return_fs:
            return data, fs
        return data

    def load_ann(self, rec: Union[str, int]) -> int:
        """Load the chagas annotations of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
            NOTE: DO NOT confuse index (int) and record name (exam_id, str).

        Returns
        -------
        chagas_ann : int
            Chagas label of the record.
            0 for negative, 1 for positive.

        """
        if isinstance(rec, int):
            rec = self[rec]
        chagas_ann = int(self._df_records.loc[rec, "chagas"])
        return chagas_ann

    def plot(
        self,
        rec: Union[str, int],
        data: Optional[np.ndarray] = None,
        ticks_granularity: int = 0,
        leads: Optional[Union[str, Sequence[str]]] = None,
        same_range: bool = False,
        **kwargs: Any,
    ) -> None:
        """Plot the signals of a record or external signals (units in μV),
        along with the annotations.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        data : numpy.ndarray, optional
            (12-lead) ECG signal to plot,
            should be of the format "channel_first",
            and compatible with `leads`.
            If is not None, data of `rec` will not be used.
            This is useful when plotting filtered data.
        ticks_granularity : int, default 0
            Granularity to plot axis ticks, the higher the more ticks.
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        leads : str or List[str], optional
            The leads of the ECG signal to plot.
        same_range : bool, default False
            If True, all leads are forced to have the same y range.
        kwargs : dict, optional
            Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot`.

        """
        if isinstance(rec, int):
            rec = self[rec]

        if "plt" not in dir():
            import matplotlib.pyplot as plt

            plt.MultipleLocator.MAXTICKS = 3000

        _leads = self._normalize_leads(leads, numeric=False)
        lead_indices = [self.all_leads.index(ld) for ld in _leads]

        if data is None:
            _data, fs = self.load_data(rec, data_format="channel_first", units="μV")
            _data = _data[lead_indices]
        else:
            units = self._auto_infer_units(data)
            fs = self._df_records.loc[rec, "fs"]
            self.logger.info(f"input data is auto detected to have units in {units}")
            if units.lower() == "mv":
                _data = 1000 * data
            else:
                _data = data
            assert _data.shape[0] == len(
                _leads
            ), f"number of leads from data of shape ({_data.shape[0]}) does not match the length ({len(_leads)}) of `leads`"

        if same_range:
            y_ranges = np.ones((_data.shape[0],)) * np.max(np.abs(_data)) + 100
        else:
            y_ranges = np.max(np.abs(_data), axis=1) + 100

        row = self._df_records.loc[rec]
        ann = f"Chagas - {row.chagas}"

        plot_alpha = 0.4
        nb_leads = len(_leads)

        t = np.arange(_data.shape[1]) / fs
        duration = len(t) / fs
        fig_sz_w = int(round(DEFAULT_FIG_SIZE_PER_SEC * duration))
        fig_sz_h = 6 * np.maximum(y_ranges, 750) / 1500
        fig, axes = plt.subplots(nb_leads, 1, sharex=False, figsize=(fig_sz_w, np.sum(fig_sz_h)))
        if nb_leads == 1:
            axes = [axes]
        for idx in range(nb_leads):
            axes[idx].plot(
                t,
                _data[idx],
                color="black",
                linewidth="2.0",
                label=f"lead - {_leads[idx]}",
            )
            axes[idx].axhline(y=0, linestyle="-", linewidth="1.0", color="red")
            # NOTE that `Locator` has default `MAXTICKS` equal to 1000
            if ticks_granularity >= 1:
                axes[idx].xaxis.set_major_locator(plt.MultipleLocator(0.2))
                axes[idx].yaxis.set_major_locator(plt.MultipleLocator(500))
                axes[idx].grid(which="major", linestyle="-", linewidth="0.4", color="red")
            if ticks_granularity >= 2:
                axes[idx].xaxis.set_minor_locator(plt.MultipleLocator(0.04))
                axes[idx].yaxis.set_minor_locator(plt.MultipleLocator(100))
                axes[idx].grid(which="minor", linestyle=":", linewidth="0.2", color="gray")
            # add extra info. to legend
            # https://stackoverflow.com/questions/16826711/is-it-possible-to-add-a-string-as-a-legend-item-in-matplotlib
            axes[idx].plot(
                [],
                [],
                " ",
                label=f"Record ID - {rec}; Age - {row.age}; Sex - {row.sex}",
            )
            axes[idx].plot([], [], " ", label=ann)
            axes[idx].legend(loc="upper left", fontsize=14)
            axes[idx].set_xlim(t[0], t[-1])
            axes[idx].set_ylim(min(-600, -y_ranges[idx]), max(600, y_ranges[idx]))
            axes[idx].set_xlabel("Time [s]", fontsize=16)
            axes[idx].set_ylabel("Voltage [μV]", fontsize=16)
        plt.subplots_adjust(hspace=0.05)
        fig.tight_layout()
        if kwargs.get("save_path", None):
            plt.savefig(kwargs["save_path"], dpi=200, bbox_inches="tight")
        else:
            plt.show()

    @property
    def database_info(self) -> DataBaseInfo:
        return _CINC2025_INFO

    @property
    def url(self) -> Dict[str, str]:
        try:
            with timeout(1):
                ptb_xl_version = wfdb_get_version("ptb-xl")
        except Exception:
            ptb_xl_version = "1.0.3"  # latest as of 2025-02-12
        links = {
            "ptb-xl": f"s3://physionet-open/ptb-xl/{ptb_xl_version}/",
            "sami-trop-exams": f"{SamiTrop.__dl_base_url__}{SamiTrop.__data_file__}?download=1",
            "sami-trop-labels": f"{SamiTrop.__dl_base_url__}{SamiTrop.__label_file__}?download=1",
            "sami-trop-chagas-labels": SamiTrop.__chagas_label_file_url__,
            "code-15-labels": f"{CODE15.__dl_base_url__}{CODE15.__label_file__}?download=1",
            "code-15-chagas-labels": CODE15.__chagas_label_file_url__,
        }
        links.update({f"code-15-exams-part{i}": f"{CODE15.__dl_base_url__}exams_part{i}.zip?download=1" for i in range(18)})
        if url_is_reachable("https://drive.google.com/"):
            links["ptb-xl-subset"] = "https://drive.google.com/u/0/uc?id=1wq9r6rbhaMhMe-GWHpi5lQQIVwBU8UPL"
        else:
            links["ptb-xl-subset"] = "https://deep-psp.tech/Data/ptb-xl-subset-tiny.zip"
        return links

    def download(self, files: Optional[Union[str, Sequence[str]]] = None, convert: bool = True) -> None:
        """Download the database files.

        Parameters
        ----------
        files : str or list of str, optional
            The files to download.
            If not specified, download all files.
            The available files are:

                - "code-15-exams-part{0-17}"
                - "code-15-labels"
                - "code-15-chagas-labels"
                - "sami-trop-exams"
                - "sami-trop-labels"
                - "sami-trop-chagas-labels"
                - "ptb-xl"
                - "ptb-xl-subset"

            or "code-15" containing all the code-15 files,
            "sami-trop" containing all the sami-trop files.
        convert : bool, default True
            Whether to convert the downloaded files to WFDB format.

        """
        if files is None:
            files = list(self.url.keys())
        elif isinstance(files, str):
            files = [files]
        files = [item for item in files if item in list(self.url.keys()) + ["code-15", "sami-trop"]]

        code15_files = [item.replace("code-15-", "") for item in files if item.startswith("code-15")]
        samitrop_files = [item.replace("sami-trop-", "") for item in files if item.startswith("sami-trop")]
        ptbxl_files = [item for item in files if item.startswith("ptb-xl")]
        if "code-15" in code15_files:
            code15_files = None  # download all code-15 files
        if "sami-trop" in samitrop_files:
            samitrop_files = None  # download all sami-trop files
        # print(f"{code15_files=}\n{samitrop_files=}\n{ptbxl_files=}")

        if code15_files is None or len(code15_files) > 0:
            (self.db_dir / CODE15.__name__).mkdir(parents=True, exist_ok=True)
            dr = CODE15(db_dir=self.db_dir / CODE15.__name__, wfdb_data_dir=self.db_dir)
            dr.download(code15_files, refresh=False)
            if convert:
                dr._ls_rec()
                dr._convert_to_wfdb_format()
            del dr
        if samitrop_files is None or len(samitrop_files) > 0:
            (self.db_dir / SamiTrop.__name__).mkdir(parents=True, exist_ok=True)
            dr = SamiTrop(db_dir=self.db_dir / SamiTrop.__name__, wfdb_data_dir=self.db_dir)
            dr.download(samitrop_files, refresh=False)
            if convert:
                dr._ls_rec()
                dr._convert_to_wfdb_format()
            del dr
        if ptbxl_files:
            (self.db_dir / PTBXL.__name__).mkdir(parents=True, exist_ok=True)
            if "ptb-xl-subset" in ptbxl_files:
                http_get(
                    self.url["ptb-xl-subset"],
                    dst_dir=self.db_dir / PTBXL.__name__,
                    filename="ptb-xl-subset-tiny.zip",
                    extract=True,
                )
                dr = PTBXL(db_dir=self.db_dir / PTBXL.__name__)
            if "ptb-xl" in ptbxl_files:
                dr = PTBXL(db_dir=self.db_dir / PTBXL.__name__)
                dr.download()
            if convert:
                dr._ls_rec()
                dr._convert_to_wfdb_format(output_path=self.db_dir)
            del dr


def load_metadata_from_header(header_file: Union[str, bytes, os.PathLike]) -> Dict[str, Any]:
    """Load metadata from the header file.

    Parameters
    ----------
    header_file : `path-like`
        Path to the header file.

    Returns
    -------
    metadata : dict
        Metadata extracted from the header file.

    """
    comment_pattern = re.compile("(?P<key>[^:\\n]+): (?P<value>.+)", re.MULTILINE)
    header = wfdb.rdheader(str(header_file))
    comments = "\n".join(header.comments)
    metadata = {key.lower().replace("label", "").strip(): val for key, val in comment_pattern.findall(comments)}
    metadata["record"] = Path(header_file).stem
    metadata["fs"] = header.fs
    metadata["sig_len"] = header.sig_len
    return metadata


if __name__ == "__main__":
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description="Process CINC2025 database.")
    parser.add_argument(
        "operations",
        nargs=argparse.ONE_OR_MORE,
        type=str,
        choices=[
            "download",
        ],
    )
    parser.add_argument(
        "-d",
        "--db-dir",
        type=str,
        help="The directory to (store) the database.",
        dest="db_dir",
    )
    parser.add_argument(
        "-c",
        "--convert",
        action="store_true",
        help="Whether to convert the downloaded files to WFDB format.",
        dest="convert",
    )
    # `remove` is added since `helper_code.find_records` might find records
    # not converted (the PTB-XL database). This will result in a failure
    # of the function `helper_code.load_label` since theses records contain
    # no chagas label.
    parser.add_argument(
        "-r",
        "--remove",
        action="store_true",
        help="Remove the downloaded files after conversion. Valid only when `convert` is True.",
        dest="remove",
    )
    parser.add_argument(
        "-w",
        "--working-dir",
        type=str,
        default=None,
        help="The working directory to store the intermediate results.",
        dest="working_dir",
    )
    parser.add_argument(
        "-f",
        "--files",
        type=str,
        help=(
            "H5 data files to download, used when `operations` contain `download`. "
            "e.g., 'code-15-exams-part0,code-15-exams-part17,sami-trop-exams,ptb-xl'."
        ),
        dest="files",
    )

    args = parser.parse_args()
    operations = args.operations
    files = args.files.split(",") if args.files else None
    if files is not None and "labels" in files:
        files.remove("labels")
        files.extend(["code-15-labels", "sami-trop-labels"])
    if files is not None and "chagas-labels" in files:
        files.remove("chagas-labels")
        files.extend(["code-15-chagas-labels", "sami-trop-chagas-labels"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dr = CINC2025(db_dir=args.db_dir, working_dir=args.working_dir)
        if "download" in operations:
            dr.download(files, convert=args.convert)
        if args.convert and args.remove:
            code15_dir = dr.db_dir / CODE15.__name__
            samitrop_dir = dr.db_dir / SamiTrop.__name__
            ptbxl_dir = dr.db_dir / PTBXL.__name__
            if code15_dir.exists():
                shutil.rmtree(code15_dir)
            if samitrop_dir.exists():
                shutil.rmtree(samitrop_dir)
            if ptbxl_dir.exists():
                shutil.rmtree(ptbxl_dir)
    del dr

    print("Done.")

    # usage examples:
    # python data_reader.py download -d /path/to/db_dir -c
    # python data_reader.py download --db-dir /path/to/db_dir --files "code-15-exams-part0,code-15-exams-part17,sami-trop,ptb-xl-subset,labels,chagas-labels" -c

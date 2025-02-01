"""
"""

import os
import re
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import wfdb
from torch_ecg.cfg import CFG
from torch_ecg.databases.base import DEFAULT_FIG_SIZE_PER_SEC, DataBaseInfo, _DataBase
from torch_ecg.utils.download import http_get
from torch_ecg.utils.misc import add_docstring
from tqdm.auto import tqdm

from cfg import BaseCfg
from prepare_code15_data import convert_dat_to_mat, fix_checksums

__all__ = [
    "CODE15",
]


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
    3. The signal files are of the format "exams_part{i}.hdf5", containing two datasets named `tracings` and other named `exam_id`. The `exam_id` is a tensor of dimension `(N,)` containing the exam id (the same as in the csv file) and the dataset `tracings` is a `(N, 4096, 12)` tensor containing the ECG tracings in the same order.
    4. The signals are sampled at 400 Hz. Some signals originally have a duration of 10 seconds (10 * 400 = 4000 samples) and others of 7 seconds (7 * 400 = 2800 samples). The latter were zero-padded (centered) to 10 seconds.
    5. The binary Chagas labels are self-reported and therefore may or may not have been validated.
    """,
    usage=[
        "ECG arrhythmia detection",
        "Self-Supervised Learning",
    ],
    note="""
    """,
    issues="""
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
        Auxilliary key word arguments

    """

    __name__ = "CODE15"
    __dl_base_url__ = "https://zenodo.org/records/4916206/files/"
    __data_files__ = {f"exams_part{i}": f"exams_part{i}.hdf5" for i in range(18)}
    __label_file__ = "exams.csv"
    __chagas_label_file__ = "code15_chagas_labels.csv"
    __chagas_label_file_url__ = "https://moody-challenge.physionet.org/2025/data/code15_chagas_labels.zip"
    __label_cols__ = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]

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
        self.wfdb_data_dir = Path(kwargs.pop("wfdb_data_dir", "wfdb_format_files"))
        self.wfdb_data_ext = kwargs.pop("wfdb_data_ext", "dat")
        self.__config = CFG(BaseCfg.copy())
        self.__config.update(kwargs)

        self.data_ext = "hdf5"
        self.ann_ext = self.__label_file__
        self.chagas_ann_ext = "code15_chagas_labels.csv"
        self.fs = 400
        self.all_leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        self._h5_data_files = []
        self._df_records = pd.DataFrame()
        self._all_records = []
        self._all_subjects = []
        self._subject_records = {}
        self._is_converted_to_wfdb_format = False
        self._ls_rec()

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in a dataframe.
        """
        # find all hdf5 files
        self._h5_data_files = list(self.db_dir.rglob("*.hdf5"))
        if len(self._h5_data_files) == 0:
            self.logger.warning("No hdf5 files found in the database directory. Call `download()` to download the database.")
            return
        assert len(set([f.parent for f in self._h5_data_files])) == 1, "All hdf5 files should be in the same directory."
        self.db_dir = self._h5_data_files[0].parent
        assert (
            self.db_dir / self.__label_file__
        ).exists(), f"Label file {self.__label_file__} not found in the database directory."
        assert (
            self.db_dir / self.__chagas_label_file__
        ).exists(), f"Chagas label file {self.__chagas_label_file__} not found in the database directory."

        self._df_records = pd.read_csv(self.db_dir / self.__label_file__)
        self._df_records["sex"] = self._df_records["is_male"].map({True: "Male", False: "Female"})
        self._df_chagas = pd.read_csv(self.db_dir / self.__chagas_label_file__)
        self._all_records = list(
            set(self._df_records.exam_id.unique().tolist()).intersection(self._df_chagas.exam_id.unique().tolist())
        )
        self._df_records = self._df_records[self._df_records.exam_id.isin(self._all_records)]
        self._df_chagas = self._df_chagas[self._df_chagas.exam_id.isin(self._all_records)]

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
        if df_wfdb_records.empty:
            self._is_converted_to_wfdb_format = False
            self._df_records["record"] = self._df_records["exam_id"].astype(str)
            self._subject_records = self._df_records.groupby("patient_id")["record"].apply(sorted).to_dict()
            self._df_records.set_index("record", inplace=True)
            self._all_records = self._df_records.index.tolist()
            self._all_subjects = self._df_records.patient_id.unique().tolist()
            self._df_chagas["record"] = self._df_chagas["exam_id"].astype(str)
            self._df_chagas.set_index("record", inplace=True)
            return

        self._is_converted_to_wfdb_format = True
        df_wfdb_records.wfdb_signal_file = df_wfdb_records.wfdb_signal_file.apply(lambda x: x.with_suffix(""))
        # note that the ".mat" files are named {exam_id}m.mat in function `convert_dat_to_mat`
        df_wfdb_records.exam_id = df_wfdb_records.wfdb_signal_file.apply(lambda x: int(re.sub("\\D", "", x.stem)))
        self._df_records = pd.merge(self._df_records, df_wfdb_records, on="exam_id", how="inner")
        self._df_records["record"] = self._df_records["exam_id"].astype(str)
        self._subject_records = self._df_records.groupby("patient_id")["record"].apply(sorted).to_dict()
        self._df_records.set_index("record", inplace=True)
        self._all_records = self._df_records.index.tolist()
        self._all_subjects = self._df_records.patient_id.unique().tolist()
        self._df_chagas = self._df_chagas[self._df_chagas.exam_id.isin(self._df_records.exam_id)]
        self._df_chagas["record"] = self._df_chagas["exam_id"].astype(str)
        self._df_chagas.set_index("record", inplace=True)

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
        if fs is not None:
            data = wfdb.processing.resample_sig(data, self.fs, fs)
        if data_format.lower() in ["channel_first", "lead_first"]:
            data = data.T
        if return_fs:
            return data, fs
        return data

    def load_ann(self, rec: Union[str, int]) -> List[str]:
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

        Returns
        -------
        ann : list of str
            List of the arrhythmia annotations.

        """
        if isinstance(rec, int):
            rec = self[rec]
        ann = self._df_records.loc[rec, self.__label_cols__].to_dict()
        ann = [k for k, v in ann.items() if v]
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
        chagas_ann = f"Chagas: {'True' if self.load_chagas_ann(rec) else 'False'}"
        diag_ann = ",".join(self.load_ann(rec))
        if diag_ann == "":
            diag_ann = "None"
        bin_ann = "Normal" if self.load_binary_ann(rec) else "Abnormal"

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
            axes[idx].plot([], [], " ", label=f"Exam ID - {rec}, Patient ID - {dem_row.patient_id}")
            axes[idx].plot([], [], " ", label=f"Age - {dem_row.age}, Sex - {dem_row.sex}")
            axes[idx].plot([], [], " ", label=f"Diagnosis - {bin_ann} - {diag_ann}")
            axes[idx].plot([], [], " ", label=f"{chagas_ann}")
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
        files = {f"exams_part{i}": f"{self.__dl_base_url__}exams_part{i}.zip?download=1" for i in range(18)}
        files.update(
            {
                "labels": f"{self.__dl_base_url__}exams.csv?download=1",
                "chagas_labels": "https://moody-challenge.physionet.org/2025/data/code15_chagas_labels.zip",
            }
        )
        return files

    def download(self, files: Optional[Union[str, Sequence[str]]]) -> None:
        """Download the database files.

        Parameters
        ----------
        files : str or list of str, optional
            The files to download.
            If not specified, download all files.
            The available files are:
                - "exams_part{i}" for i in range(18)
                - "labels"
                - "chagas_labels"

        """
        if files is None:
            files = list(self.url.keys())
        elif isinstance(files, str):
            files = [files]

        for file in files:
            if file not in self.url:
                raise ValueError(f"Unknown file: {file}")

        for file in files:
            http_get(self.url[file], self.db_dir)

    @property
    def database_info(self) -> DataBaseInfo:
        return _CODE15_INFO

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
        return CODE15.convert_to_wfdb_format(
            signal_files=self._h5_data_files,
            df_demographics=self._df_records,
            df_chagas=self._df_chagas,
            output_path=self.wfdb_data_dir,
            overwrite=overwrite,
        )

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
                            physical_signals = np.trim_zeros(physical_signals, trim="fb", axis=0)
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

                        # Add the exam ID, the patient ID, age, sex, and the Chagas label.
                        patient_id = exam_id_to_demographics[exam_id]["patient_id"]
                        age = exam_id_to_demographics[exam_id]["age"]
                        sex = exam_id_to_demographics[exam_id]["sex"]
                        chagas = exam_id_to_chagas[exam_id]
                        comments = [
                            f"Exam ID: {exam_id}",
                            f"Patient ID: {patient_id}",
                            f"Age: {age}",
                            f"Sex: {sex}",
                            f"Chagas label: {chagas}",
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

                        if signal_format == "mat":
                            convert_dat_to_mat(record, write_dir=str(output_path))

                        # Recompute the checksums for the checksum due to an error in the Python WFDB library.
                        checksums = np.sum(digital_signals, axis=0, dtype=np.int16)
                        fix_checksums(str(output_path / record), checksums)

        return excep_list

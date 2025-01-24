"""
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import wfdb
from torch_ecg.cfg import CFG
from torch_ecg.databases.base import DataBaseInfo, _DataBase
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
    1. The database contains 345,779 exams from 233,770 patients, obtained through stratified sampling from the CODE dataset ( 15% of the patients).
    2. The "exams.csv" file contains the labels and demographic information of the patients with the following columns:
        - "exam_id": id used for identifying the exam;
        - "age": patient age in years at the moment of the exam;
        - "is_male": true if the patient is male;
        - "nn_predicted_age": age predicted by a neural network to the patient. As described in the paper "Deep neural network estimated electrocardiographic-age as a mortality predictor" bellow.
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
        self.wfdb_format_dir = Path(kwargs.pop("wfdb_format_dir", "wfdb_format_files"))
        self.__config = CFG(BaseCfg.copy())
        self.__config.update(kwargs)

        self.data_ext = "hdf5"
        self.ann_ext = self.__label_file__
        self.chagas_ann_ext = "code15_chagas_labels.csv"
        self.fs = 400

        self._h5_data_files = None
        self._df_records = None
        self._df_metadata = None
        self._df_chagas = None
        self._all_records = None
        self._all_subjects = None
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

        self._df_metadata = pd.read_csv(self.db_dir / self.__label_file__)
        self._df_metadata["sex"] = self._df_metadata["is_male"].map({True: "Male", False: "Female"})
        self._df_chagas = pd.read_csv(self.db_dir / self.__chagas_label_file__)
        self._all_records = list(
            set(self._df_metadata.exam_id.unique().tolist()).intersection(self._df_chagas.exam_id.unique().tolist())
        )
        self._df_metadata = self._df_metadata[self._df_metadata.exam_id.isin(self._all_records)]
        self._df_chagas = self._df_chagas[self._df_chagas.exam_id.isin(self._all_records)]
        self._all_subjects = self._df_metadata.patient_id.unique().tolist()

        if not self.wfdb_format_dir.is_absolute():
            self.wfdb_format_dir = self.db_dir / self.wfdb_format_dir
        self.wfdb_format_dir.mkdir(parents=True, exist_ok=True)

    def load_data(
        self,
    ) -> None:
        """Load the data from the database."""
        raise NotImplementedError

    def load_ann(
        self,
    ) -> None:
        """Load the annotations from the database."""
        raise

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
            df_demographics=self._df_metadata,
            df_chagas=self._df_chagas,
            output_path=self.wfdb_format_dir,
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
                    exam_ids = list(h5_sig_file["exam_id"])
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

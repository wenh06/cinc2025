#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Union

import humanize
import numpy as np
import torch
import torch.nn as nn
import wfdb
from torch.nn.parallel import DataParallel as DP
from torch_ecg._preprocessors import PreprocManager
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import str2bool

from cfg import ModelCfg, TrainCfg
from const import MODEL_CACHE_DIR, REMOTE_MODELS
from data_reader import CINC2025
from dataset import CINC2025Dataset
from helper_code import find_records, load_age, load_sex
from models import CRNN_CINC2025, FM_CINC2025
from trainer import CINC2025Trainer
from utils.misc import remove_spikes_naive, to_dtype

################################################################################
# environment variables

# os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_CACHE_DIR)
# os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR)
# os.environ["HF_HOME"] = str(Path(MODEL_CACHE_DIR).parent)

try:
    TEST_FLAG = os.environ.get("CINC2025_REVENGER_TEST", False)
    TEST_FLAG = str2bool(TEST_FLAG)
except Exception:
    TEST_FLAG = False

if TEST_FLAG:
    print("Running in test mode.")
else:
    print("Running in submission mode.")

################################################################################


################################################################################
# NOTE: configurable options

SubmissionCfg = CFG()
SubmissionCfg.remote_model = None  # "crnn-resnet_nature_comm_bottle_neck-none-se"
# SubmissionCfg.remote_model = "crnn-resnet_nature_comm_bottle_neck-none-se"  # NOTE: for testing
SubmissionCfg.model_cls = FM_CINC2025  # CRNN_CINC2025
SubmissionCfg.final_model_name = "final_model.pth"
SubmissionCfg.use_dbs = ["CODE-15%", "SaMi-Trop", "PTB-XL"]

################################################################################


################################################################################
# NOTE: constants

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32

CINC2025.__DEBUG__ = False
CINC2025Dataset.__DEBUG__ = False
CRNN_CINC2025.__DEBUG__ = False
CINC2025Trainer.__DEBUG__ = False

print(f"Running on {DEVICE}, using data type {DTYPE}.")

################################################################################


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.


def train_model(
    data_folder: Union[str, bytes, os.PathLike], model_folder: Union[str, bytes, os.PathLike], verbose: bool = True
) -> None:
    """Train the models.

    Parameters
    ----------
    data_folder : `path_like`
        The path to the folder containing the training data.
    model_folder : `path_like`
        The path to the folder where the model will be saved.
    verbose : bool
        Whether to display progress information.

    Returns
    -------
    None

    """
    print("\n" + "*" * 100)
    msg = "   CinC2025 challenge training entry starts   ".center(100, "#")
    print(msg)
    print("*" * 100 + "\n")

    # Find the data files.
    if verbose:
        print("Finding the Challenge data...")

    # the entry will use WFDB format data
    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError("No data were provided.")
    else:
        print(f"Found {num_records} records.")

    # override the default data folder
    # if TEST_FLAG:
    #     data_folder = TEST_DATA_CACHE_DIR

    # raise error only when testing in GitHub Actions;
    # in other cases (submissions), errors are caught and printed,
    # and workarounds are used to continue the training
    # raise_error = TEST_FLAG
    raise_error = True  # early stage, always raise error
    if raise_error:
        print("Training in test mode. Any error will raise an exception.")
    else:
        print("Training in submission mode. Errors will be caught and printed.")
        print("Workarounds will be used to continue the training.")

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    model_folder = Path(model_folder).expanduser().resolve()
    data_folder = Path(data_folder).expanduser().resolve()
    (Path(model_folder) / "working_dir").mkdir(parents=True, exist_ok=True)

    # Train the models.
    if verbose:
        print("Training the model on the data...")

    ###############################################################################
    # Train the model.
    ###############################################################################

    start_time = datetime.now()

    model = None
    model_config = None
    if SubmissionCfg.remote_model is not None:
        # fine-tune the remote model
        model, train_config = SubmissionCfg.model_cls.from_checkpoint(
            Path(MODEL_CACHE_DIR) / REMOTE_MODELS[SubmissionCfg.remote_model]["filename"],
            device=DEVICE,
        )
        model_config = model.config
        # if torch.cuda.device_count() > 1:
        #     model = DP(model)
        #     # model = DDP(model)
    else:
        # general configs and logger
        train_config = deepcopy(TrainCfg)
    train_config.extra_experiment = False

    # override the default directories
    train_config.db_dir = Path(data_folder).resolve().absolute()
    train_config.model_dir = Path(model_folder).resolve().absolute()
    train_config.working_dir = train_config.model_dir / "working_dir"
    train_config.working_dir.mkdir(parents=True, exist_ok=True)
    train_config.checkpoints = train_config.working_dir / "checkpoints"
    train_config.checkpoints.mkdir(parents=True, exist_ok=True)
    train_config.log_dir = train_config.working_dir / "log"
    train_config.log_dir.mkdir(parents=True, exist_ok=True)
    train_config.final_model_name = SubmissionCfg.final_model_name
    train_config.debug = False

    train_config.use_dbs = SubmissionCfg.use_dbs

    if TEST_FLAG:
        train_config.n_epochs = 2
        train_config.batch_size = 8
        train_config.log_step = 20
        train_config.early_stopping.patience = 20
    else:
        train_config.n_epochs = 30
        if SubmissionCfg.model_cls == CRNN_CINC2025:
            train_config.batch_size = 128  # 16G (Tesla T4)
        else:  # FM_CINC2025
            train_config.batch_size = 32  # 16G (Tesla T4)
        train_config.log_step = 100
        train_config.early_stopping.patience = int(train_config.n_epochs * 0.3)

    if SubmissionCfg.remote_model is None:
        model_config = deepcopy(ModelCfg)
        model_cls = SubmissionCfg.model_cls

        # one should setup the paths to the backbone if model_cls is FM_CINC2025
        if model_cls == FM_CINC2025:
            if model_config.fm.name.lower() == "st-mem":
                model_config.fm.backbone_cache_dir = Path(MODEL_CACHE_DIR) / "ST-MEM"
            else:  # hubert-ecg
                model_config.fm.backbone_cache_dir = Path(MODEL_CACHE_DIR) / "Edoardo-BS-hubert-ecg-base"
            train_config.fs = model_config.fm.fs[model_config.fm.name]
            train_config.resample.fs = model_config.fm.fs[model_config.fm.name]
            train_config.input_len = model_config.fm.input_len[model_config.fm.name]

        model = model_cls(config=model_config)
        # NOTE: DP models might have issues:
        # the `parameters` method might not work as expected and return empty generator

        # if torch.cuda.device_count() > 1:
        #     model = DP(model)
        #     # model = DDP(model)
        model.to(DEVICE)

    assert model is not None, "Model is not initialized."
    assert model_config is not None, "Model config is not initialized."

    if verbose:
        if isinstance(model, DP):
            print("model size:", model.module.module_size, model.module.module_size_)
            print("Using devices:", model.device_ids)
        else:
            print("model size:", model.module_size, model.module_size_)
            print("Using device:", model.device)

    reader_kwargs = {
        "db_dir": Path(data_folder).expanduser().resolve(),
        "working_dir": (Path(model_folder) / "working_dir"),
    }
    ds_train = CINC2025Dataset(train_config, training=True, lazy=True, **reader_kwargs)
    ds_val = CINC2025Dataset(train_config, training=False, lazy=True, **reader_kwargs)
    if verbose:
        print(f"train size: {len(ds_train)}, val size: {len(ds_val)}")

    if isinstance(model, FM_CINC2025):
        print("Using FM_CINC2025 model, adjusting fs and input_len")
        ds_train.reset_resample_fs(model_config.fm.fs[model_config.fm.name], reload=False)
        ds_train.reset_input_len(model_config.fm.input_len[model_config.fm.name], reload=False)
        ds_val.reset_resample_fs(model_config.fm.fs[model_config.fm.name], reload=False)
        ds_val.reset_input_len(model_config.fm.input_len[model_config.fm.name], reload=False)

    trainer = CINC2025Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=True,
    )
    if TEST_FLAG:
        # switch the dataloaders to make the test faster
        # the first dataloader is used for both training and evaluation
        # the second dataloader is used for validation only
        # trainer._setup_dataloaders(ds_val, ds_train)
        trainer._setup_dataloaders(ds_val, None)
    else:
        trainer._setup_dataloaders(ds_train, ds_val)

    best_state_dict = trainer.train()  # including saving model

    trainer.log_manager.flush()
    trainer.log_manager.close()

    del trainer
    del model
    del best_state_dict

    torch.cuda.empty_cache()

    elapsed_time = humanize.naturaldelta(datetime.now() - start_time)
    if verbose:
        print(f"Training completed in {elapsed_time}.")

    print("\n" + "*" * 100)
    msg = "   CinC2025 challenge training entry ends   ".center(100, "#")
    print(msg)


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(
    model_folder: Union[str, bytes, os.PathLike], verbose: bool = True
) -> Dict[str, Union[dict, nn.Module, PreprocManager]]:
    """Load the trained models.

    Parameters
    ----------
    model_folder : `path_like`
        The path to the folder containing the trained model.
    verbose : bool
        Whether to display progress information.

    Returns
    -------
    model : Dict[str, Union[dict, nn.Module, PreprocManager]]
        The trained model, its training configurations and the preprocessor manager
        inferred from the training configurations.

    """
    model_folder = Path(model_folder).expanduser().resolve()

    print("Loading the trained model...")

    model_cls = SubmissionCfg.model_cls
    model_path = Path(model_folder) / SubmissionCfg.final_model_name
    model, train_config = model_cls.from_checkpoint(model_path)
    model.to(DEVICE)
    if isinstance(model, CRNN_CINC2025):
        print("Using CRNN_CINC2025 model.")
        train_config.fs = model.config.fs
        train_config.resample.fs = model.config.fs
    elif isinstance(model, FM_CINC2025):
        print("Using FM_CINC2025 model.")
        train_config.fs = model.config.fs[model.config.name]
        train_config.resample.fs = model.config.fs[model.config.name]
    else:
        raise ValueError("Unsupported model class.")
    ppm_config = CFG(random=False)
    ppm_config.update(deepcopy(train_config))
    ppm = PreprocManager.from_config(ppm_config)

    print(f"Chagas classification model loaded from {str(model_path)}")

    return {"model": model, "train_config": train_config, "preprocessor": ppm}


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
@torch.no_grad()
def run_model(
    record: Union[str, bytes, os.PathLike], model: Dict[str, Union[dict, nn.Module, PreprocManager]], verbose: bool = True
) -> Tuple[int, float]:
    """Run the trained model on a record.

    Parameters
    ----------
    record : `path_like`
        The path to the record to process, without the file extension.
    model : Dict[str, Union[dict, nn.Module, PreprocManager]]
        The trained model, its training configurations and the preprocessor manager
        inferred from the training configurations.
    verbose : bool
        Whether to display progress information.

    Returns
    -------
    binary_output : int
        The binary output of the model.
    probability_output : float
        The probability output of the model.

    """
    start_time = datetime.now()

    # raise error only when testing in GitHub Actions;
    # in other cases (submissions), errors are caught and printed,
    # and workarounds are used to continue the model inference
    raise_error = TEST_FLAG
    # raise_error = True  # early stage, always raise error
    if raise_error:
        if verbose:
            print("Running the models in test mode. Any error will raise an exception.")
    else:
        if verbose:
            print("Running the models in submission mode. Errors will be caught and printed.")
            print("Workarounds will be used to continue the model inference.")

    wfdb_record = wfdb.rdrecord(record)
    signal = wfdb_record.p_signal
    sig_fs = wfdb_record.fs
    if signal.shape[1] == model["train_config"].n_leads:
        signal = signal.T  # to lead-first format
    signal = to_dtype(signal, DTYPE)
    signal = remove_spikes_naive(signal)
    signal, _ = model["preprocessor"](signal, sig_fs)
    if model["model"].config.get("dem_encoder", None) is not None:
        demographics = get_demographic_features(record, model["model"].config)
        output = model["model"].inference(signal, demographics, crop_infer=True, agg="max")
    else:
        output = model["model"].inference(signal, crop_infer=True, agg="max")
    binary_output = output.chagas[0]
    probability_output = output.chagas_prob[0][1].item()

    elapsed_time = humanize.naturaldelta(datetime.now() - start_time)

    if verbose:
        print(f"Inference pipeline completed in {elapsed_time}.")

    return binary_output, probability_output


def get_demographic_features(record, model_config: CFG) -> np.ndarray:
    """Extract demographic features from the WFDB record.

    Parameters
    ----------
    record : Union[str, bytes, os.PathLike]
        The path to the record to process, without the file extension.
    model_config : CFG
        The model configuration, which may contain information
        about the required demographic features and their processing.

    Returns
    -------
    demographics : np.ndarray
        The extracted demographic features, processed according to the model configuration.
    """
    default_age = 40
    default_sex = "Male"
    demographics = {
        "age": (load_age(record) or default_age) / model_config.age_scale,
        "sex": (load_sex(record) or default_sex).capitalize(),
    }
    if demographics["sex"] not in model_config.sex_mapping:
        demographics["sex"] = default_sex  # default to "Male" if sex is unrecognized
    demographics["sex"] = model_config.sex_mapping[demographics["sex"]]
    return to_dtype(np.array([demographics[feat] for feat in model_config.demographic_features]), DTYPE)

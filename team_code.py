#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Union

import humanize
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel as DP  # noqa: F401
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import str2bool

from cfg import BaseCfg, ModelCfg, TrainCfg  # noqa: F401
from const import LABEL_CACHE_DIR, MODEL_CACHE_DIR, TEST_DATA_CACHE_DIR  # noqa: F401
from helper_code import find_records

################################################################################
# environment variables

os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HOME"] = str(Path(MODEL_CACHE_DIR).parent)

try:
    TEST_FLAG = os.environ.get("CINC2025_REVENGER_TEST", False)
    TEST_FLAG = str2bool(TEST_FLAG)
except Exception:
    TEST_FLAG = False

################################################################################


################################################################################
# NOTE: configurable options

SubmissionCfg = CFG()

################################################################################


################################################################################
# NOTE: constants

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError("No data were provided.")
    else:
        print(f"Found {num_records} records.")

    # override the default data folder
    if TEST_FLAG:
        data_folder = TEST_DATA_CACHE_DIR

    # raise error only when testing in GitHub Actions;
    # in other cases (submissions), errors are caught and printed,
    # and workarounds are used to continue the training
    raise_error = TEST_FLAG
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

    # TODO: Implement your training code here.
    raise NotImplementedError("The train_model function is not implemented yet.")

    print("\n" + "*" * 100)
    msg = "   CinC2025 challenge training entry ends   ".center(100, "#")
    print(msg)


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder: Union[str, bytes, os.PathLike], verbose: bool = True) -> Dict[str, Union[dict, nn.Module]]:
    """Load the trained models.

    Parameters
    ----------
    model_folder : `path_like`
        The path to the folder containing the trained model.
    verbose : bool
        Whether to display progress information.

    Returns
    -------
    model : Dict[str, Union[dict, nn.Module]]
        The trained model and its training configurations.

    """
    model_folder = Path(model_folder).expanduser().resolve()

    print("Loading the trained models...")

    raise NotImplementedError("The load_model function is not implemented yet.")


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
@torch.no_grad()
def run_model(
    record: Union[str, bytes, os.PathLike], model: Dict[str, Union[dict, nn.Module]], verbose: bool = True
) -> Tuple[int, float]:
    """Run the trained model on a record.

    Parameters
    ----------
    record : `path_like`
        The path to the record to process, without the file extension.
    model : Dict[str, Union[dict, nn.Module]]
        The trained model and its training configurations.
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
    if raise_error:
        print("Running the models in test mode. Any error will raise an exception.")
    else:
        print("Running the models in submission mode. Errors will be caught and printed.")
        print("Workarounds will be used to continue the model inference.")

    # TODO: Implement your inference code here.

    raise NotImplementedError("The run_model function is not implemented yet.")

    elapsed_time = humanize.naturaldelta(datetime.now() - start_time)

    print(f"Inference pipeline completed in {elapsed_time}.")

    # return binary_output, probability_output

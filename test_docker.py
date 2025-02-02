"""
"""

import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch_ecg.utils.misc import str2bool

from cfg import _BASE_DIR, ModelCfg, TrainCfg
from data_reader import CODE15
from utils.misc import func_indicator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32


tmp_data_dir = Path(os.environ.get("mount_data_dir", _BASE_DIR / "tmp" / "CINC2025")).resolve()
print(f"tmp_data_dir: {str(tmp_data_dir)}")
# create the data directory if it does not exist
tmp_data_dir.mkdir(parents=True, exist_ok=True)
# list files and folders in the data directory
print(os.listdir(tmp_data_dir))

dr = CODE15(tmp_data_dir)
# downloading is done outside the docker container
# and the data folder is mounted to the docker container as read-only
# dr.download()
# dr._ls_rec()


tmp_model_dir = Path(os.environ.get("revenger_model_dir", TrainCfg.model_dir)).resolve()

tmp_output_dir = Path(os.environ.get("revenger_output_dir", _BASE_DIR / "tmp" / "output")).resolve()


def echo_write_permission(folder: Union[str, Path]) -> None:
    is_writeable = "is writable" if os.access(str(folder), os.W_OK) else "is not writable"
    print(f"{str(folder)} {is_writeable}")


echo_write_permission(tmp_data_dir)
echo_write_permission(tmp_model_dir)
echo_write_permission(tmp_output_dir)


@func_indicator("testing dataset")
def test_dataset() -> None:
    """Test the dataset."""

    raise NotImplementedError("The dataset test is not implemented yet.")

    print("dataset test passed")


@func_indicator("testing models")
def test_models() -> None:
    """Test the models."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)
    echo_write_permission(tmp_output_dir)

    raise NotImplementedError("The models test is not implemented yet.")

    print("models test passed")


@func_indicator("testing challenge metrics")
def test_challenge_metrics() -> None:
    """Test the challenge metrics."""

    raise NotImplementedError("The challenge metrics test is not implemented yet.")

    print("challenge metrics test passed")


@func_indicator("testing trainer")
def test_trainer() -> None:
    """Test the trainer."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)
    echo_write_permission(tmp_output_dir)

    raise NotImplementedError("The trainer test is not implemented yet.")

    print("trainer test passed")


@func_indicator("testing challenge entry")
def test_entry() -> None:
    """Test Challenge entry."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)

    # run the model training function (script)
    print("   Run model training function   ".center(80, "#"))

    raise NotImplementedError("The entry test is not implemented yet.")

    print("Entry test passed")


test_team_code = test_entry  # alias


if __name__ == "__main__":
    TEST_FLAG = os.environ.get("CINC2025_REVENGER_TEST", False)
    TEST_FLAG = str2bool(TEST_FLAG)
    if not TEST_FLAG:
        # raise RuntimeError(
        #     "please set CINC2025_REVENGER_TEST to true (1, y, yes, true, etc.) to run the test"
        # )
        print("Test is skipped.")
        print("Please set CINC2025_REVENGER_TEST to true (1, y, yes, true, etc.) to run the test:")
        print("CINC2025_REVENGER_TEST=1 python test_docker.py")
        exit(0)

    print("#" * 80)
    print("testing team code")
    print("#" * 80)
    print(f"tmp_data_dir: {str(tmp_data_dir)}")
    print(f"tmp_model_dir: {str(tmp_model_dir)}")
    print(f"tmp_output_dir: {str(tmp_output_dir)}")
    print("#" * 80)

    # test_dataset()  # not implemented
    # test_models()  # not implemented
    # test_challenge_metrics()  # not implemented
    # test_trainer()  # not implemented
    # test_entry()  # not implemented
    # set_entry_test_flag(False)

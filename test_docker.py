"""
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_ecg.utils.misc import str2bool
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

from cfg import _BASE_DIR, ModelCfg, TrainCfg
from dataset import CINC2025Dataset
from models import CRNN_CINC2025
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
print("data directory signal files count:", len(list(tmp_data_dir.glob("*.hea"))))

# downloading is done outside the docker container
# and the data folder is mounted to the docker container as read-only
# dr = CODE15(tmp_data_dir)
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

    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = tmp_data_dir
    ds_config.working_dir = tmp_model_dir / "working_dir"
    ds_config.working_dir.mkdir(parents=True, exist_ok=True)

    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)
    echo_write_permission(ds_config.working_dir)

    reader_kwargs = {
        # "label_file": Path(LABEL_CACHE_DIR) / CODE15.__name__ / CODE15.__label_file__,
        # "chagas_label_file": Path(LABEL_CACHE_DIR) / CODE15.__name__ / CODE15.__chagas_label_file__,
    }

    ds_train = CINC2025Dataset(ds_config, training=True, lazy=True, **reader_kwargs)
    ds_val = CINC2025Dataset(ds_config, training=False, lazy=True, **reader_kwargs)

    print(f"{len(ds_train) = }, {len(ds_val) = }")
    assert len(ds_train) > 0, f"{len(ds_train) = }"
    assert len(ds_val) > 0, f"{len(ds_val) = }"

    # int indexing
    data = ds_val[0]
    assert isinstance(data, dict), f"{type(data) = }"
    assert "chagas" in data and "signals" in data, f"{data.keys() = }"
    assert set(data.keys()) <= ds_val.data_fields, f"{set(data.keys()) = }, {ds_val.data_fields = }"
    assert isinstance(data["signals"], np.ndarray), f"{type(data['signals']) = }"
    assert data["signals"].shape == (ds_val.config.n_leads, ds_val.config.input_len), f"{data['signals'].shape = }"
    assert isinstance(data["chagas"], int), f"{type(data['chagas']) = }"

    if "arr_diag" in data:
        assert isinstance(data["arr_diag"], np.ndarray)
        assert data["arr_diag"].shape == (len(ds_val.config.arr_diag_classes),)

    # slice indexing, everything casted to torch.Tensor
    batch_size = 4
    data = ds_val[:batch_size]
    assert isinstance(data, dict)
    assert "chagas" in data and "signals" in data
    assert set(data.keys()) <= ds_val.data_fields
    assert isinstance(data["signals"], torch.Tensor)
    assert data["signals"].shape == (batch_size, ds_val.config.n_leads, ds_val.config.input_len)
    assert isinstance(data["chagas"], torch.Tensor)
    assert data["chagas"].shape == (batch_size,)
    if "arr_diag" in data:
        assert isinstance(data["arr_diag"], torch.Tensor)
        assert data["arr_diag"].shape == (batch_size, len(ds_val.config.arr_diag_classes))

    print("dataset test passed")


@func_indicator("testing models")
def test_models() -> None:
    """Test the models."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)
    echo_write_permission(tmp_output_dir)

    model = CRNN_CINC2025(ModelCfg)
    model.to(DEVICE)

    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = tmp_data_dir
    ds_config.working_dir = tmp_model_dir / "working_dir"
    ds_config.working_dir.mkdir(parents=True, exist_ok=True)

    ds_val = CINC2025Dataset(ds_config, training=False, lazy=True)
    # ds_val._load_all_data()
    dl = DataLoader(
        dataset=ds_val,
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    for idx, input_tensors in enumerate(dl):
        if idx == 0:
            inference_output = model.inference(input_tensors["signals"])
            print(f"   {idx = }   ".center(80, "#"))
            print(f"{inference_output = }")
        elif idx == 1:
            forward_output = model.forward(input_tensors)
            print(f"   {idx = }   ".center(80, "#"))
            print(f"{forward_output = }")
        else:
            break

    # TODO: test classmethod "from_checkpoint"

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
        print("Other environment variables:")
        print("mount_data_dir: the data directory")
        # TODO: add more environment variables here
        exit(0)

    print("#" * 80)
    print("testing team code")
    print("#" * 80)
    print(f"tmp_data_dir: {str(tmp_data_dir)}")
    print(f"tmp_model_dir: {str(tmp_model_dir)}")
    print(f"tmp_output_dir: {str(tmp_output_dir)}")
    print("#" * 80)

    test_dataset()  # passed
    test_models()  # passed
    # test_challenge_metrics()  # not implemented
    # test_trainer()  # not implemented
    # test_entry()  # not implemented
    # set_entry_test_flag(False)

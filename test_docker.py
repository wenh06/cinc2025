"""
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import str2bool
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

from cfg import _BASE_DIR, ModelCfg, TrainCfg
from dataset import CINC2025Dataset
from evaluate_model import run as model_evaluator_func
from models import CRNN_CINC2025
from outputs import CINC2025Outputs
from run_model import run as model_runner_func
from team_code import train_models
from trainer import CINC2025Trainer
from utils.misc import func_indicator
from utils.scoring_metrics import compute_challenge_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32


tmp_data_dir = Path(os.environ.get("mount_data_dir", _BASE_DIR / "tmp" / "CINC2025")).resolve()
print(f"tmp_data_dir: {str(tmp_data_dir)}")
tmp_data_dir.mkdir(parents=True, exist_ok=True)
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
    # if torch.cuda.device_count() > 1:
    #     model = DP(model)
    #     # model = DDP(model)
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

    outputs = [
        CINC2025Outputs(
            chagas=[False, False, False, False],
            chagas_logits=torch.Tensor([[0.0090, -0.0561], [-0.0070, -0.0512], [-0.0080, -0.0487], [-0.0087, -0.0480]]),
            chagas_prob=np.array(
                [[0.51627094, 0.48372903], [0.5110359, 0.48896402], [0.510169, 0.48983097], [0.5098051, 0.49019495]]
            ),
        ),
        CINC2025Outputs(
            chagas=[False, False, False],
            chagas_logits=np.array([[0.00514201, -0.07293116], [0.00042483, -0.05155159], [0.0076459, -0.05799032]]),
            chagas_prob=np.array([[0.5195084, 0.48049164], [0.5129912, 0.48700884], [0.5164032, 0.48359686]]),
        ),
    ]

    labels = [
        {"chagas": [False, False, False, False]},
        {"chagas": [False, False, True]},
    ]

    metrics = compute_challenge_metrics(labels, outputs)

    print(f"{metrics = }")

    assert set(metrics.keys()) == {"challenge_score", "chagas_auroc", "chagas_auprc", "chagas_accuracy", "chagas_f_measure"}

    for k, v in metrics.items():
        assert isinstance(v, float), f"{k = }, {v = }"

    print("challenge metrics test passed")


@func_indicator("testing trainer")
def test_trainer() -> None:
    """Test the trainer."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)
    echo_write_permission(tmp_output_dir)

    train_config = deepcopy(TrainCfg)
    train_config.db_dir = tmp_data_dir
    # train_config.model_dir = model_folder
    # train_config.final_model_filename = "final_model.pth.tar"
    train_config.debug = True
    train_config.working_dir = tmp_model_dir / "working_dir"
    train_config.working_dir.mkdir(parents=True, exist_ok=True)

    train_config.n_epochs = 1
    train_config.batch_size = 4  # 16G (Tesla T4)
    # train_config.log_step = 20
    # # train_config.max_lr = 1.5e-3
    # train_config.early_stopping.patience = 20

    model_config = deepcopy(ModelCfg)
    model = CRNN_CINC2025(config=model_config)
    # if torch.cuda.device_count() > 1:
    #     model = DP(model)
    #     # model = DDP(model)
    model = model.to(device=DEVICE)
    if isinstance(model, DP):
        print("model size:", model.module.module_size, model.module.module_size_)
    else:
        print("model size:", model.module_size, model.module_size_)

    ds_train = CINC2025Dataset(train_config, training=True, lazy=True)
    ds_val = CINC2025Dataset(train_config, training=False, lazy=True)
    print(f"train size: {len(ds_train)}, val size: {len(ds_val)}")

    trainer = CINC2025Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=True,
    )
    # trainer._setup_dataloaders(ds_train, ds_val)
    # switch the dataloaders to make the test faster
    # the first dataloader is used for both training and evaluation
    # the second dataloader is used for validation only
    trainer._setup_dataloaders(ds_val, ds_train)

    best_model_state_dict = trainer.train()

    print(f"Saved models: {list(Path(train_config.model_dir).iterdir())}")

    print("trainer test passed")


@func_indicator("testing challenge entry")
def test_entry() -> None:
    """Test Challenge entry."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)

    # run the model training function (script)
    print("   Run model training function   ".center(80, "#"))
    data_folder = tmp_data_dir

    train_models(str(data_folder), str(tmp_model_dir), verbose=2)

    # run the model inference function (script)
    output_dir = tmp_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("   Run model   ".center(80, "#"))

    model_runner_args = CFG(
        data_folder=str(data_folder),
        model_folder=str(tmp_model_dir),
        output_folder=str(output_dir),
        allow_failures=False,
        verbose=2,
    )
    model_runner_func(model_runner_args)

    print("   Evaluate model   ".center(80, "#"))

    model_evaluator_args = CFG(
        folder_ref=str(data_folder),
        folder_est=str(output_dir),
        score_file=str(Path(output_dir) / "score.txt"),
    )
    model_evaluator_func(model_evaluator_args)  # metrics are printed

    print("Content of saved score file:")
    print(Path(output_dir / "score.txt").read_text())

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
        print("mount_data_dir: the data directory, usage:")
        print("CINC2025_REVENGER_TEST=1 mount_data_dir=/path/to/data python test_docker.py")
        # TODO: add more environment variables here
        exit(0)

    print("#" * 80)
    print("testing team code")
    print("#" * 80)
    print(f"tmp_data_dir: {str(tmp_data_dir)}")
    print(f"tmp_model_dir: {str(tmp_model_dir)}")
    print(f"tmp_output_dir: {str(tmp_output_dir)}")
    print("#" * 80)

    # test_dataset()  # passed
    # test_models()  # passed
    # test_challenge_metrics()  # passed
    # test_trainer()  # passed
    test_entry()  # implemented, under testing
    # set_entry_test_flag(False)

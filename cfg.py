"""
Configurations for models, training, etc., as well as some constants.
"""

import pathlib
from copy import deepcopy

import numpy as np
import torch
from torch_ecg.cfg import CFG
from torch_ecg.model_configs import linear

__all__ = [
    "BaseCfg",
    "TrainCfg",
    "ModelCfg",
]


_BASE_DIR = pathlib.Path(__file__).absolute().parent


###############################################################################
# Base Configs,
# including path, data type, classes, etc.
###############################################################################

BaseCfg = CFG()
BaseCfg.db_dir = None
BaseCfg.working_dir = None
BaseCfg.project_dir = _BASE_DIR
BaseCfg.log_dir = _BASE_DIR / "log"
BaseCfg.model_dir = _BASE_DIR / "saved_models"
BaseCfg.checkpoints = _BASE_DIR / "checkpoints"
BaseCfg.log_dir.mkdir(exist_ok=True)
BaseCfg.model_dir.mkdir(exist_ok=True)

BaseCfg.torch_dtype = torch.float32  # "double"
BaseCfg.np_dtype = np.float32
BaseCfg.fs = 400
BaseCfg.n_leads = 12
# arrhythmia diagnosis classes
BaseCfg.arr_diag_classes = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"] + ["NORM", "OTHER"]
BaseCfg.arr_diag_class_map = {c: i for i, c in enumerate(BaseCfg.arr_diag_classes)}


###############################################################################
# training configurations for machine learning and deep learning
###############################################################################

TrainCfg = deepcopy(BaseCfg)

TrainCfg.checkpoints = BaseCfg.checkpoints
TrainCfg.checkpoints.mkdir(exist_ok=True)
TrainCfg.aux_tasks = CFG(
    binary_classification=CFG(
        # merged with arrhythmia_classification by adding
        # 2 classes: "NORM" and "OTHER"
        enabled=False,
        loss="AsymmetricLoss",
        loss_weight=1.0,
        loss_kw={},  # keyword arguments for the loss function
    ),
    arrhythmia_classification=CFG(
        enabled=True,
        loss="AsymmetricLoss",
        loss_weight=1.0,
        loss_kw={},  # keyword arguments for the loss function
    ),
)

TrainCfg.train_ratio = 0.9
TrainCfg.input_len = 4096  # approximately 10s


###############################################################################
# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted
###############################################################################


ModelCfg = CFG()
# ModelCfg.num_leads = BaseCfg.num_leads
ModelCfg.torch_dtype = BaseCfg.torch_dtype
ModelCfg.model_dir = BaseCfg.model_dir
ModelCfg.checkpoints = BaseCfg.checkpoints

ModelCfg.arr_diag_head = deepcopy(linear)
ModelCfg.arr_diag_head.update(
    CFG(
        enabled=True,
        out_channels=[512] + [len(BaseCfg.arr_diag_classes)],
    )
)

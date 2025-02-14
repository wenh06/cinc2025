"""
Configurations for models, training, etc., as well as some constants.
"""

import pathlib
from copy import deepcopy

import numpy as np
import torch
from torch_ecg.cfg import CFG
from torch_ecg.model_configs import ECG_CRNN_CONFIG, linear  # noqa: F401
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths

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
BaseCfg.chagas_classes = [0, 1]
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
        include=False,
        loss="AsymmetricLoss",
        loss_weight=1.0,
        loss_kw={},  # keyword arguments for the loss function
    ),
    arrhythmia_classification=CFG(
        include=False,
        loss="AsymmetricLoss",
        loss_weight=1.0,
        loss_kw={},  # keyword arguments for the loss function
    ),
)

# preprocessing configurations
TrainCfg.resample = CFG(
    fs=TrainCfg.fs,
)
# TrainCfg.baseline_remove = {}  # default values
TrainCfg.bandpass = CFG(
    filter_type="butter",
)

# augmentations configurations
TrainCfg.label_smooth = 0.05
# TrainCfg.random_masking = False
# TrainCfg.stretch_compress = False  # stretch or compress in time axis
# TrainCfg.mixup = CFG(
#     prob=0.6,
#     alpha=0.3,
# )


TrainCfg.criterion = "AsymmetricLoss"  # "FocalLoss", "BCEWithLogitsLoss"
TrainCfg.criterion_kw = {}  # keyword arguments for the criterion

TrainCfg.train_ratio = 0.8
TrainCfg.input_len = 4096  # approximately 10s


###############################################################################
# configurations for building deep learning models
###############################################################################


_BASE_MODEL_CONFIG = CFG()
_BASE_MODEL_CONFIG.torch_dtype = BaseCfg.torch_dtype
_BASE_MODEL_CONFIG.n_leads = BaseCfg.n_leads
_BASE_MODEL_CONFIG.torch_dtype = BaseCfg.torch_dtype
_BASE_MODEL_CONFIG.fs = BaseCfg.fs
_BASE_MODEL_CONFIG.chagas_classes = BaseCfg.chagas_classes.copy()


ModelCfg = deepcopy(_BASE_MODEL_CONFIG)

# adjust filter lengths, > 1 for enlarging, < 1 for shrinking
cnn_filter_length_ratio = 1.0

ModelCfg.crnn = deepcopy(ECG_CRNN_CONFIG)
ModelCfg.crnn = adjust_cnn_filter_lengths(ModelCfg.crnn, int(ModelCfg.fs * cnn_filter_length_ratio))

# change ModelCfg.crnn.cnn.name, ModelCfg.crnn.rnn.name, ModelCfg.crnn.attn.name
# for different models,
# check `ModelCfg.cnn.keys()`, `ModelCfg.rnn.keys()`, `ModelCfg.attn.keys()` for available models
# ModelCfg.crnn.cnn.name = "tresnetN"

# ModelCfg.arr_diag_head = deepcopy(linear)
# ModelCfg.arr_diag_head.update(
#     CFG(
#         include=True,
#         out_channels=[512] + [len(BaseCfg.arr_diag_classes)],
#     )
# )

ModelCfg.criterion = TrainCfg.criterion
ModelCfg.criterion_kw = TrainCfg.criterion_kw.copy()

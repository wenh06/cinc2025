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

from const import SampleType

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
TrainCfg.normalize = CFG(  # None or False for no normalization
    method="z-score",
    mean=0.0,
    std=1.0,
)

# augmentations configurations
# TrainCfg.label_smooth = CFG(
#     prob=0.8,
#     smoothing=0.1,
# )
TrainCfg.label_smooth = CFG(
    prob=0.8,
    smoothing={
        str(SampleType.NEGATIVE_SAMPLE.value): 0.0,  # negative samples -> prob vec [1.0, 0.0]
        str(SampleType.SELF_REPORTED_POSITIVE_SAMPLE.value): 0.3,  # self-reported positive samples -> prob vec [0.15, 0.85]
        str(SampleType.SELF_REPORTED_UNCERTAIN_SAMPLE.value): 0.2,  # self-reported uncertain samples -> prob vec [0.9, 0.1]
        str(SampleType.DOCTOR_CONFIRMED_POSITIVE_SAMPLE.value): 0.0,  # doctor-confirmed positive samples -> prob vec [0.0, 1.0]
    },
)
# TrainCfg.random_masking = False
# TrainCfg.stretch_compress = False  # stretch or compress in time axis
# TrainCfg.mixup = CFG(
#     prob=0.6,
#     alpha=0.3,
# )

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 30
# TODO: automatic adjust batch size according to GPU capacity
# https://stackoverflow.com/questions/45132809/how-to-select-batch-size-automatically-to-fit-gpu
# GPU memory limit of the Challenge is 64GB
TrainCfg.batch_size = 32  # 64, 128, 256, should be adjusted according to model choices

# configs of optimizers and lr_schedulers
TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
TrainCfg.betas = (0.9, 0.999)  # default values for corresponding PyTorch optimizers
TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

# TrainCfg.learning_rate = 1e-4  # 5e-4, 1e-3
TrainCfg.learning_rate = {
    "backbone": 5e-5,
    "head": 3e-4,
}
TrainCfg.lr = TrainCfg.learning_rate


TrainCfg.lr_scheduler = "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1
TrainCfg.max_lr = {
    "backbone": 2e-4,
    "head": 1e-3,
}  # for "one_cycle", "cosine_warmup" schedulers, to adjust via expriments
TrainCfg.warmup_ratio = 0.1  # for "cosine_warmup" and "burn_in" schedulers

# number of epochs to freeze backbone at the beginning of training
# 0 for no freezing, -1 for freezing all epochs
TrainCfg.freeze_backbone_epochs = 3

TrainCfg.upsample_positive_chagas = {
    "CODE-15%": 3,
    "SaMi-Trop": 7,
}  # rate of upsampling positive samples, 1 for no upsampling
TrainCfg.use_dbs = ["CODE-15%", "SaMi-Trop", "PTB-XL"]  # CODE-15%, SaMi-Trop, PTB-XL

# configs of callbacks, including early stopping, checkpoint, etc.
TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
TrainCfg.early_stopping.patience = TrainCfg.n_epochs // 3
TrainCfg.keep_checkpoint_max = 10

# configs of loss function
# TrainCfg.loss = "AsymmetricLoss"  # "FocalLoss", "BCEWithLogitsLoss"
# TrainCfg.loss_kw = CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp")
TrainCfg.flooding_level = 0.0  # flooding performed if positive,

# configs of logging
TrainCfg.log_step = 100
# TrainCfg.eval_every = 20

# TrainCfg.criterion = "AsymmetricLoss"  # "FocalLoss", "BCEWithLogitsLoss"
# TrainCfg.criterion_kw = {
#     "gamma_neg": 4,
#     "gamma_pos": 1,
#     "prob_margin": 0.05,
# }  # keyword arguments for the criterion
TrainCfg.criterion = "ChagasLoss"  # custom loss
TrainCfg.criterion_kw = {
    "positive_weight_factor": 5.0,
}

TrainCfg.train_ratio = 0.8
TrainCfg.input_len = 4096  # approximately 10s
TrainCfg.min_len = 1200  # minimum length of the raw signal

TrainCfg.monitor = "challenge_score"
TrainCfg.final_model_name = None


###############################################################################
# configurations for building deep learning models
###############################################################################

_BASE_MODEL_CONFIG = CFG()
_BASE_MODEL_CONFIG.torch_dtype = BaseCfg.torch_dtype
_BASE_MODEL_CONFIG.n_leads = BaseCfg.n_leads
_BASE_MODEL_CONFIG.torch_dtype = BaseCfg.torch_dtype
_BASE_MODEL_CONFIG.fs = BaseCfg.fs
_BASE_MODEL_CONFIG.chagas_classes = BaseCfg.chagas_classes.copy()

_BASE_MODEL_CONFIG.criterion = TrainCfg.criterion
_BASE_MODEL_CONFIG.criterion_kw = TrainCfg.criterion_kw.copy()


ModelCfg = deepcopy(_BASE_MODEL_CONFIG)

# adjust filter lengths, > 1 for enlarging, < 1 for shrinking
cnn_filter_length_ratio = 1.0

ModelCfg.crnn = ECG_CRNN_CONFIG | _BASE_MODEL_CONFIG
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

ModelCfg.fm = deepcopy(_BASE_MODEL_CONFIG)
ModelCfg.fm.name = "st-mem"  # "st-mem", "hubert-ecg", etc.
ModelCfg.fm.fs = {
    "st-mem": 250,
    "hubert-ecg": 100,
}
ModelCfg.fm.input_len = {
    "st-mem": 75 * min(31, np.floor(4096 / 400 * 250 / 75).astype(int).item()),  # adjust to be multiple of 75
    "hubert-ecg": 4096 // 4,  # 4096 samples at 400Hz -> 1024 samples at 100Hz
}

ModelCfg.fm.freeze_backbone = False
ModelCfg.fm.dropout = 0.2
ModelCfg.fm.embed_dim = {
    "st-mem": 768,
    "hubert-ecg": 768,
}
ModelCfg.fm.backbone_cache_dir = None  # should be set before using the model

ModelCfg.fm.head = CFG(
    hidden_dim=256,
    num_layers=2,  # Linear -> ReLU -> Dropout -> Linear
)

ModelCfg.crnn.ranking = CFG(
    enable=False,
    type="adaptive",  # or "hinge", or "adaptive"
    weight=0.4,
    margin=0.1,  # logistic 0; hinge 0.5 or other values
)

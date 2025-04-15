"""Constants for the project."""

import os
from enum import Enum
from pathlib import Path

__all__ = [
    "PROJECT_DIR",
    "MODEL_CACHE_DIR",
    "DATA_CACHE_DIR",
    "LABEL_CACHE_DIR",
    "TEST_DATA_CACHE_DIR",
    "SampleType",
    "REMOTE_MODELS",
]


PROJECT_DIR = str(Path(__file__).resolve().parent)


MODEL_CACHE_DIR = str(
    Path(
        # ~/.cache/revenger_model_dir_cinc2025
        # /challenge/cache/revenger_model_dir
        os.environ.get("MODEL_CACHE_DIR", "~/.cache/cinc2025/revenger_model_dir")
    )
    .expanduser()
    .resolve()
)
Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)


DATA_CACHE_DIR = str(
    Path(
        # ~/.cache/revenger_data_dir_cinc2025
        # /challenge/cache/revenger_data_dir
        os.environ.get("DATA_CACHE_DIR", "~/.cache/cinc2025/revenger_data_dir")
    )
    .expanduser()
    .resolve()
)
Path(DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)

LABEL_CACHE_DIR = str(Path(PROJECT_DIR) / "cache")
Path(LABEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)

TEST_DATA_CACHE_DIR = str(Path(DATA_CACHE_DIR).parent / "revenger_action_test_data_dir")
Path(TEST_DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)


class SampleType(Enum):

    NEGATIVE_SAMPLE = 0
    SELF_REPORTED_POSITIVE_SAMPLE = 1
    DOCTOR_CONFIRMED_POSITIVE_SAMPLE = 2


REMOTE_MODELS = {
    "crnn-resnet_nature_comm_bottle_neck-none-se": {
        "url": {
            "google-drive": "https://drive.google.com/u/0/uc?id=11x5h_-B_fcusUhaSjeDaxF3CxRXuj_or",
            "deep-psp": (
                "https://deep-psp.tech/Models/CinC2025/"
                "BestModel_CRNN_CINC2025_resnet_nature_comm_bottle_neck_epoch8_02-21_21-41_metric_0.46.pth.tar"
            ),
        },
        "filename": "BestModel_CRNN_CINC2025_resnet_nature_comm_bottle_neck_epoch8_02-21_21-41_metric_0.46.pth.tar",
    },
    "crnn-resnet_nature_comm_bottle_neck_se-none-se": {
        "url": {
            "google-drive": "https://drive.google.com/u/0/uc?id=10jxmMUotziU6mvlsdsa2WuwWFa_By1sY",
            "deep-psp": (
                "https://deep-psp.tech/Models/CinC2025/"
                "BestModel_CRNN_CINC2025_resnet_nature_comm_bottle_neck_se_epoch9_02-22_06-21_metric_0.46.pth.tar"
            ),
        },
        "filename": "BestModel_CRNN_CINC2025_resnet_nature_comm_bottle_neck_se_epoch9_02-22_06-21_metric_0.46.pth.tar",
    },
    "crnn-tresnetF-none-se": {
        "url": {
            "google-drive": "https://drive.google.com/u/0/uc?id=1mujwFf1mLZUXXowBNIzVtlhURCYYCsMW",
            "deep-psp": (
                "https://deep-psp.tech/Models/CinC2025/"
                "BestModel_CRNN_CINC2025_tresnetF_epoch9_02-22_13-46_metric_0.45.pth.tar"
            ),
        },
        "filename": "BestModel_CRNN_CINC2025_tresnetF_epoch9_02-22_13-46_metric_0.45.pth.tar",
    },
}

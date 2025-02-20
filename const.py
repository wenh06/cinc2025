"""Constants for the project."""

import os
from pathlib import Path

__all__ = [
    "PROJECT_DIR",
    "MODEL_CACHE_DIR",
    "DATA_CACHE_DIR",
    "LABEL_CACHE_DIR",
    "TEST_DATA_CACHE_DIR",
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

TEST_DATA_CACHE_DIR = str(Path(DATA_CACHE_DIR).parent / "revenger_action_test_data_dir")
Path(TEST_DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)


REMOTE_MODELS = {
    "crnn-resnet_nature_comm_bottle_neck-none-se": {
        "url": {
            "google-drive": "https://drive.google.com/u/0/uc?id=1ZXQJkecCDQqKUfkuhJCiqYzCMConXjiy",
            "deep-psp": (
                "https://deep-psp.tech/Models/CinC2025/" "BestModel_CRNN_CINC2025_epoch13_02-20_22-00_metric_0.44.pth.tar"
            ),
        },
        "filename": "BestModel_CRNN_CINC2025_epoch13_02-20_22-00_metric_0.44.pth.tar",
    },
}

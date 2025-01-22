"""Constants for the project."""

import os
from pathlib import Path

__all__ = [
    "PROJECT_DIR",
    "MODEL_CACHE_DIR",
    "DATA_CACHE_DIR",
    "TEST_DATA_CACHE_DIR",
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
TEST_DATA_CACHE_DIR = str(Path(DATA_CACHE_DIR) / "cinc2025_action_test_data")
# SUBSET_DATA_CACHE_DIR = str(Path(DATA_CACHE_DIR) / "cinc2025_subset_data")
# FULL_DATA_CACHE_DIR = str(Path(DATA_CACHE_DIR) / "cinc2025_full_data")

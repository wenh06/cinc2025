import os
import shutil
import time
from pathlib import Path

from torch_ecg.utils.download import url_is_reachable
from torch_ecg.utils.misc import str2bool

from const import DATA_CACHE_DIR, LABEL_CACHE_DIR, MODEL_CACHE_DIR, REMOTE_MODELS, TEST_DATA_CACHE_DIR
from data_reader import CINC2025, CODE15, SamiTrop
from models import CRNN_CINC2025
from team_code import SubmissionCfg  # noqa: F401

try:
    TEST_FLAG = os.environ.get("CINC2025_REVENGER_TEST", False)
    TEST_FLAG = str2bool(TEST_FLAG)
except Exception:
    TEST_FLAG = False


def check_env():
    print("Checking the environment variables...")
    print(f"MODEL_CACHE_DIR: {MODEL_CACHE_DIR}")
    print(f"DATA_CACHE_DIR: {DATA_CACHE_DIR}")

    # for env in [
    #     "HF_ENDPOINT",
    #     "HUGGINGFACE_HUB_CACHE",
    #     "HF_HUB_CACHE",
    #     "HF_HOME",
    #     "NO_ALBUMENTATIONS_UPDATE",
    #     "ALBUMENTATIONS_DISABLE_VERSION_CHECK",
    # ]:
    #     print(f"{env}: {str(os.environ.get(env, None))}")

    print("Checking the environment variables done.")


def cache_data():
    """Cache the necessary data.

    Including: label file and chagas label file.

    """

    print("   Caching necessary data   ".center(80, "#"))
    reader_kwargs = {
        "db_dir": Path(TEST_DATA_CACHE_DIR),
    }
    dr = CINC2025(**reader_kwargs)
    # dr.download_subset()
    dr.download(files=["code-15-labels", "code-15-chagas-labels", "sami-trop-labels", "sami-trop-chagas-labels"])
    print("   Caching necessary data done.   ".center(80, "#"))

    # move the label files to `LABEL_CACHE_DIR`
    print("   Moving the label files   ".center(80, "#"))
    (Path(LABEL_CACHE_DIR) / CODE15.__name__).mkdir(parents=True, exist_ok=True)
    (Path(LABEL_CACHE_DIR) / SamiTrop.__name__).mkdir(parents=True, exist_ok=True)
    # shutil.move(dr._label_file, Path(LABEL_CACHE_DIR) / CODE15.__name__)
    # shutil.move(dr._chagas_label_file, LABEL_CACHE_DIR / CODE15.__name__)
    dr_code15 = CODE15(db_dir=Path(TEST_DATA_CACHE_DIR) / CODE15.__name__)
    shutil.move(dr_code15._label_file, Path(LABEL_CACHE_DIR) / CODE15.__name__)
    shutil.move(dr_code15._chagas_label_file, Path(LABEL_CACHE_DIR) / CODE15.__name__)
    del dr_code15
    dr_sami_trop = SamiTrop(db_dir=Path(TEST_DATA_CACHE_DIR) / SamiTrop.__name__)
    shutil.move(dr_sami_trop._label_file, Path(LABEL_CACHE_DIR) / SamiTrop.__name__)
    shutil.move(dr_sami_trop._chagas_label_file, Path(LABEL_CACHE_DIR) / SamiTrop.__name__)
    del dr_sami_trop
    print("In the LABEL_CACHE_DIR:")
    print(list((Path(LABEL_CACHE_DIR) / CODE15.__name__).rglob("*")))
    print(list((Path(LABEL_CACHE_DIR) / SamiTrop.__name__).rglob("*")))
    print("   Moving the label files done.   ".center(80, "#"))

    # re-init the reader with the new label file paths
    # del dr
    # reader_kwargs = {
    #     "db_dir": Path(TEST_DATA_CACHE_DIR),
    #     "label_file": Path(LABEL_CACHE_DIR) / CODE15.__name__ / CODE15.__label_file__,
    #     "chagas_label_file": Path(LABEL_CACHE_DIR) / CODE15.__name__ / CODE15.__chagas_label_file__,
    # }
    # dr = CODE15(**reader_kwargs)

    # print("   Checking the action test data   ".center(80, "#"))
    # print(f"{len(dr._df_records) = }")
    # print(f"{len(dr._all_records) = }")
    # print("   GitHub Action test data checking complete.   ".center(80, "#"))

    # print("   Converting to wfdb format   ".center(80, "#"))
    # dr._convert_to_wfdb_format()
    # print(f"is_converted: {dr._is_converted_to_wfdb_format}")
    # print("   Converting to wfdb format done.   ".center(80, "#"))


def cache_pretrained_models():
    """Cache the pretrained models."""
    print("Caching the pretrained models...")
    if url_is_reachable("https://drive.google.com/"):
        remote_model_source = "google-drive"
    else:
        remote_model_source = "deep-psp"
    # if SubmissionCfg.remote_model is not None:
    #     model, train_config = CRNN_CINC2025.from_remote(
    #         url=REMOTE_MODELS[SubmissionCfg.remote_model]["url"][remote_model_source],
    #         model_dir=MODEL_CACHE_DIR,
    #         filename=REMOTE_MODELS[SubmissionCfg.remote_model]["filename"],
    #     )
    #     print(f"Chagas classification model {SubmissionCfg.remote_model} cached.")
    #     print(f"Model: {model}")
    #     print(f"Train config: {train_config}")
    #     del model, train_config

    # Cache all models listed in `REMOTE_MODELS`
    for model_name, model_info in REMOTE_MODELS.items():
        model, train_config = CRNN_CINC2025.from_remote(
            url=model_info["url"][remote_model_source],
            model_dir=MODEL_CACHE_DIR,
            filename=model_info["filename"],
        )
        print(f"Chagas classification model {model_name} cached.")
        print(f"Model: {model}")
        print(f"Train config: {train_config}")
        del model, train_config


if __name__ == "__main__":
    check_env()
    time.sleep(2)
    cache_data()
    time.sleep(2)
    cache_pretrained_models()
    print("Done.")

import os
from pathlib import Path
from typing import Optional, Union

import torch
from torch_ecg.cfg import DEFAULTS
from torch_ecg.utils.download import url_is_reachable

from const import MODEL_CACHE_DIR

from .hubert_ecg import HuBERTECG, HuBERTECGConfig

if os.environ.get("HF_ENDPOINT", None) is not None and (not url_is_reachable(os.environ["HF_ENDPOINT"])):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
elif os.environ.get("HF_ENDPOINT", None) is None and (not url_is_reachable("https://huggingface.co")):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HOME"] = str(Path(MODEL_CACHE_DIR).parent)


__all__ = [
    "HuBERTECG",
    "HuBERTECGConfig",
    "load_hubert_ecg_model",
    "cache_remote_hubert_ecg_model",
]


def load_hubert_ecg_model(
    size_or_name_or_path: Union[str, Path], device: Optional[Union[str, torch.device]] = None
) -> HuBERTECG:
    """Load a HuBERT-ECG model.

    Parameters
    ----------
    size_or_name_or_path : str or Path
        The size ("small", "base", "large") or name/path of the model to load.
    device : str or torch.device, optional
        The device to load the model onto. If None, the model will be loaded onto the
        default device.

    Returns
    -------
    HuBERTECG
        The loaded HuBERT-ECG model.

    .. note::
        The pretrained models can be found at:

        - https://huggingface.co/Edoardo-BS/hubert-ecg-small
        - https://huggingface.co/Edoardo-BS/hubert-ecg-base
        - https://huggingface.co/Edoardo-BS/hubert-ecg-large

        The sampling rate of the ECG signals used to pretrain HuBERT-ECG is 100 Hz,
        as stated in the original paper, and also can be found in

        - https://github.com/Edoar-do/HuBERT-ECG/blob/master/code/dataset.py
        - https://github.com/Edoar-do/HuBERT-ECG/blob/master/code/pretrain.py
        - https://github.com/Edoar-do/HuBERT-ECG/blob/master/code/pretraining.sh

        The input to the model is expected to be of shape ``(batch_size, sequence_length)``,
        where the last dimension is flattened from standard 12-lead ECG signals of shape
        ``(12, signal_length)``.

        The output of the model is of shape ``(batch_size, time_steps, embedding_size)``.

    """
    if size_or_name_or_path == "small":
        model_name = "Edoardo-BS/hubert-ecg-small"
    elif size_or_name_or_path == "base":
        model_name = "Edoardo-BS/hubert-ecg-base"
    elif size_or_name_or_path == "large":
        model_name = "Edoardo-BS/hubert-ecg-large"
    else:
        model_name = size_or_name_or_path

    if device is None:
        device = DEFAULTS.device

    model = HuBERTECG.from_pretrained(model_name).to(device)  # type: ignore
    return model


def cache_remote_hubert_ecg_model(size_or_name: str) -> Path:
    """Cache a remote HuBERT-ECG model locally.

    Parameters
    ----------
    size_or_name : str
        The size ("small", "base", "large") or name of the model to cache.

    Returns
    -------
    Path
        The local path where the model is cached.

    """
    if size_or_name == "small":
        model_name = "Edoardo-BS/hubert-ecg-small"
    elif size_or_name == "base":
        model_name = "Edoardo-BS/hubert-ecg-base"
    elif size_or_name == "large":
        model_name = "Edoardo-BS/hubert-ecg-large"
    else:
        model_name = size_or_name

    model = HuBERTECG.from_pretrained(model_name)  # type: ignore
    local_path = Path(MODEL_CACHE_DIR) / model_name.replace("/", "-")
    model.save_pretrained(local_path)
    return local_path

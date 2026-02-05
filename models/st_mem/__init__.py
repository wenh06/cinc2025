"""
Pre-trained ST-MEM models:

- Encoder: https://drive.google.com/file/d/1E7J-A1HqWa2f08T6Sfk5uWk-_CFJhOYQ/view?usp=share_link
- Full (encoder + decoder): https://drive.google.com/file/d/14nScwPk35sFi8wc-cuLJLqudVwynKS0n/view?usp=share_link

"""

import json
from pathlib import Path
from typing import Optional, Union

import torch
from safetensors.torch import load_file
from torch_ecg.cfg import DEFAULTS
from torch_ecg.utils.download import http_get, url_is_reachable
from torch_ecg.utils.misc import get_kwargs, get_required_args

from const import MODEL_CACHE_DIR

from .st_mem import ST_MEM, ST_MEM_ViT, st_mem_vit_base_dec256d4b, st_mem_vit_small_dec256d4b

__all__ = [
    "ST_MEM",
    "ST_MEM_ViT",
    "load_st_mem_model",
    "cache_remote_st_mem_model",
]


CONFIG_FIELDS = {
    "encoder": get_required_args(ST_MEM_ViT) + list(get_kwargs(ST_MEM_ViT).keys()),
    "full": get_required_args(ST_MEM) + list(get_kwargs(ST_MEM).keys()),
}


def load_st_mem_model(
    size_or_path: Union[str, Path], encoder_only: bool = True, device: Optional[Union[str, torch.device]] = None
) -> Union[ST_MEM_ViT, ST_MEM]:
    """Load a ST-MEM model (full model or encoder only).

    Parameters
    ----------
    size_or_path : str or Path
        The size ("small", "base") or path of the model to load.
        Path can be a JSON config file or a directory containing
        the model weights file (and the config file).
    encoder_only : bool, default True
        Whether to load the encoder only or the full model (encoder + decoder).
    device : str or torch.device, optional
        The device to load the model onto. If None, the model will be loaded onto the
        default device.

    Returns
    -------
    ST_MEM or ST_MEM_ViT
        The loaded ST-MEM model.

    .. note::
        The pre-trained ST-MEM models can be found at:

        - Encoder: https://drive.google.com/file/d/1E7J-A1HqWa2f08T6Sfk5uWk-_CFJhOYQ/view?usp=share_link
        - Full (encoder + decoder): https://drive.google.com/file/d/14nScwPk35sFi8wc-cuLJLqudVwynKS0n/view?usp=share_link

        The sampling rate of the ECG signals used to pretrain ST-MEM is 250 Hz,
        as stated in the original paper, and also can be found in

        - https://github.com/vuno/ST-MEM/blob/main/util/dataset.py

        The input to the model is expected to be of shape ``(batch_size, num_leads, signal_length)``,
        where the last dimension ``signal_length`` is fixed to 75 * N, with N being an integer,
        and 75 corresponding to the patch size used in ST-MEM.

        The output of the model is of shape ``(batch_size, embedding_size)`` for the encoder-only model,
        or ``(batch_size, num_leads, patch_num, patch_size)`` for the full model.

    """
    if size_or_path == "small":
        model = st_mem_vit_small_dec256d4b(encoder_only=encoder_only)
    elif size_or_path == "base":
        model = st_mem_vit_base_dec256d4b(encoder_only=encoder_only)
    else:
        size_or_name = size_or_path  # for type checker
        path = Path(size_or_name)
        if path.is_file() and path.suffix == ".json":
            # load from config file
            config = _sanitize_config(json.loads(path.read_text()), encoder_only=encoder_only)
            model = ST_MEM_ViT(**config) if encoder_only else ST_MEM(**config)
        elif path.is_file() and path.suffix in {".pt", ".pth", ".bin", ".safetensors"}:
            if path.suffix == ".safetensors":
                raise ValueError("When loading from weights file, config file is required for safetensors format.")
            # load from weights file
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
            # assert that state_dict contains model config
            assert "config" in state_dict, "When loading from weights file, the file must contain model config."
            config = _sanitize_config(state_dict["config"], encoder_only=encoder_only)
            model = ST_MEM_ViT(**config) if encoder_only else ST_MEM(**config)
            model.load_state_dict(state_dict.get("model", state_dict.get("state_dict", state_dict)))
        elif path.is_dir():
            # load from directory
            # if there is a config file, load it
            config_path = path / "config.json"
            if config_path.is_file():
                config = _sanitize_config(json.loads(config_path.read_text()), encoder_only=encoder_only)
                model = ST_MEM_ViT(**config) if encoder_only else ST_MEM(**config)
            else:
                model = None
            # find the weights file
            weights_path = None
            use_safetensors = False
            if any(path.glob("*.pt")):
                weights_path = list(path.glob("*.pt"))[0]
            elif any(path.glob("*.pth")):
                weights_path = list(path.glob("*.pth"))[0]
            elif any(path.glob("*.bin")):
                weights_path = list(path.glob("*.bin"))[0]
            elif any(path.glob("*.safetensors")):
                weights_path = list(path.glob("*.safetensors"))[0]
                use_safetensors = True
            if weights_path is not None:
                if use_safetensors:
                    assert model is not None, "Config file is required when loading safetensors weights."
                    state_dict = load_file(weights_path)
                    model.load_state_dict(state_dict)
                else:
                    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
                    if model is None:
                        # assert that state_dict contains model config
                        assert (
                            "config" in state_dict
                        ), "When config file is not provided, the weights file must contain model config."
                        config = _sanitize_config(state_dict["config"], encoder_only=encoder_only)
                        model = ST_MEM_ViT(**config) if encoder_only else ST_MEM(**config)
                        model.load_state_dict(state_dict.get("model", state_dict.get("state_dict", state_dict)))
                    else:
                        model.load_state_dict(state_dict.get("model", state_dict.get("state_dict", state_dict)))
            elif model is None:  # neither config nor weights found
                raise ValueError(f"Neither config nor weights file found in directory: {size_or_name}")
        else:
            raise ValueError(f"Invalid size_or_path: {size_or_name}")

    if device is None:
        device = DEFAULTS.device

    model = model.to(device)
    return model


def _sanitize_config(config: dict, encoder_only: bool) -> dict:
    """Sanitize the config dictionary by keeping only the fields
    that are in the ST_MEM constructor.
    """
    config_fields = CONFIG_FIELDS["encoder" if encoder_only else "full"]
    sanitized_config = {k: v for k, v in config.items() if k in config_fields}
    return sanitized_config


def cache_remote_st_mem_model(model_cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Cache the remote ST-MEM model weights locally.

    Parameters
    ----------
    model_cache_dir : str or Path, optional
        The directory to cache the model weights. If None, the weights will be
        cached in the default model cache directory under "ST-MEM" subdirectory.

    Returns
    -------
    Path
        The local path where the model weights are cached.

    """
    if model_cache_dir is None:
        model_cache_dir = Path(MODEL_CACHE_DIR) / "ST-MEM"
        model_cache_dir.mkdir(parents=True, exist_ok=True)
    model_cache_dir = Path(model_cache_dir)

    model_urls = {  # all of base size
        "gdrive": {
            "encoder": "https://drive.google.com/file/d/1E7J-A1HqWa2f08T6Sfk5uWk-_CFJhOYQ/view?usp=share_link",
            "full": "https://drive.google.com/file/d/14nScwPk35sFi8wc-cuLJLqudVwynKS0n/view?usp=share_link",
        },
        "deep-psp": {
            "encoder": "https://deep-psp.tech/Models/ST-MEM/st_mem_vit_base_encoder.pth",
            "full": "https://deep-psp.tech/Models/ST-MEM/st_mem_vit_base_full.pth",
        },
    }

    # we only cache the encoder
    model_type = "encoder"
    if not any(model_cache_dir.glob("st_mem_vit_base_encoder.pth")):
        if url_is_reachable("https://drive.google.com"):
            print(f"Downloading ST-MEM {model_type} model from Google Drive...")
            source = "gdrive"
        elif url_is_reachable("https://deep-psp.tech"):
            print(f"Downloading ST-MEM {model_type} model from Deep-PSP...")
            source = "deep-psp"
        else:
            raise ConnectionError("No remote ST-MEM model is reachable.")

        save_path = http_get(
            model_urls[source][model_type],
            dst_dir=model_cache_dir,
            filename=f"st_mem_vit_base_{model_type}.pth",
        )
    else:
        save_path = list(model_cache_dir.glob("st_mem_vit_base_encoder.pth"))[0]

    return save_path

"""
CRNN model for CINC2025.
"""

import os
import warnings
from copy import deepcopy
from pathlib import Path, PosixPath, WindowsPath
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from easydict import EasyDict
from torch_ecg.cfg import CFG, DTYPE
from torch_ecg.components import WaveformInput  # noqa: F401
from torch_ecg.models import ECG_CRNN
from torch_ecg.models.loss import setup_criterion
from torch_ecg.utils.misc import add_docstring
from torch_ecg.utils.utils_data import one_hot_encode

from cfg import ModelCfg
from outputs import CINC2025Outputs
from utils.misc import is_stdtypes

__all__ = [
    "CRNN_CINC2025",
]


def _get_np_dtypes():
    return [eval(f"np.dtypes.{dtype}") for dtype in dir(np.dtypes) if dtype.endswith("DType")]


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # fmt: off
    _safe_globals = [
        CFG, DTYPE, EasyDict,
        Path, PosixPath, WindowsPath,
        np.core.multiarray._reconstruct,
        np.ndarray, np.dtype,
        np.float32, np.float64, np.int32, np.int64, np.uint8, np.int8,
    ] + _get_np_dtypes()
    # fmt: on

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(_safe_globals)


class CRNN_CINC2025(ECG_CRNN):
    """CRNN model for CINC2025.

    Parameters
    ----------
    config : dict
        Hyper-parameters, including backbone_name, etc.
        ref. the corresponding config file.

    """

    __DEBUG__ = True
    __name__ = "CRNN_CINC2025"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        if config is None:
            _config = deepcopy(ModelCfg.crnn)
        else:
            _config = deepcopy(config)
        chagas_classes = kwargs.pop("chagas_classes", None) or kwargs.pop("classes", None) or _config.get("chagas_classes", None) or _config.get("classes", None)
        assert chagas_classes is not None, "`chagas_classes` must be provided"
        n_leads = kwargs.pop("n_leads", None) or _config.get("n_leads", None)
        assert n_leads is not None, "`n_leads` must be provided"
        criterion = kwargs.pop("criterion", None) or _config.get("criterion", None)
        assert criterion is not None, "`criterion` must be provided"
        criterion_kw = kwargs.pop("criterion_kw", {}) or _config.get("criterion_kw", {})

        if "crnn" in _config:
            # in this case, _config.crnn should be the config
            _config = _config.crnn

        _config.chagas_classes = chagas_classes
        _config.n_leads = n_leads
        _config.criterion = criterion
        _config.criterion_kw = criterion_kw
        super().__init__(
            classes=chagas_classes,
            n_leads=n_leads,
            config=_config,
            **kwargs,
        )

        self.criterion = setup_criterion(criterion, **criterion_kw)

    def forward(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        input_tensors : dict of torch.Tensor
            Input signals and labels, including

            - "signals" : torch.Tensor
                Input signals. Required.
            - "chagas" : torch.Tensor, optional
                Labels for Chagas disease diagnosis.

        Returns
        -------
        dict
            Predictions, including "chagas", "chagas_logits", "chagas_prob",
            and "chagas_loss".

        """
        chagas_logits = super().forward(input_tensors["signals"].to(self.dtype).to(self.device))
        chagas_prob = self.softmax(chagas_logits)
        chagas_pred = torch.argmax(chagas_prob, dim=-1)
        if "chagas" in input_tensors:
            if self.criterion.__class__.__name__ != "CrossEntropyLoss":
                input_tensors["chagas"] = (
                    torch.from_numpy(one_hot_encode(input_tensors["chagas"], num_classes=self.n_classes))
                    .to(self.dtype)
                    .to(self.device)
                )
            else:
                input_tensors["chagas"] = input_tensors["chagas"].to(self.device)
            chagas_loss = self.criterion(chagas_logits, input_tensors["chagas"])
        else:
            chagas_loss = None
        return {"chagas_logits": chagas_logits, "chagas_prob": chagas_prob, "chagas": chagas_pred, "chagas_loss": chagas_loss}

    @torch.no_grad()
    def inference(self, sig: Union[np.ndarray, torch.Tensor, list]) -> CINC2025Outputs:
        """Inference on a single signal or a batch of signals.

        Parameters
        ----------
        sig : numpy.ndarray or torch.Tensor, or list
            Input signal(s).

        Returns
        -------
        CINC2025Outputs
            Predictions, including "chagas", "chagas_logits", and "chagas_prob".

        """
        training = self.training
        self.eval()

        input_tensors = {"signals": torch.as_tensor(sig, dtype=self.dtype, device=self.device)}
        if input_tensors["signals"].ndim == 2:
            input_tensors["signals"] = input_tensors["signals"].unsqueeze(0)  # add a batch dimension
        # batch_size, channels, seq_len = _input.shape
        forward_output = self.forward(input_tensors)

        output = CINC2025Outputs(**forward_output)

        # restore the training mode
        self.train(training)

        return output

    @add_docstring(inference.__doc__)
    def inference_CINC2025(self, sig: Union[np.ndarray, torch.Tensor, list]) -> CINC2025Outputs:
        """alias for `self.inference`"""
        return self.inference(sig)

    def save(self, path: Union[str, bytes, os.PathLike], train_config: CFG) -> None:
        """Save the model to disk.

        Parameters
        ----------
        path : `path-like`
            Path to save the model.
        train_config : CFG
            Config for training the model,
            used when one restores the model.

        Returns
        -------
        None

        """
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        _model_config = make_safe_globals(self.config)
        _train_config = make_safe_globals(train_config)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_config": _model_config,
                "train_config": _train_config,
            },
            path,
        )


def make_safe_globals(obj: CFG, remove_paths: bool = True) -> CFG:
    """Make a dictionary or a dictionary-like object safe for serialization.

    Parameters
    ----------
    obj : dict
        The dictionary or dictionary-like object.
    remove_paths : bool, default True
        Whether to remove paths in the dictionary.

    Returns
    -------
    CFG
        The safe dictionary.

    """
    if isinstance(obj, (CFG, dict)):
        sg = {k: make_safe_globals(v) for k, v in obj.items()}
        sg = CFG({k: v for k, v in sg.items() if v is not None})
    elif isinstance(obj, (list, tuple)):
        sg = [make_safe_globals(item) for item in obj]
        sg = [item for item in sg if item is not None]
    elif isinstance(obj, set):
        sg = {make_safe_globals(item) for item in obj}
        sg = {item for item in sg if item is not None}
    elif isinstance(obj, frozenset):
        sg = frozenset({make_safe_globals(item) for item in obj})
        sg = frozenset({item for item in sg if item is not None})
    elif isinstance(obj, tuple(item for item in _safe_globals if isinstance(item, type))):
        sg = obj
    elif type(obj).__module__ == "torch" or type(obj).__module__.startswith("torch."):
        sg = obj
    elif is_stdtypes(obj):
        sg = obj
    else:
        sg = None
    if remove_paths:
        if isinstance(sg, os.PathLike):
            sg = None
        elif isinstance(sg, (str, bytes)) and os.path.exists(sg):
            sg = None
    return sg

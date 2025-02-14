"""
Multi-head model (multi-task learning) for CINC2025.
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch_ecg.cfg import CFG
from torch_ecg.components import WaveformInput  # noqa: F401
from torch_ecg.models import ECG_CRNN
from torch_ecg.models.loss import setup_criterion  # noqa: F401
from torch_ecg.utils.misc import add_docstring
from torch_ecg.utils.utils_data import one_hot_encode  # noqa: F401

from cfg import ModelCfg
from outputs import CINC2025Outputs

__all__ = [
    "MultiHead_CINC2025",
]


class MultiHead_CINC2025(ECG_CRNN):
    """MultiHead model for CINC2025.

    Parameters
    ----------
    config : dict
        Hyper-parameters, including backbone_name, etc.
        ref. the corresponding config file.

    """

    __DEBUG__ = True
    __name__ = "MultiHead_CINC2025"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        if config is None:
            _config = deepcopy(ModelCfg)
        else:
            _config = deepcopy(config)
        super().__init__(
            classes=_config.chagas_classes,
            n_leads=_config.n_leads,
            config=_config,
            **kwargs,
        )

        raise NotImplementedError

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
            - "arr_diag" : torch.Tensor, optional
                Labels for arrhythmia diagnosis.

        Returns
        -------
        dict
            Predictions, including "chagas" and "arr_diag" (if available).

        """
        raise NotImplementedError

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
            Predictions, including "chagas" and "arr_diag" (if available).

        """
        raise NotImplementedError

    @add_docstring(inference.__doc__)
    def inference_CINC2025(self, sig: Union[np.ndarray, torch.Tensor, list]) -> CINC2025Outputs:
        """alias for `self.inference`"""
        return self.inference(sig)

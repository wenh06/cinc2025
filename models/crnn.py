"""
CINC2025 models (multi-head model for CINC2025).
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch_ecg.cfg import CFG
from torch_ecg.models import ECG_CRNN, MLP
from torch_ecg.utils.misc import add_docstring

from cfg import ModelCfg
from outputs import CINC2025Outputs  # noqa: F401

__all__ = [
    "MultiHead_CINC2025",
]


class MultiHead_CINC2025(ECG_CRNN):
    """Multi-head model for CINC2025.

    Parameters
    ----------
    config : dict
        Hyper-parameters, including backbone_name, etc.
        ref. the corresponding config file.

    """

    __DEBUG__ = True
    __name__ = "MultiHead_CINC2025"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        super().__init__()
        self.__config = deepcopy(ModelCfg)
        if config is not None:
            self.__config.update(deepcopy(config))
        self.__config.update(kwargs)

        self.extra_heads = nn.ModuleDict()
        if self.__config.arr_diag_head.enabled:
            arr_diag_head_cfg = deepcopy(self.__config.arr_diag_head)
            arr_diag_head_cfg.pop("enabled")
            self.extra_heads["arr_diag"] = MLP(**arr_diag_head_cfg)

    def forward(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        input_tensors : dict of torch.Tensor
            Input signals and labels, including

            - "signals" : torch.Tensor
                Input signals. Required.
            - "chagas_labels" : torch.Tensor, optional
                Labels for Chagas disease diagnosis.
            - "arr_diag_labels" : torch.Tensor, optional
                Labels for arrhythmia diagnosis.

        Returns
        -------
        dict
            Predictions, including "chagas" and "diag".

        """
        raise NotImplementedError

    def get_input_tensors(
        self,
        sig: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Get input tensors for the model.

        Parameters
        ----------
        sig : torch.Tensor
            Input signal tensor.
        labels : dict, optional
            Not used, but kept for compatibility with other models.

        Returns
        -------
        Dict[str, torch.Tensor]
            Input tensors for the model.

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
            Predictions, including "chagas" and "diag".

        """
        raise NotImplementedError

    @add_docstring(inference.__doc__)
    def inference_CINC2025(self, sig: Union[np.ndarray, torch.Tensor, list]) -> CINC2025Outputs:
        """alias for `self.inference`"""
        return self.inference(sig)

    @property
    def config(self) -> CFG:
        return self.__config

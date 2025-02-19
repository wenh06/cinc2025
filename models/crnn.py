"""
CRNN model for CINC2025.
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch_ecg.cfg import CFG
from torch_ecg.components import WaveformInput  # noqa: F401
from torch_ecg.models import ECG_CRNN
from torch_ecg.models.loss import setup_criterion
from torch_ecg.utils.misc import add_docstring
from torch_ecg.utils.utils_data import one_hot_encode

from cfg import ModelCfg
from outputs import CINC2025Outputs

__all__ = [
    "CRNN_CINC2025",
]


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
            _config = deepcopy(ModelCfg)
        else:
            _config = deepcopy(config)
        super().__init__(
            classes=_config.chagas_classes,
            n_leads=_config.n_leads,
            config=_config,
            **kwargs,
        )

        self.criterion = setup_criterion(_config.criterion, **_config.get("criterion_kw", {}))

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

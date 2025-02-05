"""
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np  # noqa: F401
import torch  # noqa: F401

__all__ = [
    "CINC2025Outputs",
]


@dataclass
class CINC2025Outputs:
    """Output class for CinC2025.

    Attributes
    ----------
    chagas : Sequence[bool]
        Predicted Chagas disease diagnosis.
    chagas_logits : Sequence[Sequence[float]]
        Logits of the Chagas disease diagnosis.
    chagas_prob : Sequence[Sequence[float]]
        Probabilities of the Chagas disease diagnosis.
    chagas_loss : Sequence[float]
        Loss for the Chagas disease diagnosis.
    chagas_threshold : float, default 0.5
        Threshold for the Chagas disease diagnosis.
    arr_diag : Sequence[Sequence[str]]
        Predicted arrhythmia diagnosis.
    arr_diag_logits : Sequence[Sequence[float]]
        Logits of the arrhythmia diagnosis.
    arr_diag_prob : Sequence[Sequence[float]]
        Probabilities of the arrhythmia diagnosis.
    arr_diag_loss : Sequence[float]
        Loss for the arrhythmia diagnosis.
    arr_diag_classes : Sequence[str]
        Class names for the arrhythmia diagnosis.
    arr_diag_threshold : float, default 0.5
        Threshold for the arrhythmia diagnosis.

    """

    chagas: Optional[Sequence[Union[bool, int, float]]] = None
    chagas_logits: Optional[Sequence[Sequence[float]]] = None
    chagas_prob: Optional[Sequence[Sequence[float]]] = None
    chagas_loss: Optional[Sequence[float]] = None
    chagas_threshold: float = 0.5
    arr_diag: Optional[Sequence[Sequence[str]]] = None
    arr_diag_logits: Optional[Sequence[Sequence[float]]] = None
    arr_diag_prob: Optional[Sequence[Sequence[float]]] = None
    arr_diag_loss: Optional[Sequence[float]] = None
    arr_diag_classes: Optional[Sequence[str]] = None
    arr_diag_threshold: float = 0.5

    def __post_init__(self) -> None:
        raise NotImplementedError

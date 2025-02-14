"""
"""

from dataclasses import dataclass, fields
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

__all__ = [
    "CINC2025Outputs",
]


@dataclass
class CINC2025Outputs:
    """Output class for CinC2025.

    Attributes
    ----------
    chagas : Sequence[bool] or Sequence[int] or Sequence[float]
        Predicted Chagas disease diagnosis.
    chagas_logits : Sequence[Sequence[float]]
        Logits of the Chagas disease diagnosis.
    chagas_prob : Sequence[Sequence[float]]
        Probabilities of the Chagas disease diagnosis.
    chagas_loss : Sequence[float]
        Loss for the Chagas disease diagnosis.
    chagas_threshold : float, default 0.5
        Threshold for the Chagas disease diagnosis.
    arr_diag : Sequence[Sequence[Union[str, int]]]
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
    arr_diag: Optional[Sequence[Sequence[Union[str, int]]]] = None
    arr_diag_logits: Optional[Sequence[Sequence[float]]] = None
    arr_diag_prob: Optional[Sequence[Sequence[float]]] = None
    arr_diag_loss: Optional[Sequence[float]] = None
    arr_diag_classes: Optional[Sequence[str]] = None
    arr_diag_threshold: float = 0.5

    def __post_init__(self) -> None:
        assert any(
            [
                self.chagas is not None,
                self.chagas_logits is not None,
                self.chagas_prob is not None,
            ]
        ), "at least one of `chagas`, `chagas_logits`, `chagas_prob` prediction should be provided"
        if self.chagas is not None:
            # self.chagas = np.array(self.chagas, dtype=bool).tolist()
            if isinstance(self.chagas, torch.Tensor):
                self.chagas = self.chagas.cpu().detach().numpy().astype(bool)
            else:
                self.chagas = np.asarray(self.chagas).astype(bool)
            self.chagas = self.chagas.tolist()
        if self.chagas_logits is not None:
            if self.chagas_prob is None:
                self.chagas_prob = torch.softmax(torch.tensor(self.chagas_logits), dim=-1).cpu().detach().numpy()
            elif isinstance(self.chagas_prob, torch.Tensor):
                self.chagas_prob = self.chagas_prob.cpu().detach().numpy()
            else:
                self.chagas_prob = np.array(self.chagas_prob)
            if self.chagas is None:
                self.chagas = np.argmax(self.chagas_prob, axis=-1).astype(bool).tolist()
            assert len(self.chagas) == self.chagas_logits.shape[0] == self.chagas_prob.shape[0], "inconsistent length"
            assert self.chagas_logits.shape[1] == self.chagas_prob.shape[1] == 2, "should be binary classification"
        if self.chagas_prob is not None:
            if isinstance(self.chagas_prob, torch.Tensor):
                self.chagas_prob = self.chagas_prob.cpu().detach().numpy()
            else:
                self.chagas_prob = np.array(self.chagas_prob)
            if self.chagas is None:
                self.chagas = np.argmax(self.chagas_prob, axis=-1).astype(bool).tolist()
            assert len(self.chagas) == self.chagas_prob.shape[0], "inconsistent length"
            assert self.chagas_prob.shape[1] == 2, "should be binary classification"
        if self.chagas_loss is not None:
            if isinstance(self.chagas_loss, torch.Tensor):
                self.chagas_loss = self.chagas_loss.cpu().detach().numpy()

        if self.arr_diag is not None:
            assert self.arr_diag_classes is not None, "arr_diag_classes should be provided if `arr_diag` is provided"
            assert len(self.arr_diag) == len(self.chagas), "inconsistent length"
            idx2class = {idx: cl for idx, cl in enumerate(self.arr_diag_classes)}
            # in case the arr_diag is not converted to class names
            self.arr_diag = [[idx2class.get(item, item) for item in items] for items in self.arr_diag]
        if self.arr_diag_prob is not None:
            assert self.arr_diag_classes is not None, "arr_diag_classes should be provided if `arr_diag` is provided"
            if isinstance(self.arr_diag_prob, torch.Tensor):
                self.arr_diag_prob = self.arr_diag_prob.cpu().detach().numpy()
            else:
                self.arr_diag_prob = np.array(self.arr_diag_prob)
            assert self.arr_diag_prob.shape[1] == len(self.arr_diag_classes), "inconsistent number of classes"
            if self.arr_diag is None:
                self.arr_diag = [
                    [self.arr_diag_classes[idx] for idx in np.where(np.array(items) > self.arr_diag_threshold)[0]]
                    for items in self.arr_diag_prob
                ]
            assert len(self.arr_diag) == len(self.chagas), "inconsistent length"
        elif self.arr_diag_logits is not None:
            assert self.arr_diag_classes is not None, "arr_diag_classes should be provided if `arr_diag` is provided"
            if isinstance(self.arr_diag_logits, torch.Tensor):
                self.arr_diag_logits = self.arr_diag_logits.cpu().detach().numpy()
            else:
                self.arr_diag_logits = np.array(self.arr_diag_logits)
            if self.arr_diag_prob is None:
                self.arr_diag_prob = torch.sigmoid(torch.from_numpy(self.arr_diag_logits)).cpu().detach().numpy()
            elif isinstance(self.arr_diag_prob, torch.Tensor):
                self.arr_diag_prob = self.arr_diag_prob.cpu().detach().numpy()
            else:
                self.arr_diag_prob = np.array(self.arr_diag_prob)
            assert (
                self.arr_diag_logits.shape[1] == self.arr_diag_prob.shape[1] == len(self.arr_diag_classes)
            ), "inconsistent number of classes"
            if self.arr_diag is None:
                self.arr_diag = [
                    [self.arr_diag_classes[idx] for idx in np.where(np.array(items) > self.arr_diag_threshold)[0]]
                    for items in self.arr_diag_prob
                ]
            assert len(self.arr_diag) == len(self.chagas), "inconsistent length"
        if self.arr_diag_loss is not None:
            if isinstance(self.arr_diag_loss, torch.Tensor):
                self.arr_diag_loss = self.arr_diag_loss.cpu().detach().numpy()

    def append(self, values: Union["CINC2025Outputs", Sequence["CINC2025Outputs"]]) -> None:
        """Append other :class:`CINC2025Outputs` to `self`

        Parameters
        ----------
        values : CINC2025Outputs or Sequence[CINC2025Outputs]
            The values to be appended.

        Returns
        -------
        None

        """
        if not isinstance(values, Sequence):
            values = [values]
        for v in values:
            assert v.__class__ == self.__class__, "`values` must be of the same type as `self`"
            for k in fields(v):
                v_ = getattr(v, k.name)
                self_ = getattr(self, k.name)
                if v_ is None or self_ is None:
                    continue
                if k.name in ["arr_diag_classes"]:
                    assert v_ == self_, f"the field of ordered sequence `{k.name}` must be the identical"
                    continue
                if k.name in ["chagas_threshold", "arr_diag_threshold"]:
                    assert v_ == self_, f"the field `{k.name}` must be the identical"
                    continue
                if isinstance(v_, np.ndarray):
                    # self[k] = np.concatenate((self[k], v_))
                    setattr(self, k.name, np.concatenate((self_, v_)))
                elif isinstance(v_, pd.DataFrame):
                    # self[k] = pd.concat([self[k], v_], axis=0, ignore_index=True)
                    setattr(self, k.name, pd.concat([self_, v_], axis=0, ignore_index=True))
                elif isinstance(v_, Sequence):  # list, tuple, etc.
                    # self[k] += v_
                    setattr(self, k.name, self_ + v_)
                else:
                    raise ValueError(f"field `{k.name}` of type `{type(v_)}` is not supported")

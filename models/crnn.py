"""
CRNN model for CINC2025.
"""

import os
import warnings
from copy import deepcopy
from pathlib import Path, PosixPath, WindowsPath
from typing import Any, Dict, List, Literal, Optional, Union

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

from .loss import AdaptiveLogisticPairwiseLoss, ChagasLoss, PairwiseRankingLossHinge, PairwiseRankingLossLogistic

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
        chagas_classes = (
            kwargs.pop("chagas_classes", None)
            or kwargs.pop("classes", None)
            or _config.get("chagas_classes", None)
            or _config.get("classes", None)
        )
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

        default_ranking_cfg = CFG(
            enable=False,
            type="hinge",  # or "logistic", or "adaptive"
            weight=0.3,
            margin=0.5,
        )
        if not hasattr(_config, "ranking"):
            _config.ranking = default_ranking_cfg
        else:
            # merge
            for k, v in default_ranking_cfg.items():
                _config.ranking.setdefault(k, v)

        super().__init__(
            classes=chagas_classes,
            n_leads=n_leads,
            config=_config,
            **kwargs,
        )

        if criterion == "ChagasLoss":
            self.criterion = ChagasLoss(**criterion_kw)
        else:
            self.criterion = setup_criterion(criterion, **criterion_kw)

        self.use_ranking = bool(self.config.ranking.enable)
        if self.use_ranking:
            if self.config.ranking.type.lower() == "hinge":
                self.ranking_criterion = PairwiseRankingLossHinge(margin=self.config.ranking.margin)
            elif self.config.ranking.type.lower() == "logistic":
                self.ranking_criterion = PairwiseRankingLossLogistic(margin=self.config.ranking.margin)
            elif self.config.ranking.type.lower() == "adaptive":
                self.ranking_criterion = AdaptiveLogisticPairwiseLoss(
                    margin=self.config.ranking.margin,
                    return_stats=False,
                    # hard_negative_pct=0.1,
                    # subsample_pos=32,
                    # subsample_neg=160,
                    # adaptive_margin=True,
                    # target_active_ratio=0.2,
                    # grad_threshold=0.1,
                )
            else:
                raise ValueError(f"Unknown ranking type {self.config.ranking.type}")
            self.ranking_weight = float(self.config.ranking.weight)
        else:
            self.ranking_criterion = None
            self.ranking_weight = 0.0

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

        chagas_loss = None
        ranking_loss = None
        if "chagas" in input_tensors:
            labels_in = input_tensors["chagas"]
            labels_in = labels_in.to(self.device)

            if labels_in.ndim > 1:
                # 如果已经是 one-hot
                hard_labels = torch.argmax(labels_in, dim=-1)
            else:
                hard_labels = labels_in

            if labels_in.ndim == 1 and self.criterion.__class__.__name__ != "CrossEntropyLoss":
                oh = torch.from_numpy(one_hot_encode(labels_in, num_classes=self.n_classes)).to(self.dtype).to(self.device)
                base_loss = self.criterion(chagas_logits, oh)
            else:
                base_loss = self.criterion(chagas_logits, labels_in)

            if self.use_ranking:
                pos_channel_scores = chagas_logits[:, 1]
                ranking_loss = self.ranking_criterion(pos_channel_scores, hard_labels)
                chagas_loss = base_loss + self.ranking_weight * ranking_loss
            else:
                chagas_loss = base_loss

        return {  # type: ignore
            "chagas_logits": chagas_logits,
            "chagas_prob": chagas_prob,
            "chagas": chagas_pred,
            "chagas_loss": chagas_loss,
            "ranking_loss": ranking_loss,
        }

    @torch.no_grad()
    def inference(
        self,
        sig: Union[np.ndarray, torch.Tensor, list],
        crop_infer: bool = False,
        crop_len: Optional[float] = None,
        stride: Optional[float] = None,
        agg: Literal["max", "top2_mean", "mean"] = "max",
    ) -> CINC2025Outputs:
        """Inference on a single signal or a batch of signals.

        Parameters
        ----------
        sig : array-like
            Input signal(s). Accepts:
              - (C, T)
              - (B, C, T)
              - list / np.ndarray / torch.Tensor
        crop_infer : bool, default False
            If True, apply sliding window inference per signal.
        crop_len : float, optional
            Window length in seconds. If None, defaults to 4096/fs (≈10.24s at fs=400).
        stride : float, optional
            Step length in seconds. If None, defaults to 1024/fs (≈2.56s at fs=400).
        agg : {"max","top2_mean","mean"}, default "max"
            Aggregation strategy for multi-crop probabilities.

        Returns
        -------
        CINC2025Outputs
            Predictions, including "chagas", "chagas_logits", and "chagas_prob".

        """
        training = self.training
        self.eval()

        # Normalize input to (B, C, T)
        if isinstance(sig, list):
            sig = np.asarray(sig)
        if isinstance(sig, np.ndarray):
            sig_t = torch.as_tensor(sig, dtype=self.dtype, device=self.device)
        else:
            sig_t = sig.to(self.device).to(self.dtype)

        if sig_t.ndim == 2:  # (C, T) -> (1, C, T)
            sig_t = sig_t.unsqueeze(0)
        elif sig_t.ndim != 3:
            raise ValueError(f"Unsupported input shape {sig_t.shape}, expected (C,T) or (B,C,T)")

        B, C, T = sig_t.shape

        if not crop_infer:
            forward_output = self.forward({"signals": sig_t})
            output = CINC2025Outputs.from_dict(forward_output)
            self.train(training)
            return output

        # Multi-crop path
        fs = float(self.config.get("fs", 400))
        default_crop_len_sec = 4096.0 / fs  # ≈10.24s
        default_stride_sec = 1024.0 / fs  # ≈2.56s
        crop_len_sec = crop_len if crop_len is not None else default_crop_len_sec
        stride_sec = stride if stride is not None else default_stride_sec

        crop_len_samples = int(round(crop_len_sec * fs))
        stride_samples = int(round(stride_sec * fs))
        crop_len_samples = max(1, crop_len_samples)
        stride_samples = max(1, stride_samples)

        batch_logits: List[torch.Tensor] = []
        batch_probs: List[torch.Tensor] = []
        batch_preds: List[torch.Tensor] = []
        multi_crop_probs: List[np.ndarray] = []

        for b in range(B):
            x = sig_t[b]  # (C, Tb)
            Tb = x.shape[-1]

            # Short signal: no cropping
            if Tb <= crop_len_samples:
                fo = self.forward({"signals": x.unsqueeze(0)})
                logits_full = fo["chagas_logits"]  # (1,2)
                probs_full = fo["chagas_prob"]  # (1,2)
                batch_logits.append(logits_full.squeeze(0))
                batch_probs.append(probs_full.squeeze(0))
                batch_preds.append(torch.argmax(probs_full, dim=-1))
                continue

            # Compute starting indices
            starts = list(range(0, Tb - crop_len_samples + 1, stride_samples))
            if starts[-1] != Tb - crop_len_samples:
                # Add tail window
                starts.append(Tb - crop_len_samples)

            # Stack windows: (Ncrops, C, crop_len_samples)
            windows = torch.stack(
                [x[..., s : s + crop_len_samples] for s in starts],
                dim=0,
            )
            fo = self.forward({"signals": windows})
            logits_all = fo["chagas_logits"]  # (Ncrops,2)
            probs_all = fo["chagas_prob"]  # (Ncrops,2)

            pos_probs = probs_all[:, 1]

            if agg == "max":
                idx = torch.argmax(pos_probs)
                chosen_logits = logits_all[idx]
                agg_logits = chosen_logits
                agg_probs = torch.softmax(agg_logits, dim=-1)
            elif agg == "top2_mean":
                k = min(2, probs_all.shape[0])
                vals, indices = torch.topk(pos_probs, k)
                mean_pos = vals.mean()
                # Reconstruct probability vector from aggregated pos prob
                agg_probs = torch.tensor(
                    [1.0 - mean_pos.item(), mean_pos.item()],
                    device=probs_all.device,
                    dtype=probs_all.dtype,
                )
                # Use top1 logits as representative
                agg_logits = logits_all[indices[0]]
            elif agg == "mean":
                agg_probs = probs_all.mean(dim=0)
                agg_logits = logits_all.mean(dim=0)  # Approximate
            else:
                raise ValueError(f"Unsupported agg '{agg}'. Choose from ['max','top2_mean','mean'].")

            batch_logits.append(agg_logits)
            batch_probs.append(agg_probs)
            batch_preds.append(torch.argmax(agg_probs).unsqueeze(0))

        logits_tensor = torch.stack(batch_logits, dim=0)  # (B,2)
        probs_tensor = torch.stack(batch_probs, dim=0)  # (B,2)
        preds_tensor = torch.argmax(probs_tensor, dim=-1)

        output_dict = {
            "chagas_logits": logits_tensor,
            "chagas_prob": probs_tensor,
            "chagas": preds_tensor,
            "chagas_loss": None,
            "ranking_loss": None,
        }

        output = CINC2025Outputs.from_dict(output_dict)

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

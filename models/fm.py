import os
import warnings
from copy import deepcopy
from pathlib import Path, PosixPath, WindowsPath
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from einops.layers.torch import Rearrange
from torch_ecg.cfg import CFG, DTYPE
from torch_ecg.models.loss import setup_criterion
from torch_ecg.utils.misc import add_docstring
from torch_ecg.utils.utils_data import one_hot_encode
from torch_ecg.utils.utils_nn import CkptMixin, SizeMixin

from cfg import ModelCfg
from outputs import CINC2025Outputs
from utils.misc import is_stdtypes

from .dem import DemographicEncoder
from .hubert_ecg import load_hubert_ecg_model
from .loss import AdaptiveLogisticPairwiseLoss, ChagasLoss, PairwiseRankingLossHinge, PairwiseRankingLossLogistic
from .st_mem import load_st_mem_model

__all__ = [
    "FM_CINC2025",
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


class FM_CINC2025(nn.Module, SizeMixin, CkptMixin):
    """Foundation Model based classifier for CINC2025.

    Supports ST-MEM and HuBERT backbones.
    """

    __DEBUG__ = True
    __name__ = "FM_CINC2025"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            _config = deepcopy(ModelCfg.fm)
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

        if "fm" in _config:
            _config = _config.fm

        self.config = _config
        self.n_leads = n_leads
        self.n_classes = len(chagas_classes)
        self.classes = chagas_classes

        # Backbone Setup
        model_name = self.config["name"].lower().replace("_", "-")
        self.fs = self.config["fs"][model_name]
        backbone_cache_dir = self.config.get("backbone_cache_dir", None) or kwargs.pop("backbone_cache_dir", None)
        if backbone_cache_dir is None:
            warnings.warn(
                "If `config.backbone_cache_dir` is not set before using the model, "
                "the backbone model will be randomly initialized.",
                UserWarning,
            )
            # This mechanism is to allow loading from saved checkpoints without needing the cache dir.
            # especially when the backbone is fine-tuned and saved in the checkpoint.
            backbone_cache_dir = "base"

        if "st-mem" in model_name:
            self.inputer = nn.Identity()  # ST-MEM has conventional input of shape (bs, n_leads, L)
            # Load encoder only of ST-MEM
            self.backbone = load_st_mem_model(backbone_cache_dir, encoder_only=True, device="cpu")
            self.backbone_type = "st-mem"
        elif "hubert" in model_name:
            self.inputer = Rearrange("b c l -> b (c l)")  # HuBERT usually expects (bs, n_leads * L)
            self.backbone = load_hubert_ecg_model(backbone_cache_dir, device="cpu")
            self.backbone_type = "hubert"
        else:
            raise ValueError(f"Unknown foundation model name: {model_name}")

        if self.config.dem_encoder.enable:
            if self.config.dem_encoder.mode == "film":
                feature_dim = self.config["embed_dim"][model_name]
            else:  # concat
                feature_dim = self.config.dem_encoder.feature_dim
            self.dem_encoder = DemographicEncoder(
                feature_dim=feature_dim,
                dem_input_dim=2,
                mode=self.config.dem_encoder.mode,
                hidden_dim=self.config.dem_encoder.hidden_dim,
            )

        # Classification Head
        if self.config.dem_encoder.enable and self.config.dem_encoder.mode == "concat":
            embed_dim = self.config["embed_dim"][model_name] + self.config.dem_encoder.feature_dim
        else:
            embed_dim = self.config["embed_dim"][model_name]
        hidden_dim = self.config["head"]["hidden_dim"]
        dropout = self.config.get("dropout", 0.2)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.n_classes),
        )

        # Freeze backbone if requested
        self.freeze_backbone(self.config.get("freeze_backbone", False))

        # Loss & Ranking Setup
        if criterion == "ChagasLoss":
            self.criterion = ChagasLoss(**criterion_kw)
        else:
            self.criterion = setup_criterion(criterion, **criterion_kw)

        # Merge ranking config if missing
        default_ranking_cfg = CFG(
            enable=False,
            type="hinge",
            weight=0.3,
            margin=0.5,
        )
        if not hasattr(self.config, "ranking"):
            self.config.ranking = default_ranking_cfg
        else:
            for k, v in default_ranking_cfg.items():
                self.config.ranking.setdefault(k, v)

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
                )
            else:
                raise ValueError(f"Unknown ranking type {self.config.ranking.type}")
            self.ranking_weight = float(self.config.ranking.weight)
        else:
            self.ranking_criterion = None
            self.ranking_weight = 0.0

        self.softmax = nn.Softmax(dim=-1)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze the backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

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
            - "demographics" : torch.Tensor, optional
                Demographic features, required if demographic encoder is enabled.

        Returns
        -------
        dict
            Predictions, including "chagas", "chagas_logits", "chagas_prob",
            and "chagas_loss".

        """
        batch_size, n_leads, sig_len = input_tensors["signals"].shape
        x = input_tensors["signals"].to(self.dtype).to(self.device)  # (B, 12, L)
        x = self.inputer(x)

        # Backbone
        if self.backbone_type == "st-mem":
            # ST-MEM input: (B, 12, L) -> output: (B, embed_dim)
            features = self.backbone(x)
        else:  # self.backbone_type == "hubert":
            features_seq = self.backbone(x)  # (B*C, T, D)
            features = features_seq.last_hidden_state.mean(dim=1)  # Global Pool over time -> (B*C, D)

        if self.dem_encoder is not None:
            if "demographics" not in input_tensors:
                raise ValueError("Demographic features are required by the model but not found in input_tensors.")
            x_dem = input_tensors["demographics"].to(self.dtype).to(self.device)
            if self.dem_encoder.mode == "film":
                scale, shift = self.dem_encoder(x_dem)
                features = self.dem_encoder.modulate_features(features, scale, shift)
            else:  # concat
                dem_feats = self.dem_encoder(x_dem)
                features = torch.cat([features, dem_feats], dim=1)

        # Head
        chagas_logits = self.head(features)
        chagas_prob = self.softmax(chagas_logits)
        chagas_pred = torch.argmax(chagas_prob, dim=-1)

        # Loss Calculation
        chagas_loss = None
        ranking_loss = None
        if "chagas" in input_tensors:
            labels_in = input_tensors["chagas"]
            labels_in = labels_in.to(self.device)

            if labels_in.ndim > 1:
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
                ranking_loss = self.ranking_criterion(pos_channel_scores, hard_labels)  # type: ignore
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
        demographics: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
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
        demographics : array-like, optional
            Demographic features corresponding to the input signals. Required if demographic encoder is enabled.
            Accepts:
              - (n_demographic_features,) for single signal
              - (B, n_demographic_features) for batch of signals
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

        forward_input = {"signals": sig_t}

        if self.config.dem_encoder.enable:
            if demographics is None:
                raise ValueError("Demographic features are required by the model but not provided.")
            if isinstance(demographics, list):
                demographics = np.asarray(demographics)
            if isinstance(demographics, np.ndarray):
                demographics_t = torch.as_tensor(demographics, dtype=self.dtype, device=self.device)
            else:
                demographics_t = demographics.to(self.device).to(self.dtype)

            if demographics_t.ndim == 1:  # (n_demographic_features,) -> (1, n_demographic_features)
                demographics_t = demographics_t.unsqueeze(0)
            elif demographics_t.ndim != 2:
                raise ValueError(
                    f"Unsupported demographics shape {demographics_t.shape}, expected "
                    "(n_demographic_features,) or (B, n_demographic_features)"
                )
            if demographics_t.shape[0] != B:
                raise ValueError(
                    f"Batch size of demographics ({demographics_t.shape[0]}) does not match " f"that of signals ({B})"
                )
            forward_input["demographics"] = demographics_t
        else:
            demographics_t = None

        if self.backbone_type == "st-mem" and not crop_infer:
            crop_infer = True
            # ST-MEM was pretrained on 75*31=2325 samples at 250Hz (≈9.3s),
            # so multi-crop inference is needed for longer signals.
            warnings.warn(
                "ST-MEM backbone is designed for inputs of length 75*N samples, "
                "where N is at most 31. For longer signals, multi-crop inference will be applied automatically.",
                UserWarning,
            )

        if not crop_infer:
            # Full signal inference (Foundation models handle variable length usually via pooling/attention)
            forward_output = self.forward(forward_input)
            output = CINC2025Outputs.from_dict(forward_output)
            self.train(training)
            return output

        # Multi-crop path
        fs = float(self.fs)
        default_crop_len_sec = 4096.0 / fs if self.backbone_type != "st-mem" else 75 * 31 / fs
        default_stride_sec = 1024.0 / fs if self.backbone_type != "st-mem" else 75 * 8 / fs
        crop_len_sec = crop_len if crop_len is not None else default_crop_len_sec
        stride_sec = stride if stride is not None else default_stride_sec

        crop_len_samples = int(round(crop_len_sec * fs))
        stride_samples = int(round(stride_sec * fs))
        crop_len_samples = max(1, crop_len_samples)
        stride_samples = max(1, stride_samples)

        batch_logits: List[torch.Tensor] = []
        batch_probs: List[torch.Tensor] = []
        # batch_preds: List[torch.Tensor] = [] # unused locally

        for b in range(B):
            x = sig_t[b]  # (C, Tb)
            Tb = x.shape[-1]

            # Short signal: no cropping
            if Tb <= crop_len_samples:
                if self.backbone_type == "st-mem":
                    # ST-MEM was pretrained on 75*N samples, so we can pad to the nearest 75*N if shorter than crop_len_samples
                    pad_len = (crop_len_samples - Tb) % 75
                    if pad_len > 0:
                        x = torch.nn.functional.pad(x, (0, pad_len), mode="constant", value=0)
                fo = self.forward({"signals": x.unsqueeze(0)})
                logits_full = fo["chagas_logits"]
                probs_full = fo["chagas_prob"]
                batch_logits.append(logits_full.squeeze(0))
                batch_probs.append(probs_full.squeeze(0))
                continue

            # Compute starting indices
            starts = list(range(0, Tb - crop_len_samples + 1, stride_samples))
            if starts[-1] != Tb - crop_len_samples:
                starts.append(Tb - crop_len_samples)

            # Stack windows: (Ncrops, C, crop_len_samples)
            windows = torch.stack(
                [x[..., s : s + crop_len_samples] for s in starts],
                dim=0,
            )
            forward_input = {"signals": windows}
            if self.config.dem_encoder.enable:
                assert demographics_t is not None, "Demographic features are required by the model but not provided."
                dem_feats = demographics_t[b].unsqueeze(0).repeat(windows.shape[0], 1)
                forward_input["demographics"] = dem_feats
            fo = self.forward(forward_input)
            logits_all = fo["chagas_logits"]
            probs_all = fo["chagas_prob"]

            pos_probs = probs_all[:, 1]

            if agg == "max":
                idx = torch.argmax(pos_probs)
                agg_logits = logits_all[idx]
                agg_probs = torch.softmax(agg_logits, dim=-1)
            elif agg == "top2_mean":
                k = min(2, probs_all.shape[0])
                vals, indices = torch.topk(pos_probs, k)
                mean_pos = vals.mean()
                agg_probs = torch.tensor(
                    [1.0 - mean_pos.item(), mean_pos.item()],
                    device=probs_all.device,
                    dtype=probs_all.dtype,
                )
                agg_logits = logits_all[indices[0]]  # Just for shape/type
            elif agg == "mean":
                agg_probs = probs_all.mean(dim=0)
                agg_logits = logits_all.mean(dim=0)
            else:
                raise ValueError(f"Unsupported agg '{agg}'")

            batch_logits.append(agg_logits)
            batch_probs.append(agg_probs)

        logits_tensor = torch.stack(batch_logits, dim=0)
        probs_tensor = torch.stack(batch_probs, dim=0)
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
    def inference_CINC2025(
        self,
        sig: Union[np.ndarray, torch.Tensor, list],
        demographics: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
        crop_infer: bool = False,
        crop_len: Optional[float] = None,
        stride: Optional[float] = None,
        agg: Literal["max", "top2_mean", "mean"] = "max",
    ) -> CINC2025Outputs:
        """alias for `self.inference`"""
        return self.inference(sig, demographics, crop_infer, crop_len, stride, agg)

    def save(self, path: Union[str, bytes, os.PathLike], train_config: CFG) -> None:
        """Save the model to disk."""
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


# Helper function (Same as yours)
def make_safe_globals(obj: CFG, remove_paths: bool = True) -> CFG:
    """Make a dictionary or a dictionary-like object safe for serialization."""
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

"""
Auxiliary loss functions for training.
"""

from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PairwiseRankingLossHinge",
    "PairwiseRankingLossLogistic",
    "AdaptiveLogisticPairwiseLoss",
    "ChagasLoss",
]


class PairwiseRankingLossHinge(nn.Module):
    """Hinge pairwise ranking loss.

    loss = mean( max(0, margin - (s_pos - s_neg)) )

    Parameters
    ----------
    margin : float
        Margin value for the hinge loss.

    """

    def __init__(self, margin: float = 0.5) -> None:
        print("Using PairwiseRankingLossHinge")
        super().__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        scores : torch.Tensor
            Predicted scores (logits) for each sample, shape (B,).
        labels : torch.Tensor
            Ground truth binary labels (0 or 1), shape (B,).

        Returns
        -------
        torch.Tensor
            Computed hinge pairwise ranking loss.

        """
        # scores: (B,) logit for positive class; labels: (B,) 0/1
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            return scores.sum() * 0.0  # keep differentiable
        # broadcast: (P,1) - (1,N) -> (P,N)
        diff = self.margin - (pos.unsqueeze(1) - neg.unsqueeze(0))
        loss = F.relu(diff)
        return loss.mean()


class PairwiseRankingLossLogistic(nn.Module):
    """Logistic pairwise ranking loss.

    loss = mean( softplus( - (s_pos - s_neg) ) )

    Parameters
    ----------
    margin : float
        Margin value (default 0.0).

    """

    def __init__(self, margin: float = 0.0) -> None:
        print("Using PairwiseRankingLossLogistic")
        super().__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        scores : torch.Tensor
            Predicted scores (logits) for each sample, shape (B,).
        labels : torch.Tensor
            Ground truth binary labels (0 or 1), shape (B,).

        Returns
        -------
        torch.Tensor
            Computed logistic pairwise ranking loss.

        """
        # scores: (B,) logit for positive class; labels: (B,)
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            return scores.sum() * 0.0
        # broadcast: (P,1) - (1,N) => (P,N)
        diff = pos.unsqueeze(1) - neg.unsqueeze(0) - self.margin
        loss = F.softplus(-diff).mean()  # softplus(-diff) = log(1+exp(-diff))
        return loss


class AdaptiveLogisticPairwiseLoss(nn.Module):
    """Adaptive margin + hard negative + subsampling for logistic pairwise ranking loss.

    loss = softplus(margin - (s_pos - s_neg))

    Parameters
    ----
    margin : float, default 0.0
        Initial margin value.
    hard_negative_pct : float, default 0.1
        Percentage of top-scoring negatives to consider (0 < p <= 1.0).
    subsample_pos : int, default 32
        Maximum positive pairs to sample (uses all if insufficient).
    subsample_neg : int, default 160
        Maximum negative pairs to sample (applied after hard negative selection).
    adaptive_margin : bool, default True
        Enable adaptive margin adjustment.
    target_active_ratio : float, default 0.2
        Target active pair ratio (0-1).
    grad_threshold : float, default 0.1
        Gradient threshold for determining "active" pairs (g = sigmoid(-(diff - margin))).
    adapt_lr : float, default 0.05
        Learning rate for margin updates.
    margin_min, margin_max : float, default -0.5, 2.0
        Clipping range for margin values.
    margin_update_interval : int, default 50
        Update interval for margin (number of forward passes).
    return_stats : bool, default False
        Return statistics (active_ratio, margin) for logging.

    """

    def __init__(
        self,
        margin: float = 0.0,
        hard_negative_pct: float = 0.1,
        subsample_pos: int = 32,
        subsample_neg: int = 160,
        adaptive_margin: bool = True,
        target_active_ratio: float = 0.2,
        grad_threshold: float = 0.1,
        adapt_lr: float = 0.05,
        margin_min: float = -0.5,
        margin_max: float = 2.0,
        margin_update_interval: int = 50,
        return_stats: bool = False,
    ) -> None:
        print("Using AdaptiveLogisticPairwiseLoss")
        super().__init__()
        assert 0 < hard_negative_pct <= 1.0
        self.margin = nn.Parameter(torch.tensor(float(margin)), requires_grad=False)
        self.hard_negative_pct = hard_negative_pct
        self.subsample_pos = subsample_pos
        self.subsample_neg = subsample_neg
        self.adaptive_margin = adaptive_margin
        self.target_active_ratio = target_active_ratio
        self.grad_threshold = grad_threshold
        self.adapt_lr = adapt_lr
        self.margin_min = margin_min
        self.margin_max = margin_max
        self.margin_update_interval = margin_update_interval
        self.return_stats = return_stats

        self.register_buffer("_step", torch.zeros(1, dtype=torch.long))

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        scores : torch.Tensor
            Predicted scores (logits) for each sample, shape (B,).
        labels : torch.Tensor
            Ground truth binary labels (0 or 1), shape (B,).
        Returns
        -------
        torch.Tensor or dict
            Computed logistic pairwise ranking loss.
            If `return_stats` is True, returns a dict with keys:
            - "loss": the computed loss
            - "ranking_active_ratio": the active pair ratio
            - "ranking_margin": the current margin value

        """
        device = scores.device
        pos = scores[labels == 1]  # Positive samples
        neg = scores[labels == 0]  # Negative samples
        if pos.numel() == 0 or neg.numel() == 0:
            dummy = scores.sum() * 0.0  # Zero loss if missing positives/negatives
            if self.return_stats:
                return {
                    "loss": dummy,
                    "ranking_active_ratio": torch.tensor(0.0, device=device),
                    "ranking_margin": self.margin.detach(),
                }
            return dummy

        # 1. Select hard negatives
        if self.hard_negative_pct < 1.0 and neg.numel() > 1:
            k = max(1, int(neg.numel() * self.hard_negative_pct))
            # Top-k highest scoring negatives
            topk_vals, _ = torch.topk(neg, k)
            neg = topk_vals

        # 2. Subsampling (to reduce O(P*N) complexity)
        if self.subsample_pos > 0 and pos.numel() > self.subsample_pos:
            idx = torch.randint(0, pos.numel(), (self.subsample_pos,), device=device)
            pos = pos[idx]
        if self.subsample_neg > 0 and neg.numel() > self.subsample_neg:
            idx = torch.randint(0, neg.numel(), (self.subsample_neg,), device=device)
            neg = neg[idx]

        # 3. Compute pairwise differences
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)  # (P, N_hard)
        margin_val = self.margin
        loss_mat = F.softplus(margin_val - diff)  # (P, N)
        loss = loss_mat.mean()

        # 4. Calculate active pair ratio
        with torch.no_grad():
            # Gradient strength w.r.t diff:
            # d/diff [softplus(margin - diff)] = -sigmoid(margin - diff)
            grad_strength = torch.sigmoid(margin_val - diff)  # (P, N)
            active_ratio = (grad_strength > self.grad_threshold).float().mean()

            # 5. Adaptive margin update
            self._step += 1
            if self.adaptive_margin and (self._step.item() % self.margin_update_interval == 0):
                # Adjust margin to approach target active ratio
                delta = self.target_active_ratio - active_ratio
                new_margin = torch.clamp(
                    margin_val + self.adapt_lr * delta,
                    self.margin_min,
                    self.margin_max,
                )
                self.margin.data.copy_(new_margin.detach())

        if self.return_stats:
            return {
                "loss": loss,
                "ranking_active_ratio": active_ratio.detach(),
                "ranking_margin": self.margin.detach(),
            }
        return loss


class ChagasLoss(nn.Module):
    """
    Custom Loss function for Chagas Disease Detection Challenge.

    It wraps BCEWithLogitsLoss with class-specific positive weights to handle
    severe class imbalance and maximize the Top-5% Recall metric.
    This implementation supports One-Hot Encoded targets and logits.

    Parameters
    ----------
    positive_weight_factor : float, optional
        The weight multiplier for the positive class(es).
        Default is 5.0 (aggressive weighting for high recall).
    num_classes : int, optional
        Number of output classes. Default is 2.

    Attributes
    ----------
    pos_weight : torch.Tensor
        Buffer containing the weight vector [1.0, factor, ...].
    bce_loss : nn.BCEWithLogitsLoss
        The underlying PyTorch loss function.

    """

    def __init__(self, positive_weight_factor: float = 5.0, num_classes: int = 2):
        super().__init__()
        # Construct the weight vector.
        # Index 0 is assumed to be 'Negative' (Background), weight = 1.0.
        # Index 1+ are 'Positive' classes, weight = positive_weight_factor.
        # For binary classification (one-hot), weights will be [1.0, 5.0].
        weights = torch.ones(num_classes)
        weights[1:] = positive_weight_factor

        # Register as a buffer so it moves to device automatically and is saved in state_dict.
        self.register_buffer("pos_weight", weights)

        # Initialize BCEWithLogitsLoss.
        # reduction='none' allows for future extensions (e.g., hard example mining).
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted binary cross entropy loss.

        Parameters
        ----------
        logits : torch.Tensor
            Model output (raw scores before Sigmoid/Softmax).
            Shape: ``(batch_size, num_classes)``.
        targets : torch.Tensor
            Soft or Hard One-Hot encoded labels.
            Shape: ``(batch_size, num_classes)``.
            Example: ``[[0.9, 0.1], [0.3, 0.7]]``.

        Returns
        -------
        torch.Tensor
            Scalar loss value (averaged over the batch).

        Raises
        ------
        AssertionError
            If the shapes of logits and targets do not match.
            If the number of classes in logits does not match `num_classes`.

        """
        # Shape Assertions
        # Check if input shapes match (B, C) vs (B, C)
        assert logits.shape == targets.shape, (
            f"Shape Mismatch! Logits: {logits.shape}, Targets: {targets.shape}. "
            f"Ensure your model output dimension matches the label dimension."
        )
        # Check if class dimension matches the initialized weights
        assert logits.shape[1] == self.pos_weight.shape[0], (
            f"Dimension Mismatch! Logits have {logits.shape[1]} classes, "
            f"but Loss was initialized for {self.pos_weight.shape[0]} classes."
        )

        loss = self.bce_loss(logits, targets)

        return loss.mean()

"""
Auxiliary loss functions for training.
"""

import torch
import torch.nn.functional as F

__all__ = [
    "PairwiseRankingLossHinge",
    "PairwiseRankingLossLogistic",
]


class PairwiseRankingLossHinge(torch.nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # scores: (B,) logit for positive class; labels: (B,) 0/1
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            return scores.sum() * 0.0  # keep differentiable
        # broadcast: (P,1) - (1,N) -> (P,N)
        diff = self.margin - (pos.unsqueeze(1) - neg.unsqueeze(0))
        loss = F.relu(diff)
        return loss.mean()


class PairwiseRankingLossLogistic(torch.nn.Module):
    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            return scores.sum() * 0.0
        # broadcast: (P,1) - (1,N) => (P,N)
        diff = pos.unsqueeze(1) - neg.unsqueeze(0) - self.margin
        loss = F.softplus(-diff).mean()  # softplus(-diff) = log(1+exp(-diff))
        return loss

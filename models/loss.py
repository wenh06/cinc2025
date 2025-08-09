"""
Auxiliary loss functions for training.
"""

import torch

__all__ = ["PairwiseRankingLoss"]


class PairwiseRankingLoss(torch.nn.Module):

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pos_indices = (y_true == 1).nonzero(as_tuple=True)[0]
        neg_indices = (y_true == 0).nonzero(as_tuple=True)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return torch.tensor(0.0, requires_grad=True, device=y_pred.device)

        pos_preds = y_pred[pos_indices]
        neg_preds = y_pred[neg_indices]

        pos_preds = pos_preds.unsqueeze(1)  # [num_pos, 1]
        neg_preds = neg_preds.unsqueeze(0)  # [1, num_neg]

        diff = self.margin - (pos_preds - neg_preds)  # [num_pos, num_neg]
        loss = torch.relu(diff)

        return loss.mean()

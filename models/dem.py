import torch
import torch.nn as nn

__all__ = ["DemographicEncoder"]


class DemographicEncoder(nn.Module):
    """
    Demographic features encoder with two fusion modes.

    Parameters
    ----------
    feature_dim : int
        Dimension of the ECG feature vector.
    dem_input_dim : int, optional
        Number of demographic features. Default is 2 (Age, Sex).
    mode : str, optional
        Fusion mode: 'film' or 'concat'. Default is 'film'.
    hidden_dim : int, optional
        Hidden layer dimension. Default is 64.
    """

    def __init__(self, feature_dim: int, dem_input_dim: int = 2, mode: str = "film", hidden_dim: int = 64) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.mode = mode.lower()

        if self.mode == "film":
            self.encoder = nn.Sequential(
                nn.Linear(dem_input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim * 2),
            )
        elif self.mode == "concat":
            self.encoder = nn.Sequential(
                nn.Linear(dem_input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, feature_dim)
            )
        else:
            raise ValueError(f"mode must be 'film' or 'concat', got {mode}")

    def forward(self, x_dem: torch.Tensor):
        """
        Forward pass.

        Returns
        -------
        If mode='film': (scale, shift) tuple
        If mode='concat': demographic feature vector
        """
        if self.mode == "film":
            dem_out = self.encoder(x_dem)
            scale, shift = dem_out.chunk(2, dim=1)
            return scale, shift
        else:  # concat
            return self.encoder(x_dem)

    @staticmethod
    def modulate_features(features: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation: y = x * (1 + gamma) + beta"""
        return features * (1.0 + scale) + shift

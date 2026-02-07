"""
Powerline Noise Augmentation
"""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_ecg.augmenters.base import Augmenter

__all__ = [
    "PowerlineNoise",
]


class PowerlineNoise(Augmenter):
    """
    Add synthetic sinusoidal powerline noise (of specific frequencies) to ECGs.

    This augmentation simulates powerline interference, preventing the model
    from relying on specific noise frequencies (which might correlate with
    datasets/regions) as spurious features.

    Parameters
    ----------
    fs : int
        Sampling frequency of the ECGs to be augmented.
    prob : float, default 0.5
        Probability of applying the augmentation to a sample.
    amplitude_ratio_range : Sequence[float], default ``[0.01, 0.05]``
        Range ``[min, max]`` of noise amplitude relative to the signal's
        absolute maximum amplitude.
    frequencies : Sequence[int], default ``[50, 60]``
        List of possible powerline frequencies to sample from.
    inplace : bool, default True
        Whether to perform the augmentation inplace.
    kwargs : dict, optional
        Additional keyword arguments.

    Examples
    --------
    .. code-block:: python

        pln = PowerlineNoise(fs=500, prob=0.5)
        sig = torch.randn(32, 12, 5000)
        sig, _ = pln(sig, None)

    """

    __name__ = "PowerlineNoise"

    def __init__(
        self,
        fs: int,
        prob: float = 0.5,
        amplitude_ratio_range: Sequence[float] = [0.02, 0.08],
        frequencies: Sequence[int] = [50, 60],
        inplace: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.fs = fs
        self.prob = prob
        self.amplitude_ratio_range = amplitude_ratio_range
        self.frequencies = frequencies
        self.inplace = inplace

    def forward(
        self,
        sig: Tensor,
        label: Optional[Tensor],
        *extra_tensors: Sequence[Tensor],
        **kwargs: Any,
    ) -> Tuple[Tensor, ...]:
        """
        Forward method of the PowerlineNoise augmenter.

        Parameters
        ----------
        sig : torch.Tensor
            Batched ECGs to be augmented, of shape ``(batch, lead, siglen)``.
        label : torch.Tensor, optional
            Batched labels of the ECGs.
        *extra_tensors : Sequence[torch.Tensor], optional
            Extra tensors to be augmented.
        **kwargs: dict, optional
            Additional keyword arguments.

        Returns
        -------
        sig : torch.Tensor
            The augmented ECGs.
        label : torch.Tensor
            The labels (unchanged).
        extra_tensors : Sequence[torch.Tensor], optional
            The extra tensors (unchanged).
        """
        batch, lead, siglen = sig.shape
        device = sig.device
        dtype = sig.dtype

        if not self.inplace:
            sig = sig.clone()

        # Generate Probability Mask
        # Shape: (batch,) -> True if we apply noise, False otherwise
        # We perform calculations for the whole batch but zero out noise later
        # to avoid branching and keep it fully vectorized.
        apply_mask = torch.rand(batch, device=device, dtype=dtype) < self.prob

        # If no samples need augmentation, return early (optimization)
        if not apply_mask.any():
            return (sig, label, *extra_tensors)

        # Select Frequencies
        # Convert frequencies list to tensor on device
        freq_pool = torch.tensor(self.frequencies, device=device, dtype=dtype)
        # Randomly select indices: Shape (batch,)
        freq_indices = torch.randint(0, len(self.frequencies), (batch,), device=device)
        # Gather frequencies: Shape (batch, 1) for broadcasting
        batch_freqs = freq_pool[freq_indices].view(batch, 1)

        # Generate Phases
        # Random phase in [0, 2pi]: Shape (batch, 1)
        batch_phases = torch.rand(batch, 1, device=device, dtype=dtype) * 2 * np.pi

        # Generate Time Vector
        # Shape (1, siglen)
        t = torch.arange(siglen, device=device, dtype=dtype).view(1, siglen) / self.fs

        # Compute Sinusoid
        # Broadcasting: (B, 1) * (1, L) + (B, 1) -> (B, L)
        # noise shape: (batch, siglen)
        noise = torch.sin(2 * np.pi * batch_freqs * t + batch_phases)

        # Compute Amplitudes
        # Get max amplitude per sample across all leads: Shape (batch,)
        # Note: dim=(1,2) reduces (batch, lead, siglen) to (batch,)
        sample_max_val = sig.abs().amax(dim=(1, 2))

        # Generate random scale ratios: Shape (batch,)
        min_r, max_r = self.amplitude_ratio_range
        ratios = torch.rand(batch, device=device, dtype=dtype) * (max_r - min_r) + min_r

        # Final noise amplitude: Shape (batch, 1)
        noise_amps = (sample_max_val * ratios).view(batch, 1)

        # Apply Mask and Scale
        # Zero out noise for samples that shouldn't be augmented
        noise = noise * noise_amps * apply_mask.view(batch, 1)

        # Add to Signal
        # Broadcast noise (batch, siglen) -> (batch, 1, siglen) to match (batch, lead, siglen)
        # This adds the same noise pattern (Common Mode) to all leads.
        sig = sig + noise.unsqueeze(1)

        return (sig, label, *extra_tensors)

    def extra_repr_keys(self) -> List[str]:
        return [
            "fs",
            "prob",
            "amplitude_ratio_range",
            "frequencies",
            "inplace",
        ] + super().extra_repr_keys()

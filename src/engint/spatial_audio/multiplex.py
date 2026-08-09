from __future__ import annotations

import numpy as np

from .panning import RingPanner


class SpatiotemporalMultiplexer:
    """Distributes one signal across speakers as its direction moves in time.

    Panning gains are evaluated at block boundaries and linearly
    interpolated per sample inside each block, so energy crossfades
    smoothly between speakers (no zipper noise) while remaining
    sample-accurate to the trajectory at every boundary.
    """

    def __init__(self, panner: RingPanner, block_size: int = 256) -> None:
        if block_size < 1:
            raise ValueError("block size must be >= 1")
        self.panner = panner
        self.block_size = block_size

    def render(self, signal: np.ndarray, azimuths_rad: np.ndarray) -> np.ndarray:
        """Return shape [n_speakers, n_samples] speaker feeds."""
        signal = np.asarray(signal, dtype=float)
        azimuths = np.asarray(azimuths_rad, dtype=float)
        if signal.ndim != 1:
            raise ValueError("signal must be 1D")
        if azimuths.shape != signal.shape:
            raise ValueError("azimuth trajectory must match signal length")
        n = signal.size
        if n == 0:
            raise ValueError("signal is empty")

        boundaries = np.arange(0, n, self.block_size)
        boundaries = np.append(boundaries, n - 1)
        boundary_gains = np.stack([self.panner.gains(azimuths[i]) for i in boundaries])
        sample_index = np.arange(n)
        gains = np.empty((boundary_gains.shape[1], n))
        for ch in range(boundary_gains.shape[1]):
            gains[ch] = np.interp(sample_index, boundaries, boundary_gains[:, ch])
        return gains * signal[None, :]

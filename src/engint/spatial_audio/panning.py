from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


class RingPanner:
    """Pairwise amplitude panner (2D VBAP) over azimuth-sorted speakers.

    Gains are power-normalized (sum of squares == 1) so perceived
    loudness is constant while energy sweeps around the ring.
    Panning accuracy degrades for pairs subtending more than pi
    radians; such gaps are allowed but gains are clamped nonnegative.
    """

    def __init__(self, azimuths_rad: np.ndarray, min_gap_rad: float = 1e-6) -> None:
        azimuths = np.asarray(azimuths_rad, dtype=float) % TWO_PI
        if azimuths.size < 2:
            raise ValueError("panner requires at least 2 azimuths")
        self._order = np.argsort(azimuths)
        self._sorted = azimuths[self._order]
        gaps = np.diff(np.concatenate([self._sorted, self._sorted[:1] + TWO_PI]))
        if np.any(gaps < min_gap_rad):
            raise ValueError("two speakers share the same azimuth; panning is singular")
        self._count = azimuths.size

    def gains(self, azimuth_rad: float) -> np.ndarray:
        target = float(azimuth_rad) % TWO_PI
        lo = int(np.searchsorted(self._sorted, target, side="right")) - 1
        lo %= self._count
        hi = (lo + 1) % self._count

        a_lo = self._sorted[lo]
        a_hi = self._sorted[hi]
        basis = np.array(
            [
                [np.cos(a_lo), np.cos(a_hi)],
                [np.sin(a_lo), np.sin(a_hi)],
            ]
        )
        direction = np.array([np.cos(target), np.sin(target)])
        pair = np.linalg.solve(basis, direction)
        pair = np.maximum(pair, 0.0)
        norm = np.linalg.norm(pair)
        if norm == 0.0:
            # Target diametrically opposes a wide pair; fall back to the nearer edge.
            pair = np.array([1.0, 0.0]) if _arc(a_lo, target) <= _arc(target, a_hi) else np.array([0.0, 1.0])
            norm = 1.0
        pair /= norm

        out = np.zeros(self._count)
        out[self._order[lo]] = pair[0]
        out[self._order[hi]] = pair[1]
        return out


def energy_vector_error_rad(gains: np.ndarray, azimuths_rad: np.ndarray, target_azimuth_rad: float) -> float:
    """Angle between the Gerzon energy vector of `gains` and the target direction.

    This is the truthful localization metric: 0 means the energy
    centroid points exactly where the source was panned.
    """
    gains = np.asarray(gains, dtype=float)
    energy = np.square(gains)
    total = energy.sum()
    if total <= 0.0:
        raise ValueError("gains carry no energy")
    directions = np.stack([np.cos(azimuths_rad), np.sin(azimuths_rad)], axis=1)
    r_e = energy @ directions / total
    magnitude = np.linalg.norm(r_e)
    if magnitude == 0.0:
        return np.pi
    target = np.array([np.cos(target_azimuth_rad), np.sin(target_azimuth_rad)])
    cosine = float(np.clip(r_e @ target / magnitude, -1.0, 1.0))
    return float(np.arccos(cosine))


def _arc(from_rad: float, to_rad: float) -> float:
    return (to_rad - from_rad) % TWO_PI

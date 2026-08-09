from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import SPEED_OF_SOUND_M_S, SpeakerLayout


@dataclass(slots=True)
class AcousticMap:
    """Measured geometry of the array relative to one listening position.

    All arrays are indexed by speaker in layout order.
    - distances_m: straight-line speaker->listener distance.
    - propagation_delays_s: acoustic travel time speaker->listener.
    - alignment_delays_s: extra electronic delay per speaker so every
      speaker's wavefront arrives at the listener simultaneously
      (closer speakers are delayed more; the farthest gets zero).
    - compensation_gains: linear gain equalizing arrival level at the
      listener under 1/r spreading, normalized so the maximum is 1.0
      (compensation never amplifies, only attenuates).
    - unit_vectors: unit direction listener->speaker.
    - azimuths_rad: horizontal-plane angle of each unit vector, atan2(y, x).
    """

    distances_m: np.ndarray
    propagation_delays_s: np.ndarray
    alignment_delays_s: np.ndarray
    compensation_gains: np.ndarray
    unit_vectors: np.ndarray
    azimuths_rad: np.ndarray


def build_acoustic_map(
    layout: SpeakerLayout,
    listener_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
    min_distance_m: float = 1e-3,
) -> AcousticMap:
    if speed_of_sound_m_s <= 0.0:
        raise ValueError("speed of sound must be positive")
    listener = np.asarray(listener_position_m, dtype=float)
    offsets = layout.positions() - listener[None, :]
    distances = np.linalg.norm(offsets, axis=1)
    if np.any(distances < min_distance_m):
        raise ValueError("a speaker coincides with the listening position")

    unit_vectors = offsets / distances[:, None]
    propagation_delays = distances / speed_of_sound_m_s
    alignment_delays = propagation_delays.max() - propagation_delays
    gains = distances / distances.max()
    azimuths = np.arctan2(unit_vectors[:, 1], unit_vectors[:, 0])
    return AcousticMap(
        distances_m=distances,
        propagation_delays_s=propagation_delays,
        alignment_delays_s=alignment_delays,
        compensation_gains=gains,
        unit_vectors=unit_vectors,
        azimuths_rad=azimuths,
    )

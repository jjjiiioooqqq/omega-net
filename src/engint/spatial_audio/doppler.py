from __future__ import annotations

import numpy as np

from .geometry import SPEED_OF_SOUND_M_S


def arrival_times_samples(
    distances_m: np.ndarray,
    sample_rate_hz: int,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
) -> np.ndarray:
    """Arrival time (in output samples) of each emitted sample.

    Sample n emitted at distance d[n] arrives at a[n] = n + fs*d[n]/c.
    Arrival times must be strictly increasing, which holds exactly when
    the radial closing speed stays below the speed of sound; supersonic
    approach is rejected rather than silently folded.
    """
    distances = np.asarray(distances_m, dtype=float)
    if distances.ndim != 1 or distances.size == 0:
        raise ValueError("distances must be non-empty 1D")
    if np.any(distances < 0.0):
        raise ValueError("distances must be nonnegative")
    if speed_of_sound_m_s <= 0.0:
        raise ValueError("speed of sound must be positive")
    arrivals = np.arange(distances.size) + distances * sample_rate_hz / speed_of_sound_m_s
    if np.any(np.diff(arrivals) <= 0.0):
        raise ValueError("radial closing speed reaches the speed of sound; Doppler model invalid")
    return arrivals


def resample_at_arrival(
    values: np.ndarray,
    arrivals: np.ndarray,
    out_len: int,
    fill_edges: bool = False,
) -> np.ndarray:
    """Re-index an emission-time array onto the output (arrival) time axis.

    With fill_edges=False samples outside the arrival span are zero
    (correct for signals: silence before the first wavefront). With
    fill_edges=True the first/last values are held (correct for
    trajectory data like azimuth or distance).
    """
    values = np.asarray(values, dtype=float)
    if values.shape != arrivals.shape:
        raise ValueError("values must match arrival times length")
    t = np.arange(out_len, dtype=float)
    if fill_edges:
        return np.interp(t, arrivals, values)
    return np.interp(t, arrivals, values, left=0.0, right=0.0)


def apply_propagation_delay(
    signal: np.ndarray,
    distances_m: np.ndarray,
    sample_rate_hz: int,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
) -> np.ndarray:
    """Delay `signal` by the time-varying acoustic travel time source->listener.

    Emission-time formulation: each input sample is placed at its true
    arrival time and the waveform is linearly interpolated back onto the
    uniform output grid. A closing source compresses arrivals, so the
    exact Doppler ratio f' = f * c / (c - v_radial) emerges from the
    physics rather than a pitch-shifter.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("signal must be non-empty 1D")
    distances = np.asarray(distances_m, dtype=float)
    if distances.shape != signal.shape:
        raise ValueError("distances must match signal length")
    arrivals = arrival_times_samples(distances, sample_rate_hz, speed_of_sound_m_s)
    out_len = int(np.ceil(arrivals[-1])) + 1
    return resample_at_arrival(signal, arrivals, out_len)

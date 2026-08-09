from __future__ import annotations

import numpy as np

REFERENCE_PRESSURE_KPA = 101.325
_T0_K = 293.15
_T01_K = 273.16


def iso9613_alpha_db_per_m(
    frequencies_hz: np.ndarray,
    temperature_c: float = 20.0,
    relative_humidity_pct: float = 50.0,
    pressure_kpa: float = REFERENCE_PRESSURE_KPA,
) -> np.ndarray:
    """Pure-tone atmospheric absorption coefficient per ISO 9613-1.

    Combines classical (viscous/thermal) absorption with the O2 and N2
    molecular relaxation terms. Valid roughly for 50 Hz-10 kHz per octave
    of the standard's stated ranges; outside them the formula is still
    evaluated but accuracy is not certified.
    """
    if not (-73.0 <= temperature_c <= 100.0):
        raise ValueError("temperature outside plausible range")
    if not (0.0 <= relative_humidity_pct <= 100.0):
        raise ValueError("relative humidity must be 0-100%")
    if pressure_kpa <= 0.0:
        raise ValueError("pressure must be positive")

    f = np.asarray(frequencies_hz, dtype=float)
    t_k = 273.15 + temperature_c
    p_ratio = pressure_kpa / REFERENCE_PRESSURE_KPA
    t_ratio = t_k / _T0_K

    psat_ratio = 10.0 ** (-6.8346 * (_T01_K / t_k) ** 1.261 + 4.6151)
    h_molar_pct = relative_humidity_pct * psat_ratio / p_ratio

    fr_o = p_ratio * (24.0 + 4.04e4 * h_molar_pct * (0.02 + h_molar_pct) / (0.391 + h_molar_pct))
    fr_n = (
        p_ratio
        * t_ratio ** (-0.5)
        * (9.0 + 280.0 * h_molar_pct * np.exp(-4.170 * (t_ratio ** (-1.0 / 3.0) - 1.0)))
    )

    classical = 1.84e-11 * (1.0 / p_ratio) * np.sqrt(t_ratio)
    oxygen = 0.01275 * np.exp(-2239.1 / t_k) * fr_o / (fr_o**2 + f**2)
    nitrogen = 0.1068 * np.exp(-3352.0 / t_k) * fr_n / (fr_n**2 + f**2)
    return 8.686 * f**2 * (classical + t_ratio ** (-2.5) * (oxygen + nitrogen))


class AirAbsorptionFilter:
    """Distance-dependent atmospheric absorption applied in overlapped blocks.

    The signal is cut into 50%-overlapped periodic-Hann blocks; each
    block's spectrum is attenuated by 10^(-alpha(f) * d_block / 20) using
    the mean propagation distance over that block, then overlap-added
    (the window pair sums to exactly 1). Zero-phase magnitude-only
    filtering per block — an explicit approximation: no absorption
    dispersion, and distance is quantized to the block hop.
    """

    def __init__(
        self,
        sample_rate_hz: int,
        temperature_c: float = 20.0,
        relative_humidity_pct: float = 50.0,
        pressure_kpa: float = REFERENCE_PRESSURE_KPA,
        block_size: int = 1024,
    ) -> None:
        if block_size < 8 or block_size % 2 != 0:
            raise ValueError("block size must be an even integer >= 8")
        self.sample_rate_hz = sample_rate_hz
        self.block_size = block_size
        self.conditions = {
            "temperature_c": temperature_c,
            "relative_humidity_pct": relative_humidity_pct,
            "pressure_kpa": pressure_kpa,
        }
        freqs = np.fft.rfftfreq(block_size, 1.0 / sample_rate_hz)
        self.alpha_db_per_m = iso9613_alpha_db_per_m(
            freqs, temperature_c, relative_humidity_pct, pressure_kpa
        )
        k = np.arange(block_size)
        self._window = 0.5 * (1.0 - np.cos(2.0 * np.pi * k / block_size))

    def apply(self, signal: np.ndarray, distances_m: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=float)
        distances = np.asarray(distances_m, dtype=float)
        if signal.ndim != 1 or signal.size == 0:
            raise ValueError("signal must be non-empty 1D")
        if distances.shape != signal.shape:
            raise ValueError("distances must match signal length")

        n = signal.size
        hop = self.block_size // 2
        padded = np.zeros(hop + n + self.block_size)
        padded[hop : hop + n] = signal
        dist_padded = np.concatenate(
            [
                np.full(hop, distances[0]),
                distances,
                np.full(self.block_size, distances[-1]),
            ]
        )

        out = np.zeros_like(padded)
        for start in range(0, hop + n, hop):
            block = padded[start : start + self.block_size] * self._window
            d_block = float(np.mean(dist_padded[start : start + self.block_size]))
            gain = 10.0 ** (-self.alpha_db_per_m * d_block / 20.0)
            out[start : start + self.block_size] += np.fft.irfft(
                np.fft.rfft(block) * gain, self.block_size
            )
        return out[hop : hop + n]

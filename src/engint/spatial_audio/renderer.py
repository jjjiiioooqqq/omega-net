from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .acoustic_map import AcousticMap, build_acoustic_map
from .geometry import SPEED_OF_SOUND_M_S, SpeakerLayout
from .multiplex import SpatiotemporalMultiplexer
from .panning import RingPanner, energy_vector_error_rad


@dataclass(slots=True)
class SourceTrajectory:
    """One dry source signal plus its per-sample position relative to the listener."""

    name: str
    signal: np.ndarray
    azimuths_rad: np.ndarray
    distances_m: np.ndarray

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal, dtype=float)
        self.azimuths_rad = np.asarray(self.azimuths_rad, dtype=float)
        self.distances_m = np.asarray(self.distances_m, dtype=float)
        if self.signal.ndim != 1 or self.signal.size == 0:
            raise ValueError(f"source '{self.name}': signal must be non-empty 1D")
        if self.azimuths_rad.shape != self.signal.shape or self.distances_m.shape != self.signal.shape:
            raise ValueError(f"source '{self.name}': trajectory arrays must match signal length")
        if np.any(self.distances_m <= 0.0):
            raise ValueError(f"source '{self.name}': distances must be positive")


@dataclass(slots=True)
class RenderResult:
    output: np.ndarray  # [n_speakers, n_samples]
    acoustic_map: AcousticMap
    sample_rate_hz: int
    metrics: dict[str, float] = field(default_factory=dict)


class SofInityRenderer:
    """SoF-inity: acoustically mapped, spatiotemporally multiplexed speaker rendering.

    Pipeline per source:
      1. distance attenuation (1/r referenced to `reference_distance_m`),
      2. spatiotemporal multiplexing of the signal across the ring (VBAP
         gains interpolated per sample),
    then per speaker channel:
      3. time-alignment delay and level compensation from the acoustic
         map, so all wavefronts arrive coherently at the listener.

    Known limitations (explicit): no Doppler on moving sources, no air
    absorption or room reflections, horizontal-plane panning only
    (speaker elevation is measured but not used for panning).
    """

    def __init__(
        self,
        layout: SpeakerLayout,
        listener_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        sample_rate_hz: int = 48_000,
        block_size: int = 256,
        speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
        reference_distance_m: float = 1.0,
        min_source_distance_m: float = 0.1,
    ) -> None:
        if sample_rate_hz < 8_000:
            raise ValueError("sample rate must be >= 8000 Hz")
        self.layout = layout
        self.sample_rate_hz = sample_rate_hz
        self.reference_distance_m = reference_distance_m
        self.min_source_distance_m = min_source_distance_m
        self.acoustic_map = build_acoustic_map(
            layout, listener_position_m, speed_of_sound_m_s=speed_of_sound_m_s
        )
        self.panner = RingPanner(self.acoustic_map.azimuths_rad)
        self.multiplexer = SpatiotemporalMultiplexer(self.panner, block_size=block_size)

    def render(self, sources: list[SourceTrajectory]) -> RenderResult:
        if not sources:
            raise ValueError("at least one source is required")
        n_samples = max(s.signal.size for s in sources)
        n_speakers = self.layout.count
        mix = np.zeros((n_speakers, n_samples))
        metrics: dict[str, float] = {}

        for source in sources:
            attenuation = self.reference_distance_m / np.maximum(
                source.distances_m, self.min_source_distance_m
            )
            feeds = self.multiplexer.render(source.signal * attenuation, source.azimuths_rad)
            mix[:, : feeds.shape[1]] += feeds
            metrics[f"{source.name}_max_energy_vector_error_rad"] = self._localization_error(source)

        align_samples = self.acoustic_map.alignment_delays_s * self.sample_rate_hz
        pad = int(np.ceil(align_samples.max()))
        output = np.zeros((n_speakers, n_samples + pad))
        t = np.arange(n_samples + pad, dtype=float)
        for ch in range(n_speakers):
            delayed = np.interp(t - align_samples[ch], np.arange(n_samples), mix[ch], left=0.0, right=0.0)
            output[ch] = delayed * self.acoustic_map.compensation_gains[ch]

        peak = float(np.max(np.abs(output))) if output.size else 0.0
        metrics["peak_abs"] = peak
        metrics["headroom_db"] = float(-20.0 * np.log10(peak)) if peak > 0.0 else np.inf
        metrics["output_rms"] = float(np.sqrt(np.mean(np.square(output))))
        return RenderResult(
            output=output,
            acoustic_map=self.acoustic_map,
            sample_rate_hz=self.sample_rate_hz,
            metrics=metrics,
        )

    def _localization_error(self, source: SourceTrajectory) -> float:
        """Worst-case Gerzon energy-vector error over trajectory block boundaries."""
        step = max(self.multiplexer.block_size, 1)
        probes = source.azimuths_rad[::step]
        errors = [
            energy_vector_error_rad(self.panner.gains(a), self.acoustic_map.azimuths_rad, a)
            for a in probes
        ]
        return float(np.max(errors))

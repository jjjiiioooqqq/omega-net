from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .acoustic_map import AcousticMap, build_acoustic_map
from .air_absorption import AirAbsorptionFilter
from .doppler import arrival_times_samples, resample_at_arrival
from .geometry import SPEED_OF_SOUND_M_S, SpeakerLayout
from .multiplex import SpatiotemporalMultiplexer
from .panning import RingPanner, energy_vector_error_rad
from .reflections import ShoeboxRoom


@dataclass(slots=True)
class SourceTrajectory:
    """One dry source signal plus its per-sample position relative to the listener.

    `positions_m` (room-frame [n, 3] coordinates) is optional and only
    required when rendering with room reflections; build via
    `SourceTrajectory.from_positions` to keep azimuth/distance consistent
    with it.
    """

    name: str
    signal: np.ndarray
    azimuths_rad: np.ndarray
    distances_m: np.ndarray
    positions_m: np.ndarray | None = None

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
        if self.positions_m is not None:
            self.positions_m = np.asarray(self.positions_m, dtype=float)
            if self.positions_m.shape != (self.signal.size, 3):
                raise ValueError(f"source '{self.name}': positions must have shape [n_samples, 3]")

    @classmethod
    def from_positions(
        cls,
        name: str,
        signal: np.ndarray,
        positions_m: np.ndarray,
        listener_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> SourceTrajectory:
        positions = np.asarray(positions_m, dtype=float)
        offsets = positions - np.asarray(listener_position_m, dtype=float)[None, :]
        distances = np.linalg.norm(offsets, axis=1)
        azimuths = np.arctan2(offsets[:, 1], offsets[:, 0])
        return cls(
            name=name,
            signal=signal,
            azimuths_rad=azimuths,
            distances_m=distances,
            positions_m=positions,
        )


@dataclass(slots=True)
class RenderResult:
    output: np.ndarray  # [n_speakers, n_samples]
    acoustic_map: AcousticMap
    sample_rate_hz: int
    metrics: dict[str, float] = field(default_factory=dict)


class SofInityRenderer:
    """SoF-inity: acoustically mapped, spatiotemporally multiplexed speaker rendering.

    Each source is expanded into propagation paths (the direct path plus,
    when a room is configured, image-source reflections). Per path:
      1. time-varying propagation delay from the path distance — Doppler
         shift on moving sources falls out of this step,
      2. 1/r distance attenuation (referenced to `reference_distance_m`)
         times the path's wall-reflection gain,
      3. ISO 9613-1 air absorption filtering (when configured),
      4. spatiotemporal multiplexing across the ring (VBAP gains
         interpolated per sample),
    then per speaker channel: time-alignment delay and level compensation
    from the acoustic map, so all wavefronts arrive coherently at the
    listener.

    Known limitations (explicit): horizontal-plane panning only (path
    elevation collapses to its azimuth), specular shoebox reflections
    with uniform frequency-independent wall absorption, and block-wise
    zero-phase air absorption (no dispersion).
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
        air_absorption: AirAbsorptionFilter | None = None,
        room: ShoeboxRoom | None = None,
        reflection_order: int = 1,
        propagation_delay: bool = True,
    ) -> None:
        if sample_rate_hz < 8_000:
            raise ValueError("sample rate must be >= 8000 Hz")
        if air_absorption is not None and air_absorption.sample_rate_hz != sample_rate_hz:
            raise ValueError("air absorption filter sample rate must match renderer")
        if room is not None and not room.contains(np.asarray(listener_position_m, dtype=float)):
            raise ValueError("listener must be inside the room")
        if room is not None and not propagation_delay:
            raise ValueError("room reflections require propagation delay (arrival-time ordering)")
        self.layout = layout
        self.listener_position_m = tuple(float(v) for v in listener_position_m)
        self.sample_rate_hz = sample_rate_hz
        self.speed_of_sound_m_s = speed_of_sound_m_s
        self.reference_distance_m = reference_distance_m
        self.min_source_distance_m = min_source_distance_m
        self.air_absorption = air_absorption
        self.room = room
        self.reflection_order = reflection_order
        self.propagation_delay = propagation_delay
        self.acoustic_map = build_acoustic_map(
            layout, listener_position_m, speed_of_sound_m_s=speed_of_sound_m_s
        )
        self.panner = RingPanner(self.acoustic_map.azimuths_rad)
        self.multiplexer = SpatiotemporalMultiplexer(self.panner, block_size=block_size)

    def render(self, sources: list[SourceTrajectory]) -> RenderResult:
        if not sources:
            raise ValueError("at least one source is required")
        metrics: dict[str, float] = {}
        path_feeds: list[np.ndarray] = []

        for source in sources:
            paths = [(source.azimuths_rad, source.distances_m, 1.0)]
            if self.room is not None:
                if source.positions_m is None:
                    raise ValueError(
                        f"source '{source.name}': room reflections require positions_m "
                        "(build with SourceTrajectory.from_positions)"
                    )
                if not self.room.contains(source.positions_m):
                    raise ValueError(f"source '{source.name}': trajectory leaves the room")
                listener = np.asarray(self.listener_position_m)
                images = self.room.image_paths(source.positions_m, self.reflection_order)
                for image in images:
                    offsets = image.positions_m - listener[None, :]
                    paths.append(
                        (
                            np.arctan2(offsets[:, 1], offsets[:, 0]),
                            np.linalg.norm(offsets, axis=1),
                            image.gain,
                        )
                    )
                metrics[f"{source.name}_image_count"] = float(len(images))

            for azimuths, distances, path_gain in paths:
                path_feeds.append(self._render_path(source.signal, azimuths, distances, path_gain))
            metrics[f"{source.name}_max_energy_vector_error_rad"] = self._localization_error(source)

        n_samples = max(f.shape[1] for f in path_feeds)
        mix = np.zeros((self.layout.count, n_samples))
        for feeds in path_feeds:
            mix[:, : feeds.shape[1]] += feeds

        align_samples = self.acoustic_map.alignment_delays_s * self.sample_rate_hz
        pad = int(np.ceil(align_samples.max()))
        output = np.zeros((self.layout.count, n_samples + pad))
        t = np.arange(n_samples + pad, dtype=float)
        for ch in range(self.layout.count):
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

    def _render_path(
        self,
        signal: np.ndarray,
        azimuths_rad: np.ndarray,
        distances_m: np.ndarray,
        path_gain: float,
    ) -> np.ndarray:
        if self.propagation_delay:
            # Emission-time physics: place each sample at its arrival time
            # (exact Doppler) and pan it toward where the source was when
            # it emitted, not where it is now.
            arrivals = arrival_times_samples(
                distances_m, self.sample_rate_hz, self.speed_of_sound_m_s
            )
            out_len = int(np.ceil(arrivals[-1])) + 1
            signal = resample_at_arrival(signal, arrivals, out_len)
            # Unwrap before interpolating so a pi -> -pi crossing doesn't
            # sweep through the opposite side of the ring.
            azimuths_rad = resample_at_arrival(
                np.unwrap(azimuths_rad), arrivals, out_len, fill_edges=True
            )
            distances_m = resample_at_arrival(distances_m, arrivals, out_len, fill_edges=True)
        attenuation = path_gain * self.reference_distance_m / np.maximum(
            distances_m, self.min_source_distance_m
        )
        signal = signal * attenuation
        if self.air_absorption is not None:
            signal = self.air_absorption.apply(signal, distances_m)
        return self.multiplexer.render(signal, azimuths_rad)

    def _localization_error(self, source: SourceTrajectory) -> float:
        """Worst-case Gerzon energy-vector error over direct-path block boundaries."""
        step = max(self.multiplexer.block_size, 1)
        probes = source.azimuths_rad[::step]
        errors = [
            energy_vector_error_rad(self.panner.gains(a), self.acoustic_map.azimuths_rad, a)
            for a in probes
        ]
        return float(np.max(errors))

"""SoF-inity (Sound of Freedom Infinity) spatial audio for speaker arrays.

Accurate acoustic mapping (measured speaker geometry -> per-speaker
time-alignment and level compensation) plus spatiotemporal multiplexing
(sample-interpolated amplitude panning of moving sources across the
array), with physical propagation modeling — Doppler via time-varying
delay, ISO 9613-1 air absorption, shoebox image-source reflections —
and explicit localization metrics.
"""

from .acoustic_map import AcousticMap, build_acoustic_map
from .air_absorption import AirAbsorptionFilter, iso9613_alpha_db_per_m
from .doppler import apply_propagation_delay, arrival_times_samples, resample_at_arrival
from .geometry import SPEED_OF_SOUND_M_S, Speaker, SpeakerLayout, ring_layout
from .multiplex import SpatiotemporalMultiplexer
from .panning import RingPanner, energy_vector_error_rad
from .reflections import ImagePath, ShoeboxRoom
from .renderer import RenderResult, SofInityRenderer, SourceTrajectory

__all__ = [
    "AcousticMap",
    "AirAbsorptionFilter",
    "ImagePath",
    "RenderResult",
    "RingPanner",
    "ShoeboxRoom",
    "SofInityRenderer",
    "SourceTrajectory",
    "SPEED_OF_SOUND_M_S",
    "SpatiotemporalMultiplexer",
    "Speaker",
    "SpeakerLayout",
    "apply_propagation_delay",
    "arrival_times_samples",
    "build_acoustic_map",
    "resample_at_arrival",
    "energy_vector_error_rad",
    "iso9613_alpha_db_per_m",
    "ring_layout",
]

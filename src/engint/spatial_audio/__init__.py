"""SoF-inity (Sound of Freedom Infinity) spatial audio for speaker arrays.

Accurate acoustic mapping (measured speaker geometry -> per-speaker
time-alignment and level compensation) plus spatiotemporal multiplexing
(sample-interpolated amplitude panning of moving sources across the
array) with explicit localization metrics.
"""

from .acoustic_map import AcousticMap, build_acoustic_map
from .geometry import SPEED_OF_SOUND_M_S, Speaker, SpeakerLayout, ring_layout
from .multiplex import SpatiotemporalMultiplexer
from .panning import RingPanner, energy_vector_error_rad
from .renderer import RenderResult, SofInityRenderer, SourceTrajectory

__all__ = [
    "AcousticMap",
    "RenderResult",
    "RingPanner",
    "SofInityRenderer",
    "SourceTrajectory",
    "SPEED_OF_SOUND_M_S",
    "SpatiotemporalMultiplexer",
    "Speaker",
    "SpeakerLayout",
    "build_acoustic_map",
    "energy_vector_error_rad",
    "ring_layout",
]

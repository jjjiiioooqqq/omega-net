from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEED_OF_SOUND_M_S = 343.0


@dataclass(slots=True)
class Speaker:
    name: str
    position_m: tuple[float, float, float]


@dataclass(slots=True)
class SpeakerLayout:
    speakers: list[Speaker]

    def __post_init__(self) -> None:
        if len(self.speakers) < 2:
            raise ValueError("layout requires at least 2 speakers")
        names = [s.name for s in self.speakers]
        if len(set(names)) != len(names):
            raise ValueError("speaker names must be unique")

    @property
    def count(self) -> int:
        return len(self.speakers)

    def positions(self) -> np.ndarray:
        return np.array([s.position_m for s in self.speakers], dtype=float)


def ring_layout(
    count: int,
    radius_m: float = 2.0,
    height_m: float = 0.0,
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SpeakerLayout:
    """Evenly spaced horizontal ring around `center_m`, speaker 0 toward +x."""
    if count < 2:
        raise ValueError("ring layout requires at least 2 speakers")
    if radius_m <= 0.0:
        raise ValueError("radius must be positive")
    azimuths = 2.0 * np.pi * np.arange(count) / count
    speakers = [
        Speaker(
            name=f"ring_{i:02d}",
            position_m=(
                center_m[0] + radius_m * float(np.cos(a)),
                center_m[1] + radius_m * float(np.sin(a)),
                center_m[2] + height_m,
            ),
        )
        for i, a in enumerate(azimuths)
    ]
    return SpeakerLayout(speakers=speakers)

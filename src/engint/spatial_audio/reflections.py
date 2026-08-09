from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ImagePath:
    """One specular reflection path as a virtual (image) source."""

    positions_m: np.ndarray  # [n_samples, 3]
    wall_hits: int
    gain: float


@dataclass(slots=True)
class ShoeboxRoom:
    """Rectangular room [0,Lx]x[0,Ly]x[0,Lz] with uniform wall absorption.

    Reflections follow the image-source method (Allen & Berkley):
    mirroring the source across walls yields virtual sources whose
    straight-line paths reproduce specular reflections. Each wall hit
    multiplies pressure by sqrt(1 - absorption). Explicit limitations:
    specular-only (no scattering/diffraction), uniform absorption on all
    six walls, and no frequency dependence of the wall reflection.
    """

    dimensions_m: tuple[float, float, float]
    wall_absorption: float = 0.3

    def __post_init__(self) -> None:
        if any(d <= 0.0 for d in self.dimensions_m):
            raise ValueError("room dimensions must be positive")
        if not (0.0 < self.wall_absorption <= 1.0):
            raise ValueError("wall absorption must be in (0, 1]")

    def contains(self, points_m: np.ndarray, margin_m: float = 0.0) -> bool:
        points = np.atleast_2d(np.asarray(points_m, dtype=float))
        dims = np.asarray(self.dimensions_m)
        return bool(np.all(points >= margin_m) and np.all(points <= dims[None, :] - margin_m))

    def image_paths(self, positions_m: np.ndarray, max_order: int = 1) -> list[ImagePath]:
        """Image-source trajectories for a (possibly moving) source.

        positions_m: [n_samples, 3] source positions in room coordinates.
        Returns every image with 1..max_order total wall hits; the direct
        path (0 hits) is excluded because the caller renders it itself.
        """
        positions = np.asarray(positions_m, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape [n_samples, 3]")
        if max_order < 1:
            raise ValueError("max_order must be >= 1")
        if not self.contains(positions):
            raise ValueError("source trajectory leaves the room")

        reflection_gain = float(np.sqrt(1.0 - self.wall_absorption))
        # 1D images of x in [0, L]: 2mL + x with |2m| hits, 2mL - x with |2m-1| hits.
        per_axis: list[list[tuple[np.ndarray, int]]] = []
        for axis in range(3):
            length = self.dimensions_m[axis]
            coord = positions[:, axis]
            candidates: list[tuple[np.ndarray, int]] = []
            for m in range(-(max_order + 1), max_order + 2):
                even_hits = abs(2 * m)
                if even_hits <= max_order:
                    candidates.append((2.0 * m * length + coord, even_hits))
                odd_hits = abs(2 * m - 1)
                if odd_hits <= max_order:
                    candidates.append((2.0 * m * length - coord, odd_hits))
            per_axis.append(candidates)

        paths: list[ImagePath] = []
        for (cx, hx), (cy, hy), (cz, hz) in itertools.product(*per_axis):
            hits = hx + hy + hz
            if hits == 0 or hits > max_order:
                continue
            paths.append(
                ImagePath(
                    positions_m=np.stack([cx, cy, cz], axis=1),
                    wall_hits=hits,
                    gain=reflection_gain**hits,
                )
            )
        return paths

"""Render a source orbiting an 8-speaker ring inside a shoebox room.

Full SoF-inity stack: propagation delay (Doppler), ISO 9613-1 air
absorption, first-order image-source reflections, VBAP spatiotemporal
multiplexing, and acoustic-map speaker alignment. Writes speaker feeds
and truthful metrics.
"""

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from engint.spatial_audio import (
    AirAbsorptionFilter,
    ShoeboxRoom,
    SofInityRenderer,
    SourceTrajectory,
    ring_layout,
)

SAMPLE_RATE_HZ = 48_000
DURATION_S = 4.0


def main(output_dir: Path = Path("output_data")) -> dict:
    n = int(SAMPLE_RATE_HZ * DURATION_S)
    t = np.arange(n) / SAMPLE_RATE_HZ
    signal = 0.5 * np.sin(2 * np.pi * 220.0 * t) * np.hanning(n)

    room = ShoeboxRoom(dimensions_m=(10.0, 8.0, 3.0), wall_absorption=0.35)
    listener = (5.0, 4.0, 1.5)
    orbit_angle = 2 * np.pi * t / DURATION_S
    orbit_radius = 2.0 + 0.5 * np.sin(2 * np.pi * t / DURATION_S)
    positions = np.stack(
        [
            listener[0] + orbit_radius * np.cos(orbit_angle),
            listener[1] + orbit_radius * np.sin(orbit_angle),
            np.full(n, listener[2]),
        ],
        axis=1,
    )
    orbit = SourceTrajectory.from_positions("orbit", signal, positions, listener)

    renderer = SofInityRenderer(
        ring_layout(8, radius_m=2.0, center_m=listener),
        listener_position_m=listener,
        sample_rate_hz=SAMPLE_RATE_HZ,
        air_absorption=AirAbsorptionFilter(SAMPLE_RATE_HZ),
        room=room,
        reflection_order=1,
    )
    result = renderer.render([orbit])

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "sofinity_demo_8ch.wav"
    wavfile.write(wav_path, SAMPLE_RATE_HZ, result.output.T.astype(np.float32))
    metrics_path = output_dir / "sofinity_demo_metrics.json"
    metrics = {
        "sample_rate_hz": result.sample_rate_hz,
        "n_speakers": int(result.output.shape[0]),
        "n_samples": int(result.output.shape[1]),
        **{k: float(v) for k, v in result.metrics.items()},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {wav_path} and {metrics_path}")
    return metrics


if __name__ == "__main__":
    print(main())

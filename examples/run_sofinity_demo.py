"""Render a source orbiting an 8-speaker ring and write speaker feeds + metrics."""

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from engint.spatial_audio import SofInityRenderer, SourceTrajectory, ring_layout

SAMPLE_RATE_HZ = 48_000
DURATION_S = 4.0


def main(output_dir: Path = Path("output_data")) -> dict:
    n = int(SAMPLE_RATE_HZ * DURATION_S)
    t = np.arange(n) / SAMPLE_RATE_HZ
    signal = 0.5 * np.sin(2 * np.pi * 220.0 * t) * np.hanning(n)
    orbit = SourceTrajectory(
        name="orbit",
        signal=signal,
        azimuths_rad=2 * np.pi * t / DURATION_S,
        distances_m=2.0 + 0.5 * np.sin(2 * np.pi * t / DURATION_S),
    )

    renderer = SofInityRenderer(ring_layout(8, radius_m=2.0), sample_rate_hz=SAMPLE_RATE_HZ)
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

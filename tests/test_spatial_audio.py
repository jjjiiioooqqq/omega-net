import numpy as np
import pytest

pytest.importorskip("numpy")

from engint.spatial_audio import (
    RingPanner,
    SofInityRenderer,
    SourceTrajectory,
    SpatiotemporalMultiplexer,
    Speaker,
    SpeakerLayout,
    build_acoustic_map,
    energy_vector_error_rad,
    ring_layout,
)
from engint.spatial_audio.geometry import SPEED_OF_SOUND_M_S


def test_acoustic_map_delays_and_gains():
    layout = SpeakerLayout(
        speakers=[
            Speaker("near", (1.0, 0.0, 0.0)),
            Speaker("far", (0.0, 2.0, 0.0)),
        ]
    )
    amap = build_acoustic_map(layout)
    assert amap.distances_m == pytest.approx([1.0, 2.0])
    assert amap.propagation_delays_s == pytest.approx([1.0 / SPEED_OF_SOUND_M_S, 2.0 / SPEED_OF_SOUND_M_S])
    # Farthest speaker gets zero extra delay; the near one is delayed to match.
    assert amap.alignment_delays_s[1] == pytest.approx(0.0)
    assert amap.alignment_delays_s[0] == pytest.approx(1.0 / SPEED_OF_SOUND_M_S)
    # Level compensation attenuates the near speaker, never amplifies.
    assert amap.compensation_gains == pytest.approx([0.5, 1.0])
    assert amap.azimuths_rad == pytest.approx([0.0, np.pi / 2])


def test_acoustic_map_rejects_listener_on_speaker():
    layout = SpeakerLayout(
        speakers=[Speaker("a", (0.0, 0.0, 0.0)), Speaker("b", (1.0, 0.0, 0.0))]
    )
    with pytest.raises(ValueError):
        build_acoustic_map(layout, listener_position_m=(0.0, 0.0, 0.0))


def test_layout_validation():
    with pytest.raises(ValueError):
        SpeakerLayout(speakers=[Speaker("solo", (1.0, 0.0, 0.0))])
    with pytest.raises(ValueError):
        SpeakerLayout(
            speakers=[Speaker("dup", (1.0, 0.0, 0.0)), Speaker("dup", (0.0, 1.0, 0.0))]
        )


def test_panner_one_hot_at_speaker_direction():
    azimuths = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    panner = RingPanner(azimuths)
    gains = panner.gains(np.pi / 2)
    assert gains == pytest.approx([0.0, 1.0, 0.0, 0.0], abs=1e-12)


def test_panner_equal_power_between_pair():
    azimuths = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    panner = RingPanner(azimuths)
    gains = panner.gains(np.pi / 4)
    assert gains[0] == pytest.approx(gains[1])
    assert np.sum(np.square(gains)) == pytest.approx(1.0)
    assert energy_vector_error_rad(gains, azimuths, np.pi / 4) == pytest.approx(0.0, abs=1e-6)


def test_panner_wraparound_pair():
    azimuths = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    panner = RingPanner(azimuths)
    gains = panner.gains(-np.pi / 4)  # between last speaker and first
    assert gains[3] == pytest.approx(gains[0])
    assert gains[1] == pytest.approx(0.0, abs=1e-12)
    assert gains[2] == pytest.approx(0.0, abs=1e-12)


def test_panner_rejects_coincident_azimuths():
    with pytest.raises(ValueError):
        RingPanner(np.array([0.1, 0.1, 2.0]))


def test_multiplexer_static_azimuth_matches_static_gains():
    layout = ring_layout(4)
    amap = build_acoustic_map(layout)
    panner = RingPanner(amap.azimuths_rad)
    mux = SpatiotemporalMultiplexer(panner, block_size=64)
    signal = np.sin(2 * np.pi * 440 * np.arange(1000) / 48_000)
    feeds = mux.render(signal, np.full(1000, np.pi / 2))
    expected = panner.gains(np.pi / 2)[:, None] * signal[None, :]
    assert feeds == pytest.approx(expected)


def test_multiplexer_moving_source_shifts_energy():
    layout = ring_layout(4)
    amap = build_acoustic_map(layout)
    mux = SpatiotemporalMultiplexer(RingPanner(amap.azimuths_rad), block_size=32)
    n = 2000
    signal = np.ones(n)
    azimuths = np.linspace(0.0, np.pi / 2, n)  # sweep speaker 0 -> speaker 1
    feeds = mux.render(signal, azimuths)
    first_half = np.sum(np.square(feeds[:, : n // 4]), axis=1)
    last_half = np.sum(np.square(feeds[:, -n // 4 :]), axis=1)
    assert first_half[0] > first_half[1]
    assert last_half[1] > last_half[0]


def test_multiplexer_input_validation():
    layout = ring_layout(4)
    amap = build_acoustic_map(layout)
    mux = SpatiotemporalMultiplexer(RingPanner(amap.azimuths_rad))
    with pytest.raises(ValueError):
        mux.render(np.ones(10), np.zeros(9))
    with pytest.raises(ValueError):
        mux.render(np.ones((2, 5)), np.zeros((2, 5)))


def test_renderer_end_to_end_ring():
    layout = ring_layout(8, radius_m=2.0)
    renderer = SofInityRenderer(layout, sample_rate_hz=48_000, block_size=128)
    n = 4800
    t = np.arange(n) / 48_000
    source = SourceTrajectory(
        name="orbit",
        signal=np.sin(2 * np.pi * 220 * t),
        azimuths_rad=np.linspace(0.0, 2 * np.pi, n),
        distances_m=np.full(n, 2.0),
    )
    result = renderer.render([source])
    assert result.output.shape[0] == 8
    assert result.output.shape[1] >= n
    assert np.all(np.isfinite(result.output))
    # Equidistant ring: no alignment padding beyond rounding.
    assert result.acoustic_map.alignment_delays_s == pytest.approx(np.zeros(8))
    # VBAP on a ring keeps the energy centroid within the half-pair angle.
    assert result.metrics["orbit_max_energy_vector_error_rad"] <= np.pi / 8 + 1e-9
    assert result.metrics["peak_abs"] > 0.0


def test_renderer_static_source_lands_on_speaker_channel():
    layout = ring_layout(4, radius_m=1.5)
    renderer = SofInityRenderer(layout, block_size=64)
    n = 1024
    source = SourceTrajectory(
        name="static",
        signal=np.random.default_rng(7).standard_normal(n),
        azimuths_rad=np.full(n, np.pi),  # exactly at speaker 2
        distances_m=np.full(n, 1.0),
    )
    result = renderer.render([source])
    energy = np.sum(np.square(result.output), axis=1)
    assert energy[2] > 0.0
    assert energy[0] == pytest.approx(0.0, abs=1e-18)
    assert energy[1] == pytest.approx(0.0, abs=1e-18)
    assert energy[3] == pytest.approx(0.0, abs=1e-18)


def test_renderer_validates_sources():
    layout = ring_layout(4)
    renderer = SofInityRenderer(layout)
    with pytest.raises(ValueError):
        renderer.render([])
    with pytest.raises(ValueError):
        SourceTrajectory(
            name="bad",
            signal=np.ones(10),
            azimuths_rad=np.zeros(10),
            distances_m=np.zeros(10),  # non-positive distance
        )

import numpy as np
import pytest

pytest.importorskip("numpy")

from engint.spatial_audio import (
    AirAbsorptionFilter,
    ShoeboxRoom,
    SofInityRenderer,
    SourceTrajectory,
    apply_propagation_delay,
    iso9613_alpha_db_per_m,
    ring_layout,
)
from engint.spatial_audio.geometry import SPEED_OF_SOUND_M_S

FS = 48_000


def _dominant_frequency_hz(signal: np.ndarray, fs: int) -> float:
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    return float(np.fft.rfftfreq(signal.size, 1.0 / fs)[np.argmax(spectrum)])


def test_static_distance_is_pure_delay():
    n = 4000
    signal = np.sin(2 * np.pi * 500 * np.arange(n) / FS)
    distance = 3.43  # exactly 480 samples at 343 m/s and 48 kHz
    out = apply_propagation_delay(signal, np.full(n, distance), FS)
    delay = int(round(distance * FS / SPEED_OF_SOUND_M_S))
    assert out[:delay] == pytest.approx(np.zeros(delay), abs=1e-12)
    assert out[delay : delay + n] == pytest.approx(signal, abs=1e-9)


def test_doppler_shift_of_approaching_source():
    # Source closing at 34.3 m/s -> f' = f * c / (c - v) = 1.1 * f.
    duration = 1.0
    n = int(FS * duration)
    t = np.arange(n) / FS
    f0 = 1000.0
    v = 34.3
    distances = 60.0 - v * t  # stays > 25 m
    out = apply_propagation_delay(np.sin(2 * np.pi * f0 * t), distances, FS)
    # Analyze a steady interior window (skip the initial silent delay).
    window = out[n // 2 : n // 2 + 16_384]
    expected = f0 * SPEED_OF_SOUND_M_S / (SPEED_OF_SOUND_M_S - v)
    assert _dominant_frequency_hz(window, FS) == pytest.approx(expected, rel=0.01)


def test_doppler_input_validation():
    with pytest.raises(ValueError):
        apply_propagation_delay(np.ones(10), np.ones(9), FS)
    with pytest.raises(ValueError):
        apply_propagation_delay(np.ones(10), -np.ones(10), FS)


def test_iso9613_alpha_reference_magnitude_and_monotonicity():
    freqs = np.array([500.0, 1000.0, 4000.0, 8000.0])
    alpha = iso9613_alpha_db_per_m(freqs, temperature_c=20.0, relative_humidity_pct=50.0)
    assert np.all(alpha > 0.0)
    assert np.all(np.diff(alpha) > 0.0)  # absorption grows with frequency
    # ISO 9613-1 tabulates ~4.7 dB/km at 1 kHz, 20 C, 50% RH.
    assert alpha[1] * 1000.0 == pytest.approx(4.7, rel=0.25)


def test_air_absorption_filters_highs_more_than_lows():
    n = FS // 2
    t = np.arange(n) / FS
    low = np.sin(2 * np.pi * 500 * t)
    high = np.sin(2 * np.pi * 8000 * t)
    filt = AirAbsorptionFilter(FS)
    distances = np.full(n, 100.0)
    low_ratio = np.std(filt.apply(low, distances)) / np.std(low)
    high_ratio = np.std(filt.apply(high, distances)) / np.std(high)
    assert high_ratio < low_ratio < 1.0
    # At 100 m and 8 kHz (~0.09 dB/m) expect several dB of loss.
    assert 20 * np.log10(high_ratio) < -5.0


def test_air_absorption_near_transparent_at_short_distance():
    n = FS // 4
    signal = np.sin(2 * np.pi * 1000 * np.arange(n) / FS)
    filt = AirAbsorptionFilter(FS)
    out = filt.apply(signal, np.full(n, 0.5))
    assert np.std(out) / np.std(signal) == pytest.approx(1.0, abs=1e-3)


def test_shoebox_first_order_images():
    room = ShoeboxRoom(dimensions_m=(4.0, 5.0, 3.0), wall_absorption=0.36)
    positions = np.array([[1.0, 2.0, 1.5]])
    paths = room.image_paths(positions, max_order=1)
    assert len(paths) == 6
    image_points = sorted(tuple(p.positions_m[0]) for p in paths)
    assert image_points == sorted(
        [
            (-1.0, 2.0, 1.5),
            (7.0, 2.0, 1.5),
            (1.0, -2.0, 1.5),
            (1.0, 8.0, 1.5),
            (1.0, 2.0, -1.5),
            (1.0, 2.0, 4.5),
        ]
    )
    assert all(p.wall_hits == 1 for p in paths)
    assert all(p.gain == pytest.approx(np.sqrt(1.0 - 0.36)) for p in paths)


def test_shoebox_second_order_counts_and_validation():
    room = ShoeboxRoom(dimensions_m=(4.0, 5.0, 3.0))
    positions = np.array([[1.0, 2.0, 1.5]])
    paths = room.image_paths(positions, max_order=2)
    # 6 first-order + (6 axial second-order + 12 cross-axis) = 24 images.
    assert len(paths) == 24
    with pytest.raises(ValueError):
        room.image_paths(np.array([[9.0, 2.0, 1.5]]), max_order=1)
    with pytest.raises(ValueError):
        ShoeboxRoom(dimensions_m=(4.0, -1.0, 3.0))
    with pytest.raises(ValueError):
        ShoeboxRoom(dimensions_m=(4.0, 5.0, 3.0), wall_absorption=0.0)


def test_renderer_with_room_adds_delayed_reflections():
    room = ShoeboxRoom(dimensions_m=(8.0, 8.0, 3.0), wall_absorption=0.3)
    listener = (4.0, 4.0, 1.5)
    layout = ring_layout(8, radius_m=1.5, center_m=listener)
    n = 8000
    rng = np.random.default_rng(11)
    positions = np.tile(np.array([5.5, 4.0, 1.5]), (n, 1))
    source = SourceTrajectory.from_positions("burst", rng.standard_normal(n), positions, listener)

    common = dict(listener_position_m=listener, sample_rate_hz=FS, block_size=128)
    dry = SofInityRenderer(layout, **common).render([source])
    wet = SofInityRenderer(layout, room=room, reflection_order=1, **common).render([source])

    assert wet.metrics["burst_image_count"] == 6.0
    wet_energy = float(np.sum(np.square(wet.output)))
    dry_energy = float(np.sum(np.square(dry.output)))
    assert wet_energy > dry_energy  # reflections add energy
    assert wet.output.shape[1] > dry.output.shape[1]  # later arrivals extend the tail


def test_renderer_room_requires_positions():
    room = ShoeboxRoom(dimensions_m=(8.0, 8.0, 3.0))
    layout = ring_layout(4)
    renderer = SofInityRenderer(
        layout, listener_position_m=(4.0, 4.0, 1.5), room=room
    )
    n = 512
    bare = SourceTrajectory(
        name="bare",
        signal=np.ones(n),
        azimuths_rad=np.zeros(n),
        distances_m=np.full(n, 1.0),
    )
    with pytest.raises(ValueError, match="positions_m"):
        renderer.render([bare])


def test_renderer_rejects_listener_outside_room():
    room = ShoeboxRoom(dimensions_m=(4.0, 4.0, 3.0))
    with pytest.raises(ValueError, match="inside the room"):
        SofInityRenderer(ring_layout(4), listener_position_m=(9.0, 1.0, 1.0), room=room)


def test_full_stack_render_is_finite_and_attenuated():
    room = ShoeboxRoom(dimensions_m=(10.0, 10.0, 4.0), wall_absorption=0.4)
    listener = (5.0, 5.0, 1.5)
    layout = ring_layout(8, radius_m=2.0, center_m=listener)
    n = FS // 2
    t = np.arange(n) / FS
    # Orbiting source 3 m from the listener.
    positions = np.stack(
        [
            5.0 + 3.0 * np.cos(2 * np.pi * t),
            5.0 + 3.0 * np.sin(2 * np.pi * t),
            np.full(n, 1.5),
        ],
        axis=1,
    )
    source = SourceTrajectory.from_positions(
        "orbit", 0.25 * np.sin(2 * np.pi * 440 * t), positions, listener
    )
    renderer = SofInityRenderer(
        layout,
        listener_position_m=listener,
        sample_rate_hz=FS,
        air_absorption=AirAbsorptionFilter(FS),
        room=room,
        reflection_order=2,
    )
    result = renderer.render([source])
    assert np.all(np.isfinite(result.output))
    assert result.metrics["orbit_image_count"] == 24.0
    assert result.metrics["peak_abs"] > 0.0
